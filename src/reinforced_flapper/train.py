"""Training loop: config in, run directory and ledger row out.

Each run gets a directory under runs/<run_id> containing the exact config,
a structured log, the Monitor episode log, the periodic-evaluation learning
curve, and the best and final models. The headline result is appended to
the results ledger.
"""

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from reinforced_flapper.config import RunConfig
from reinforced_flapper.env import FlappyBirdEnv
from reinforced_flapper.evaluate import EvalResult, evaluate_model
from reinforced_flapper.ledger import LedgerRow, append_row, git_sha
from reinforced_flapper.runlog import new_run_id, setup_run_logging

if TYPE_CHECKING:
    from loguru import Logger

DEFAULT_RUNS_DIR = Path("runs")


class PeriodicScoreEval(BaseCallback):
    """Periodically evaluate the live model and keep the best checkpoint.

    Evaluation uses game score under the fixed-seed protocol (a shortened
    episode count for speed), writes one learning-curve row per evaluation
    to curve.jsonl, and saves the model whenever mean score improves.
    """

    def __init__(self, config: RunConfig, run_dir: Path, log: "Logger") -> None:
        """Initialize the callback.

        Args:
            config: The run configuration.
            run_dir: The run directory.
            log: Bound loguru logger.
        """
        super().__init__()
        self._config = config
        self._run_dir = run_dir
        self._log = log
        self._eval_env = FlappyBirdEnv(config.env)
        self._next_eval = config.train.eval_every_steps
        self.best_mean_score = float("-inf")

    def _on_step(self) -> bool:
        """Evaluate when the step threshold is crossed.

        Returns:
            True to continue training.
        """
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self._config.train.eval_every_steps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Run one periodic evaluation and checkpoint if improved."""
        protocol = self._config.evaluation.model_copy(
            update={"n_episodes": self._config.train.eval_episodes}
        )
        result = evaluate_model(
            self.model,
            self._config.env,
            protocol,
            deterministic=True,
            env=self._eval_env,
        )
        summary = result.summary()
        point = {"timesteps": self.num_timesteps, **summary}
        with (self._run_dir / "curve.jsonl").open("a") as f:
            f.write(json.dumps(point) + "\n")

        improved = summary["score_mean"] > self.best_mean_score
        if improved:
            self.best_mean_score = summary["score_mean"]
            self.model.save(str(self._run_dir / "best_model"))

        self._log.info(
            "eval at {steps} steps: score mean {mean:.2f} median {median:.1f} "
            "max {max:.0f}{best}",
            steps=self.num_timesteps,
            mean=summary["score_mean"],
            median=summary["score_median"],
            max=summary["score_max"],
            best=" (new best, checkpointed)" if improved else "",
        )

    def _on_training_end(self) -> None:
        """Close the evaluation environment."""
        self._eval_env.close()


def train_run(
    config: RunConfig,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    ledger_path: Path | None = None,
) -> tuple[Path, EvalResult]:
    """Train one agent and record the result.

    Args:
        config: The run configuration.
        runs_dir: Directory that run directories are created under.
        ledger_path: Ledger to append the result row to. None uses the
            default ledger.

    Returns:
        Tuple of (run directory, deterministic eval result of the best model).
    """
    run_id = new_run_id()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log = setup_run_logging(run_id, run_dir / "run.log")

    config.to_yaml(run_dir / "config.yaml")
    sha = git_sha()
    log.info(
        "starting training: name={name} config_hash={hash} seed={seed} "
        "timesteps={steps} git={sha}",
        name=config.name,
        hash=config.config_hash(),
        seed=config.train.seed,
        steps=config.train.total_timesteps,
        sha=sha[:10],
    )

    set_random_seed(config.train.seed)
    env = Monitor(FlappyBirdEnv(config.env), str(run_dir / "monitor.csv"))

    dqn = config.dqn
    model = DQN(
        dqn.policy,
        env,
        seed=config.train.seed,
        verbose=0,
        learning_rate=dqn.learning_rate,
        buffer_size=dqn.buffer_size,
        learning_starts=dqn.learning_starts,
        batch_size=dqn.batch_size,
        gamma=dqn.gamma,
        train_freq=dqn.train_freq,
        gradient_steps=dqn.gradient_steps,
        target_update_interval=dqn.target_update_interval,
        exploration_fraction=dqn.exploration_fraction,
        exploration_final_eps=dqn.exploration_final_eps,
        policy_kwargs={"net_arch": list(dqn.net_arch)},
    )

    callback = PeriodicScoreEval(config, run_dir, log)
    start = time.time()
    model.learn(total_timesteps=config.train.total_timesteps, callback=callback)
    wall_clock = time.time() - start
    model.save(str(run_dir / "final_model"))
    env.close()

    best_path = run_dir / "best_model.zip"
    final_path = run_dir / "final_model.zip"
    eval_path = best_path if best_path.exists() else final_path
    log.info("training done in {s:.0f}s, evaluating {p}", s=wall_clock, p=eval_path)

    best_model = DQN.load(str(eval_path))
    det = evaluate_model(best_model, config.env, config.evaluation, deterministic=True)
    sto = evaluate_model(best_model, config.env, config.evaluation, deterministic=False)
    det_summary = det.summary()
    sto_summary = sto.summary()

    (run_dir / "final_eval.json").write_text(
        json.dumps(
            {"deterministic": det.model_dump(), "stochastic": sto.model_dump()},
            indent=2,
        )
    )

    metrics = {f"det_{k}": v for k, v in det_summary.items()}
    metrics.update({f"sto_{k}": v for k, v in sto_summary.items()})
    row = LedgerRow(
        run_id=run_id,
        kind="train",
        git_sha=sha,
        config_hash=config.config_hash(),
        config_name=config.name,
        seed=config.train.seed,
        frames=config.train.total_timesteps,
        wall_clock_s=round(wall_clock, 1),
        metrics=metrics,
        model_path=str(eval_path),
        config=config.model_dump(mode="json"),
    )
    if ledger_path is None:
        append_row(row)
    else:
        append_row(row, ledger_path)

    log.info(
        "final eval (best model, deterministic, n={n}): score mean {mean:.2f} "
        "median {median:.1f} min {min:.0f} max {max:.0f} capped {capped:.0f}",
        n=det.n_episodes,
        mean=det_summary["score_mean"],
        median=det_summary["score_median"],
        min=det_summary["score_min"],
        max=det_summary["score_max"],
        capped=det_summary["episodes_capped"],
    )
    return run_dir, det
