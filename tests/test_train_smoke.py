"""End-to-end training smoke test with a tiny budget."""

from pathlib import Path

from reinforced_flapper.config import (
    DQNConfig,
    EnvConfig,
    EvalProtocolConfig,
    RunConfig,
    TrainConfig,
)
from reinforced_flapper.ledger import load_rows
from reinforced_flapper.train import train_run


def test_train_run_end_to_end(tmp_path: Path) -> None:
    """A tiny training run produces a run dir, models, curve, and ledger row."""
    config = RunConfig(
        name="smoke",
        env=EnvConfig(max_episode_steps=200),
        dqn=DQNConfig(
            buffer_size=2_000,
            learning_starts=200,
            batch_size=32,
            net_arch=[32, 32],
            target_update_interval=200,
        ),
        train=TrainConfig(
            total_timesteps=1_200,
            seed=0,
            eval_every_steps=600,
            eval_episodes=2,
        ),
        evaluation=EvalProtocolConfig(n_episodes=3, seed_base=42),
    )

    ledger_path = tmp_path / "ledger.jsonl"
    run_dir, result = train_run(
        config, runs_dir=tmp_path / "runs", ledger_path=ledger_path
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "curve.jsonl").exists()
    assert (run_dir / "final_eval.json").exists()
    assert (run_dir / "run.log").exists()
    assert len(result.scores) == 3

    rows = load_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0].kind == "train"
    assert rows[0].config_hash == config.config_hash()
    assert "det_score_mean" in rows[0].metrics
    assert "sto_score_mean" in rows[0].metrics
