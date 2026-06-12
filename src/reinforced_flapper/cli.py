"""Command-line interface for playing, training, evaluating, and rendering."""

import argparse
from pathlib import Path
import time

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from reinforced_flapper.config import EnvConfig, RunConfig
from reinforced_flapper.env import FlappyBirdEnv
from reinforced_flapper.ledger import LedgerRow, append_row, git_sha, load_rows
from reinforced_flapper.runlog import new_run_id, setup_run_logging

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _load_config(args: argparse.Namespace) -> RunConfig:
    """Load the run config, applying CLI overrides.

    Args:
        args: Parsed CLI arguments with config and optional seed.

    Returns:
        The validated run config.
    """
    config = RunConfig.from_yaml(args.config)
    if getattr(args, "seed", None) is not None:
        config.train.seed = args.seed
    return config


def cmd_play(_args: argparse.Namespace) -> None:
    """Run the game in human playable mode."""
    env = FlappyBirdEnv(render_mode="human")
    env.reset()

    running = True
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
                break
            is_flap_key = event.type == KEYDOWN and event.key in (K_SPACE, K_UP)
            if is_flap_key or event.type == pygame.MOUSEBUTTONDOWN:
                action = 1

        if not running:
            break

        _obs, _reward, terminated, _truncated, _info = env.step(action)
        if terminated:
            env.reset()

    env.close()


def cmd_train(args: argparse.Namespace) -> None:
    """Train one agent from a config."""
    from reinforced_flapper.train import train_run

    config = _load_config(args)
    train_run(config, runs_dir=Path(args.runs_dir))


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model or the random baseline, and append to the ledger."""
    from stable_baselines3 import DQN

    from reinforced_flapper.evaluate import evaluate_model, evaluate_random

    config = _load_config(args)
    protocol = config.evaluation
    if args.episodes is not None:
        protocol = protocol.model_copy(update={"n_episodes": args.episodes})

    run_id = new_run_id()
    log = setup_run_logging(run_id)
    start = time.time()

    if args.policy == "random":
        result = evaluate_random(config.env, protocol)
        model_path = None
    else:
        if args.model is None:
            raise SystemExit("evaluate: --model is required when --policy=model")
        model_path = args.model
        model = DQN.load(str(Path(model_path)))
        result = evaluate_model(
            model, config.env, protocol, deterministic=not args.stochastic
        )

    summary = result.summary()
    log.info(
        "{policy} ({mode}, n={n}): score mean {mean:.2f} median {median:.1f} "
        "min {min:.0f} max {max:.0f} capped {capped:.0f}",
        policy=result.policy,
        mode="deterministic" if result.deterministic else "stochastic",
        n=result.n_episodes,
        mean=summary["score_mean"],
        median=summary["score_median"],
        min=summary["score_min"],
        max=summary["score_max"],
        capped=summary["episodes_capped"],
    )

    if not args.no_ledger:
        prefix = "det_" if result.deterministic else "sto_"
        row = LedgerRow(
            run_id=run_id,
            kind="eval",
            git_sha=git_sha(),
            config_hash=config.config_hash(),
            config_name=config.name,
            seed=protocol.seed_base,
            frames=sum(result.lengths),
            wall_clock_s=round(time.time() - start, 1),
            metrics={f"{prefix}{k}": v for k, v in summary.items()},
            model_path=model_path,
            config=config.model_dump(mode="json"),
            notes=f"standalone eval, policy={result.policy}",
        )
        append_row(row)
        log.info("ledger row appended for run {id}", id=run_id)


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch a trained model play with rendering."""
    from stable_baselines3 import DQN

    model = DQN.load(str(Path(args.model)))
    env = FlappyBirdEnv(render_mode="human")
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode if args.seed else None)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                print(f"episode {episode + 1}: score {info['score']}")
                break
    env.close()


def cmd_video(args: argparse.Namespace) -> None:
    """Record a model playing to an mp4 file."""
    from reinforced_flapper.video import record_video

    out = record_video(
        args.model,
        args.out,
        EnvConfig(),
        n_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"video written to {out}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate learning-curve and final-score plots."""
    from reinforced_flapper.plots import plot_final_scores, plot_learning_curves

    run_dirs = [Path(d) for d in args.runs]
    out_dir = Path(args.out_dir)
    curve_path = plot_learning_curves(run_dirs, out_dir / "learning_curves.png")
    print(f"plot written to {curve_path}")

    rows = [
        row
        for row in load_rows()
        if row.kind == "train" and any(row.run_id == d.name for d in run_dirs)
    ]
    if rows:
        baselines = {"random": 0.0}
        scores_path = plot_final_scores(
            rows, baselines, out_dir / "final_scores.png", target=args.target
        )
        print(f"plot written to {scores_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="flapper", description="Flappy Bird reinforcement learning pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("play", help="play the game yourself").set_defaults(func=cmd_play)

    p_train = sub.add_parser("train", help="train an agent from a config")
    p_train.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_train.add_argument("--seed", type=int, default=None, help="override config seed")
    p_train.add_argument("--runs-dir", default="runs")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a model or baseline")
    p_eval.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p_eval.add_argument("--model", default=None, help="path to a saved model")
    p_eval.add_argument(
        "--policy",
        choices=["model", "random"],
        default="model",
        help="what to evaluate",
    )
    p_eval.add_argument(
        "--stochastic",
        action="store_true",
        help="epsilon-greedy at the model's saved exploration rate",
    )
    p_eval.add_argument("--episodes", type=int, default=None)
    p_eval.add_argument("--no-ledger", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_watch = sub.add_parser("watch", help="watch a trained model play")
    p_watch.add_argument("--model", required=True)
    p_watch.add_argument("--episodes", type=int, default=5)
    p_watch.add_argument("--seed", type=int, default=None)
    p_watch.set_defaults(func=cmd_watch)

    p_video = sub.add_parser("video", help="record a model playing to mp4")
    p_video.add_argument("--model", required=True)
    p_video.add_argument("--out", default="videos/eval.mp4")
    p_video.add_argument("--episodes", type=int, default=1)
    p_video.add_argument("--seed", type=int, default=10_000)
    p_video.add_argument("--max-steps", type=int, default=3_000)
    p_video.set_defaults(func=cmd_video)

    p_plot = sub.add_parser("plot", help="generate plots from runs")
    p_plot.add_argument("--runs", nargs="+", required=True, help="run directories")
    p_plot.add_argument("--out-dir", default="plots")
    p_plot.add_argument("--target", type=float, default=None)
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
