"""Plots built from run directories and the ledger."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reinforced_flapper.ledger import LedgerRow


def _load_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a run's learning curve.

    Args:
        run_dir: The run directory containing curve.jsonl.

    Returns:
        Tuple of (timesteps, mean scores).
    """
    points = [
        json.loads(line)
        for line in (run_dir / "curve.jsonl").read_text().splitlines()
        if line.strip()
    ]
    steps = np.array([p["timesteps"] for p in points])
    means = np.array([p["score_mean"] for p in points])
    return steps, means


SEED_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3")
PERFECT_SCORE = 255


def plot_learning_curves(run_dirs: list[Path], out_path: Path) -> Path:
    """Plot best-checkpoint-so-far curves per seed, raw evals faint behind.

    The live DQN policy oscillates, so the curve that matters (and the one
    the pipeline actually uses for checkpoint selection) is the running
    best of the periodic evaluations.

    Args:
        run_dirs: Run directories, one per seed.
        out_path: Destination png path.

    Returns:
        The written plot path.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for run_dir, color in zip(run_dirs, SEED_COLORS, strict=False):
        steps, means = _load_curve(run_dir)
        seed = _run_seed(run_dir)
        millions = steps / 1e6
        ax.plot(millions, means, color=color, alpha=0.22, linewidth=1)
        ax.plot(
            millions,
            np.maximum.accumulate(means),
            color=color,
            linewidth=2.2,
            label=f"seed {seed}",
        )

    ax.axhline(PERFECT_SCORE, color="#999999", linestyle="--", linewidth=1, zorder=0)
    ax.text(
        0.02,
        PERFECT_SCORE - 6,
        "perfect play (eval ceiling)",
        color="#777777",
        fontsize=9,
        va="top",
    )

    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Eval score (pipes passed)")
    ax.set_title("Best checkpoint so far (bold), raw periodic evals (faint)")
    ax.set_ylim(bottom=-8)
    ax.legend(frameon=False, loc="center left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_final_scores(
    rows: list[LedgerRow],
    baselines: dict[str, float],
    out_path: Path,
    target: float | None = None,
) -> Path:
    """Plot final deterministic scores per seed against baselines.

    Args:
        rows: Training ledger rows to plot.
        baselines: Label to score for reference policies.
        out_path: Destination png path.
        target: Optional solved-target line.

    Returns:
        The written plot path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [f"seed {row.seed}" for row in rows] + list(baselines)
    values = [row.metrics["det_score_mean"] for row in rows] + list(baselines.values())
    colors = ["tab:blue"] * len(rows) + ["tab:gray"] * len(baselines)
    ax.bar(labels, values, color=colors)

    if target is not None:
        ax.axhline(target, color="tab:red", linestyle="--", label=f"target {target:g}")
        ax.legend()

    ax.set_ylabel("Mean score (deterministic, fixed protocol)")
    ax.set_title("Final evaluation scores")
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _run_seed(run_dir: Path) -> int:
    """Read the training seed from a run directory's config.

    Args:
        run_dir: The run directory.

    Returns:
        The training seed.
    """
    import yaml

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return int(config["train"]["seed"])
