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


def plot_learning_curves(run_dirs: list[Path], out_path: Path) -> Path:
    """Plot per-seed learning curves with a mean band.

    Args:
        run_dirs: Run directories, one per seed.
        out_path: Destination png path.

    Returns:
        The written plot path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    curves = []
    for run_dir in run_dirs:
        steps, means = _load_curve(run_dir)
        seed = _run_seed(run_dir)
        ax.plot(steps, means, alpha=0.5, linewidth=1, label=f"seed {seed}")
        curves.append((steps, means))

    # Aggregate onto the shortest common grid (same eval cadence per seed).
    n_common = min(len(c[0]) for c in curves)
    if n_common > 1 and len(curves) > 1:
        grid = curves[0][0][:n_common]
        stacked = np.stack([c[1][:n_common] for c in curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        ax.plot(grid, mean, color="black", linewidth=2, label="mean")
        ax.fill_between(grid, mean - std, mean + std, color="black", alpha=0.15)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval score (mean over eval episodes)")
    ax.set_title("Learning curves, deterministic periodic evaluation")
    ax.legend()
    ax.grid(alpha=0.3)
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
