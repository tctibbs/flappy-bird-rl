"""Append-only JSONL results ledger.

The ledger is the source of truth for every reported number. Each row
carries the run id, git SHA, config hash, seed, and full config, so any
result traces back to the exact code and settings that produced it.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_LEDGER_PATH = Path("results/ledger.jsonl")


class LedgerRow(BaseModel):
    """One result row.

    Attributes:
        run_id: Unique run id.
        kind: Whether the row records a training run or a standalone eval.
        timestamp: UTC ISO timestamp when the row was written.
        git_sha: Git commit the code ran at.
        config_hash: Hash of the config (excluding seed).
        config_name: Human-readable config name.
        seed: Training seed (or evaluation seed base for standalone evals).
        frames: Environment steps consumed (training) or played (eval).
        wall_clock_s: Wall-clock duration in seconds.
        metrics: Flat metric names to values.
        model_path: Path to the model the row refers to, if any.
        config: Full config dump, so the row survives without the YAML file.
        notes: Free-form context.
    """

    run_id: str
    kind: Literal["train", "eval"]
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    git_sha: str
    config_hash: str
    config_name: str
    seed: int
    frames: int
    wall_clock_s: float
    metrics: dict[str, float]
    model_path: str | None = None
    config: dict[str, Any]
    notes: str = ""


def git_sha() -> str:
    """Return the current git commit SHA, or "unknown" outside a repo.

    Returns:
        The full commit SHA or "unknown".
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def append_row(row: LedgerRow, path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append a row to the ledger.

    Args:
        row: The row to append.
        path: Ledger file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row.model_dump(mode="json")) + "\n")


def load_rows(path: Path = DEFAULT_LEDGER_PATH) -> list[LedgerRow]:
    """Load all rows from the ledger.

    Args:
        path: Ledger file path.

    Returns:
        All rows, oldest first. Empty if the ledger does not exist.
    """
    if not path.exists():
        return []
    with path.open() as f:
        return [
            LedgerRow.model_validate(json.loads(line)) for line in f if line.strip()
        ]
