"""Ledger tests: append and load round trip."""

from pathlib import Path

from reinforced_flapper.config import RunConfig
from reinforced_flapper.ledger import LedgerRow, append_row, load_rows


def _row(run_id: str, seed: int) -> LedgerRow:
    """Build a minimal valid row."""
    config = RunConfig()
    return LedgerRow(
        run_id=run_id,
        kind="train",
        git_sha="deadbeef",
        config_hash=config.config_hash(),
        config_name=config.name,
        seed=seed,
        frames=1000,
        wall_clock_s=1.5,
        metrics={"det_score_mean": 12.0},
        config=config.model_dump(mode="json"),
    )


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    """Rows read back equal the rows written."""
    path = tmp_path / "ledger.jsonl"
    rows = [_row("run-a", 0), _row("run-b", 1)]
    for row in rows:
        append_row(row, path)

    loaded = load_rows(path)
    assert len(loaded) == 2
    assert loaded[0].run_id == "run-a"
    assert loaded[1].seed == 1
    assert loaded[0].metrics["det_score_mean"] == 12.0
    assert loaded[0].config["env"]["reward_alive"] == 1.0


def test_load_missing_ledger_is_empty(tmp_path: Path) -> None:
    """A missing ledger loads as an empty list."""
    assert load_rows(tmp_path / "nope.jsonl") == []
