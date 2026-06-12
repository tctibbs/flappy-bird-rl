"""Structured logging with a run id bound to every line."""

from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import uuid

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[run_id]}</cyan> | "
    "{message}"
)


def new_run_id() -> str:
    """Create a sortable, unique run id.

    Returns:
        A timestamp plus short random suffix, e.g. 20260612-153045-a1b2.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


def setup_run_logging(run_id: str, log_file: Path | None = None) -> "Logger":
    """Configure loguru for a run and return a bound logger.

    Args:
        run_id: The run id to bind to every line.
        log_file: Optional file to also write the log to.

    Returns:
        A logger with the run id bound.
    """
    logger.remove()
    logger.add(sys.stderr, format=_FORMAT)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, format=_FORMAT, colorize=False)
    return logger.bind(run_id=run_id)
