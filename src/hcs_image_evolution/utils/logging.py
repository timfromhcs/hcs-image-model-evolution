"""Structured event logging and error handling conforming to PLAN.md section 3.2."""

import datetime
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("hcs_image_evolution")


def setup_logger(log_dir: Path = Path("state"), level: int = logging.INFO) -> logging.Logger:
    """Configures structured console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
        logger.addHandler(console)

        file_handler = logging.FileHandler(log_dir / "evolution.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def log_event(
    event_type: str,
    payload: dict[str, Any],
    events_file: Path = Path("state/events.jsonl"),
) -> None:
    """Appends an event to the append-only events.jsonl ledger."""
    events_file.parent.mkdir(parents=True, exist_ok=True)
    event_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "type": event_type,
        "payload": payload,
    }
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_record, default=str) + "\n")


def record_error(
    stage: str,
    run_id: str,
    error: Exception,
    git_commit: str | None = None,
    last_checkpoint: str | None = None,
    recoverable: bool = True,
    retry_count: int = 0,
    errors_dir: Path = Path("state/errors"),
) -> dict[str, Any]:
    """Records an explicit, structured error report to state/errors/."""
    errors_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    trace_path = errors_dir / f"trace_{timestamp}.txt"

    with open(trace_path, "w", encoding="utf-8") as f:
        traceback.print_exception(type(error), error, error.__traceback__, file=f)

    error_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": stage,
        "run_id": run_id,
        "git_commit": git_commit or "unknown",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback_path": str(trace_path),
        "last_checkpoint": last_checkpoint,
        "recoverable": recoverable,
        "retry_count": retry_count,
    }

    report_path = errors_dir / f"error_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, indent=2, default=str)

    log_event("stage_error", error_data)
    logger.error("Error recorded in %s: %s", stage, error)
    return error_data
