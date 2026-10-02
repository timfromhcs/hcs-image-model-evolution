"""Atomic file operations and safe pointer management."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_bytes(target_path: str | Path, data: bytes) -> None:
    """Writes bytes to a temporary file and atomically renames it to target_path."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target.parent
    with tempfile.NamedTemporaryFile("wb", dir=temp_dir, delete=False) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name

    shutil.move(temp_name, target)


def atomic_write_json(target_path: str | Path, data: Any, indent: int = 2) -> None:
    """Atomically serializes data to a JSON file."""
    encoded = json.dumps(data, indent=indent, default=str).encode("utf-8")
    atomic_write_bytes(target_path, encoded)


def update_pointer(pointer_file: str | Path, checkpoint_dir: str | Path, metadata: dict[str, Any]) -> None:
    """Atomically updates a pointer file (e.g. latest.json, best.json)."""
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "target_name": Path(checkpoint_dir).name,
        "metadata": metadata,
    }
    atomic_write_json(pointer_file, payload)
