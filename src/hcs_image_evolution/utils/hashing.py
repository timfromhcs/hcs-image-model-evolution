"""Cryptographic and dataset hashing utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes the SHA-256 hash of a file efficiently using chunking."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Computes the SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_dict_hash(data: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash for a dictionary structure."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
