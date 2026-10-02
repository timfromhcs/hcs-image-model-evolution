"""Unit tests for cryptographic SHA-256 and dictionary hashing."""

from pathlib import Path

from hcs_image_evolution.utils.hashing import (
    compute_bytes_sha256,
    compute_dict_hash,
    compute_file_sha256,
)


def test_bytes_and_file_hashing(tmp_path: Path):
    test_data = b"HCS Autonomous Evolution Lab Test Payload"
    expected_hash = compute_bytes_sha256(test_data)

    test_file = tmp_path / "test.bin"
    test_file.write_bytes(test_data)

    file_hash = compute_file_sha256(test_file)
    assert file_hash == expected_hash
    assert len(file_hash) == 64


def test_dict_hash_determinism():
    dict1 = {"b": 2, "a": 1, "c": [3, 4]}
    dict2 = {"a": 1, "c": [3, 4], "b": 2}
    assert compute_dict_hash(dict1) == compute_dict_hash(dict2)
