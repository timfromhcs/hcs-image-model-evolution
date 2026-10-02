"""Unit tests for GGUF exporter binary structure."""

import struct
from pathlib import Path

import torch

from hcs_image_evolution.conversion.gguf_export import GGUFExporter


def test_gguf_export_header_and_checksum(tmp_path: Path):
    out_file = tmp_path / "model.gguf"
    state_dict = {
        "weight_a": torch.randn(4, 4),
        "bias_b": torch.zeros(4),
    }

    path, sha = GGUFExporter.export_state_dict_to_gguf(state_dict, out_file)
    assert path.exists()
    assert len(sha) == 64

    # Validate header magic
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        assert magic == GGUFExporter.GGUF_MAGIC
        assert version == GGUFExporter.GGUF_VERSION
