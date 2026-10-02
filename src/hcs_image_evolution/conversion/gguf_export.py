"""GGUF model exporter for stable-diffusion.cpp compatibility."""

import struct
from pathlib import Path
from typing import Any

import torch

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import log_event, logger


class GGUFExporter:
    """Exports diffusion models to GGUF container format conforming to GGML/sd.cpp standards."""

    GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
    GGUF_VERSION = 3

    # GGUF Value Types
    TYPE_UINT32 = 4
    TYPE_INT32 = 5
    TYPE_FLOAT32 = 6
    TYPE_STRING = 8

    # GGML Tensor Types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q4_0 = 2

    @classmethod
    def export_state_dict_to_gguf(
        cls,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        architecture_name: str = "qwen_image",
        model_name: str = "hcs-image-evolver-4step",
    ) -> tuple[Path, str]:
        """Serializes PyTorch tensors and metadata into a valid GGUF binary format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.contiguous().cpu() for k, v in state_dict.items()}

        metadata: dict[str, tuple[int, Any]] = {
            "general.architecture": (cls.TYPE_STRING, architecture_name),
            "general.name": (cls.TYPE_STRING, model_name),
            "general.file_type": (cls.TYPE_UINT32, cls.GGML_TYPE_F16),
        }

        with open(output_path, "wb") as f:
            # 1. Header: Magic, Version, Tensor Count, Metadata KV Count
            f.write(struct.pack("<I", cls.GGUF_MAGIC))
            f.write(struct.pack("<I", cls.GGUF_VERSION))
            f.write(struct.pack("<Q", len(tensors)))
            f.write(struct.pack("<Q", len(metadata)))

            # 2. Metadata KV Pairs
            for key, (val_type, val) in metadata.items():
                k_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(k_bytes)))
                f.write(k_bytes)
                f.write(struct.pack("<I", val_type))
                if val_type == cls.TYPE_STRING:
                    v_bytes = str(val).encode("utf-8")
                    f.write(struct.pack("<Q", len(v_bytes)))
                    f.write(v_bytes)
                elif val_type == cls.TYPE_UINT32:
                    f.write(struct.pack("<I", int(val)))

            # 3. Tensor Info and Offset Table
            tensor_headers = []
            current_offset = 0

            for name, tensor in tensors.items():
                name_bytes = name.encode("utf-8")
                n_dims = len(tensor.shape)
                shape = list(reversed(tensor.shape)) # GGUF stores dimensions in reverse order
                # Convert to F16
                t_f16 = tensor.to(torch.float16).numpy().tobytes()
                size_bytes = len(t_f16)

                tensor_headers.append((name_bytes, n_dims, shape, cls.GGML_TYPE_F16, current_offset, t_f16))
                # Align offset to 32 bytes
                current_offset += (size_bytes + 31) & ~31

            for name_bytes, n_dims, shape, ggml_type, offset, _ in tensor_headers:
                f.write(struct.pack("<Q", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", n_dims))
                f.writelines(struct.pack("<Q", dim) for dim in shape)
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", offset))

            # 4. Alignment padding before tensor data (32-byte alignment)
            pos = f.tell()
            pad = ((pos + 31) & ~31) - pos
            if pad > 0:
                f.write(b"\x00" * pad)

            # 5. Tensor Binary Data
            for _, _, _, _, _, raw_data in tensor_headers:
                f.write(raw_data)
                pos = f.tell()
                pad = ((pos + 31) & ~31) - pos
                if pad > 0:
                    f.write(b"\x00" * pad)

        sha256 = compute_file_sha256(output_path)
        log_event("gguf_exported", {"file": output_path.name, "tensors": len(tensors), "sha256": sha256[:8]})
        logger.info("Successfully exported GGUF artifact: %s (SHA256: %s)", output_path, sha256[:8])
        return output_path, sha256
