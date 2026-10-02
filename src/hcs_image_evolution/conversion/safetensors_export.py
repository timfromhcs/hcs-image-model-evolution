"""SafeTensors weight serialization with embedded provenance metadata."""

from pathlib import Path

from safetensors.torch import save_file
from torch import nn

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import log_event, logger


class SafeTensorsExporter:
    """Exports model weights to SafeTensors format with embedded model card metadata."""

    @staticmethod
    def export(
        model: nn.Module,
        output_path: Path,
        metadata: dict[str, str] | None = None,
    ) -> tuple[Path, str]:
        """Saves model state dict into safetensors and computes SHA-256."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()

        # Ensure all tensors are contiguous on CPU
        cpu_state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        string_metadata = {k: str(v) for k, v in (metadata or {}).items()}

        save_file(cpu_state_dict, str(output_path), metadata=string_metadata)
        sha256 = compute_file_sha256(output_path)
        log_event("safetensors_exported", {"file": output_path.name, "sha256": sha256[:8]})
        logger.info("Exported SafeTensors artifact: %s (SHA256: %s)", output_path, sha256[:8])
        return output_path, sha256
