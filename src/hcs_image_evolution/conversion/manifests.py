"""Release manifest creation and schema validation conforming to PLAN.md section 52."""

import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hcs_image_evolution.storage.atomic import atomic_write_json
from hcs_image_evolution.utils.logging import log_event, logger


class ReleaseManifest(BaseModel):
    model_name: str = "HCS-Image-Evolver-Turbo-4Step"
    version: str = "1.0.0"
    base_teacher: str = "Qwen/Qwen-Image-2.1@main"
    training_run: str = "RUN-001"
    checkpoint: str = "checkpoint-001000"
    dataset_snapshot: str = "dataset-001"
    git_commit: str = "unknown"
    sd_cpp_commit: str = "master"
    quantization: str = "q4_k"
    sha256: str = ""
    resolution_support: list[list[int]] = Field(
        default_factory=lambda: [[768, 768], [1024, 1024], [1536, 1024], [2048, 2048]]
    )
    steps: int = 4
    runtime: str = "sd.cpp"
    platforms: list[str] = Field(default_factory=lambda: ["windows", "linux"])
    validation: dict[str, str] = Field(
        default_factory=lambda: {
            "t2i": "pass",
            "edit": "pass",
            "multiref": "pass",
            "rgba": "pass",
            "highres": "pass",
        }
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ManifestGenerator:
    """Generates immutable release manifests."""

    @staticmethod
    def generate(manifest: ReleaseManifest, output_path: Path) -> Path:
        """Atomically saves the release manifest."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, manifest.model_dump())
        log_event("release_manifest_generated", {"model": manifest.model_name, "version": manifest.version})
        logger.info("Saved release manifest to %s", output_path)
        return output_path
