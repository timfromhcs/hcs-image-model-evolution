"""Dataset provenance tracking and immutable record generation."""

from pathlib import Path
from typing import Optional
from PIL import Image

from hcs_image_evolution.data.schemas import SampleRecord
from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import log_event


class ProvenanceTracker:
    """Creates fully traceable SampleRecord instances with SHA256 hashes and metadata."""

    @staticmethod
    def create_record(
        sample_id: str,
        image_path: Path,
        source_prompt: str,
        caption: str,
        source_type: str = "synthetic",
        source_dataset: Optional[str] = None,
        source_model: Optional[str] = "Qwen/Qwen-Image-2.1",
        source_revision: Optional[str] = "main",
        seed: int = 42,
        license_str: str = "apache-2.0",
        judge_scores: Optional[dict[str, float]] = None,
        dataset_snapshot: str = "snapshot-001",
        edit_instruction: Optional[str] = None,
        reference_images: Optional[list[str]] = None,
    ) -> SampleRecord:
        """Constructs an immutable provenance-verified SampleRecord."""
        with Image.open(image_path) as img:
            width, height = img.size

        sha256_hash = compute_file_sha256(image_path)
        record = SampleRecord(
            sample_id=sample_id,
            source_type=source_type,
            source_dataset=source_dataset,
            source_model=source_model,
            source_revision=source_revision,
            source_prompt=source_prompt,
            caption=caption,
            edit_instruction=edit_instruction,
            reference_images=reference_images or [],
            seed=seed,
            width=width,
            height=height,
            sha256=sha256_hash,
            license=license_str,
            accepted=True,
            judge_scores=judge_scores or {},
            dataset_snapshot=dataset_snapshot,
        )
        log_event("provenance_created", {"sample_id": sample_id, "sha256": sha256_hash[:8]})
        return record
