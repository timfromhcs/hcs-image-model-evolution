"""Generation metadata schemas and batch tracking manifests."""

import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hcs_image_evolution.storage.atomic import atomic_write_json


class GenerationItem(BaseModel):
    item_id: str
    prompt: str
    difficulty_bucket: str
    seed: int
    width: int = 1024
    height: int = 1024
    steps: int = 40
    guidance_scale: float = 4.5
    teacher_model_id: str = "Qwen/Qwen-Image-2.1"
    teacher_revision: str = "main"
    image_path: str | None = None
    generation_time_seconds: float = 0.0
    device_name: str = "unknown"
    completed: bool = False
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class GenerationBatchManifest(BaseModel):
    batch_id: str
    total_items: int = 0
    completed_items: int = 0
    items: list[GenerationItem] = Field(default_factory=list)
    manifest_path: str | None = None

    def save(self, target_path: Path) -> None:
        """Atomically saves the batch manifest."""
        self.manifest_path = str(target_path)
        atomic_write_json(target_path, self.model_dump())
