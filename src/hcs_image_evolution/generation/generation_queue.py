"""Persistent generation queue for batch processing and resume support."""

from pathlib import Path

from hcs_image_evolution.generation.generation_manifest import (
    GenerationBatchManifest,
    GenerationItem,
)
from hcs_image_evolution.utils.logging import logger


class GenerationQueue:
    """Manages pending and completed generation tasks with disk persistence."""

    def __init__(self, manifest_file: Path = Path("state/generation_queue.json")):
        self.manifest_file = manifest_file
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> GenerationBatchManifest:
        if self.manifest_file.exists():
            try:
                content = self.manifest_file.read_text(encoding="utf-8")
                return GenerationBatchManifest.model_validate_json(content)
            except Exception as e:
                logger.error("Failed to parse generation queue manifest: %s", e)

        return GenerationBatchManifest(batch_id=f"batch-{self.manifest_file.stem}")

    def enqueue(self, item: GenerationItem) -> None:
        """Adds a generation task to the queue."""
        self.manifest.items.append(item)
        self.manifest.total_items = len(self.manifest.items)
        self.save()

    def get_pending_items(self) -> list[GenerationItem]:
        """Returns all uncompleted generation tasks."""
        return [i for i in self.manifest.items if not i.completed]

    def mark_completed(self, item_id: str, image_path: Path, duration_sec: float, device: str) -> None:
        """Marks a task as completed and updates progress."""
        for item in self.manifest.items:
            if item.item_id == item_id:
                item.completed = True
                item.image_path = str(image_path)
                item.generation_time_seconds = duration_sec
                item.device_name = device
                break
        self.manifest.completed_items = sum(1 for i in self.manifest.items if i.completed)
        self.save()

    def save(self) -> None:
        """Persists the queue manifest."""
        self.manifest.save(self.manifest_file)
