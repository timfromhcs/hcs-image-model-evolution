"""Teacher target generation, caching, and revision invalidation."""

from pathlib import Path

import torch

from hcs_image_evolution.utils.hashing import compute_dict_hash


class TeacherTargetCache:
    """Caches teacher velocity predictions and intermediate states to eliminate redundant compute."""

    def __init__(self, cache_dir: Path = Path("cache/teacher_targets"), teacher_revision: str = "main"):
        self.cache_dir = cache_dir
        self.teacher_revision = teacher_revision
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, sample_id: str, timestep: float, seed: int) -> str:
        data = {
            "sample_id": sample_id,
            "timestep": round(timestep, 4),
            "seed": seed,
            "revision": self.teacher_revision,
        }
        return compute_dict_hash(data)

    def get_cached_velocity(self, sample_id: str, timestep: float, seed: int) -> torch.Tensor | None:
        """Retrieves cached velocity vector if it exists."""
        key = self._get_cache_key(sample_id, timestep, seed)
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
        return None

    def store_velocity(self, sample_id: str, timestep: float, seed: int, velocity: torch.Tensor) -> None:
        """Stores computed teacher velocity."""
        key = self._get_cache_key(sample_id, timestep, seed)
        path = self.cache_dir / f"{key}.pt"
        torch.save(velocity.detach().cpu(), path)
