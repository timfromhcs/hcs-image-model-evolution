"""Local cache management with disk quota enforcement."""

from pathlib import Path

from hcs_image_evolution.utils.logging import logger


class CacheManager:
    """Controls local disk caching for teacher representations, latents, and shards."""

    def __init__(self, cache_dir: Path = Path("cache"), max_size_gb: float = 50.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * (1024**3))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_current_size(self) -> int:
        """Returns total size of the cache directory in bytes."""
        total = 0
        for entry in self.cache_dir.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def prune_if_needed(self) -> None:
        """Removes oldest cache files if cache exceeds max size limit."""
        size = self.get_current_size()
        if size <= self.max_size_bytes:
            return

        logger.info("Cache size (%s GB) exceeds limit (%s GB). Pruning oldest items...",
                    round(size / (1024**3), 2), round(self.max_size_bytes / (1024**3), 2))

        files = sorted(
            [p for p in self.cache_dir.rglob("*") if p.is_file()],
            key=lambda p: p.stat().st_mtime
        )
        for f in files:
            f.unlink()
            size = self.get_current_size()
            if size <= self.max_size_bytes * 0.8:
                break
