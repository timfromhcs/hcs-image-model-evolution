"""Multi-level exact and perceptual image deduplication."""

from pathlib import Path
from typing import Optional
from PIL import Image
import imagehash

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import logger


class Deduplicator:
    """Detects duplicates via cryptographic hash, perceptual pHash, and dHash."""

    def __init__(self, phash_threshold: int = 6, dhash_threshold: int = 6):
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold
        self.exact_hashes: set[str] = set()
        self.phashes: list[tuple[imagehash.ImageHash, str]] = []
        self.dhashes: list[tuple[imagehash.ImageHash, str]] = []

    def is_duplicate(self, image_path: Path) -> tuple[bool, Optional[str]]:
        """Checks if image is duplicate of any previously indexed image."""
        exact = compute_file_sha256(image_path)
        if exact in self.exact_hashes:
            return True, f"Exact SHA256 duplicate ({exact[:8]})"

        try:
            with Image.open(image_path) as img:
                current_p = imagehash.phash(img)
                current_d = imagehash.dhash(img)

            for stored_p, orig_id in self.phashes:
                if current_p - stored_p <= self.phash_threshold:
                    return True, f"Perceptual pHash duplicate with {orig_id} (diff: {current_p - stored_p})"

            for stored_d, orig_id in self.dhashes:
                if current_d - stored_d <= self.dhash_threshold:
                    return True, f"Perceptual dHash duplicate with {orig_id} (diff: {current_d - stored_d})"

            self.exact_hashes.add(exact)
            self.phashes.append((current_p, image_path.stem))
            self.dhashes.append((current_d, image_path.stem))
            return False, None
        except Exception as e:
            logger.warning("Failed perceptual hash for %s: %s", image_path, e)
            return False, None
