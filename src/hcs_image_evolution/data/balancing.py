"""Dataset balancing across aspect ratios, sources, and styles."""

from collections import Counter
from typing import Any, Sequence
from hcs_image_evolution.data.schemas import SampleRecord
from hcs_image_evolution.utils.logging import logger


class DatasetBalancer:
    """Computes distribution balances and selects representative subsets."""

    def __init__(self, target_synthetic_ratio: float = 0.50):
        self.target_synthetic = target_synthetic_ratio

    def compute_distribution_metrics(self, samples: Sequence[SampleRecord]) -> dict[str, Any]:
        """Calculates current ratios of source types and aspect ratios."""
        total = len(samples)
        if total == 0:
            return {"total": 0}

        types = Counter([s.source_type for s in samples])
        aspects = Counter([f"{s.width}x{s.height}" for s in samples])

        metrics = {
            "total_samples": total,
            "type_ratios": {k: round(v / total, 3) for k, v in types.items()},
            "aspect_ratios": {k: round(v / total, 3) for k, v in aspects.items()},
        }
        logger.info("Dataset distribution: %s", metrics)
        return metrics
