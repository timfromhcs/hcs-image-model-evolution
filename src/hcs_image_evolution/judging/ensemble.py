"""Multi-VLM judge ensemble with robust trimmed aggregation and disagreement penalty."""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from hcs_image_evolution.judging.schemas import EnsembleEvaluationResult, VLMJudgeScore
from hcs_image_evolution.judging.vlm_rater import VLMRater
from hcs_image_evolution.utils.logging import log_event, logger


class JudgeEnsemble:
    """Combines evaluations from multiple independent vision-language models."""

    def __init__(
        self,
        raters: Sequence[VLMRater],
        weights: Optional[dict[str, float]] = None,
        min_overall_score: float = 0.70,
        max_disagreement: float = 0.35,
    ):
        self.raters = raters
        self.weights = weights or {
            "quality": 0.25,
            "prompt_alignment": 0.20,
            "aesthetics": 0.15,
            "composition": 0.15,
            "detail": 0.10,
            "edit_fidelity": 0.10,
            "typography": 0.05,
        }
        self.min_overall_score = min_overall_score
        self.max_disagreement = max_disagreement

    def evaluate(self, image_path: Path, prompt: str) -> EnsembleEvaluationResult:
        """Collects evaluations from all judges and aggregates robustly."""
        raw_scores: dict[str, VLMJudgeScore] = {}
        for rater in self.raters:
            try:
                score = rater.rate_image(image_path, prompt)
                raw_scores[rater.model_id] = score
            except Exception as e:
                logger.error("Judge %s evaluation failed: %s", rater.model_id, e)

        if not raw_scores:
            raise RuntimeError(f"All judges failed to evaluate image {image_path}")

        dimensions = list(self.weights.keys())
        dim_medians: dict[str, float] = {}
        dimension_variances = []

        for dim in dimensions:
            values = [getattr(s, dim, 0.0) for s in raw_scores.values()]
            dim_medians[dim] = float(np.median(values))
            if len(values) > 1:
                dimension_variances.append(float(np.std(values)))

        disagreement = float(np.mean(dimension_variances)) if dimension_variances else 0.0

        # Weighted calculation
        raw_composite = sum(dim_medians[dim] * self.weights[dim] for dim in dimensions)
        artifact_penalties = [s.artifact_penalty for s in raw_scores.values()]
        mean_artifact = float(np.mean(artifact_penalties))

        final_score = max(0.0, raw_composite - (0.5 * disagreement) - mean_artifact)

        rejections = []
        if final_score < self.min_overall_score:
            rejections.append(f"Final aggregated score {final_score:.2f} below threshold {self.min_overall_score}")
        if disagreement > self.max_disagreement:
            rejections.append(f"Judge disagreement {disagreement:.2f} exceeds threshold {self.max_disagreement}")

        accepted = len(rejections) == 0
        result = EnsembleEvaluationResult(
            aggregated_score=round(final_score, 3),
            dimension_scores={k: round(v, 3) for k, v in dim_medians.items()},
            judge_disagreement=round(disagreement, 3),
            accepted=accepted,
            rejection_reasons=rejections,
            raw_judge_scores=raw_scores,
        )

        log_event(
            "ensemble_evaluation",
            {"image": image_path.name, "score": result.aggregated_score, "accepted": accepted},
        )
        return result
