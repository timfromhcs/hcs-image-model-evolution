"""Calibration suite measuring inter-rater agreement across anchor cases."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hcs_image_evolution.judging.ensemble import JudgeEnsemble
from hcs_image_evolution.utils.logging import logger


class CalibrationSuite:
    """Evaluates inter-rater agreement and bias on a fixed anchor set."""

    def __init__(self, ensemble: JudgeEnsemble):
        self.ensemble = ensemble

    def run_calibration(self, test_cases: Sequence[tuple[Path, str]]) -> dict[str, float]:
        """Runs the ensemble across test cases and computes inter-rater agreement."""
        disagreements = []
        scores = []

        for img_path, prompt in test_cases:
            if not img_path.exists():
                continue
            res = self.ensemble.evaluate(img_path, prompt)
            disagreements.append(res.judge_disagreement)
            scores.append(res.aggregated_score)

        if not disagreements:
            return {"mean_disagreement": 0.0, "mean_score": 0.0}

        metrics = {
            "mean_disagreement": float(np.mean(disagreements)),
            "max_disagreement": float(np.max(disagreements)),
            "mean_calibrated_score": float(np.mean(scores)),
        }
        logger.info("Calibration metrics: %s", metrics)
        return metrics
