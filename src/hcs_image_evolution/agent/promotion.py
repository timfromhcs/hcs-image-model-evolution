"""Promotion gating logic comparing candidate models against baselines."""

from pathlib import Path
from typing import Any

import yaml

from hcs_image_evolution.utils.logging import log_event


class PromotionManager:
    """Applies strict gates to candidate checkpoints before marking them promoted/release-ready."""

    def __init__(self, config_path: Path = Path("configs/benchmark.yaml")):
        self.config_path = config_path
        self.gates = self._load_gates()

    def _load_gates(self) -> dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("benchmark", {}).get("gates", {})
        return {
            "min_quality_ratio_vs_teacher": 0.85,
            "min_prompt_alignment": 0.70,
            "min_editing_fidelity": 0.70,
            "max_artifact_rate": 0.15,
        }

    def evaluate_promotion(
        self,
        candidate_metrics: dict[str, float],
        teacher_baseline_metrics: dict[str, float],
        current_champion_metrics: dict[str, float] | None = None,
    ) -> tuple[bool, list[str]]:
        """Evaluates whether candidate beats baseline gates. Returns (passed, failure_reasons)."""
        reasons = []

        cand_qual = candidate_metrics.get("quality", 0.0)
        teach_qual = teacher_baseline_metrics.get("quality", 1.0)
        min_ratio = self.gates.get("min_quality_ratio_vs_teacher", 0.85)

        if cand_qual < (teach_qual * min_ratio):
            reasons.append(
                f"Quality ratio {cand_qual / max(teach_qual, 1e-5):.2f} below required {min_ratio}"
            )

        cand_align = candidate_metrics.get("prompt_alignment", 0.0)
        min_align = self.gates.get("min_prompt_alignment", 0.70)
        if cand_align < min_align:
            reasons.append(f"Prompt alignment {cand_align:.2f} below gate {min_align}")

        cand_artifact = candidate_metrics.get("artifact_rate", 1.0)
        max_artifact = self.gates.get("max_artifact_rate", 0.15)
        if cand_artifact > max_artifact:
            reasons.append(f"Artifact rate {cand_artifact:.2f} exceeds gate {max_artifact}")

        if current_champion_metrics:
            champ_qual = current_champion_metrics.get("quality", 0.0)
            if cand_qual < champ_qual:
                reasons.append(
                    f"Candidate quality ({cand_qual:.2f}) does not beat champion ({champ_qual:.2f})"
                )

        passed = len(reasons) == 0
        log_event("promotion_evaluated", {"passed": passed, "reasons": reasons})
        return passed, reasons
