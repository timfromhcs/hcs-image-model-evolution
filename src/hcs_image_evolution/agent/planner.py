"""Autonomous research planner directing data curriculum and training experiments."""

from pathlib import Path
from typing import Any

from hcs_image_evolution.agent.state_machine import EvolutionPhase, RunState
from hcs_image_evolution.utils.logging import logger


class ResearchPlanner:
    """Decides the curriculum adjustments and next experiment based on measured failure modes."""

    def __init__(self, config_dir: Path = Path("configs")):
        self.config_dir = config_dir

    def plan_next_action(self, state: RunState, recent_metrics: dict[str, Any]) -> tuple[EvolutionPhase, dict[str, Any]]:
        """Determines next execution phase and parameter adjustments."""
        current = state.stage

        # Autonomous state progression
        if current == EvolutionPhase.BOOT:
            return EvolutionPhase.DISCOVER, {}
        elif current == EvolutionPhase.DISCOVER:
            return EvolutionPhase.VERIFY, {}
        elif current == EvolutionPhase.VERIFY:
            return EvolutionPhase.PLAN, {}
        elif current == EvolutionPhase.PLAN:
            return EvolutionPhase.SMOKE_TEST, {}
        elif current == EvolutionPhase.SMOKE_TEST:
            return EvolutionPhase.GENERATION, {}
        elif current == EvolutionPhase.GENERATION:
            return EvolutionPhase.DATA_CURATION, {}
        elif current == EvolutionPhase.DATA_CURATION:
            return EvolutionPhase.TRAIN, {}
        elif current == EvolutionPhase.TRAIN:
            return EvolutionPhase.DISTILL, {}
        elif current == EvolutionPhase.DISTILL:
            return EvolutionPhase.BENCHMARK, {}
        elif current == EvolutionPhase.BENCHMARK:
            # Check if metrics warrant promotion or curriculum adaptation
            quality = recent_metrics.get("quality", 0.0)
            typography = recent_metrics.get("typography", 0.0)
            if typography < 0.65:
                logger.info("Typography score low (%s). Adapting curriculum for next iteration.", typography)
                return EvolutionPhase.GENERATION, {"focus": "typography", "ratio_boost": 0.25}
            return EvolutionPhase.PROMOTE, {}
        elif current == EvolutionPhase.PROMOTE:
            return EvolutionPhase.QUANTIZE, {}
        elif current == EvolutionPhase.QUANTIZE:
            return EvolutionPhase.SDCPP_TEST, {}
        elif current == EvolutionPhase.SDCPP_TEST:
            return EvolutionPhase.PACKAGE, {}
        elif current == EvolutionPhase.PACKAGE:
            return EvolutionPhase.DOCUMENT, {}
        elif current == EvolutionPhase.DOCUMENT:
            return EvolutionPhase.RELEASE, {}
        elif current == EvolutionPhase.RELEASE:
            return EvolutionPhase.COMPLETE, {}

        return current, {}
