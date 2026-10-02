"""Persistent run state machine and autonomous phase transitions."""

import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hcs_image_evolution.storage.atomic import atomic_write_json
from hcs_image_evolution.utils.logging import log_event, logger
from hcs_image_evolution.utils.system import get_environment_manifest


class EvolutionPhase(str, Enum):
    BOOT = "BOOT"
    DISCOVER = "DISCOVER"
    VERIFY = "VERIFY"
    PLAN = "PLAN"
    IMPLEMENT = "IMPLEMENT"
    SMOKE_TEST = "SMOKE_TEST"
    GENERATION = "GENERATION"
    DATA_CURATION = "DATA_CURATION"
    TRAIN = "TRAIN"
    DISTILL = "DISTILL"
    BENCHMARK = "BENCHMARK"
    PROMOTE = "PROMOTE"
    QUANTIZE = "QUANTIZE"
    SDCPP_TEST = "SDCPP_TEST"
    PACKAGE = "PACKAGE"
    DOCUMENT = "DOCUMENT"
    RELEASE = "RELEASE"
    COMPLETE = "COMPLETE"


class RunState(BaseModel):
    run_id: str
    stage: EvolutionPhase = EvolutionPhase.BOOT
    experiment_id: str = "exp-001"
    model_revision: str = "student-001"
    global_step: int = 0
    epoch: int = 0
    dataset_snapshot: str = "dataset-001"
    teacher_revision: str = "Qwen/Qwen-Image-2.1@main"
    best_checkpoint: str | None = None
    latest_checkpoint: str | None = None
    rng: dict[str, Any] = Field(default_factory=dict)
    environment_hash: str = ""
    config_hash: str = ""
    status: str = "running"
    last_updated: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class StateManager:
    """Manages atomic loading, updating, and persistence of the run state."""

    def __init__(self, state_file: Path = Path("state/run_state.json")):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_or_initialize()

    def load_or_initialize(self) -> RunState:
        """Loads state from file or initializes a new RunState."""
        if self.state_file.exists():
            try:
                content = self.state_file.read_text(encoding="utf-8")
                state = RunState.model_validate_json(content)
                logger.info("Resumed active run state: %s at stage %s", state.run_id, state.stage)
                return state
            except Exception as e:
                logger.error("Failed to parse existing run_state.json: %s. Creating recovery state.", e)

        run_id = f"RUN-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        env = get_environment_manifest()
        new_state = RunState(
            run_id=run_id,
            environment_hash=env["environment_hash"],
        )
        self.persist(new_state)
        logger.info("Initialized new run state: %s", run_id)
        return new_state

    def persist(self, state: RunState | None = None) -> None:
        """Atomically persists run state to disk."""
        if state is not None:
            self.state = state
        self.state.last_updated = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_write_json(self.state_file, self.state.model_dump())
        log_event("state_persisted", {"stage": self.state.stage, "step": self.state.global_step})

    def transition_to(self, next_phase: EvolutionPhase) -> None:
        """Transitions to the next phase and records event."""
        prev = self.state.stage
        self.state.stage = next_phase
        self.persist()
        log_event("phase_transition", {"from": prev, "to": next_phase})
        logger.info("State transition: %s -> %s", prev, next_phase)
