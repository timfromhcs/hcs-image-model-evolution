"""Experiment registry, hypothesis ledger, and challenger tracking."""

import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hcs_image_evolution.utils.logging import log_event, logger


class ExperimentRecord(BaseModel):
    id: str
    parent: str | None = None
    purpose: str
    hypothesis: str
    config_hash: str = ""
    dataset_snapshot: str = ""
    teacher_revision: str = "Qwen/Qwen-Image-2.1@main"
    checkpoint_in: str | None = None
    checkpoint_out: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    decision: str = "pending" # promote | reject | retry | pending
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ExperimentManager:
    """Maintains append-only experiment records and research ledger."""

    def __init__(
        self,
        registry_file: Path = Path("state/experiments.jsonl"),
        ledger_file: Path = Path("state/ledger.jsonl"),
    ):
        self.registry_file = registry_file
        self.ledger_file = ledger_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

    def register_experiment(self, exp: ExperimentRecord) -> None:
        """Appends a new experiment to experiments.jsonl."""
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp.model_dump(), default=str) + "\n")
        log_event("experiment_registered", {"experiment_id": exp.id, "hypothesis": exp.hypothesis})
        logger.info("Registered experiment: %s (%s)", exp.id, exp.hypothesis)

    def record_finding(
        self,
        observation: str,
        source: str,
        experiment_id: str,
        conclusion: str,
        confidence: float,
    ) -> None:
        """Records a research discovery to ledger.jsonl."""
        finding = {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "observation": observation,
            "source": source,
            "experiment": experiment_id,
            "conclusion": conclusion,
            "confidence": confidence,
        }
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(finding, default=str) + "\n")
        log_event("finding_recorded", finding)
        logger.info("Recorded research finding for %s: %s", experiment_id, conclusion)

    def list_experiments(self) -> list[ExperimentRecord]:
        """Reads all recorded experiments."""
        if not self.registry_file.exists():
            return []
        records = []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(ExperimentRecord.model_validate_json(line))
        return records
