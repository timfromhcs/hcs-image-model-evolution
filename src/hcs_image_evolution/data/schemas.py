"""Data schemas conforming strictly to PLAN.md section 18 and 19."""

import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DatasetSourceRecord(BaseModel):
    dataset_id: str
    revision: str = "main"
    license: str
    source_url: str
    retrieved_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    hash: str = ""
    allowed_for_training: bool = True
    allowed_for_redistribution: bool = False
    notes: str = ""


class SampleRecord(BaseModel):
    sample_id: str
    source_type: str = "synthetic" # "synthetic" | "real" | "edit"
    source_dataset: Optional[str] = None
    source_model: Optional[str] = "Qwen/Qwen-Image-2.1"
    source_revision: Optional[str] = "main"
    source_prompt: str
    caption: str
    edit_instruction: Optional[str] = None
    reference_images: list[str] = Field(default_factory=list)
    seed: int = 42
    width: int = 1024
    height: int = 1024
    sha256: str
    license: str = "apache-2.0"
    accepted: bool = True
    judge_scores: dict[str, float] = Field(default_factory=dict)
    dataset_snapshot: str = "snapshot-001"
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class DatasetSnapshotManifest(BaseModel):
    snapshot_id: str
    version: str = "1.0.0"
    total_samples: int = 0
    synthetic_samples: int = 0
    real_samples: int = 0
    edit_samples: int = 0
    shards: list[str] = Field(default_factory=list)
    sha256_checksums: dict[str, str] = Field(default_factory=dict)
    licenses: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
