"""Image editing dataset schemas and paired sample management."""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class EditPair(BaseModel):
    pair_id: str
    source_image_path: str
    target_image_path: str
    edit_instruction: str
    source_caption: Optional[str] = None
    target_caption: Optional[str] = None
    change_mask_path: Optional[str] = None
    identity_preservation_score: float = 1.0
    requested_change_score: float = 1.0
    accepted: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)
