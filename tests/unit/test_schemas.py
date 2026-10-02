"""Unit tests for Pydantic data and evaluation schemas."""

import pytest
from pydantic import ValidationError

from hcs_image_evolution.data.schemas import SampleRecord
from hcs_image_evolution.judging.schemas import VLMJudgeScore


def test_sample_record_valid():
    rec = SampleRecord(
        sample_id="test-001",
        source_type="synthetic",
        source_prompt="A red ball",
        caption="A vibrant red sphere",
        sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        license="apache-2.0",
    )
    assert rec.sample_id == "test-001"
    assert rec.width == 1024


def test_vlm_judge_score_ranges():
    with pytest.raises(ValidationError):
        # quality must be <= 1.0
        VLMJudgeScore(
            quality=1.5,
            aesthetics=0.8,
            prompt_alignment=0.9,
            composition=0.8,
            detail=0.7,
            confidence=0.9,
        )
