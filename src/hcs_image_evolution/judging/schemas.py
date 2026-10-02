"""Strict Pydantic schema for multi-VLM evaluations conforming to PLAN.md section 14."""

from pydantic import BaseModel, Field


class VLMJudgeScore(BaseModel):
    quality: float = Field(..., ge=0.0, le=1.0, description="Overall visual clarity and fidelity")
    aesthetics: float = Field(..., ge=0.0, le=1.0, description="Artistic and photographic aesthetics")
    prompt_alignment: float = Field(..., ge=0.0, le=1.0, description="Adherence to prompt conditioning")
    composition: float = Field(..., ge=0.0, le=1.0, description="Scene framing and balance")
    detail: float = Field(..., ge=0.0, le=1.0, description="Micro-detail and texture definition")
    anatomy: float = Field(1.0, ge=0.0, le=1.0, description="Anatomical correctness for beings")
    object_consistency: float = Field(1.0, ge=0.0, le=1.0, description="Structural consistency of objects")
    text_rendering: float = Field(1.0, ge=0.0, le=1.0, description="Legibility of requested typography")
    realism: float = Field(1.0, ge=0.0, le=1.0, description="Photographic realism if applicable")
    edit_fidelity: float = Field(1.0, ge=0.0, le=1.0, description="Fidelity to requested edits")
    artifact_penalty: float = Field(0.0, ge=0.0, le=1.0, description="Penalty for distortions, tearing, glitches")
    safety: float = Field(1.0, ge=0.0, le=1.0, description="Content safety score (1.0 = safe)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model self-reported evaluation confidence")
    description: str = Field("", description="Concise visual description of the generated image")
    failure_reasons: list[str] = Field(default_factory=list, description="Specific defect observations")


class EnsembleEvaluationResult(BaseModel):
    aggregated_score: float
    dimension_scores: dict[str, float]
    judge_disagreement: float
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    raw_judge_scores: dict[str, VLMJudgeScore] = Field(default_factory=dict)
