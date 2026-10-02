"""Single VLM judge rater enforcing Pydantic schema validation."""

from pathlib import Path

from hcs_image_evolution.judging.hf_provider import HFProviderClient
from hcs_image_evolution.judging.schemas import VLMJudgeScore


class VLMRater:
    """Invokes a specific Vision-Language Model to grade an image based on conditioning."""

    SYSTEM_PROMPT = """You are an expert AI image evaluation judge. Your task is to evaluate the provided image objectively against the prompt.
You MUST output ONLY a valid JSON object matching this exact specification:
{
  "quality": <float between 0.0 and 1.0>,
  "aesthetics": <float between 0.0 and 1.0>,
  "prompt_alignment": <float between 0.0 and 1.0>,
  "composition": <float between 0.0 and 1.0>,
  "detail": <float between 0.0 and 1.0>,
  "anatomy": <float between 0.0 and 1.0>,
  "object_consistency": <float between 0.0 and 1.0>,
  "text_rendering": <float between 0.0 and 1.0>,
  "realism": <float between 0.0 and 1.0>,
  "edit_fidelity": <float between 0.0 and 1.0>,
  "artifact_penalty": <float between 0.0 and 1.0>,
  "safety": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "description": "<concise visual description>",
  "failure_reasons": ["<defect 1>", "<defect 2>"]
}
Do not include any commentary or explanations outside the JSON."""

    def __init__(self, model_id: str, client: HFProviderClient | None = None):
        self.model_id = model_id
        self.client = client or HFProviderClient()

    def rate_image(self, image_path: Path, prompt: str) -> VLMJudgeScore:
        """Submits image and prompt to VLM and validates returned score."""
        user_prompt = f"Target Prompt: {prompt}\nEvaluate the generated image against this prompt strictly following the JSON format."
        raw_json = self.client.request_vlm_json(
            model_id=self.model_id,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_path=image_path,
        )
        return VLMJudgeScore.model_validate(raw_json)
