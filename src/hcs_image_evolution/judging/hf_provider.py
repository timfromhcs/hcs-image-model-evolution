"""Hugging Face Inference Providers client with exponential backoff and 429 handling."""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

from hcs_image_evolution.utils.logging import log_event, logger


class HFProviderClient:
    """Manages resilient multimodal requests to Hugging Face Inference Providers."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(token=self.token)

    def image_to_base64_url(self, image_path: Path) -> str:
        """Encodes an image to a base64 data URL."""
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def request_vlm_json(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        max_retries: int = 4,
        backoff_sec: float = 2.0,
    ) -> dict[str, Any]:
        """Sends an image and prompt to a VLM on HF Inference Provider, expecting valid JSON."""
        image_url = self.image_to_base64_url(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        for attempt in range(max_retries):
            try:
                start_t = time.time()
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.1,
                )
                duration = time.time() - start_t
                raw_text = response.choices[0].message.content.strip()

                # Clean markdown codeblocks if returned
                raw_text = raw_text.removeprefix("```json")
                raw_text = raw_text.removeprefix("```")
                raw_text = raw_text.removesuffix("```")

                parsed = json.loads(raw_text.strip())
                log_event("hf_vlm_request_success", {"model": model_id, "duration": round(duration, 2), "attempt": attempt})
                return parsed

            except Exception as e:
                logger.warning(
                    "VLM request attempt %d failed for model %s: %s", attempt + 1, model_id, e
                )
                if attempt == max_retries - 1:
                    log_event("hf_vlm_request_exhausted", {"model": model_id, "error": str(e)})
                    raise RuntimeError(f"VLM request failed after {max_retries} attempts: {e}") from e
                time.sleep(backoff_sec * (2**attempt))

        return {}
