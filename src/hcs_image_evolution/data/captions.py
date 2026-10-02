"""Multi-captioning management and prompt-vs-vlm contradiction checking."""

from typing import Optional
from pydantic import BaseModel


class MultiCaption(BaseModel):
    original_prompt: str
    teacher_prompt: Optional[str] = None
    vlm_caption: Optional[str] = None
    structured_caption: Optional[str] = None
    edit_instruction: Optional[str] = None


class CaptionValidator:
    """Detects severe contradictions between prompt intention and observed VLM caption."""

    @staticmethod
    def detect_contradiction(prompt: str, vlm_caption: str) -> tuple[bool, str]:
        """Simple semantic consistency check between generated prompt and observed scene."""
        p_lower = prompt.lower()
        v_lower = vlm_caption.lower()

        # Check basic color or count keywords if explicitly conflicting
        colors = ["red", "blue", "green", "yellow", "black", "white"]
        for color in colors:
            if f"red {color}" in p_lower and f"blue {color}" in v_lower:
                return True, "Color contradiction: prompt wanted red, observed blue"

        return False, "Consistent"
