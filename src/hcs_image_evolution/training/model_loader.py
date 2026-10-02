"""Model loader supporting Qwen-Image-2.1 teacher and student adapters."""

import torch
from torch import nn

from hcs_image_evolution.utils.logging import logger


class ModelLoader:
    """Loads diffusion components, applies PEFT LoRA adapters, or initializes compact DiT students."""

    @staticmethod
    def load_transformer_backbone(
        model_id: str = "Qwen/Qwen-Image-2.1",
        subfolder: str = "transformer",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Loads the diffusion transformer backbone from Hugging Face or local path."""
        try:
            from diffusers import DiffusionPipeline
            logger.info("Loading transformer backbone from %s...", model_id)
            # Diffusers modular loader
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
            return pipe.transformer
        except Exception as e:
            logger.warning("Falling back to generic DiT backbone loader: %s", e)
            # Lightweight fallback module for testing or student initialization
            return nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 4, kernel_size=3, padding=1),
            )

    @staticmethod
    def apply_lora(
        model: nn.Module,
        rank: int = 64,
        alpha: int = 64,
        target_modules: list[str] | None = None,
    ) -> nn.Module:
        """Injects trainable LoRA adapters into diffusion transformer layers."""
        try:
            from peft import LoraConfig, get_peft_model
            targets = target_modules or ["to_q", "to_k", "to_v", "to_out.0"]
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=targets,
                lora_dropout=0.05,
                bias="none",
            )
            peft_model = get_peft_model(model, lora_config)
            logger.info("Applied PEFT LoRA with rank=%d, alpha=%d", rank, alpha)
            return peft_model
        except Exception as e:
            logger.warning("PEFT LoRA initialization fallback: %s", e)
            return model
