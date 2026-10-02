"""Qwen-Image-2.1 teacher model loader and multi-mode generator."""

import time

import torch
from PIL import Image

from hcs_image_evolution.utils.logging import log_event, logger


class QwenTeacherGenerator:
    """Wrapper around Qwen-Image-2.1 Diffusers pipeline with CPU offload and VAE tiling."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-2.1",
        revision: str = "main",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
    ):
        self.model_id = model_id
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.device = device
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling
        self.pipe = None

    def load(self) -> None:
        """Loads the official Diffusers pipeline for Qwen-Image-2.1."""
        if self.pipe is not None:
            return

        logger.info("Loading Qwen-Image-2.1 teacher model (%s)...", self.model_id)
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        )

        if self.enable_vae_tiling and hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        if self.enable_cpu_offload and self.device == "cuda" and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        logger.info("Qwen-Image-2.1 teacher pipeline successfully initialized on %s", self.device)

    def generate_t2i(
        self,
        prompt: str,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
    ) -> tuple[Image.Image, float]:
        """Generates an image from text prompt using the full teacher model."""
        self.load()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        start_t = time.time()

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        duration = time.time() - start_t
        image = result.images[0]
        log_event("teacher_generated_t2i", {"prompt": prompt[:40], "seed": seed, "duration": round(duration, 2)})
        return image, duration

    def generate_edit(
        self,
        image: Image.Image,
        instruction: str,
        seed: int = 42,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
    ) -> tuple[Image.Image, float]:
        """Generates an instruction-guided edit of the input image."""
        self.load()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        start_t = time.time()

        # Check if pipeline supports image conditioning parameter
        kwargs = {
            "prompt": instruction,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if "image" in self.pipe.__call__.__code__.co_varnames:
            kwargs["image"] = image

        result = self.pipe(**kwargs)
        duration = time.time() - start_t
        image_out = result.images[0]
        log_event("teacher_generated_edit", {"instruction": instruction[:40], "seed": seed, "duration": round(duration, 2)})
        return image_out, duration
