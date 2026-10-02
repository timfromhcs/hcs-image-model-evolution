"""Local CLI execution wrapper for stable-diffusion.cpp (sd-cli.exe)."""

import subprocess
import time
from pathlib import Path

from hcs_image_evolution.utils.logging import log_event, logger


class SDCPPRunner:
    """Invokes compiled sd-cli binary for local Windows CPU or Vulkan inference."""

    def __init__(self, binary_path: Path):
        self.binary_path = binary_path

    def run_t2i(
        self,
        model_path: Path,
        prompt: str,
        output_path: Path,
        steps: int = 4,
        cfg_scale: float = 1.0,
        sampling_method: str = "euler",
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
    ) -> tuple[bool, float, str]:
        """Executes text-to-image inference with sd-cli.exe."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.binary_path),
            "-m", str(model_path),
            "-p", prompt,
            "-o", str(output_path),
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampling_method,
            "-W", str(width),
            "-H", str(height),
            "-s", str(seed),
        ]

        logger.info("Executing sd-cli command: %s", " ".join(cmd))
        start_t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        duration = time.time() - start_t

        success = res.returncode == 0 and output_path.exists()
        log_event(
            "sdcpp_t2i_execution",
            {"success": success, "duration": round(duration, 2), "returncode": res.returncode},
        )
        return success, duration, res.stdout + "\n" + res.stderr

    def run_i2i(
        self,
        model_path: Path,
        input_image: Path,
        prompt: str,
        output_path: Path,
        steps: int = 4,
        cfg_scale: float = 1.0,
        strength: float = 0.75,
        seed: int = 42,
    ) -> tuple[bool, float, str]:
        """Executes image-to-image editing inference with sd-cli.exe."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(self.binary_path),
            "-m", str(model_path),
            "-i", str(input_image),
            "-p", prompt,
            "-o", str(output_path),
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--strength", str(strength),
            "-s", str(seed),
        ]

        start_t = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        duration = time.time() - start_t

        success = res.returncode == 0 and output_path.exists()
        log_event("sdcpp_i2i_execution", {"success": success, "duration": round(duration, 2)})
        return success, duration, res.stdout + "\n" + res.stderr
