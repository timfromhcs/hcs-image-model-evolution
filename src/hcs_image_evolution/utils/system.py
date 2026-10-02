"""System diagnostics, environment introspection, and GPU memory detection."""

import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil

from hcs_image_evolution.utils.hashing import compute_dict_hash


def get_git_commit() -> str:
    """Returns current git commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def detect_vulkan_support() -> bool:
    """Detects whether Vulkan runtime is available on the system."""
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        try:
            res = subprocess.run(
                [vulkaninfo, "--summary"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return True
    return False


def get_system_specs() -> dict[str, Any]:
    """Collects complete local or remote system specification."""
    mem = psutil.virtual_memory()
    specs: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "vulkan_available": detect_vulkan_support(),
        "cuda_available": False,
        "gpu_devices": [],
    }

    try:
        import torch
        if torch.cuda.is_available():
            specs["cuda_available"] = True
            specs["cuda_version"] = torch.version.cuda
            specs["gpu_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_total_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return specs


def get_environment_manifest() -> dict[str, Any]:
    """Collects installed package versions and generates environment hash."""
    packages = {}
    key_pkgs = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "pydantic",
        "huggingface_hub",
        "pillow",
        "numpy",
        "psutil",
    ]
    for pkg in key_pkgs:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = "not_installed"

    env_data = {
        "python": sys.version.split()[0],
        "system": platform.platform(),
        "packages": packages,
        "git_commit": get_git_commit(),
    }
    env_data["environment_hash"] = compute_dict_hash(env_data)
    return env_data
