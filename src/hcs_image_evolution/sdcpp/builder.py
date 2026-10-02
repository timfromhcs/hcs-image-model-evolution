"""Source compilation manager for stable-diffusion.cpp with Vulkan/CPU backend."""

import subprocess
from pathlib import Path

from hcs_image_evolution.utils.logging import log_event, logger
from hcs_image_evolution.utils.system import detect_vulkan_support


class SDCPPBuilder:
    """Clones and compiles stable-diffusion.cpp using CMake and native compilers."""

    def __init__(
        self,
        repo_url: str = "https://github.com/leejet/stable-diffusion.cpp",
        repo_dir: Path = Path("build/stable-diffusion.cpp"),
        build_dir: Path = Path("build/stable-diffusion.cpp/build"),
    ):
        self.repo_url = repo_url
        self.repo_dir = repo_dir
        self.build_dir = build_dir

    def clone_or_update(self, commit: str = "master") -> bool:
        """Clones the repository with submodules if not already present."""
        if not (self.repo_dir / ".git").is_dir():
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Cloning stable-diffusion.cpp from %s...", self.repo_url)
            cmd = ["git", "clone", "--recursive", self.repo_url, str(self.repo_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                logger.error("Failed to clone sd.cpp: %s", res.stderr)
                return False
        return True

    def build(self, enable_vulkan: bool | None = None) -> Path | None:
        """Configures and builds sd-cli using CMake."""
        self.clone_or_update()
        self.build_dir.mkdir(parents=True, exist_ok=True)

        use_vulkan = enable_vulkan if enable_vulkan is not None else detect_vulkan_support()
        logger.info("Building stable-diffusion.cpp (Vulkan=%s)...", use_vulkan)

        cmake_args = [
            "cmake",
            "-B", str(self.build_dir),
            "-S", str(self.repo_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSD_BUILD_EXAMPLES=ON",
        ]
        if use_vulkan:
            cmake_args.append("-DSD_VULKAN=ON")

        res = subprocess.run(cmake_args, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            logger.warning("CMake config warning/error: %s", res.stderr)

        build_cmd = ["cmake", "--build", str(self.build_dir), "--config", "Release", "--parallel"]
        build_res = subprocess.run(build_cmd, capture_output=True, text=True, check=False)

        # Look for sd-cli.exe in build outputs
        for name in ["sd-cli.exe", "sd.exe", "Release/sd-cli.exe", "bin/sd-cli.exe"]:
            candidate = self.build_dir / name
            if candidate.is_file():
                logger.info("Built binary located at %s", candidate)
                log_event("sdcpp_build_success", {"binary": str(candidate), "vulkan": use_vulkan})
                return candidate

        return None
