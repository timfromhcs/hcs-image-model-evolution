"""Builds standalone, offline-ready distribution packages for Windows/Linux."""

import shutil
from pathlib import Path

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import log_event, logger


class ReleasePackager:
    """Assembles self-contained offline distribution zip/folder."""

    def __init__(self, dist_root: Path = Path("dist")):
        self.dist_root = dist_root

    def create_package(
        self,
        package_name: str,
        sd_cli_path: Path,
        gguf_model_path: Path,
        manifest_path: Path,
    ) -> Path:
        """Assembles all runtime binaries, models, presets, and scripts into a distribution folder."""
        pkg_dir = self.dist_root / package_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        bin_dir = pkg_dir / "bin"
        models_dir = pkg_dir / "models"
        scripts_dir = pkg_dir / "scripts"
        presets_dir = pkg_dir / "presets"

        for d in [bin_dir, models_dir, scripts_dir, presets_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. Copy binary if exists
        if sd_cli_path.exists():
            shutil.copy2(sd_cli_path, bin_dir / sd_cli_path.name)

        # 2. Copy model
        if gguf_model_path.exists():
            shutil.copy2(gguf_model_path, models_dir / gguf_model_path.name)

        # 3. Copy manifest
        if manifest_path.exists():
            shutil.copy2(manifest_path, pkg_dir / "MANIFEST.json")

        # 4. Generate batch runner scripts for Windows
        t2i_bat = scripts_dir / "run_t2i.bat"
        t2i_bat.write_text(
            f'@echo off\n..\\bin\\{sd_cli_path.name} -m ..\\models\\{gguf_model_path.name} -p "%~1" -o output.png --steps 4 --cfg-scale 1.0\n',
            encoding="utf-8",
        )

        edit_bat = scripts_dir / "run_edit.bat"
        edit_bat.write_text(
            f'@echo off\n..\\bin\\{sd_cli_path.name} -m ..\\models\\{gguf_model_path.name} -i "%~1" -p "%~2" -o output_edit.png --steps 4 --cfg-scale 1.0\n',
            encoding="utf-8",
        )

        # 5. Write package README
        readme = pkg_dir / "README.md"
        readme.write_text(
            f"# {package_name}\n\nStandalone offline distribution powered by stable-diffusion.cpp.\n\n"
            f"## Quickstart\n"
            f"Run text-to-image:\n"
            f"```cmd\n"
            f"scripts\\run_t2i.bat \"A futuristic flying car in a cyberpunk city\"\n"
            f"```\n",
            encoding="utf-8",
        )

        # 6. Generate SHA256SUMS.txt
        checksum_lines = []
        for f in pkg_dir.rglob("*"):
            if f.is_file() and f.name != "SHA256SUMS.txt":
                sha = compute_file_sha256(f)
                rel = f.relative_to(pkg_dir)
                checksum_lines.append(f"{sha}  {rel.as_posix()}")

        (pkg_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        log_event("release_package_assembled", {"package": package_name, "files": len(checksum_lines)})
        logger.info("Successfully packaged standalone distribution: %s", pkg_dir)
        return pkg_dir
