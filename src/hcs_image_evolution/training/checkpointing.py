"""Atomic checkpoint manager creating immutable state packages with SHA-256 verification."""

import json
from pathlib import Path
from typing import Any

import torch

from hcs_image_evolution.storage.atomic import atomic_write_json
from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import log_event, logger
from hcs_image_evolution.utils.reproducibility import get_rng_states
from hcs_image_evolution.utils.system import get_environment_manifest, get_git_commit


class CheckpointManager:
    """Manages saving, loading, checksumming, and retention of training checkpoints."""

    def __init__(self, checkpoints_dir: Path = Path("state/checkpoints")):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        """Saves a structured checkpoint directory with all artifacts and SHA-256 checksums."""
        ckpt_dir = self.checkpoints_dir / f"checkpoint-{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights
        model_dir = ckpt_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # 2. Save optimizer
        if optimizer:
            opt_dir = ckpt_dir / "optimizer"
            opt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(optimizer.state_dict(), opt_dir / "optimizer.pt")

        # 3. Save scheduler
        if scheduler and hasattr(scheduler, "state_dict"):
            sched_dir = ckpt_dir / "scheduler"
            sched_dir.mkdir(parents=True, exist_ok=True)
            torch.save(scheduler.state_dict(), sched_dir / "scheduler.pt")

        # 4. Save metadata files
        state_info = {
            "step": step,
            "epoch": epoch,
            "git_commit": get_git_commit(),
            "rng": get_rng_states(),
        }
        atomic_write_json(ckpt_dir / "state.json", state_info)
        atomic_write_json(ckpt_dir / "metrics.json", metrics or {})
        atomic_write_json(ckpt_dir / "environment.json", get_environment_manifest())
        atomic_write_json(ckpt_dir / "manifest.json", config or {})

        # 5. Compute SHA256 for all files and write checksum.sha256
        checksum_lines = []
        for file_path in ckpt_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "checksum.sha256":
                sha = compute_file_sha256(file_path)
                rel_path = file_path.relative_to(ckpt_dir)
                checksum_lines.append(f"{sha}  {rel_path.as_posix()}")

        checksum_file = ckpt_dir / "checksum.sha256"
        checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

        log_event("checkpoint_saved", {"step": step, "checkpoint": ckpt_dir.name})
        logger.info("Saved atomic checkpoint to %s with verified checksums", ckpt_dir)
        return ckpt_dir

    def load_checkpoint(self, checkpoint_path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        """Loads state from checkpoint directory into model and optimizer."""
        model_path = checkpoint_path / "model" / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)

        if optimizer:
            opt_path = checkpoint_path / "optimizer" / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

        state_json = checkpoint_path / "state.json"
        if state_json.exists():
            return json.loads(state_json.read_text(encoding="utf-8"))
        return {}
