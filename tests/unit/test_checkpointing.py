"""Unit tests for checkpoint saving, checksumming, and reload."""

from pathlib import Path

from torch import nn

from hcs_image_evolution.agent.recovery import RecoveryManager
from hcs_image_evolution.training.checkpointing import CheckpointManager


def test_checkpoint_roundtrip_and_verification(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(checkpoints_dir=ckpt_dir)
    recovery = RecoveryManager(checkpoints_dir=ckpt_dir)

    model = nn.Linear(10, 2)
    saved_path = mgr.save_checkpoint(
        step=100,
        epoch=1,
        model=model,
        metrics={"loss": 0.123},
    )

    assert saved_path.exists()
    assert (saved_path / "checksum.sha256").exists()

    # Verify recovery manager validates integrity
    assert recovery.verify_checkpoint(saved_path)
    latest = recovery.find_latest_valid_checkpoint()
    assert latest == saved_path
