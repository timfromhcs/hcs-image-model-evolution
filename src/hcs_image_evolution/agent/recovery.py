"""Disaster recovery and checkpoint rollback mechanism."""

from pathlib import Path

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import logger


class RecoveryManager:
    """Validates checkpoint integrity and executes rollbacks on corruption."""

    def __init__(self, checkpoints_dir: Path = Path("state/checkpoints")):
        self.checkpoints_dir = checkpoints_dir

    def verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verifies the SHA-256 checksum file inside a checkpoint directory."""
        if not checkpoint_path.is_dir():
            return False

        checksum_file = checkpoint_path / "checksum.sha256"
        if not checksum_file.exists():
            logger.warning("Checkpoint %s missing checksum.sha256", checkpoint_path.name)
            return False

        try:
            with open(checksum_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        expected_hash, rel_path = parts[0], parts[1]
                        target_file = checkpoint_path / rel_path
                        if not target_file.is_file():
                            return False
                        actual_hash = compute_file_sha256(target_file)
                        if actual_hash != expected_hash:
                            logger.error(
                                "Checksum mismatch in checkpoint %s for file %s",
                                checkpoint_path.name, rel_path
                            )
                            return False
            return True
        except Exception as e:
            logger.error("Error verifying checkpoint %s: %s", checkpoint_path.name, e)
            return False

    def find_latest_valid_checkpoint(self) -> Path | None:
        """Finds the newest valid uncorrupted checkpoint directory."""
        if not self.checkpoints_dir.is_dir():
            return None

        candidates = sorted(
            [d for d in self.checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: d.name,
            reverse=True,
        )

        for candidate in candidates:
            if self.verify_checkpoint(candidate):
                logger.info("Found valid checkpoint for recovery: %s", candidate.name)
                return candidate
            else:
                logger.warning("Checkpoint %s failed verification. Checking earlier checkpoint.", candidate.name)

        return None
