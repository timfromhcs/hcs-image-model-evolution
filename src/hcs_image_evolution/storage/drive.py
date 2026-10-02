"""Google Drive synchronization, verification, and disaster recovery manager."""

import os
import shutil
from pathlib import Path

from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import logger


class DriveManager:
    """Manages backup and restoration between local working storage and Google Drive."""

    def __init__(self, drive_mount: Path = Path("/content/drive/MyDrive/HCS_Image_Evolution")):
        self.drive_mount = drive_mount
        self.is_mounted = self._check_mount()

    def _check_mount(self) -> bool:
        """Verifies if Google Drive is mounted and accessible."""
        try:
            return self.drive_mount.is_dir() or self.drive_mount.parent.is_dir()
        except Exception:
            return False

    def sync_file_to_drive(
        self,
        local_path: Path,
        relative_dest_path: str,
        verify_checksum: bool = True,
    ) -> bool:
        """Backs up a local file to Drive with explicit SHA-256 integrity verification."""
        if not self.is_mounted:
            logger.warning("Drive not mounted. Skipping backup for %s", local_path)
            return False

        dest_path = self.drive_mount / relative_dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        local_hash = compute_file_sha256(local_path)
        shutil.copy2(local_path, dest_path)

        if verify_checksum:
            drive_hash = compute_file_sha256(dest_path)
            if local_hash != drive_hash:
                raise OSError(
                    f"Drive backup verification failed for {local_path}: hash mismatch ({local_hash} != {drive_hash})"
                )

        logger.info("Successfully backed up %s to Drive: %s (SHA256: %s)", local_path.name, dest_path, local_hash[:8])
        return True

    def sync_directory_to_drive(
        self,
        local_dir: Path,
        relative_dest_dir: str,
        verify_checksums: bool = True,
    ) -> bool:
        """Recursively copies and verifies directory backup to Drive."""
        if not self.is_mounted or not local_dir.is_dir():
            return False

        for root, _, files in os.walk(local_dir):
            for file in files:
                l_file = Path(root) / file
                rel_file = l_file.relative_to(local_dir)
                dest_rel = Path(relative_dest_dir) / rel_file
                self.sync_file_to_drive(l_file, str(dest_rel), verify_checksum=verify_checksums)
        return True

    def restore_from_drive(self, relative_src_dir: str, local_dest_dir: Path) -> bool:
        """Restores checkpoints or dataset files from Google Drive to local high-speed SSD."""
        drive_src = self.drive_mount / relative_src_dir
        if not drive_src.exists():
            logger.warning("Drive path does not exist for restore: %s", drive_src)
            return False

        local_dest_dir.mkdir(parents=True, exist_ok=True)
        if drive_src.is_file():
            shutil.copy2(drive_src, local_dest_dir / drive_src.name)
        else:
            for root, _, files in os.walk(drive_src):
                for file in files:
                    d_file = Path(root) / file
                    rel = d_file.relative_to(drive_src)
                    out = local_dest_dir / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(d_file, out)

        logger.info("Restored %s from Drive to %s", relative_src_dir, local_dest_dir)
        return True
