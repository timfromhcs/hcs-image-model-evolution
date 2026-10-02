"""Licensing validation gates protecting against unsafe or non-permissive datasets."""

from typing import Optional
from hcs_image_evolution.data.schemas import DatasetSourceRecord
from hcs_image_evolution.utils.logging import log_event, logger

PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "openrail",
    "openrail++",
}


class LicenseGate:
    """Enforces license whitelist checks before data ingestion."""

    def __init__(self, allowed_licenses: Optional[set[str]] = None):
        self.allowed = allowed_licenses or PERMISSIVE_LICENSES

    def is_license_allowed(self, license_str: str) -> bool:
        """Checks if a license identifier is compatible with training releases."""
        norm = license_str.strip().lower()
        return norm in self.allowed

    def validate_source(self, source: DatasetSourceRecord) -> tuple[bool, str]:
        """Validates that a dataset source has documented and permitted licensing."""
        if not source.license:
            return False, "Dataset missing explicit license identifier"

        if not self.is_license_allowed(source.license):
            msg = f"License '{source.license}' is not in the approved permissive whitelist"
            logger.warning("License check failed for %s: %s", source.dataset_id, msg)
            log_event("license_gate_rejected", {"dataset_id": source.dataset_id, "license": source.license})
            return False, msg

        return True, "License approved"
