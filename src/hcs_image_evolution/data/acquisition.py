"""Hugging Face dataset search and verified permissive dataset acquisition."""

from typing import Optional
from huggingface_hub import HfApi
from hcs_image_evolution.data.licensing import LicenseGate
from hcs_image_evolution.data.schemas import DatasetSourceRecord
from hcs_image_evolution.utils.logging import logger


class DatasetAcquisition:
    """Discovers and validates real image datasets from Hugging Face Hub."""

    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.license_gate = LicenseGate()

    def search_permissive_datasets(self, query: str = "image", limit: int = 10) -> list[DatasetSourceRecord]:
        """Searches Hugging Face for public image datasets matching permissive licenses."""
        results = []
        try:
            datasets = self.api.list_datasets(search=query, limit=limit, full=True)
            for d in datasets:
                license_id = getattr(d.card_data, "license", "") if d.card_data else ""
                if not license_id:
                    license_id = "unknown"

                is_allowed = self.license_gate.is_license_allowed(license_id)
                rec = DatasetSourceRecord(
                    dataset_id=d.id,
                    revision=d.sha or "main",
                    license=license_id,
                    source_url=f"https://huggingface.co/datasets/{d.id}",
                    allowed_for_training=is_allowed,
                    allowed_for_redistribution=is_allowed,
                    notes="Automated discovery",
                )
                results.append(rec)
        except Exception as e:
            logger.error("Failed to query Hugging Face datasets: %s", e)

        return results
