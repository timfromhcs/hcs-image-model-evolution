"""Unit tests for licensing validation gates."""

from hcs_image_evolution.data.licensing import LicenseGate
from hcs_image_evolution.data.schemas import DatasetSourceRecord


def test_license_gate():
    gate = LicenseGate()

    valid_src = DatasetSourceRecord(
        dataset_id="permissive_dataset",
        license="apache-2.0",
        source_url="https://huggingface.co/datasets/test",
    )
    ok, _ = gate.validate_source(valid_src)
    assert ok

    invalid_src = DatasetSourceRecord(
        dataset_id="restrictive_dataset",
        license="non-commercial-only",
        source_url="https://huggingface.co/datasets/test2",
    )
    ok2, msg = gate.validate_source(invalid_src)
    assert not ok2
    assert "not in the approved permissive whitelist" in msg
