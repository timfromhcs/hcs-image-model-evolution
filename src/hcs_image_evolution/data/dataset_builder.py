"""Orchestrates dataset curation, deduplication, balancing, and snapshot creation."""

from pathlib import Path
from typing import Sequence
from hcs_image_evolution.data.balancing import DatasetBalancer
from hcs_image_evolution.data.dedupe import Deduplicator
from hcs_image_evolution.data.schemas import DatasetSnapshotManifest, SampleRecord
from hcs_image_evolution.data.shards import ShardBuilder
from hcs_image_evolution.storage.atomic import atomic_write_json
from hcs_image_evolution.utils.logging import log_event, logger


class DatasetBuilder:
    """End-to-end dataset curation and snapshot pipeline."""

    def __init__(
        self,
        snapshot_id: str = "dataset-001",
        output_dir: Path = Path("manifests/datasets"),
        shards_dir: Path = Path("state/shards"),
    ):
        self.snapshot_id = snapshot_id
        self.output_dir = output_dir
        self.shards_dir = shards_dir
        self.deduplicator = Deduplicator()
        self.balancer = DatasetBalancer()
        self.shard_builder = ShardBuilder(output_dir=shards_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_snapshot(
        self,
        candidate_samples: Sequence[tuple[SampleRecord, Path]],
        min_quality_score: float = 0.70,
    ) -> DatasetSnapshotManifest:
        """Filters, deduplicates, shards, and produces an immutable snapshot manifest."""
        accepted: list[tuple[SampleRecord, Path]] = []

        for record, img_path in candidate_samples:
            # 1. Quality gate
            overall_score = record.judge_scores.get("overall", 1.0)
            if overall_score < min_quality_score:
                continue

            # 2. Deduplication check
            is_dup, reason = self.deduplicator.is_duplicate(img_path)
            if is_dup:
                logger.debug("Deduplication rejected sample %s: %s", record.sample_id, reason)
                continue

            accepted.append((record, img_path))

        # Write shards
        shard_paths = []
        checksums = {}
        batch_size = 1000
        for i in range(0, len(accepted), batch_size):
            chunk = accepted[i : i + batch_size]
            shard_path, sha256 = self.shard_builder.write_shard(i // batch_size, chunk)
            shard_paths.append(shard_path.name)
            checksums[shard_path.name] = sha256

        accepted_records = [r for r, _ in accepted]
        synthetic_count = sum(1 for r in accepted_records if r.source_type == "synthetic")
        real_count = sum(1 for r in accepted_records if r.source_type == "real")
        edit_count = sum(1 for r in accepted_records if r.source_type == "edit")

        manifest = DatasetSnapshotManifest(
            snapshot_id=self.snapshot_id,
            total_samples=len(accepted),
            synthetic_samples=synthetic_count,
            real_samples=real_count,
            edit_samples=edit_count,
            shards=shard_paths,
            sha256_checksums=checksums,
            licenses=list(set(r.license for r in accepted_records)),
        )

        manifest_file = self.output_dir / f"{self.snapshot_id}.json"
        atomic_write_json(manifest_file, manifest.model_dump())
        log_event("dataset_snapshot_built", {"snapshot_id": self.snapshot_id, "samples": len(accepted)})
        logger.info("Successfully created dataset snapshot %s with %d samples", self.snapshot_id, len(accepted))
        return manifest
