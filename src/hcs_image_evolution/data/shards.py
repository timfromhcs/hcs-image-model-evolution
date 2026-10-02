"""Dataset sharding into WebDataset tar archives and metadata tables."""

import json
import tarfile
from pathlib import Path
from typing import Sequence
from hcs_image_evolution.data.schemas import SampleRecord
from hcs_image_evolution.utils.hashing import compute_file_sha256
from hcs_image_evolution.utils.logging import logger


class ShardBuilder:
    """Packs images and metadata records into standardized WebDataset tar shards."""

    def __init__(self, output_dir: Path = Path("state/shards"), max_samples_per_shard: int = 1000):
        self.output_dir = output_dir
        self.max_samples = max_samples_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_shard(
        self,
        shard_idx: int,
        samples: Sequence[tuple[SampleRecord, Path]],
    ) -> tuple[Path, str]:
        """Creates a single tar.gz shard containing images and .json metadata."""
        shard_name = f"shard-{shard_idx:06d}.tar.gz"
        shard_path = self.output_dir / shard_name

        with tarfile.open(shard_path, "w:gz") as tar:
            for record, img_path in samples:
                base_name = f"{record.sample_id}"
                # Add image
                tar.add(img_path, arcname=f"{base_name}.png")
                # Add JSON metadata
                meta_json = json.dumps(record.model_dump(), default=str).encode("utf-8")
                ti = tarfile.TarInfo(name=f"{base_name}.json")
                ti.size = len(meta_json)
                import io
                tar.addfile(ti, io.BytesIO(meta_json))

        sha256 = compute_file_sha256(shard_path)
        logger.info("Wrote shard %s with %d samples (SHA256: %s)", shard_name, len(samples), sha256[:8])
        return shard_path, sha256
