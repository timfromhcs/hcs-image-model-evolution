# Dataset Pipeline & Provenance

## Data Composition
- **Synthetic Teacher Generations (50%)**: Generated via Qwen-Image-2.1 with D0-D9 difficulty curriculum.
- **Permissive Licensed Real Images (25%)**: Verified MIT, Apache 2.0, CC0, CC-BY-4.0 imagery.
- **Synthetic Editing Pairs (15%)**: Real and synthetic images with paired instruction transforms.
- **Real Paired Editing (10%)**: Documented benchmark pairs.

## Integrity & Quality Gates
- **Multi-Level Deduplication**: Exact SHA-256 and perceptual pHash/dHash checks.
- **Multi-VLM Ensemble**: Acceptance requires passing quality thresholds ($\ge 0.70$) and low rater disagreement ($\le 0.35$).
- **WebDataset Sharding**: Sharded `.tar.gz` packages with companion Parquet/JSON manifests.
