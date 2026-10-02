# Reproducibility Guide

## Core Tenets
Every released model checkpoint, dataset shard, and benchmark report is fully deterministic and traceable:

1. **Git Commit Hash**: Stored in every checkpoint and release manifest.
2. **Environment Manifest**: Hash of all installed Python dependencies and CUDA/hardware parameters.
3. **Dataset Snapshot ID**: Exact WebDataset shard references and SHA-256 integrity sums.
4. **Seed Policy**: Full RNG serialization across Python, NumPy, and PyTorch.
5. **No Synthetic Hallucinations**: All reported benchmarks represent physical executions against verified hardware.
