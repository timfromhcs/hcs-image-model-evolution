#!/usr/bin/env bash
set -euo pipefail

echo "=== Bootstrapping Google Colab Environment ==="

# 1. GPU Diagnostics
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "[WARNING] No NVIDIA GPU detected! Training requires a GPU runtime."
fi

# 2. Install pinned dependencies
if [ -f "requirements-colab.lock" ]; then
    echo "Installing pinned dependencies from requirements-colab.lock..."
    pip install --no-cache-dir -r requirements-colab.lock
else
    pip install -e ".[colab]"
fi

# 3. Create persistent directories
mkdir -p state/errors state/checkpoints state/shards state/artifacts benchmarks/reports

echo "=== Google Colab Bootstrap Complete ==="
