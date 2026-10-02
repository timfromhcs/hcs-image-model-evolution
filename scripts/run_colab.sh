#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Autonomous Evolution Cycle on Colab ==="
python -m hcs_image_evolution.cli agent run --config configs/base.yaml
