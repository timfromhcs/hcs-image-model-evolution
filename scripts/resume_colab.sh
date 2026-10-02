#!/usr/bin/env bash
set -euo pipefail

DRIVE_DIR="/content/drive/MyDrive/HCS_Image_Evolution"
echo "=== Resuming Evolution Run from Google Drive ==="

if [ -d "$DRIVE_DIR" ]; then
    echo "Found Drive mount at $DRIVE_DIR. Syncing latest state..."
    mkdir -p state
    if [ -f "$DRIVE_DIR/latest/run_state.json" ]; then
        cp "$DRIVE_DIR/latest/run_state.json" state/run_state.json
        echo "Restored run_state.json from Drive."
    fi
fi

python -m hcs_image_evolution.cli agent run --config configs/base.yaml
