# Launch script for Autonomous Evolution Agent on Windows
$ErrorActionPreference = "Stop"

Write-Host "Starting HCS Image Evolution Agent..." -ForegroundColor Cyan

$PYTHON_EXE = ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

& $PYTHON_EXE -m hcs_image_evolution.cli agent run --config configs/base.yaml
