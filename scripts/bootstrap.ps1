# Windows 11 Local Bootstrap Script for HCS Image Evolution Lab
$ErrorActionPreference = "Stop"

Write-Host "=== Bootstrapping Local Environment ===" -ForegroundColor Cyan

# 1. Check Python
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[OK] uv is available. Setting up virtual environment..." -ForegroundColor Green
    if (-not (Test-Path ".venv")) {
        uv venv .venv --python 3.12
    }
    & .venv\Scripts\python.exe -m pip install -e ".[dev]"
} else {
    Write-Host "[INFO] uv not found, checking standard python..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\python.exe -m pip install -e ".[dev]"
}

# 2. Check Git and Vulkan
if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Git is available." -ForegroundColor Green
}
if (Get-Command vulkaninfo -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Vulkan runtime detected." -ForegroundColor Green
} else {
    Write-Host "[WARNING] vulkaninfo not in PATH. Vulkan inference may fall back to CPU." -ForegroundColor Yellow
}

Write-Host "=== Local Bootstrap Complete ===" -ForegroundColor Cyan
