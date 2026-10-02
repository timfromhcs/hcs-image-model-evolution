# Windows Release Validation Script
$ErrorActionPreference = "Stop"

Write-Host "=== Validating Release Artifacts ===" -ForegroundColor Cyan

# 1. Run Tests
Write-Host "Running test suite..." -ForegroundColor Yellow
& .venv\Scripts\pytest.exe tests/unit tests/integration

# 2. Secret Scan
Write-Host "Running secret scan..." -ForegroundColor Yellow
$leakFound = $false
$patterns = @("ghp_", "hf_", "AIzaSy")
foreach ($p in $patterns) {
    $matches = git grep -I "$p" -- ":!scripts/validate_release.ps1" 2>$null
    if ($matches) {
        Write-Host "[ERROR] Potential secret detected matching $p" -ForegroundColor Red
        $leakFound = $true
    }
}

if ($leakFound) {
    Write-Host "[FAIL] Secret scan failed. Aborting release." -ForegroundColor Red
    exit 1
}

Write-Host "[PASS] All pre-release verification gates cleared." -ForegroundColor Green
