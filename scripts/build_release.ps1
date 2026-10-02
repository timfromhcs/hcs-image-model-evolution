# Build Release Packaging Script for Windows
$ErrorActionPreference = "Stop"

Write-Host "=== Assembling Final Release Package ===" -ForegroundColor Cyan

$PYTHON_EXE = ".venv\Scripts\python.exe"
if (-not (Test-Path $PYTHON_EXE)) {
    $PYTHON_EXE = "python"
}

& $PYTHON_EXE -c "
from pathlib import Path
from hcs_image_evolution.sdcpp.package import ReleasePackager
from hcs_image_evolution.conversion.manifests import ReleaseManifest, ManifestGenerator

dist_dir = Path('dist')
dist_dir.mkdir(parents=True, exist_ok=True)

manifest = ReleaseManifest(sha256='placeholder')
manifest_path = ManifestGenerator.generate(manifest, dist_dir / 'MANIFEST.json')

packager = ReleasePackager(dist_root=dist_dir)
packager.create_package(
    package_name='HCS-Image-Evolver-v1.0',
    sd_cli_path=Path('build/sd-cli.exe'),
    gguf_model_path=Path('dist/hcs-image-evolver-q4.gguf'),
    manifest_path=manifest_path,
)
print('Package assembled successfully.')
"

Write-Host "[OK] Release assembled in dist/" -ForegroundColor Green
