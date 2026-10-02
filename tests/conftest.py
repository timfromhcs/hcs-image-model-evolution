"""Pytest fixtures for unit and integration testing."""

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Provides an isolated temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Creates a sample test PNG image."""
    img_path = tmp_path / "test_image.png"
    img = Image.new("RGB", (256, 256), color=(128, 64, 200))
    img.save(img_path)
    return img_path
