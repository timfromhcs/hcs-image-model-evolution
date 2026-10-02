"""Unit tests for exact and perceptual image deduplication."""

from pathlib import Path

from PIL import Image

from hcs_image_evolution.data.dedupe import Deduplicator


def test_deduplicator(tmp_path: Path):
    dedup = Deduplicator(phash_threshold=6)

    # 1. Create original image
    img1_path = tmp_path / "img1.png"
    img1 = Image.new("RGB", (100, 100), color="blue")
    img1.save(img1_path)

    is_dup1, _ = dedup.is_duplicate(img1_path)
    assert not is_dup1

    # 2. Exact copy
    img2_path = tmp_path / "img2.png"
    img1.save(img2_path)
    is_dup2, reason2 = dedup.is_duplicate(img2_path)
    assert is_dup2
    assert "Exact" in reason2

    # 3. Highly distinct image
    img3_path = tmp_path / "img3.png"
    img3 = Image.new("RGB", (100, 100), color="yellow")
    # Add random pattern
    for x in range(50):
        img3.putpixel((x, x), (255, 0, 0))
    img3.save(img3_path)

    is_dup3, _ = dedup.is_duplicate(img3_path)
    assert not is_dup3
