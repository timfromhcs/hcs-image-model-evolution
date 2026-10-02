"""PyTorch dataset loader for curated images and text prompts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from hcs_image_evolution.data.schemas import SampleRecord


class EvolutionDataset(Dataset):
    """Dataset providing conditioned image-text pairs with resolution normalization."""

    def __init__(
        self,
        samples: Sequence[tuple[SampleRecord, Path]],
        target_resolution: int = 1024,
    ):
        self.samples = samples
        self.target_resolution = target_resolution
        self.transform = T.Compose([
            T.Resize((target_resolution, target_resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record, img_path = self.samples[idx]
        with Image.open(img_path).convert("RGB") as img:
            image_tensor = self.transform(img)

        return {
            "image": image_tensor,
            "prompt": record.source_prompt,
            "caption": record.caption,
            "sample_id": record.sample_id,
            "source_type": record.source_type,
        }
