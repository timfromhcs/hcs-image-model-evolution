"""Integration test verifying data assembly, tiny training step, and GGUF export."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from hcs_image_evolution.conversion.gguf_export import GGUFExporter
from hcs_image_evolution.data.dataset_builder import DatasetBuilder
from hcs_image_evolution.data.provenance import ProvenanceTracker
from hcs_image_evolution.distillation.few_step import FewStepSampler
from hcs_image_evolution.training.checkpointing import CheckpointManager
from hcs_image_evolution.training.datasets import EvolutionDataset
from hcs_image_evolution.training.trainer import EvolutionTrainer


def test_integration_evolution_flow(tmp_path: Path):
    # 1. Create a dummy synthetic image & record
    img_path = tmp_path / "sample_001.png"
    Image.new("RGB", (64, 64), color="red").save(img_path)

    rec = ProvenanceTracker.create_record(
        sample_id="sample-001",
        image_path=img_path,
        source_prompt="A red cube",
        caption="A geometric red cube",
    )

    # 2. Build dataset snapshot
    builder = DatasetBuilder(
        snapshot_id="test-snap",
        output_dir=tmp_path / "manifests",
        shards_dir=tmp_path / "shards",
    )
    manifest = builder.build_snapshot([(rec, img_path)])
    assert manifest.total_samples == 1

    # 3. Initialize tiny model and train 2 steps
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, kernel_size=3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = EvolutionDataset([(rec, img_path)], target_resolution=64)
    loader = DataLoader(dataset, batch_size=1)
    ckpt_mgr = CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")

    trainer = EvolutionTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        checkpoint_manager=ckpt_mgr,
        device="cpu",
    )
    metrics = trainer.train_steps(num_steps=2, checkpoint_interval=2)
    assert metrics["steps"] == 2

    # 4. Low-NFE inference pass
    sampler = FewStepSampler(student_model=model, device="cpu")
    out_tensor = sampler.sample(batch_size=1, channels=3, height=64, width=64, num_steps=2)
    assert out_tensor.shape == (1, 3, 64, 64)

    # 5. GGUF Export
    gguf_path = tmp_path / "test_model.gguf"
    exported_path, sha = GGUFExporter.export_state_dict_to_gguf(model.state_dict(), gguf_path)
    assert exported_path.exists()
