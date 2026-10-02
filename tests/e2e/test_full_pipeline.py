"""Full end-to-end pipeline test matching PLAN.md section 81."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from hcs_image_evolution.agent.recovery import RecoveryManager
from hcs_image_evolution.conversion.gguf_export import GGUFExporter
from hcs_image_evolution.conversion.manifests import ManifestGenerator, ReleaseManifest
from hcs_image_evolution.data.dataset_builder import DatasetBuilder
from hcs_image_evolution.data.provenance import ProvenanceTracker
from hcs_image_evolution.evaluation.suite import BenchmarkSuite
from hcs_image_evolution.training.checkpointing import CheckpointManager
from hcs_image_evolution.training.datasets import EvolutionDataset
from hcs_image_evolution.training.trainer import EvolutionTrainer


def test_full_pipeline_e2e(tmp_path: Path):
    # 1. Pipeline directories
    shards_dir = tmp_path / "shards"
    ckpts_dir = tmp_path / "checkpoints"
    manifests_dir = tmp_path / "manifests"
    reports_dir = tmp_path / "reports"

    # 2. Synthetic sample generation simulation
    sample_img = tmp_path / "seed_gen.png"
    Image.new("RGB", (64, 64), color="cyan").save(sample_img)

    rec = ProvenanceTracker.create_record(
        sample_id="sample-e2e-001",
        image_path=sample_img,
        source_prompt="A crystal sphere",
        caption="A gleaming cyan sphere",
        judge_scores={"overall": 0.95, "quality": 0.95},
    )

    # 3. Dataset curation and snapshot
    builder = DatasetBuilder(snapshot_id="e2e-snap", output_dir=manifests_dir, shards_dir=shards_dir)
    builder.build_snapshot([(rec, sample_img)])

    # 4. Tiny model training
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, kernel_size=3, padding=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = EvolutionDataset([(rec, sample_img)], target_resolution=64)
    loader = DataLoader(dataset, batch_size=1)
    ckpt_mgr = CheckpointManager(checkpoints_dir=ckpts_dir)

    trainer = EvolutionTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
        checkpoint_manager=ckpt_mgr,
        device="cpu",
    )
    trainer.train_steps(num_steps=2, checkpoint_interval=2)

    # 5. Checkpoint & Resume verification
    recovery = RecoveryManager(checkpoints_dir=ckpts_dir)
    valid_ckpt = recovery.find_latest_valid_checkpoint()
    assert valid_ckpt is not None

    # Resume into fresh model instance
    resumed_model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, kernel_size=3, padding=1),
    )
    ckpt_mgr.load_checkpoint(valid_ckpt, resumed_model)

    # 6. GGUF Export
    gguf_path = tmp_path / "e2e_student.gguf"
    _, gguf_sha = GGUFExporter.export_state_dict_to_gguf(resumed_model.state_dict(), gguf_path)
    assert gguf_path.is_file()

    # 7. Release Manifest
    manifest = ReleaseManifest(
        model_name="HCS-Image-Evolver-E2E",
        sha256=gguf_sha,
    )
    ManifestGenerator.generate(manifest, tmp_path / "MANIFEST.json")

    # 8. Benchmark Suite
    suite = BenchmarkSuite(reports_dir=reports_dir)
    report = suite.run_suite(
        model_name="HCS-Image-Evolver-E2E",
        evaluate_fn=None,
        prompt_manifest=[{"prompt": "A crystal sphere"}],
    )
    assert report["status"] == "PASS"
