# PLAN.md

# Autonomous Image Model Evolution Lab

## Mission

Build a fully autonomous, reproducible, self-improving image-model research and production pipeline from an empty directory.

The system must:

1. run locally on Windows 11 as an autonomous CLI coding/research agent;
2. use Google Colab GPU runtimes for heavy image generation, training and distillation;
3. persist every important state transition and checkpoint to Google Drive;
4. automatically recover after Colab runtime termination;
5. generate large synthetic image datasets with Qwen-Image-2.1;
6. evaluate generated images through multiple independent VLM judges through Hugging Face Inference Providers;
7. retain only sufficiently high-quality, diverse, license-safe training data;
8. combine synthetic data with real images from explicitly compatible public datasets;
9. create T2I, I2I, editing, multi-reference and high-resolution training examples;
10. train an original image-generation model derived from the Qwen-Image-2.1 teacher;
11. distill the model toward 4-step inference;
12. investigate compact student architectures and low-bit quantization;
13. convert the final diffusion model to GGUF;
14. run the final artifact through real `stable-diffusion.cpp` inference;
15. benchmark quality, speed, memory use and failure modes;
16. automatically reject bad checkpoints;
17. automatically promote good checkpoints;
18. publish only reproducible, tested releases to GitHub and optionally Hugging Face;
19. create a complete README containing exact installation, inference, training, benchmark and provenance information;
20. never use mocks, fake benchmark numbers, fabricated outputs or simulated success states.

The agent is not allowed to say that something works merely because code compiles.

A component is considered working only after it has been executed against the real dependency/model/data path and produced verifiable output.

---

# 1. Core Principle

The project is a closed-loop model evolution system:

```text
                     ┌─────────────────────────────┐
                     │       Research Planner      │
                     │ curriculum / experiments   │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Prompt Generator   │
                         │ T2I / I2I / Edit   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Qwen Image 2.1     │
                         │ Teacher Generator  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                     ┌─────────────────────────────┐
                     │ Multi-VLM Evaluation        │
                     │ quality / alignment / art   │
                     │ anatomy / text / artifacts  │
                     │ edit fidelity / safety      │
                     └──────────────┬──────────────┘
                                    │
                      rejected ◄────┴────► accepted
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ Dataset Builder    │
                         │ dedupe / balance   │
                         │ provenance / caps  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Student Training   │
                         │ flow + distill     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Benchmark Engine   │
                         │ quality / speed    │
                         │ VRAM / edit / T2I  │
                         └─────────┬──────────┘
                                   │
                        worse ◄────┴────► better
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Promote Checkpoint │
                         │ package / GGUF     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ sd.cpp Validation  │
                         │ real local runtime │
                         └─────────┬──────────┘
                                   │
                                   ▼
                              GitHub Release

                                   ▲
                                   │
                         persistent state
                                   │
                         Google Drive Backup
```

This loop is the core product.

Do not implement it as a single giant Python script.

Use explicit modules, persistent state, checkpoints, manifests and resumable jobs.

---

# 2. Project Identity

Suggested project name:

```text
HCS Image Model Evolution Lab
```

Suggested repository:

```text
hcs-image-model-evolution
```

Suggested package/model naming convention:

```text
HCS-Image-Evolver
HCS-Image-Evolver-Turbo-4Step
HCS-Image-Evolver-Edit
HCS-Image-Evolver-HD
```

Every generated artifact must carry:

```text
project_name
model_name
model_version
training_run_id
git_commit
dataset_snapshot
teacher_revision
training_config_hash
environment_hash
checkpoint_step
seed_policy
```

Never allow an anonymous checkpoint to become a release.

---

# 3. Non-Negotiable Rules

## 3.1 No mocks

Never create:

* fake VLM scores;
* fake benchmark results;
* fake model outputs;
* dummy GGUF files;
* placeholder checkpoints presented as trained artifacts;
* simulated Colab execution;
* fabricated GPU statistics;
* hard-coded "PASS" results.

Test doubles are allowed only for unit tests of isolated parsing/state-management logic.

Anything claiming model functionality must use the real model/runtime.

---

## 3.2 No silent failure

Every failure must be persisted.

Use:

```text
state/errors/
state/events.jsonl
state/run_state.json
state/checkpoints/
state/artifacts/
```

Every exception must contain:

```json
{
  "timestamp": "...",
  "stage": "...",
  "run_id": "...",
  "git_commit": "...",
  "error_type": "...",
  "message": "...",
  "traceback_path": "...",
  "last_checkpoint": "...",
  "recoverable": true,
  "retry_count": 2
}
```

---

## 3.3 No destructive overwrite

Never overwrite the only checkpoint.

Write:

```text
checkpoint-000100/
checkpoint-000200/
checkpoint-000300/
```

Then create:

```text
latest.json
best.json
```

as pointers.

Use atomic file replacement.

---

## 3.4 Every promoted checkpoint must beat a baseline

A model is not promoted simply because the loss decreased.

Promotion requires passing configured gates for:

* image quality;
* prompt alignment;
* diversity;
* editing fidelity;
* text rendering;
* anatomy/object consistency;
* resolution;
* runtime stability;
* memory usage;
* inference latency;
* regression tests.

---

# 4. Reality About Google Colab

The system must assume runtime termination.

Colab's official FAQ states that free runtimes can run for up to 12 hours depending on availability and usage. Pro/Pro+/Pay As You Go increase availability, and Pro+ can support continuous execution up to 24 hours under the documented conditions. This still does not constitute an unlimited 24/7 runtime.

Therefore:

```text
247 != one permanent Colab process

247 == permanent resumable state machine
```

The agent must be capable of:

```text
Runtime starts
        ↓
mount Drive
        ↓
read run_state.json
        ↓
verify checkpoint
        ↓
verify environment
        ↓
verify data shard
        ↓
resume exact unfinished job
        ↓
checkpoint
        ↓
backup
        ↓
continue
```

After runtime death:

```text
new runtime
    ↓
same notebook
    ↓
same Drive
    ↓
same run state
    ↓
resume
```

Never require a human to manually determine where training stopped.

---

# 5. Repository Structure

Create the repository from an empty directory.

Target structure:

```text
hcs-image-model-evolution/
│
├── AGENTS.md
├── GEMINI.md
├── PLAN.md
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-colab.lock
├── .gitignore
├── .gitattributes
├── .python-version
│
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── generation.yaml
│   ├── judging.yaml
│   ├── training.yaml
│   ├── distillation.yaml
│   ├── benchmark.yaml
│   ├── sdcpp.yaml
│   └── experiments/
│
├── src/
│   └── hcs_image_evolution/
│       ├── __init__.py
│       ├── cli.py
│       │
│       ├── agent/
│       │   ├── planner.py
│       │   ├── state_machine.py
│       │   ├── experiment_manager.py
│       │   ├── recovery.py
│       │   └── promotion.py
│       │
│       ├── data/
│       │   ├── schemas.py
│       │   ├── acquisition.py
│       │   ├── provenance.py
│       │   ├── licensing.py
│       │   ├── dedupe.py
│       │   ├── balancing.py
│       │   ├── captions.py
│       │   ├── edits.py
│       │   ├── shards.py
│       │   └── dataset_builder.py
│       │
│       ├── generation/
│       │   ├── qwen_teacher.py
│       │   ├── prompt_curriculum.py
│       │   ├── generation_queue.py
│       │   └── generation_manifest.py
│       │
│       ├── judging/
│       │   ├── hf_provider.py
│       │   ├── vlm_rater.py
│       │   ├── ensemble.py
│       │   ├── schemas.py
│       │   └── calibration.py
│       │
│       ├── training/
│       │   ├── model_loader.py
│       │   ├── datasets.py
│       │   ├── losses.py
│       │   ├── trainer.py
│       │   ├── checkpointing.py
│       │   └── schedules.py
│       │
│       ├── distillation/
│       │   ├── teacher_targets.py
│       │   ├── flow_matching.py
│       │   ├── trajectory_matching.py
│       │   ├── dmd.py
│       │   ├── feature_distill.py
│       │   ├── few_step.py
│       │   └── compact_student.py
│       │
│       ├── evaluation/
│       │   ├── suite.py
│       │   ├── text_to_image.py
│       │   ├── image_editing.py
│       │   ├── multiref.py
│       │   ├── typography.py
│       │   ├── highres.py
│       │   ├── diversity.py
│       │   └── performance.py
│       │
│       ├── conversion/
│       │   ├── safetensors_export.py
│       │   ├── gguf_export.py
│       │   ├── quantize.py
│       │   ├── manifests.py
│       │   └── checksums.py
│       │
│       ├── sdcpp/
│       │   ├── downloader.py
│       │   ├── builder.py
│       │   ├── runner.py
│       │   ├── validator.py
│       │   └── package.py
│       │
│       ├── storage/
│       │   ├── drive.py
│       │   ├── hf.py
│       │   ├── atomic.py
│       │   └── cache.py
│       │
│       └── utils/
│           ├── hashing.py
│           ├── logging.py
│           ├── system.py
│           └── reproducibility.py
│
├── scripts/
│   ├── bootstrap.ps1
│   ├── bootstrap_colab.sh
│   ├── run_agent.ps1
│   ├── run_colab.sh
│   ├── resume_colab.sh
│   ├── validate_release.ps1
│   └── build_release.ps1
│
├── notebooks/
│   ├── HCS_Image_Evolution_Colab.ipynb
│   ├── HCS_Image_Evolution_Resume.ipynb
│   └── HCS_Image_Evolution_Eval.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
├── benchmarks/
│   ├── prompts/
│   ├── edit_cases/
│   ├── expected/
│   └── reports/
│
├── manifests/
│   ├── datasets/
│   ├── models/
│   └── releases/
│
├── docs/
│   ├── architecture.md
│   ├── training.md
│   ├── distillation.md
│   ├── dataset.md
│   ├── sdcpp.md
│   └── reproducibility.md
│
└── state/
    └── .gitkeep
```

Runtime state and huge datasets must never be committed to Git.

---

# 6. Agent Responsibilities

The autonomous agent must operate in phases.

```text
BOOT
 ↓
DISCOVER
 ↓
VERIFY
 ↓
PLAN
 ↓
IMPLEMENT
 ↓
SMOKE TEST
 ↓
GENERATION
 ↓
DATA CURATION
 ↓
TRAIN
 ↓
DISTILL
 ↓
BENCHMARK
 ↓
PROMOTE / ROLLBACK
 ↓
QUANTIZE
 ↓
SD.CPP TEST
 ↓
PACKAGE
 ↓
DOCUMENT
 ↓
GITHUB RELEASE
```

The agent may return to any previous phase when new evidence requires it.

---

# 7. Local Bootstrap From Empty Folder

The agent starts with:

```text
empty/
```

and must autonomously:

1. inspect operating system;
2. inspect CPU;
3. inspect available GPUs;
4. inspect RAM;
5. inspect storage;
6. inspect Python;
7. inspect Git;
8. inspect GitHub CLI;
9. inspect Hugging Face CLI;
10. inspect internet connectivity;
11. inspect Vulkan if local `sd.cpp` validation is enabled;
12. initialize repository;
13. initialize Python package;
14. create configs;
15. create tests;
16. create Colab notebook;
17. create CI;
18. create documentation skeleton.

Local hardware must not be assumed to be capable of training the teacher.

The local machine is primarily:

```text
agent controller
developer environment
artifact validator
sd.cpp inference machine
GitHub client
dataset/model manager
```

Colab is:

```text
GPU generation
GPU training
GPU distillation
GPU evaluation
```

---

# 8. Environment Reproducibility

Create one explicit version lock.

Do not use uncontrolled:

```text
pip install -U everything
```

inside the final training pipeline.

Instead:

```text
pyproject.toml
requirements-colab.lock
environment-manifest.json
```

must contain exact package versions.

Qwen-Image-2.1 currently documents a recent `diffusers` installation path and recent Transformers requirements, while community 4-/6-step Qwen-2.1 work may pin a specific Git revision. The agent must therefore discover a real compatible revision during bootstrap and freeze it after a successful end-to-end test.

Record:

```text
python_version
torch_version
cuda_version
diffusers_git_revision
transformers_version
accelerate_version
peft_version
pillow_version
huggingface_hub_version
numpy_version
xformers_version_if_used
sdcpp_commit
compiler_version
```

Store the manifest with every checkpoint.

---

# 9. Google Drive Layout

Mount Drive in Colab.

Use:

```text
Google Drive/
└── HCS_Image_Evolution/
    ├── runs/
    │   └── RUN-YYYYMMDD-HHMMSS/
    │       ├── state/
    │       ├── checkpoints/
    │       ├── optimizer/
    │       ├── datasets/
    │       ├── generations/
    │       ├── judgments/
    │       ├── benchmarks/
    │       ├── artifacts/
    │       ├── manifests/
    │       └── logs/
    │
    ├── cache/
    ├── models/
    ├── datasets/
    └── latest/
```

Use the Drive copy as recovery storage, not as the active high-throughput training filesystem.

Recommended pattern:

```text
Colab local SSD
     ↓
write checkpoint
     ↓
verify checksum
     ↓
atomic finalize
     ↓
copy to Drive
     ↓
verify Drive checksum
     ↓
update latest.json
```

Do not train directly from huge Drive-mounted files when local staging is practical.

---

# 10. Persistent Run State

Create:

```text
run_state.json
```

Example conceptual schema:

```json
{
  "run_id": "RUN-20261002-001",
  "stage": "distillation",
  "experiment_id": "exp-014",
  "model_revision": "student-014",
  "global_step": 12400,
  "epoch": 3,
  "dataset_snapshot": "dataset-008",
  "teacher_revision": "Qwen/Qwen-Image-2.1@REV",
  "best_checkpoint": "checkpoint-12000",
  "latest_checkpoint": "checkpoint-12400",
  "rng": {
    "python": "...",
    "numpy": "...",
    "torch": "..."
  },
  "environment_hash": "...",
  "config_hash": "...",
  "status": "running"
}
```

Also save:

```text
events.jsonl
```

for append-only event history.

---

# 11. Synthetic Data Engine

## Goal

Use Qwen-Image-2.1 as the high-quality teacher.

Qwen-Image-2.1 officially supports both text-to-image and image-conditioned editing, supports multiple reference images, and supports RGBA generation. The model card documents native 2048-scale examples and several larger aspect-ratio configurations.

The teacher must generate:

```text
T2I
I2I
instruction editing
mask/local editing
multi-reference
RGBA
typography
photography
illustration
product imagery
architecture
landscape
portrait
character
environment
concept art
graphic design
```

---

# 12. Prompt Curriculum Generator

Do not generate random prompts only.

Create a curriculum engine.

Prompt dimensions:

```text
subject
action
pose
environment
foreground
midground
background
composition
camera
lens
perspective
lighting
materials
texture
color
style
mood
typography
spatial relations
counting
object attributes
negative constraints
editing instruction
reference-image relation
```

Use difficulty buckets:

```text
D0 = simple single subject
D1 = multiple objects
D2 = detailed scene
D3 = relationships
D4 = counting
D5 = typography
D6 = complex composition
D7 = multi-reference editing
D8 = difficult real-world editing
D9 = adversarial edge cases
```

The agent should automatically increase the proportion of prompts corresponding to the model's current weaknesses.

Example:

```text
benchmark detects poor text rendering
        ↓
planner creates more typography prompts
        ↓
teacher generates candidate data
        ↓
VLM judges typography
        ↓
best examples enter next training shard
        ↓
model retrains
        ↓
benchmark tests typography again
```

That is the beginning of the "miniAGI" behavior.

---

# 13. Teacher Generation Strategy

Never generate one image and blindly store it.

Generate candidates.

Example:

```text
prompt
 ├── seed A
 ├── seed B
 ├── seed C
 └── seed D
```

Then judge all candidates.

The generator must save:

```text
image
prompt
seed
resolution
aspect_ratio
teacher_revision
scheduler
steps
guidance
generation_time
GPU
config_hash
```

Store metadata beside every image.

---

# 14. Multi-VLM Judging

Hugging Face Inference Providers currently provide a unified interface to multiple hosted inference providers and support VLM inference through several providers. The Hugging Face client supports provider selection through an API such as `InferenceClient`.

Do not trust one VLM.

Create an ensemble:

```text
Judge A
Judge B
Judge C
Judge D
Judge E
```

The exact active models/providers must be dynamically discovered from available Hugging Face providers at run time.

Do not hard-code model names indefinitely.

Each judge receives the same image and task description.

Each judge returns strict JSON:

```json
{
  "quality": 0,
  "aesthetics": 0,
  "prompt_alignment": 0,
  "composition": 0,
  "detail": 0,
  "anatomy": 0,
  "object_consistency": 0,
  "text_rendering": 0,
  "realism": 0,
  "edit_fidelity": 0,
  "artifact_penalty": 0,
  "safety": 0,
  "confidence": 0,
  "description": "",
  "failure_reasons": []
}
```

No free-form score parsing.

Validate JSON with Pydantic.

---

# 15. Judge Calibration

Before trusting VLM scores:

1. create a small calibration set;
2. include obviously good images;
3. include broken images;
4. include blurry images;
5. include anatomy failures;
6. include prompt-mismatch cases;
7. include typography failures;
8. include excellent but unusual artistic cases.

Measure inter-rater agreement.

Do not simply average scores if one model behaves differently.

Possible aggregation:

```text
normalized_judge_scores
        ↓
median
        ↓
trimmed mean
        ↓
uncertainty estimate
        ↓
disagreement penalty
```

Example conceptual score:

```text
final =
    0.25 quality
  + 0.20 alignment
  + 0.15 aesthetics
  + 0.15 composition
  + 0.10 detail
  + 0.10 edit_fidelity
  + 0.05 typography
  - disagreement_penalty
  - artifact_penalty
```

The exact weights must live in config, not source code.

---

# 16. Data Selection

Use three levels:

```text
REJECT
REVIEW
ACCEPT
```

Accept only samples satisfying configured hard gates.

Example hard gates:

```text
quality >= threshold
alignment >= threshold
artifact_penalty <= maximum
safety == pass
confidence >= minimum
judge_disagreement <= maximum
```

Do not only keep the top 1% of images.

That creates a narrow, repetitive training distribution.

Instead use quality + diversity.

Pipeline:

```text
quality filtering
      ↓
embedding extraction
      ↓
clustering
      ↓
representative selection
      ↓
dataset balance
```

---

# 17. Dataset Diversity

Balance by:

```text
subject
style
camera
lighting
color
scene complexity
resolution
aspect ratio
language
typography
editing type
real/synthetic source
```

Do not allow synthetic Qwen images to dominate the dataset.

Use adaptive mixture weights.

Start conservatively:

```text
real licensed images
+
teacher synthetic images
+
real-image editing examples
+
synthetic editing examples
```

The exact mixture is an experiment variable.

---

# 18. Real Dataset Acquisition

Search Hugging Face datasets programmatically.

Hugging Face supports image datasets with metadata such as captions and supports Parquet and WebDataset formats; streaming is particularly useful for large datasets.

The agent must never automatically ingest an image dataset solely because it is popular.

Every dataset must pass:

```text
license available?
license compatible?
image provenance documented?
commercial-use ambiguity?
redistribution restrictions?
known removal requirements?
metadata available?
duplicates?
corrupted files?
unsafe content?
```

Store:

```json
{
  "dataset_id": "...",
  "revision": "...",
  "license": "...",
  "source_url": "...",
  "retrieved_at": "...",
  "hash": "...",
  "allowed_for_training": true,
  "allowed_for_redistribution": false,
  "notes": "..."
}
```

If the license cannot be established, do not ingest the dataset into the release-training pool.

---

# 19. Provenance

Every training sample must be traceable.

Example:

```json
{
  "sample_id": "sample-000001",
  "source_type": "synthetic",
  "source_dataset": null,
  "source_model": "Qwen/Qwen-Image-2.1",
  "source_revision": "...",
  "source_prompt": "...",
  "caption": "...",
  "edit_instruction": null,
  "reference_images": [],
  "seed": 12345,
  "width": 2048,
  "height": 2048,
  "sha256": "...",
  "license": "...",
  "accepted": true,
  "judge_scores": {},
  "dataset_snapshot": "..."
}
```

Real data must include original dataset provenance.

This makes later model cards and dataset cards possible.

Hugging Face supports dataset cards and metadata for documenting dataset contents, license and provenance.

---

# 20. Synthetic Captioning

Do not train only on the original teacher prompt.

Store multiple textual representations:

```text
original_prompt
teacher_prompt
vlm_caption
structured_caption
edit_instruction
reference_description
```

For synthetic data:

```text
prompt = desired condition
VLM caption = observed visual result
```

If the VLM caption contradicts the prompt, flag the sample.

Example:

```text
Prompt:
"a red motorcycle next to a blue bicycle"

VLM:
"one red bicycle with another bicycle"

=> reject or place into hard-negative pool
```

This improves conditioning fidelity.

---

# 21. Editing Dataset

Create a separate editing pipeline.

Three modes:

### A. Real → synthetic edit

```text
real image
   +
instruction
   ↓
Qwen Image 2.1
   ↓
edited target
   ↓
VLM verification
```

### B. Synthetic → synthetic edit

```text
teacher image
   +
instruction
   ↓
teacher edit
   ↓
judge
```

### C. Real paired edit datasets

Use only datasets with verified paired-image licenses.

Do not force every image into editing training.

---

# 22. Editing Quality Tests

The edited image must preserve what was not requested to change.

Judge:

```text
requested_change_present
unrequested_change_penalty
identity_preservation
geometry_preservation
background_consistency
lighting_consistency
semantic_consistency
```

A useful edit dataset entry is:

```text
source image
+
instruction
+
target image
+
source caption
+
target caption
+
change mask if available
```

---

# 23. High-Resolution Strategy

Do not blindly train everything at maximum resolution.

Use multi-scale training:

```text
Stage A:
768-ish latent/image training

Stage B:
1024

Stage C:
1536

Stage D:
2048+

Stage E:
selected high-resolution refinement
```

Qwen-Image-2.1 itself documents 2048-level generation and larger aspect-specific dimensions, so the pipeline should preserve the model's multi-aspect behavior rather than locking the student to 1024×1024.

Use:

```text
random aspect bucket
random resolution bucket
crop strategy
latent caching
mixed resolution batches
```

High-resolution samples should be a smaller but strategically selected portion of training.

---

# 24. Hybrid Generation Architecture

The intended final system is not simply:

```text
4-step diffusion → final image
```

Instead:

```text
Prompt
  ↓
compact 4-step base generator
  ↓
latent refinement / detail preservation
  ↓
optional high-resolution pass
  ↓
VAE decode
  ↓
output
```

The agent must investigate several variants:

### Variant A

Single distilled transformer.

### Variant B

4-step base + lightweight refinement.

### Variant C

4-step base + latent high-resolution refinement.

### Variant D

single model with multi-scale training.

### Variant E

compact student + optional detail stage.

Promote only variants that genuinely improve measured results.

Do not call a pipeline "hybrid" simply because it contains multiple scripts.

---

# 25. Training Philosophy

There are two separate research tracks.

## Track A — Safe production track

Start from the existing Qwen-Image-2.1 architecture.

Train:

```text
LoRA
→ merge / compatible full transformer representation
→ few-step student
→ quantization
→ GGUF
```

This gives the highest probability of obtaining a functional model.

## Track B — Architecture evolution

Investigate a smaller original student:

```text
Qwen teacher
      ↓
teacher features / flow targets
      ↓
smaller DiT
      ↓
trajectory distillation
      ↓
few-step model
```

Candidate:

```text
~1.5B
~2.5B
~3B
~4B
```

with a configurable number of layers and hidden dimensions.

Do not assume a smaller student will retain teacher quality.

Benchmark it.

---

# 26. Distillation

The model must be explicitly optimized for few-step inference.

The base Qwen-Image-2.1 pipeline documents 40 steps as its standard configuration. The project therefore has to learn the mapping from a high-NFE teacher trajectory to a low-NFE student trajectory instead of merely using fewer inference steps at test time.

Implement a distillation stack with pluggable losses.

## Loss 1 — Flow matching target

Student predicts the teacher's velocity/flow target at selected noise levels.

Concept:

```text
teacher(x_t, t, condition)
        ↓
target velocity
        ↓
student(x_t, t, condition)
        ↓
velocity regression
```

---

## Loss 2 — Trajectory matching

Instead of matching only one timestep:

```text
teacher:
t0 → t1 → t2 → ... → tn
```

student:

```text
s0 → s1 → s2 → s3 → output
```

Train the student to approximate teacher trajectory segments.

---

## Loss 3 — Distribution matching

Add a distribution-level distillation objective.

The agent may use DMD-style methods as one candidate implementation.

Current community Qwen-Image-2.1 few-step work demonstrates DMD plus velocity/trajectory guidance as a viable direction for 4–6 step students. Treat such implementations as research references, not ground truth.

---

## Loss 4 — Feature distillation

Match intermediate teacher/student representations where architecture permits.

Example:

```text
teacher hidden feature
        ↕
student hidden feature
```

with projection adapters if dimensions differ.

---

## Loss 5 — Perceptual image loss

Decode selected predictions and compare:

```text
LPIPS or equivalent
DINO/image embedding
VLM semantic similarity
```

Do not use these as the only objective.

---

## Loss 6 — Editing consistency

For editing examples:

```text
source
+
instruction
↓
teacher target
```

student must reproduce requested edits while preserving unrelated content.

---

# 27. 4-Step Target

Primary production target:

```text
4 NFE
CFG ≈ 1 / distilled conditioning
```

Do not assume that four arbitrary scheduler steps equal a four-step distilled model.

A dedicated student must be trained for the exact four-step schedule.

The schedule must be stored inside:

```text
distillation.yaml
```

and embedded in the model manifest.

---

# 28. Experimental 5/6/8-Step Fallback

Because current Qwen-Image-2.1 community distillation research indicates that 5/6-step variants can recover some detail that plain 4-step students lose, the agent must benchmark:

```text
4 steps
5 steps
6 steps
8 steps
```

without automatically declaring a winner.

The production target remains 4-step, but the benchmark should establish the quality/speed tradeoff.

---

# 29. Training Schedule

Use curriculum training.

### Phase 0 — Smoke training

Tiny:

```text
50–500 samples
few hundred steps
```

Purpose:

* verify forward pass;
* verify backward pass;
* verify checkpoint;
* verify resume;
* verify loss decreases;
* verify generated image changes.

### Phase 1 — Prototype

```text
thousands of examples
```

Purpose:

* verify learning behavior;
* verify teacher targets;
* verify distillation;
* verify image quality.

### Phase 2 — Main training

```text
tens of thousands+
```

depending on available GPU budget.

### Phase 3 — refinement

Focus only on weaknesses.

### Phase 4 — release candidate

Freeze:

```text
dataset snapshot
training config
teacher revision
code revision
seed
environment
```

---

# 30. Checkpoint Mechanism

Checkpoint must contain:

```text
model weights
optimizer
scheduler
grad scaler if used
training step
epoch
RNG states
config
environment manifest
dataset snapshot
teacher revision
Git commit
metrics
```

Suggested:

```text
checkpoint-001000/
    model/
    optimizer/
    scheduler/
    state.json
    metrics.json
    environment.json
    manifest.json
    checksum.sha256
```

---

# 31. Automatic Checkpoint Frequency

The agent must save on:

```text
fixed step interval
validation improvement
before environment changes
before runtime shutdown if detectable
before high-risk operations
```

Additionally:

```text
best_quality
best_alignment
best_speed
best_memory
best_overall
```

Each pointer references an immutable checkpoint.

---

# 32. Crash Recovery

On notebook start:

```text
mount Drive
↓
locate latest run
↓
validate run_state.json
↓
find newest valid checkpoint
↓
verify SHA256
↓
verify environment hash
↓
resume
```

If latest checkpoint is corrupt:

```text
rollback to previous valid checkpoint
```

Do not continue from an unverified checkpoint.

---

# 33. Dataset Sharding

Large image sets must be sharded.

Preferred formats:

```text
WebDataset TAR
Parquet metadata
or equivalent streaming format
```

Hugging Face documents WebDataset streaming and Parquet-based efficient filtering for large datasets.

Example:

```text
train-000000.tar
train-000001.tar
train-000002.tar
...
```

with metadata:

```text
dataset.parquet
```

The dataset builder must be able to rebuild shards from the manifest.

---

# 34. Deduplication

Perform multiple levels.

## Exact

```text
SHA256
```

## Perceptual

```text
pHash
dHash
```

## Semantic

```text
image embedding cosine distance
```

## Prompt

```text
prompt similarity
```

## Source

```text
same original dataset + same source image
```

Never use only pHash.

---

# 35. Hard Example Mining

The agent must actively find failure cases.

Examples:

```text
bad typography
wrong count
wrong color
wrong relation
missing object
extra object
wrong pose
identity drift
bad hands
bad reflections
incorrect lighting
wrong background
edit leakage
```

Then generate additional examples targeting those failures.

This creates:

```text
self-generated curriculum
```

---

# 36. MiniAGI Loop

The autonomous research controller should operate like:

```text
observe
  ↓
diagnose
  ↓
hypothesize
  ↓
design experiment
  ↓
run experiment
  ↓
measure
  ↓
compare
  ↓
retain or rollback
  ↓
update curriculum
  ↓
repeat
```

Persist hypotheses:

```json
{
  "hypothesis": "Typography is underrepresented.",
  "evidence": {
    "benchmark": 0.61
  },
  "intervention": "increase difficult typography samples",
  "experiment": "exp-031",
  "result": 0.69,
  "status": "supported"
}
```

The agent must never declare a research hypothesis true merely because one random sample looked better.

Require repeated benchmark evidence.

---

# 37. Experiment Registry

Use:

```text
experiments.jsonl
```

Every experiment:

```json
{
  "id": "exp-031",
  "parent": "exp-030",
  "purpose": "...",
  "hypothesis": "...",
  "config_hash": "...",
  "dataset_snapshot": "...",
  "teacher_revision": "...",
  "checkpoint_in": "...",
  "checkpoint_out": "...",
  "metrics": {},
  "decision": "promote|reject|retry"
}
```

---

# 38. Benchmark Suite

The benchmark must contain fixed prompts never used for training.

Split:

```text
benchmark/
├── t2i/
├── typography/
├── counting/
├── composition/
├── realism/
├── editing/
├── multireference/
├── highres/
└── adversarial/
```

Never train on benchmark images/prompts.

---

# 39. Current Benchmark References

Use a combination of:

```text
Qwen-Image-Bench
GenEval2
T2I-CompBench++
project-specific fixed benchmark
```

Qwen-Image-Bench currently evaluates generation/creation using a hierarchical rubric with quality, aesthetics, alignment, real-world fidelity and creative generation, while GenEval2 includes VQA-based evaluation such as Soft-TIFA.

Do not depend on one benchmark.

---

# 40. Internal Benchmark Dimensions

Every model generation must receive:

```text
quality
aesthetics
prompt alignment
composition
detail
realism
object fidelity
relation fidelity
count accuracy
typography
editing fidelity
identity preservation
multi-reference fidelity
diversity
artifact rate
latency
VRAM
throughput
```

Record raw results.

Never report only one aggregate score.

---

# 41. Baselines

Always benchmark against:

```text
Qwen-Image-2.1 teacher
current internal best
latest student
candidate student
public 4-step reference where licensing permits
```

A public few-step Qwen-Image-2.1 model can be useful as a regression/reference point, but it must not silently become part of your own released model.

---

# 42. Quality Gates

Example release gate:

```text
T2I alignment >= baseline - tolerance
overall quality >= baseline target
edit fidelity >= target
artifact rate <= maximum
diversity >= minimum
4-step stability = PASS
high-resolution stability = PASS
sd.cpp inference = PASS
checksum validation = PASS
```

The actual thresholds must be configurable.

---

# 43. Quantization

After the best floating-point student is frozen:

```text
BF16/F16
 ↓
Q8_0
 ↓
Q6/Q5
 ↓
Q4
 ↓
Q3/Q2
```

Test each.

Do not assume the smallest file is the best release.

`stable-diffusion.cpp` supports explicit quantization and conversion to GGUF, with several quantization formats available in its current codebase.

Also test current low-bit types supported by the exact `sd.cpp` commit selected for the release.

---

# 44. Quantization Selection

Produce:

```text
HCS-Image-Evolver-F16
HCS-Image-Evolver-Q8_0
HCS-Image-Evolver-Q6
HCS-Image-Evolver-Q5
HCS-Image-Evolver-Q4
```

Potentially:

```text
Q3/Q2
```

only if quality remains acceptable.

Benchmark:

```text
file_size
load_time
VRAM
RAM
latency
image quality
text rendering
editing fidelity
```

---

# 45. GGUF Requirement

Primary final artifact:

```text
*.gguf
```

The conversion pipeline must:

1. start with the exact frozen checkpoint;
2. verify tensor names;
3. convert to supported GGUF format;
4. verify metadata;
5. verify tensor count;
6. verify tensor shapes;
7. compute SHA256;
8. load with real `sd.cpp`;
9. generate real images;
10. compare output against source inference.

No "conversion successful" message counts as validation.

---

# 46. Self-Contained sd.cpp Release

The release package should look like:

```text
HCS-Image-Evolver-sd.cpp/
├── sd-cli.exe
├── models/
│   ├── hcs-image-evolver-q4.gguf
│   ├── qwen-image-2.1-vae...
│   ├── qwen3-vl...
│   └── mmproj...
│
├── presets/
│   ├── t2i.json
│   ├── edit.json
│   ├── 4step.json
│   └── hd.json
│
├── examples/
├── run_t2i.bat
├── run_edit.bat
├── sha256sums.txt
└── README.md
```

No:

```text
Python
CUDA Python packages
Diffusers
Transformers runtime
ComfyUI
Internet connection
```

should be required for normal local inference.

The final package is therefore offline and runtime-independent from the training stack.

---

# 47. Exact sd.cpp Verification

Current `stable-diffusion.cpp` explicitly documents Qwen-Image-2.1 support, including GGUF diffusion weights, Qwen3-VL text encoding and vision support for editing.

The agent must test:

### T2I

```text
simple prompt
complex prompt
typography
portrait
landscape
16:9
9:16
```

### I2I

```text
simple edit
local edit
identity preservation
background change
object change
```

### Multi-reference

```text
2 refs
3 refs
```

### RGBA

```text
transparent output
```

### HD

```text
2048+
```

Only then can the release be marked `sd.cpp validated`.

---

# 48. sd.cpp Build

Build the exact selected commit from source.

Do not blindly use the latest binary.

Record:

```text
commit
compiler
CMake version
backend
Vulkan version
build flags
```

Current `sd.cpp` documentation supports source builds and configurable GGML backends.

For local AMD/Vulkan:

```text
SD_VULKAN=ON
```

where supported by the chosen build.

The agent must detect actual device availability at runtime.

---

# 49. LoRA Handling

`stable-diffusion.cpp` currently supports LoRA and runtime/application modes, including specialized handling for quantized models.

Use LoRA during research when useful.

For final release:

```text
preferred:
merged student GGUF
```

rather than requiring users to manually attach a training adapter.

However:

```text
research adapter
```

may still be published separately.

Never merge blindly into heavily quantized weights if the resulting delta becomes smaller than quantization noise.

Instead:

```text
train
→ merge in high precision
→ quantize once
```

when supported.

---

# 50. Local Regression Test

After GGUF generation:

```text
teacher diffusers image
       ↓
candidate diffusers image
       ↓
candidate gguf sd.cpp image
```

Do not expect pixel equality.

Compare:

```text
semantic similarity
VLM quality
image embeddings
composition
text rendering
visual artifacts
```

with the same prompt and seed where reproducibility permits.

---

# 51. Numerical Reproducibility

Store:

```text
seed
prompt
negative prompt
scheduler
sigmas
steps
width
height
guidance
model revision
software revision
```

Exact pixel reproducibility is not required across different backends unless demonstrably possible.

Semantic regression is mandatory.

---

# 52. Release Manifest

Every release gets:

```text
release_manifest.json
```

containing:

```json
{
  "model_name": "...",
  "version": "...",
  "base_teacher": "...",
  "training_run": "...",
  "checkpoint": "...",
  "dataset_snapshot": "...",
  "git_commit": "...",
  "sd_cpp_commit": "...",
  "quantization": "...",
  "sha256": "...",
  "resolution_support": [],
  "steps": 4,
  "runtime": "sd.cpp",
  "platforms": ["windows"],
  "validation": {
    "t2i": "pass",
    "edit": "pass",
    "multiref": "pass",
    "rgba": "pass",
    "highres": "pass"
  }
}
```

---

# 53. Hugging Face Publishing

The agent may publish:

```text
model repository
dataset repository
benchmark repository
```

only after the GitHub repository is internally consistent.

Recommended release hierarchy:

```text
GitHub
    source code
    configs
    notebooks
    benchmark definitions
    documentation

Hugging Face
    model weights
    GGUF
    dataset shards
    model card
    dataset card

Google Drive
    recovery checkpoints
    optimizer state
    training logs
    experiment artifacts
```

---

# 54. GitHub Publishing

The final agent must:

```text
git status
git diff
git secret scan
git lfs validation
pytest
integration tests
build
benchmark
package validation
README generation
```

Then:

```text
git add
git commit
git push
git tag
git push --tags
```

Do not commit:

```text
HF_TOKEN
GitHub tokens
Google credentials
Drive credentials
private datasets
huge local caches
```

---

# 55. README Requirements

The generated README must document:

```text
project purpose
architecture
model lineage
base teacher
training method
dataset provenance
license
known limitations
installation
Windows usage
sd.cpp usage
T2I command
I2I command
4-step settings
high-resolution usage
model download
checksums
benchmark results
training results
Colab notebook
Drive resume workflow
reproduction instructions
citation
```

It must explicitly distinguish:

```text
measured
estimated
planned
experimental
unsupported
```

---

# 56. Colab Notebook Requirements

The notebook must be usable from a fresh Google Colab runtime.

Top-level cells:

```text
00 - Runtime diagnostics
01 - Mount Google Drive
02 - Configure secrets
03 - Clone repository
04 - Restore state
05 - Install locked environment
06 - Verify Qwen model
07 - Verify generation
08 - Generate synthetic batch
09 - VLM judging
10 - Dataset construction
11 - Training
12 - Checkpoint
13 - Resume verification
14 - Distillation
15 - Benchmark
16 - GGUF conversion
17 - sd.cpp validation
18 - Backup
19 - Release summary
```

Every important cell must be independently rerunnable.

---

# 57. Colab Secrets

Never store tokens in notebook text.

Use:

```text
HF_TOKEN
GITHUB_TOKEN if required
```

through Colab secrets or equivalent secure mechanisms.

The notebook must fail with an actionable message when a required secret is missing.

Never print secrets.

Never log bearer tokens.

---

# 58. Drive Resume Notebook

Create a dedicated:

```text
HCS_Image_Evolution_Resume.ipynb
```

Its job:

```text
mount
↓
find latest run
↓
verify
↓
resume
```

This makes runtime replacement painless.

---

# 59. Checkpoint Backup Policy

On every saved checkpoint:

```text
local checkpoint
→ SHA256
→ Drive backup
→ verify
→ update state
```

For best checkpoints:

```text
Drive
+
Hugging Face private storage if enabled
```

Never delete all previous checkpoints automatically.

Apply retention:

```text
last N
+
best N
+
milestone checkpoints
```

---

# 60. 24/7 Autonomous Operation

Create a resumable worker:

```text
worker.run()
```

that chooses the next task based on state.

Example:

```text
if no dataset:
    build dataset

elif dataset incomplete:
    generate more

elif judge queue incomplete:
    judge

elif dataset ready and no student:
    train

elif student exists and not distilled:
    distill

elif student exists and not benchmarked:
    benchmark

elif benchmark improved:
    promote

elif promoted and not gguf:
    quantize

elif gguf and not sd.cpp tested:
    validate

elif validated:
    package

else:
    create next experiment
```

The worker never relies on notebook position.

State is authoritative.

---

# 61. Adaptive Budgeting

Do not waste GPU time equally.

Example:

```text
cheap:
prompt generation
metadata
dedupe

medium:
VLM judging
small benchmark

expensive:
teacher generation
training
high-res generation
distillation
```

The agent should first use cheap tests to eliminate bad ideas.

Example:

```text
candidate config
 ↓
100-image mini benchmark
 ↓
bad
 ↓
discard
```

instead of:

```text
candidate config
 ↓
20 hours training
 ↓
bad
```

---

# 62. Data Generation Efficiency

Implement:

```text
batched prompts
batched VLM requests
async provider requests
retry with backoff
rate-limit handling
image caching
metadata caching
sharded output
```

Do not regenerate an image if:

```text
same teacher revision
same prompt
same seed
same resolution
same generation config
```

already exists.

---

# 63. VLM API Reliability

Every request gets:

```text
request_id
provider
model
attempt
latency
status
retry_count
response_hash
```

Implement:

```text
429 handling
5xx retry
timeout
invalid JSON retry
provider unavailable fallback
```

Do not silently substitute another model without recording the substitution.

---

# 64. Judge Ensemble Evolution

The agent can experiment with:

```text
3 judges
4 judges
5 judges
```

and different provider mixtures.

But benchmark the judges themselves.

The judge ensemble is part of the research infrastructure.

It must not become an uncontrolled moving target.

Freeze the judge set for official benchmark runs.

---

# 65. Anti-Reward-Hacking

This is critical.

A self-improving image system can learn to exploit the judge.

Protect against:

```text
VLM preference hacking
prompt overfitting
style collapse
dataset collapse
judge-specific artifacts
```

Use:

```text
hidden benchmark prompts
multiple judges
fixed benchmark
randomized prompts
human spot-checks where practical
diversity metrics
unseen evaluation prompts
```

Never allow the same VLM to be both:

```text
training selector
```

and sole:

```text
final evaluator
```

---

# 66. Diversity Protection

Track:

```text
image embedding entropy
cluster coverage
style distribution
subject distribution
color distribution
prompt family distribution
```

Reject candidate generations if model quality improves only by collapsing diversity.

---

# 67. Model Selection

Maintain:

```text
champion
challenger
```

Example:

```text
champion = current release candidate
challenger = newest experiment
```

A challenger replaces the champion only after benchmark gates.

If challenger fails:

```text
rollback
record reason
update experiment registry
```

---

# 68. Automatic Research Decisions

The agent should automatically choose the next experiment from a finite experiment grammar:

```text
TRAINING:
- learning rate
- batch size
- gradient accumulation
- LoRA rank
- layer targeting
- loss weights

DISTILLATION:
- timestep schedule
- trajectory loss
- DMD strength
- feature loss
- number of steps

DATA:
- synthetic ratio
- real ratio
- edit ratio
- typography ratio
- highres ratio

MODEL:
- layer count
- hidden size
- attention configuration
- quantization
```

Do not perform combinatorial brute force.

Use evidence-driven search.

---

# 69. Experiment Priority

Prioritize experiments with:

```text
high expected gain
+
low GPU cost
+
high information value
```

An experiment that can rule out five approaches is valuable even if it does not produce a better model.

Persist the reason.

---

# 70. Compact Student Research

After a stable 7B 4-step student exists, investigate:

```text
7B teacher
      ↓
knowledge distillation
      ↓
4B student
      ↓
2–3B student
      ↓
GGUF Q4/Q3
```

The compact student should be judged by:

```text
quality per GB
quality per second
quality per VRAM GB
```

rather than raw quality alone.

This is where the model can become genuinely differentiated rather than being merely a fine-tuned Qwen checkpoint.

---

# 71. Model Architecture Search

Create a small search space.

For example:

```yaml
student:
  layers:
    - 12
    - 16
    - 20
    - 24
  hidden_size:
    - 1024
    - 1280
    - 1536
  heads:
    - auto
  vocab:
    - inherited
```

The architecture search must first pass:

```text
shape validation
forward
backward
memory
checkpoint
resume
```

before expensive training.

---

# 72. Teacher Target Caching

Do not run the full teacher every training step forever.

Create teacher target caches.

Possible cached content:

```text
noise
timestep
conditioning
teacher velocity
teacher hidden states
selected decoded targets
```

Shard them.

This converts repetitive teacher work into reusable training data.

---

# 73. Teacher Refresh

However, teacher targets must not become stale forever.

Support:

```text
teacher revision
target cache revision
```

When teacher revision changes:

```text
invalidate affected cache
```

---

# 74. Editing Teacher Targets

For I2I:

```text
reference image
instruction
teacher output
teacher metadata
```

The student should learn:

```text
semantic instruction following
+
preservation
```

not simply reproduce training images.

---

# 75. Normal Photos

Normal real photographs should be treated differently from synthetic teacher data.

Use them for:

```text
visual grounding
lighting
materials
real-world composition
editing preservation
```

Do not assume a real photo is automatically a valid generation target.

A real image can be:

```text
high quality visually
but poor as a captioned generative training example
```

Use VLM captioning and metadata quality gates.

---

# 76. Synthetic-to-Real Balance

Track metrics separately:

```text
synthetic validation
real validation
edit validation
```

A model that improves only on synthetic data but worsens on real images must not be promoted as an overall improvement.

---

# 77. High-Resolution Validation

Every release candidate should generate:

```text
1024
1536
2048
```

and selected native aspect ratios.

Check:

```text
texture
edges
fine detail
faces
text
repetition
tiling artifacts
memory
runtime
```

---

# 78. Memory Optimization

Use:

```text
mixed precision
gradient checkpointing
attention optimizations
latent caching
CPU offload where useful
VAE tiling
resolution buckets
gradient accumulation
```

Only enable optimizations after measuring them.

Do not add a dependency solely because a blog post says it is faster.

Benchmark.

Qwen-Image-2.1 itself documents model CPU offload as a memory optimization, and `sd.cpp` provides VAE tiling and related runtime optimizations.

---

# 79. Automatic Memory Detection

At startup detect:

```text
GPU VRAM
system RAM
GPU backend
compute capability / equivalent
```

Then choose:

```text
resolution
batch size
gradient accumulation
precision
offload policy
```

based on a config policy.

Never hard-code:

```text
batch_size = 8
```

for every GPU.

---

# 80. Real Smoke Tests

Before the first expensive training job:

### Test 1

Download teacher.

### Test 2

Generate one real image.

### Test 3

Generate one real edit.

### Test 4

Call one real VLM.

### Test 5

Create one valid dataset record.

### Test 6

Perform one training step.

### Test 7

Save checkpoint.

### Test 8

Restart and resume.

### Test 9

Generate from resumed checkpoint.

### Test 10

Export checkpoint.

### Test 11

Convert GGUF.

### Test 12

Run GGUF in sd.cpp.

### Test 13

Generate image again.

### Test 14

Run benchmark.

Only after all tests pass may the full run begin.

---

# 81. E2E Test

Create:

```text
tests/e2e/test_full_pipeline.py
```

that executes:

```text
download
→ generation
→ judge
→ dataset entry
→ train tiny model
→ checkpoint
→ resume
→ export
→ GGUF
→ sd.cpp
→ benchmark
```

Use a tiny real test configuration.

It must not use fake model inference.

---

# 82. CI

GitHub Actions should execute at least:

```text
ruff
mypy where applicable
pytest unit
pytest integration
schema validation
config validation
manifest validation
secret scan
build test
```

GPU-heavy tests can be separated from CPU CI.

Never make CI claim that a GPU model works if no GPU test actually ran.

Label:

```text
CPU CI
GPU validation
Colab E2E
local Vulkan validation
```

separately.

---

# 83. Windows Validation

The release pipeline must test:

```text
Windows 11
sd-cli.exe
Vulkan backend
model path
Unicode prompt handling
UTF-8 metadata
spaces in paths
long paths
batch files
```

Test both:

```text
absolute model path
relative model path
```

---

# 84. Release Package

Create:

```text
dist/
└── HCS-Image-Evolver-v1.0/
    ├── bin/
    ├── models/
    ├── configs/
    ├── examples/
    ├── scripts/
    ├── README.md
    ├── MODEL_CARD.md
    ├── LICENSES.md
    ├── MANIFEST.json
    └── SHA256SUMS.txt
```

---

# 85. Hash Everything

Every important artifact:

```text
SHA256
```

including:

```text
checkpoint
GGUF
VAE
text encoder
dataset shard
manifest
release archive
```

Store checksums in a signed or immutable release manifest when practical.

---

# 86. Reproducibility

A release should be reproducible from:

```text
Git commit
+
config
+
dataset snapshot
+
teacher revision
+
environment lock
+
seed policy
```

The agent does not have to regenerate the exact same pixels, but it must be possible to reproduce the same experimental procedure.

---

# 87. Documentation Generated by Agent

When development completes, the agent must automatically write:

```text
README.md
docs/architecture.md
docs/training.md
docs/distillation.md
docs/dataset.md
docs/sdcpp.md
docs/reproducibility.md
```

Documentation must be generated from actual manifests and benchmark results.

Do not write fake claims such as:

```text
"This model is state of the art"
```

unless a documented benchmark establishes the claim.

Prefer:

```text
"On our fixed benchmark, model X achieved ..."
```

---

# 88. Final README Metrics

Include tables like:

```text
| Model | Steps | Resolution | Quality | Alignment | Edit | VRAM | Latency |
|------|------:|------------|--------:|----------:|-----:|-----:|--------:|
| Teacher | 40 | ... | ... | ... | ... | ... | ... |
| Student | 4 | ... | ... | ... | ... | ... | ... |
| Student Q4 | 4 | ... | ... | ... | ... | ... | ... |
```

Values must be generated from real benchmark files.

---

# 89. Failure Recovery

Possible failure classes:

```text
DOWNLOAD_FAILED
LICENSE_FAILED
MODEL_LOAD_FAILED
OOM
VLM_RATE_LIMIT
INVALID_VLM_JSON
CHECKPOINT_CORRUPT
DRIVE_COPY_FAILED
GGUF_CONVERSION_FAILED
SDCPP_LOAD_FAILED
SDCPP_RUNTIME_FAILED
QUALITY_REGRESSION
```

Each has a recovery strategy.

Example:

```text
OOM
→ reduce batch
→ increase accumulation
→ enable checkpointing
→ lower training resolution
→ resume
```

Example:

```text
VLM 429
→ exponential backoff
→ switch allowed provider
→ preserve exact request
→ retry
```

Example:

```text
quality regression
→ do not promote
→ preserve checkpoint
→ inspect metrics
→ create next experiment
```

---

# 90. No Infinite Self-Healing Loop

Autonomy must still be controlled.

Each operation has:

```text
retry budget
```

After repeated failures:

```text
save state
save error report
save logs
mark blocker
```

Do not endlessly burn compute credits on one broken dependency.

---

# 91. Research Ledger

Create:

```text
research/ledger.jsonl
```

Every meaningful discovery:

```json
{
  "date": "...",
  "observation": "...",
  "source": "...",
  "experiment": "...",
  "conclusion": "...",
  "confidence": "..."
}
```

This gives the agent a long-term memory of what has already been tested.

---

# 92. Model Lineage

Never lose track of ancestry.

Example:

```text
Qwen-Image-2.1
    │
    ├── exp-001 LoRA
    │
    ├── exp-004 full student
    │
    ├── exp-008 8-step
    │
    ├── exp-012 4-step
    │       │
    │       ├── q8
    │       ├── q6
    │       └── q4
    │
    └── exp-021 compact 3B
```

Represent this programmatically.

---

# 93. Benchmark Regression

Every new candidate gets compared against:

```text
parent
champion
teacher
previous release
```

Track:

```text
delta_quality
delta_speed
delta_memory
delta_edit
delta_diversity
```

A candidate may be accepted for a specific purpose without becoming overall champion.

Example:

```text
compact model:
better memory
slightly lower quality
```

This can become a separate release rather than replacing the main model.

---

# 94. Multiple Final Models

The project may eventually produce:

```text
HCS Image Evolver 4S
```

for ultra-fast generation,

```text
HCS Image Evolver Edit
```

for editing,

```text
HCS Image Evolver HD
```

for high-resolution quality,

```text
HCS Image Evolver Compact
```

for low-memory systems.

Do not force one architecture to solve every problem if benchmarks show that specialized variants are materially better.

---

# 95. Final Goal

The successful project should eventually look like:

```text
User
 ↓
HCS Image Model
 ↓
T2I / I2I / Edit / MultiRef
 ↓
4-step generation
 ↓
HD/high-resolution path
 ↓
offline sd.cpp
```

while training operates as:

```text
Agent
 ↓
Qwen teacher
 ↓
synthetic data
 ↓
VLM ensemble
 ↓
curated dataset
 ↓
student
 ↓
distillation
 ↓
benchmark
 ↓
checkpoint promotion
 ↓
GGUF
 ↓
sd.cpp
 ↓
release
 ↓
next experiment
```

---

# 96. Definition of Done

The agent may mark the project COMPLETE only when all of the following are true:

```text
[ ] repository created
[ ] architecture implemented
[ ] Windows tooling works
[ ] Colab notebook runs from fresh runtime
[ ] Drive mounting works
[ ] checkpoint backup works
[ ] checkpoint restore works
[ ] automatic resume works
[ ] real Qwen Image 2.1 generation works
[ ] real Qwen Image 2.1 editing works
[ ] real HF VLM judging works
[ ] multiple judges work
[ ] dataset curation works
[ ] provenance works
[ ] license gate works
[ ] real image datasets work
[ ] synthetic dataset works
[ ] edit dataset works
[ ] training forward works
[ ] backward works
[ ] checkpoint works
[ ] resume works
[ ] few-step distillation works
[ ] 4-step model generates real images
[ ] benchmark suite runs
[ ] benchmark reports are generated
[ ] best checkpoint selection works
[ ] rollback works
[ ] GGUF conversion works
[ ] GGUF checksum works
[ ] real sd.cpp build works
[ ] real sd.cpp GGUF loading works
[ ] sd.cpp T2I works
[ ] sd.cpp editing works
[ ] sd.cpp high-resolution path works
[ ] sd.cpp offline package works
[ ] Windows inference works
[ ] release package works
[ ] GitHub tests pass
[ ] no secrets committed
[ ] README generated
[ ] model lineage documented
[ ] dataset provenance documented
[ ] license information documented
[ ] benchmark results are real
[ ] no mock output exists in release code
[ ] GitHub push succeeds
[ ] release tag created
```

One unchecked box means:

```text
NOT COMPLETE
```

---

# 97. Agent Execution Order

The autonomous agent must execute in this exact strategic order:

```text
1. DISCOVER LOCAL ENVIRONMENT

2. CREATE REPOSITORY

3. WRITE TESTS AND SCHEMAS

4. IMPLEMENT STATE MACHINE

5. IMPLEMENT COLAB BOOTSTRAP

6. TEST GOOGLE DRIVE RECOVERY

7. TEST REAL QWEN IMAGE 2.1 GENERATION

8. TEST REAL QWEN IMAGE 2.1 EDITING

9. IMPLEMENT HF VLM ENSEMBLE

10. BUILD MINI DATASET

11. RUN VLM QUALITY FILTER

12. BUILD CURATED DATASET

13. RUN TINY REAL TRAINING

14. TEST CHECKPOINT/RESUME

15. IMPLEMENT DISTILLATION

16. TRAIN FIRST 4-STEP STUDENT

17. BENCHMARK AGAINST TEACHER

18. IMPROVE DATA/CURRICULUM

19. RUN SECOND-GENERATION TRAINING

20. BENCHMARK

21. EXPLORE COMPACT STUDENT

22. QUANTIZE

23. CONVERT GGUF

24. BUILD SD.CPP

25. RUN REAL LOCAL INFERENCE

26. RUN OFFLINE PACKAGE TEST

27. GENERATE RELEASE MANIFEST

28. GENERATE README

29. RUN FULL TEST SUITE

30. GIT COMMIT

31. GIT PUSH

32. TAG RELEASE

33. FINAL RELEASE VALIDATION

34. STOP ONLY AFTER ALL DEFINITION-OF-DONE CHECKS PASS
```

---

# 98. Important Technical Interpretation

The target is not:

```text
"take Qwen Image 2.1,
fine-tune it a little,
quantize it,
call it done."
```

The target is:

```text
teacher
→ data engine
→ visual judge ensemble
→ automatic curriculum
→ student training
→ trajectory/flow distillation
→ 4-step specialization
→ multi-scale learning
→ compact architecture research
→ quantitative selection
→ quantization
→ standalone runtime package
→ continuous autonomous evolution
```

The differentiating component is the **evolution loop**, not one particular loss function.

---

# 99. Current External Technical Baseline

The agent should keep a small compatibility document describing the currently observed external state:

```text
Qwen Image 2.1
- 7B visual generation component
- T2I
- I2I/editing
- multi-reference
- RGBA
- native high-resolution configurations
- Qwen3-VL-based conditioning
```

```text
stable-diffusion.cpp
- current Qwen Image 2.1 support
- GGUF diffusion support
- Vulkan/backend support
- LoRA support
- quantization
- VAE tiling / memory optimizations
```

The exact versions must always be frozen by the successful build rather than inferred from this document.

Qwen's model card and current `sd.cpp` documentation should be treated as the authoritative compatibility references.

---

# 100. Final Agent Behavior

The agent should behave like a combination of:

```text
ML engineer
research engineer
dataset engineer
VLM evaluator
DevOps engineer
release engineer
C++ runtime engineer
QA engineer
research notebook manager
```

It must continuously ask internally:

```text
What is currently true?
What evidence proves it?
What failed?
What is the cheapest experiment that gives useful information?
What checkpoint is currently best?
Can I reproduce this result?
Can I resume after a crash?
Can the final model run outside Python?
```

The agent must never confuse:

```text
implemented
```

with:

```text
validated
```

and never confuse:

```text
validated once
```

with:

```text
release quality
```

---

# 101. Final Success Definition

Success is:

```text
A genuinely trained image model,
derived through a documented training/evolution process,
using real data and real evaluations,
optimized for few-step generation,
tested against fixed benchmarks,
converted to a real GGUF,
validated through real stable-diffusion.cpp inference,
packaged for offline use,
recoverable through Google Drive,
and reproducibly released on GitHub.
```

The final agent should finish with a machine-generated report:

```text
FINAL_REPORT.md
```

containing:

```text
best checkpoint
best 4-step model
compact model if successful
GGUF variants
training duration
dataset size
synthetic/real composition
judge ensemble
benchmark results
memory results
speed results
sd.cpp results
known failures
license/provenance notes
Git commit
release tag
SHA256
```

No fabricated claims.

No hidden dependencies.

No fake benchmark.

No missing lineage.

No unverified "works".

Only evidence.
