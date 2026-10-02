# System Architecture

## Overview

The HCS Image Model Evolution Lab is an autonomous closed-loop research and engineering framework designed to evolve and distill diffusion models derived from the `Qwen/Qwen-Image-2.1` teacher architecture.

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
```

## Storage & Runtime Decoupling
- **Heavy Compute**: Google Colab GPU runtimes for synthetic generation, teacher inference, and model training.
- **State Recovery**: Google Drive mirror storing `run_state.json`, `events.jsonl`, checkpoints, and dataset shards.
- **Local Runtime**: Windows 11 AMD Vulkan/CPU machine hosting the autonomous controller and offline `stable-diffusion.cpp` inference.
