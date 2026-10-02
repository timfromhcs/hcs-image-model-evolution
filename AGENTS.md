# AGENTS.md

## Autonomous Image Model Evolution Lab

Welcome, autonomous agent. This repository is an autonomous, reproducible, self-improving image-model research and production pipeline targeting few-step student diffusion models distilled from the Qwen-Image-2.1 teacher architecture.

### Non-Negotiable Operational Principles

1. **No Mocks / No Hallucinated Metrics**:
   - Never fabricate benchmark results, VLM scores, or placeholder GGUF files.
   - Every metric reported must be derived from verified physical execution.
2. **Explicit State Persistence & Google Drive Recovery**:
   - Training & evolution runs are designed for Google Colab GPU runtimes subject to preemption/disconnection.
   - Run state (`run_state.json`), event log (`events.jsonl`), checkpoints, and datasets are atomically saved and synced with Google Drive.
   - New runtimes must resume seamlessly without human intervention.
3. **Continuous Benchmarking & Promotion Gates**:
   - Every candidate model must beat baseline quality, alignment, and latency gates before promotion.
   - Automated rollback occurs if regressions are detected.
4. **Offline Packaging**:
   - The production target artifact is GGUF packaged for local, offline inference using `stable-diffusion.cpp` (`sd-cli.exe`), with Vulkan / CPU support.
5. **No Secret Leaks**:
   - Do not commit access tokens (`HF_TOKEN`, GitHub PATs, Google auth) to Git. Always load them via environment variables or secret vaults.

### CLI Entrypoint

```powershell
# Run the local evolution controller agent:
python -m hcs_image_evolution.cli agent run --config configs/base.yaml
```
