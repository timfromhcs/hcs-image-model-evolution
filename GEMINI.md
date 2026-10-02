# GEMINI.md

## Evolution Lab Architecture & Guidelines for Gemini

This project implements an autonomous closed-loop research lab:
`Teacher Generation -> Multi-VLM Judging -> Dataset Construction -> Distillation Training -> Benchmarking -> GGUF Quantization -> sd.cpp Validation -> Packaging & Release`.

### Key Constraints & Protocols
- Primary teacher: `Qwen/Qwen-Image-2.1` (7B parameter visual backbone with multi-modal condition encoder).
- Distillation target: 4-step low-NFE inference (with 5/6/8-step fallback evaluations).
- Local runtime: Windows 11 AMD Vulkan / CPU inference via `sd.cpp`.
- Heavy runtime: Google Colab GPU (L4 / A100 / T4) backed up to Google Drive.
- Strictly adhere to `PLAN.md` instructions and non-negotiable rules.
