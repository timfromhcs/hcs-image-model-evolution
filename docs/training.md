# Training Strategy

## Curriculum Phases

1. **Phase 0 — Smoke Training**:
   - 50–500 samples, ~200 steps.
   - Verifies gradient flow, loss reduction, and atomic checkpoint serialization.
2. **Phase 1 — Prototype**:
   - Thousands of examples.
   - Evaluates flow matching dynamics and LoRA adaptation stability.
3. **Phase 2 — Main Distillation**:
   - Full scale student training targeting 4-step convergence.
4. **Phase 3 — Multi-Resolution Refinement**:
   - Multi-scale fine-tuning spanning 768px, 1024px, 1536px, and 2048px aspect buckets.

## Memory Optimizations
- **Mixed Precision**: BFloat16 native tensors.
- **Gradient Checkpointing**: Drastically reduces VRAM footprint during backward pass.
- **VAE Tiling**: Prevents OOM errors when decoding large 2048px latent fields.
- **AdamW 8-bit**: Cuts optimizer state memory by ~75%.
