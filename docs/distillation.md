# Distillation & 4-Step Formulation

## Distillation Objectives

1. **Velocity Flow Matching**:
   The student predicts velocity $v_\theta(x_t, t) \approx x_1 - x_0$.
2. **Trajectory Matching**:
   Direct endpoint and intermediate trajectory segment matching between high-NFE teacher paths and low-NFE student leaps.
3. **Editing Preservation**:
   Directional instruction following with an inverse change mask preserving unmodified background content.

## Low-NFE Scheduling
- **Target**: 4 steps, guidance scale = 1.0 (CFG incorporated via learned conditioning).
- **Evaluation Variants**: 4, 5, 6, and 8 steps to profile the visual quality vs. latency trade-off curve.
