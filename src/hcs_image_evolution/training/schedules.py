"""Flow matching noise and timestep schedules for continuous and few-step distillation."""

import torch


class FlowMatchingSchedule:
    """Linear flow matching schedule parameterizing paths between noise x_1 and data x_0."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Samples continuous timesteps t in [0, 1] using logit-normal or uniform distribution."""
        # Continuous time sampling
        u = torch.rand(batch_size, device=device)
        return u

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes interpolated state x_t and velocity target v_t.
        x_t = (1 - t) * x_0 + t * x_1
        v_t = x_1 - x_0
        """
        # Expand t to broadcast with image tensors (B, C, H, W)
        t_expanded = t.view(-1, *([1] * (x_0.ndim - 1)))
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
        v_target = x_1 - x_0
        return x_t, v_target

    @staticmethod
    def get_few_step_timesteps(num_steps: int = 4) -> list[float]:
        """Calculates discrete timesteps for low-NFE inference."""
        # Linear spacing from 1.0 (noise) down to 0.0 (clean)
        return [1.0 - (i / num_steps) for i in range(num_steps)]
