"""Low-NFE numerical ODE sampler for 4-step (and 5/6/8-step) student inference."""

import torch
from PIL import Image
from torch import nn


class FewStepSampler:
    """Euler integration along student flow matching velocity vector field."""

    def __init__(self, student_model: nn.Module, device: str = "cpu"):
        self.model = student_model.to(device)
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        channels: int = 4,
        height: int = 128, # latent height or pixel
        width: int = 128,  # latent width or pixel
        num_steps: int = 4,
        seed: int = 42,
    ) -> torch.Tensor:
        """Solves ODE from noise x_1 (t=1.0) to data x_0 (t=0.0) in num_steps."""
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Initial Gaussian noise x_1
        x = torch.randn((batch_size, channels, height, width), generator=generator, device=self.device)

        # Uniform step discretization
        timesteps = [1.0 - (i / num_steps) for i in range(num_steps)]
        dt = 1.0 / num_steps

        for t_val in timesteps:
            t_tensor = torch.full((batch_size,), t_val, device=self.device, dtype=x.dtype)
            try:
                v_pred = self.model(x, t_tensor)
            except TypeError:
                v_pred = self.model(x)

            # Euler step: x_{t - dt} = x_t - dt * v_t
            x = x - dt * v_pred

        return x

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts [-1, 1] RGB image tensor to PIL Image."""
        t = tensor.detach().cpu().squeeze(0).clamp(-1.0, 1.0)
        t = (t + 1.0) / 2.0 * 255.0
        t = t.permute(1, 2, 0).to(torch.uint8).numpy()
        if t.shape[2] == 1:
            return Image.fromarray(t.squeeze(2), mode="L")
        return Image.fromarray(t, mode="RGB")
