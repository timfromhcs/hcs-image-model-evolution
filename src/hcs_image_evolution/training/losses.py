"""Distillation and training loss functions."""

import torch
import torch.nn.functional as F
from torch import nn


class FlowMatchingLoss(nn.Module):
    """Computes mean squared error between student predicted velocity and target velocity."""

    def __init__(self, weighting: str = "uniform"):
        super().__init__()
        self.weighting = weighting

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(v_pred, v_target, reduction="none")
        if self.weighting == "snr":
            weight = (1.0 - t) / (t + 1e-4)
            weight = weight.view(-1, *([1] * (v_pred.ndim - 1)))
            loss = loss * weight
        return loss.mean()


class TrajectoryMatchingLoss(nn.Module):
    """Multi-step segment trajectory approximation loss."""

    def __init__(self):
        super().__init__()

    def forward(self, student_endpoint: torch.Tensor, teacher_endpoint: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_endpoint, teacher_endpoint)


class EditingConsistencyLoss(nn.Module):
    """Preserves unedited regions using an inverse change mask."""

    def __init__(self, mask_weight: float = 1.0):
        super().__init__()
        self.mask_weight = mask_weight

    def forward(
        self,
        predicted: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # mask: 1 where changed, 0 where preserved
        preserve_mask = 1.0 - mask
        diff = (predicted - source) * preserve_mask
        return (diff**2).mean() * self.mask_weight
