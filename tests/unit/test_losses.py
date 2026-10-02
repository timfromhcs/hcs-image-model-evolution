"""Unit tests for flow matching loss and schedule math."""

import torch

from hcs_image_evolution.training.losses import FlowMatchingLoss
from hcs_image_evolution.training.schedules import FlowMatchingSchedule


def test_flow_matching_interpolation():
    schedule = FlowMatchingSchedule()
    x_0 = torch.zeros((2, 4, 16, 16))
    x_1 = torch.ones((2, 4, 16, 16))
    t = torch.tensor([0.5, 0.5])

    x_t, v_target = schedule.interpolate(x_0, x_1, t)
    assert torch.allclose(x_t, torch.full_like(x_t, 0.5))
    assert torch.allclose(v_target, torch.ones_like(v_target))


def test_flow_matching_loss():
    loss_fn = FlowMatchingLoss()
    v_pred = torch.ones((2, 4, 16, 16))
    v_target = torch.ones((2, 4, 16, 16))
    t = torch.tensor([0.5, 0.5])

    loss = loss_fn(v_pred, v_target, t)
    assert torch.isclose(loss, torch.tensor(0.0))
