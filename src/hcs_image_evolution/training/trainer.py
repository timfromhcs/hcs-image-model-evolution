"""Training loop orchestrator for flow matching diffusion and few-step distillation."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from hcs_image_evolution.training.checkpointing import CheckpointManager
from hcs_image_evolution.training.losses import FlowMatchingLoss
from hcs_image_evolution.training.schedules import FlowMatchingSchedule
from hcs_image_evolution.utils.logging import log_event, logger


class EvolutionTrainer:
    """Manages forward passes, velocity loss computations, backward passes, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        schedule: FlowMatchingSchedule | None = None,
        loss_fn: FlowMatchingLoss | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.schedule = schedule or FlowMatchingSchedule()
        self.loss_fn = loss_fn or FlowMatchingLoss()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_steps(self, num_steps: int, checkpoint_interval: int = 100) -> dict[str, float]:
        """Executes a defined number of training iterations."""
        self.model.train()
        step = 0
        total_loss = 0.0
        start_time = time.time()

        data_iter = iter(self.dataloader)
        logger.info("Starting training loop for %d steps on %s...", num_steps, self.device)

        while step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            images = batch["image"].to(self.device)
            batch_size = images.shape[0]

            # Flow matching formulation:
            # Noise x_1 ~ N(0, I)
            x_1 = torch.randn_like(images)
            x_0 = images

            t = self.schedule.sample_timesteps(batch_size, device=torch.device(self.device))
            x_t, v_target = self.schedule.interpolate(x_0, x_1, t)

            # Student forward pass predicting velocity
            try:
                # If model expects (x, t, conditioning), call accordingly
                v_pred = self.model(x_t)
            except TypeError:
                v_pred = self.model(x_t, t)

            loss = self.loss_fn(v_pred, v_target, t)
            loss_scaled = loss / self.gradient_accumulation_steps
            loss_scaled.backward()

            total_loss += loss.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            step += 1

            if step % 20 == 0 or step == num_steps:
                avg_loss = total_loss / step
                logger.info("Step %d/%d - Loss: %.4f", step, num_steps, avg_loss)
                log_event("train_step", {"step": step, "loss": round(avg_loss, 4)})

            if step % checkpoint_interval == 0 or step == num_steps:
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    epoch=0,
                    model=self.model,
                    optimizer=self.optimizer,
                    metrics={"loss": total_loss / step},
                )

        duration = time.time() - start_time
        metrics = {
            "final_loss": round(total_loss / max(step, 1), 4),
            "steps": step,
            "duration_seconds": round(duration, 2),
        }
        log_event("train_completed", metrics)
        return metrics
