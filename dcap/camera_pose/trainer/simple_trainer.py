"""Minimal training loop utilities for camera prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


class Trainer:
    """Lightweight trainer handling a single-optimizer setup."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        writer: Optional[SummaryWriter] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.writer = writer
        self.scheduler = scheduler
        self.global_step = 0

    def train_one_epoch(self, dataloader: DataLoader, epoch: int, log_interval: int = 10) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}

        progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch)
                    loss_dict = self.criterion(predictions, batch)
                    loss = loss_dict["objective"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch)
                loss_dict = self.criterion(predictions, batch)
                loss = loss_dict["objective"]
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in loss_dict.items():
                running[key] = running.get(key, 0.0) + value.item()
                if self.writer is not None:
                    self.writer.add_scalar(f"train/{key}", value.item(), self.global_step)

            if self.writer is not None:
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    lr = param_group.get("lr")
                    if lr is not None:
                        self.writer.add_scalar(f"train/lr_group_{group_idx}", lr, self.global_step)

            self.global_step += 1

            if (step + 1) % log_interval == 0:
                averaged = {k: v / (step + 1) for k, v in running.items()}
                progress.set_postfix({k: f"{val:.4f}" for k, val in averaged.items()})

        num_steps = max(len(dataloader), 1)
        epoch_metrics = {k: v / num_steps for k, v in running.items()}
        if self.writer is not None:
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
            self.writer.flush()
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        aggregated: Dict[str, float] = {}
        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            predictions = self.model(batch)
            loss_dict = self.criterion(predictions, batch)
            for key, value in loss_dict.items():
                aggregated[key] = aggregated.get(key, 0.0) + value.item()
        num_steps = max(len(dataloader), 1)
        eval_metrics = {k: v / num_steps for k, v in aggregated.items()}
        if self.writer is not None:
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, self.global_step)
            self.writer.flush()
        return eval_metrics
