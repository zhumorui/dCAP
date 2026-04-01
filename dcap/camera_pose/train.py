#!/usr/bin/env python3
"""Training entrypoint for truck-trailer camera prediction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from dcap.camera_pose.datasets.truck_trailer_dataset import TruckTrailerDataset
from dcap.camera_pose.models import LossConfig, TruckTrailerCriterion, build_model
from dcap.camera_pose.trainer import OptimConfig, Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train truck-trailer camera predictor")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--tensorboard-logdir", type=str, default=None, help="Directory for TensorBoard logs")
    parser.add_argument("--tensorboard-port", type=int, default=None, help="Launch TensorBoard on the given port")
    parser.add_argument("--tensorboard-host", type=str, default="0.0.0.0", help="Host interface for TensorBoard server")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def build_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"][split]

    common_conf = SimpleNamespace(
        img_size=data_cfg.get("img_size", 518),
        patch_size=data_cfg.get("patch_size", 14),
        augs=SimpleNamespace(scales=data_cfg.get("scale_range", [0.8, 1.2])),
        rescale=data_cfg.get("rescale", True),
        rescale_aug=data_cfg.get("rescale_aug", True),
        landscape_check=data_cfg.get("landscape_check", True),
    )

    dataset_conf = SimpleNamespace(
        data_root=data_cfg["root"],
        version=data_cfg.get("version", "v1.0-mini"),
        seq_len=data_cfg.get("seq_len", 6),
        sample_stride=data_cfg.get("sample_stride", 1),
        split=split,
        queue_length=data_cfg.get("queue_length", 1),
    )

    dataset = TruckTrailerDataset(common_conf, dataset_conf)
    dataset.train(mode=(split == "train"))
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=data_cfg.get("shuffle", True),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=data_cfg.get("drop_last", False),
    )
    return loader


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optim_cfg = OptimConfig(**cfg)
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    schedule_cfg = train_cfg.get("lr_schedule")
    if not schedule_cfg or steps_per_epoch <= 0:
        return None

    warmup_epochs = float(schedule_cfg.get("warmup_epochs", 0.0))
    min_lr_ratio = float(schedule_cfg.get("min_lr_ratio", 0.0))

    total_epochs = int(train_cfg.get("epochs", 1))
    total_steps = max(int(total_epochs * steps_per_epoch), 1)
    warmup_steps = max(int(warmup_epochs * steps_per_epoch), 0)
    warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps == warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    checkpoint_path = model_cfg.get("checkpoint")
    model = build_model(model_cfg["backbone"], model_cfg["head"], checkpoint=checkpoint_path)
    if model_cfg.get("freeze_aggregator", False):
        model.backbone.aggregator.requires_grad_(False)
    model.to(device)

    loss_cfg = LossConfig(**cfg["loss"]) if "loss" in cfg else LossConfig()
    criterion = TruckTrailerCriterion(loss_cfg)
    criterion.to(device)

    optim_cfg = cfg.get("optim", {})
    optimizer = build_optimizer(model, optim_cfg)

    scaler = torch.cuda.amp.GradScaler() if cfg.get("train", {}).get("amp", False) and device.type == "cuda" else None

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val") if "val" in cfg.get("data", {}) else None

    steps_per_epoch = len(train_loader)
    train_cfg = cfg.get("train", {})
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch)

    epochs = train_cfg.get("epochs", 1)
    log_interval = train_cfg.get("log_interval", 10)
    override_checkpoint = train_cfg.get("override_checkpoint", True)
    resume_from = train_cfg.get("resume_from")
    start_epoch_cfg = train_cfg.get("start_epoch")

    work_dir = Path(train_cfg.get("work_dir", "work_dirs/camera_pose"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logdir = Path(args.tensorboard_logdir) if args.tensorboard_logdir else work_dir / "tensorboard"
    logdir.mkdir(parents=True, exist_ok=True)

    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(logdir))
    writer.add_text("config/raw", yaml.safe_dump(cfg))

    tb_program = None
    if args.tensorboard_port is not None:
        try:
            from tensorboard import program

            tb_program = program.TensorBoard()
            tb_program.configure(
                argv=[
                    None,
                    f"--logdir={logdir}",
                    f"--port={args.tensorboard_port}",
                    f"--host={args.tensorboard_host}",
                ]
            )
            url = tb_program.launch()
            print(f"TensorBoard is live at {url}")
        except Exception as exc:  # pragma: no cover - runtime convenience
            tb_program = None
            print(f"Failed to launch TensorBoard server: {exc}")

    trainer = Trainer(model, criterion, optimizer, device, scaler, writer, scheduler)

    last_epoch = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler_state = checkpoint["scaler"]
            if isinstance(scaler_state, dict):
                scaler.load_state_dict(scaler_state)
        trainer.global_step = checkpoint.get("global_step", 0)
        last_epoch = checkpoint.get("epoch", 0)
        if trainer.global_step == 0 and steps_per_epoch > 0:
            trainer.global_step = last_epoch * steps_per_epoch
        for group in optimizer.param_groups:
            group["lr"] = optim_cfg.get("lr", group.get("lr", 0.0))
    else:
        trainer.global_step = 0

    if start_epoch_cfg is not None:
        last_epoch = start_epoch_cfg
    if steps_per_epoch > 0:
        expected_steps = last_epoch * steps_per_epoch
        if trainer.global_step < expected_steps:
            trainer.global_step = expected_steps

    if epochs < last_epoch:
        raise ValueError(f"Configured total epochs ({epochs}) is smaller than start epoch ({last_epoch}).")

    try:
        best_val = None
        best_ckpt_path: Optional[Path] = None

        def save_checkpoint(path: Path) -> None:
            checkpoint_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "global_step": trainer.global_step,
            }
            if scheduler is not None:
                checkpoint_payload["scheduler"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint_payload["scaler"] = scaler.state_dict()
            torch.save(checkpoint_payload, path)
            print(f"Saved checkpoint to {path}")

        for epoch in range(last_epoch + 1, epochs + 1):
            train_metrics = trainer.train_one_epoch(train_loader, epoch, log_interval=log_interval)
            print(f"Epoch {epoch} train: {train_metrics}")
            latest_ckpt_path = work_dir / "checkpoint_latest.pth"
            save_checkpoint(latest_ckpt_path)

            eval_this_epoch = val_loader is not None and (epoch % 12 == 0 or epoch == epochs)
            if eval_this_epoch:
                val_metrics = trainer.evaluate(val_loader) if val_loader is not None else None
                if val_metrics is not None:
                    print(f"Epoch {epoch} val: {val_metrics}")
                    metric_value = val_metrics.get("objective")
                    if metric_value is None:
                        metric_value = val_metrics.get("loss")
                    if metric_value is not None:
                        is_better = best_val is None or metric_value < best_val
                        if is_better:
                            best_val = metric_value
                            best_ckpt_path = work_dir / "checkpoint_best.pth"
                            save_checkpoint(best_ckpt_path)
            elif not override_checkpoint:
                epoch_ckpt_path = work_dir / f"epoch_{epoch:03d}.pth"
                save_checkpoint(epoch_ckpt_path)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if tb_program is not None:
            try:
                tb_program._terminate()  # type: ignore[attr-defined]
            except Exception:
                pass


if __name__ == "__main__":
    main()
