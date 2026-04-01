#!/usr/bin/env python3
"""Batch evaluation script for truck-trailer camera prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcap.camera_pose.datasets.truck_trailer_dataset import TruckTrailerDataset
from dcap.camera_pose.models import LossConfig, TruckTrailerCriterion, build_model
from dcap.camera_pose.utils.pose_enc import extri_intri_to_pose_encoding
from dcap.camera_pose.utils.rotation import quat_to_mat


BACK_CAMERA_INDEX = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate truck-trailer camera predictor")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Optional path to dump aggregated metrics as YAML",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def build_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    data_cfgs = cfg.get("data")
    if data_cfgs is None or "val" not in data_cfgs:
        raise KeyError("Config must define a 'data.val' section for evaluation")
    data_cfg = data_cfgs["val"]

    split_value = data_cfg.get("split", "val")
    if split_value != "val":
        raise ValueError("Evaluation expects the validation split; update config to use split='val'")

    common_conf = SimpleNamespace(
        img_size=data_cfg.get("img_size", 518),
        patch_size=data_cfg.get("patch_size", 14),
        augs=SimpleNamespace(scales=[1.0, 1.0]),
        rescale=True,
        rescale_aug=False,
        landscape_check=True,
    )
    dataset_conf = SimpleNamespace(
        data_root=data_cfg["root"],
        version=data_cfg.get("version", "v1.0-mini"),
        seq_len=data_cfg.get("seq_len", 6),
        sample_stride=data_cfg.get("sample_stride", 1),
        split=split_value,
        queue_length=data_cfg.get("queue_length", 1),
    )

    dataset = TruckTrailerDataset(common_conf, dataset_conf)
    dataset.train(mode=False)

    batch_size = data_cfg.get("batch_size", 1)
    if batch_size < 1:
        raise ValueError("data.val.batch_size must be >= 1 for evaluation")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=False,
    )
    return loader

def extract_ground_truth_pose(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    if extrinsics.dim() >= 5:
        extrinsics = extrinsics[:, -1]
    if intrinsics.dim() >= 5:
        intrinsics = intrinsics[:, -1]

    pose_encoding = extri_intri_to_pose_encoding(extrinsics, intrinsics, image_hw)
    return pose_encoding[:, BACK_CAMERA_INDEX, :7]


def compute_pose_metrics_batch(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
    delta_xyz = pred_pose[:, :3] - gt_pose[:, :3]
    delta_T = torch.linalg.norm(delta_xyz, dim=-1)
    pred_rot = quat_to_mat(pred_pose[:, 3:7])
    gt_rot = quat_to_mat(gt_pose[:, 3:7])
    rel_rot = pred_rot.transpose(-1, -2) @ gt_rot
    trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    rra = torch.acos(cos_theta)

    return {
        "delta_xyz": delta_xyz,
        "delta_T": delta_T,
        "rra": rra,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for evaluation; no compatible GPU was found.")

    device = torch.device(args.device)

    model_cfg = cfg["model"].copy()
    checkpoint = Path(args.checkpoint)

    model = build_model(model_cfg["backbone"], model_cfg["head"], checkpoint=str(checkpoint))
    state = torch.load(checkpoint, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    dataloader = build_dataloader(cfg)

    use_amp = bool(cfg.get("train", {}).get("amp", False))

    loss_cfg = LossConfig(**cfg.get("loss", {}))
    criterion = TruckTrailerCriterion(loss_cfg).to(device)

    all_delta_T = []
    all_delta_xyz = []
    all_rra = []
    objective_sum = torch.zeros(1, device=device)

    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch["images"]
            extrinsics = batch["extrinsics"]
            intrinsics = batch["intrinsics"]

            image_hw = images.shape[-2:]

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(batch)
                loss_dict = criterion(predictions, batch)

            pred_pose = predictions["back_pose_list"][-1][:, 0, :7].to(extrinsics.dtype)
            gt_pose = extract_ground_truth_pose(extrinsics, intrinsics, image_hw)

            metrics = compute_pose_metrics_batch(pred_pose, gt_pose)
            print(metrics)

            all_delta_T.append(metrics["delta_T"].detach())
            all_delta_xyz.append(metrics["delta_xyz"].detach())
            all_rra.append(metrics["rra"].detach())
            batch_size = metrics["delta_T"].shape[0]
            objective_sum += loss_dict["objective"].detach().to(torch.float32) * batch_size

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation dataset is empty; nothing to evaluate.")

    delta_T_all = torch.cat(all_delta_T, dim=0)
    delta_xyz_all = torch.cat(all_delta_xyz, dim=0)
    rra_all = torch.cat(all_rra, dim=0)

    mean_delta_T = delta_T_all.mean()
    mean_delta_xyz = torch.mean(torch.abs(delta_xyz_all), dim=0)
    mean_rra = rra_all.mean()
    mean_objective = objective_sum / total_samples

    summary = {
        "delta_T": float(mean_delta_T.cpu().item()),
        "delta_x": float(mean_delta_xyz[0].cpu().item()),
        "delta_y": float(mean_delta_xyz[1].cpu().item()),
        "delta_z": float(mean_delta_xyz[2].cpu().item()),
        "RRA": float(mean_rra.cpu().item()),
        "objective_loss": float(mean_objective.cpu().item()),
    }

    print("Evaluation summary:")
    print(f"  delta_T: {summary['delta_T']:.6f}")
    print(f"  delta_x: {summary['delta_x']:.6f}")
    print(f"  delta_y: {summary['delta_y']:.6f}")
    print(f"  delta_z: {summary['delta_z']:.6f}")
    print(f"  RRA: {summary['RRA']:.6f}")
    print(f"  objective_loss: {summary['objective_loss']:.6f}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle)
        print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
