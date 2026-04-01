#!/usr/bin/env python3
"""Interactive visualization script for truck-trailer back camera pose prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from dcap.camera_pose.datasets.truck_trailer_dataset import TruckTrailerDataset
from dcap.camera_pose.models import build_model
from dcap.camera_pose.utils.pose_enc import extri_intri_to_pose_encoding
from dcap.camera_pose.utils.rotation import quat_to_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict back camera pose for validation samples and render visualizations",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of validation samples to visualize",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start visualization from",
    )
    parser.add_argument(
        "--vis-dir",
        default="vis_outputs",
        help="Directory used to store visualization frames",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(cfg: Dict[str, Any]) -> TruckTrailerDataset:
    data_cfgs = cfg.get("data")
    if data_cfgs is None or "val" not in data_cfgs:
        raise KeyError("Config must define a 'data.val' section for inference")
    data_cfg = data_cfgs["val"]

    split_value = data_cfg.get("split", "val")
    if split_value != "val":
        raise ValueError("Inference expects the validation split; update config to use split='val'")

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
    return dataset


def quaternion_to_forward_xy(quaternion: np.ndarray) -> np.ndarray:
    """Return the camera forward direction projected onto the ground plane."""
    quat_tensor = torch.from_numpy(quaternion).float().unsqueeze(0)
    rot = quat_to_mat(quat_tensor).squeeze(0).cpu().numpy()
    forward_world = rot.T @ np.array([0.0, 0.0, 1.0])
    forward_xy = forward_world[[0, 2]]
    norm = np.linalg.norm(forward_xy)
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return (forward_xy / norm).astype(np.float32)


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to roll, pitch, yaw in radians using SciPy."""

    quat_tensor = torch.from_numpy(quaternion).float().unsqueeze(0)
    rot = quat_to_mat(quat_tensor).squeeze(0).cpu().numpy()

    rotation = R.from_matrix(rot)
    euler = rotation.as_euler("xyz", degrees=True)
    return euler.astype(np.float32)


def compute_pose_stats(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Return minimal statistics for legend annotations."""

    pred_xyz = pred_pose[:3]
    gt_xyz = gt_pose[:3]
    delta_xyz = pred_xyz - gt_xyz
    pred_T_norm = float(np.linalg.norm(pred_xyz))

    pred_rpy = quaternion_to_euler(pred_pose[3:7])
    gt_rpy = quaternion_to_euler(gt_pose[3:7])

    return {
        "pred_xyz": pred_xyz,
        "gt_xyz": gt_xyz,
        "delta_xyz": delta_xyz,
        "pred_T_norm": pred_T_norm,
        "pred_rpy": pred_rpy,
        "gt_rpy": gt_rpy,
    }


def render_sample_visualization(
    sample_idx: int,
    sample: Dict[str, Any],
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    pose_stats: Dict[str, np.ndarray | float],
    output_dir: Path,
) -> Path:
    """Render camera mosaics plus BEV pose comparison into a single figure."""

    output_dir.mkdir(parents=True, exist_ok=True)

    images_tensor = sample["images"]
    if images_tensor.dim() == 5:
        images_tensor = images_tensor[-1]
    images = images_tensor.detach().cpu().numpy()
    camera_names = sample.get("camera_names")
    images = np.clip(images, 0.0, 1.0)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.3], wspace=0.1, hspace=0.15)

    for idx_img in range(images.shape[0]):
        row = idx_img // 3
        col = idx_img % 3
        ax_img = fig.add_subplot(gs[row, col])
        img = np.transpose(images[idx_img], (1, 2, 0))
        ax_img.imshow(img)
        title = camera_names[idx_img] if camera_names is not None else f"Camera {idx_img}"
        ax_img.set_title(title, fontsize=10)
        ax_img.axis("off")

    bev_ax = fig.add_subplot(gs[:, 3])
    bev_ax.set_aspect("equal")
    bev_ax.set_title(f"Sample {sample_idx}")
    bev_ax.set_xlabel("X (m)")
    bev_ax.set_ylabel("Z (m)")
    bev_ax.grid(True, linestyle="--", alpha=0.3)

    pred_xy = pred_pose[[0, 2]]
    gt_xy = gt_pose[[0, 2]]
    pred_forward = quaternion_to_forward_xy(pred_pose[3:7])
    gt_forward = quaternion_to_forward_xy(gt_pose[3:7])

    positions = np.vstack([pred_xy, gt_xy])
    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    span = np.maximum(max_vals - min_vals, 1e-3)
    margin = np.maximum(span * 0.3, 0.5)

    gt_handle = bev_ax.scatter(gt_xy[0], gt_xy[1], c="tab:green", label="GT back cam", s=60, marker="o")
    pred_handle = bev_ax.scatter(pred_xy[0], pred_xy[1], c="tab:blue", label="Pred back cam", s=60, marker="x")

    arrow_len = 1.0
    bev_ax.arrow(
        gt_xy[0],
        gt_xy[1],
        gt_forward[0] * arrow_len,
        gt_forward[1] * arrow_len,
        color="tab:green",
        head_width=0.15,
        length_includes_head=True,
        alpha=0.7,
    )
    bev_ax.arrow(
        pred_xy[0],
        pred_xy[1],
        pred_forward[0] * arrow_len,
        pred_forward[1] * arrow_len,
        color="tab:blue",
        head_width=0.15,
        length_includes_head=True,
        alpha=0.7,
    )

    bev_ax.legend(handles=[gt_handle, pred_handle], loc="upper right")
    bev_ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    bev_ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])

    pred_rpy_deg = np.degrees(pose_stats["pred_rpy"])
    gt_rpy_deg = np.degrees(pose_stats["gt_rpy"])

    text_lines = [
        f"Pred T: [{pose_stats['pred_xyz'][0]:.2f}, {pose_stats['pred_xyz'][1]:.2f}, {pose_stats['pred_xyz'][2]:.2f}] m",
        f"|Pred T|: {pose_stats['pred_T_norm']:.2f} m",
        f"Pred R: [{pred_rpy_deg[0]:.1f}, {pred_rpy_deg[1]:.1f}, {pred_rpy_deg[2]:.1f}]°",
        f"GT   T: [{pose_stats['gt_xyz'][0]:.2f}, {pose_stats['gt_xyz'][1]:.2f}, {pose_stats['gt_xyz'][2]:.2f}] m",
        f"GT   R: [{gt_rpy_deg[0]:.1f}, {gt_rpy_deg[1]:.1f}, {gt_rpy_deg[2]:.1f}]°",
        f"ΔT   : [{pose_stats['delta_xyz'][0]:.2f}, {pose_stats['delta_xyz'][1]:.2f}, {pose_stats['delta_xyz'][2]:.2f}] m",
    ]
    bev_ax.text(
        0.02,
        0.02,
        "\n".join(text_lines),
        transform=bev_ax.transAxes,
        fontsize=9,
        family="monospace",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    frame_path = output_dir / f"sample_{sample_idx:05d}.png"
    fig.savefig(frame_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return frame_path


def infer_single_sample(model: torch.nn.Module, sample: Dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run the back-pose head for a dataset sample and return prediction and GT pose."""

    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value

    with torch.no_grad():
        back_pose = model.predict_back_pose(batch)

    translation = back_pose[0, 0, :3].detach().cpu().numpy()
    quaternion = back_pose[0, 0, 3:].detach().cpu().numpy()
    pred_pose = np.concatenate([translation, quaternion], axis=0)

    extrinsics = sample["extrinsics"]
    intrinsics = sample["intrinsics"]
    if extrinsics.dim() == 4:
        extrinsics = extrinsics.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
    gt_pose_enc = extri_intri_to_pose_encoding(
        extrinsics[-1:].to(device),
        intrinsics[-1:].to(device),
        sample["images"].shape[-2:],
    )
    gt_pose = gt_pose_enc[0, 3, :7].detach().cpu().numpy()

    return pred_pose, gt_pose


def compute_sample_range(length: int, start: int, count: int) -> range:
    if count <= 0:
        raise ValueError("--num-samples must be positive")
    if start < 0:
        raise ValueError("--start-index must be non-negative")
    end = min(start + count, length)
    if start >= length or start >= end:
        raise ValueError("Requested sample range is empty; adjust --start-index/--num-samples")
    return range(start, end)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for inference; no compatible GPU was found.")

    device = torch.device(args.device)

    model_cfg = cfg["model"].copy()
    checkpoint = Path(args.checkpoint)

    model = build_model(model_cfg["backbone"], model_cfg["head"], checkpoint=str(checkpoint))
    state = torch.load(checkpoint, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    dataset = build_dataset(cfg)
    vis_dir = Path(args.vis_dir)

    sample_range = compute_sample_range(len(dataset), args.start_index, args.num_samples)

    for idx in sample_range:
        sample = dataset[idx]
        pred_pose, gt_pose = infer_single_sample(model, sample, device)
        display_sample = dict(sample)
        if torch.is_tensor(sample["images"]) and sample["images"].dim() == 5:
            display_sample["images"] = sample["images"][-1]
        stats = compute_pose_stats(pred_pose, gt_pose)
        frame_path = render_sample_visualization(idx, display_sample, pred_pose, gt_pose, stats, vis_dir)
        print(
            f"Sample {idx}: saved visualization to {frame_path}\n"
            f"  pred_xyz={stats['pred_xyz']} m\n"
            f"  delta_xyz={stats['delta_xyz']} m"
        )


if __name__ == "__main__":
    main()
