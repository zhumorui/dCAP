#!/usr/bin/env python3
"""Export predicted trailer camera calibrations for BEVFormer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcap.camera_pose.datasets.truck_trailer_dataset import TruckTrailerDataset
from dcap.camera_pose.models import build_model
from dcap.camera_pose.utils.rotation import mat_to_quat, quat_to_mat


CAM_BACK = "CAM_BACK"
CAM_BACK_LEFT = "CAM_BACK_LEFT"
CAM_BACK_RIGHT = "CAM_BACK_RIGHT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predicted trailer calibrations")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output", required=True, help="Destination calibrated_sensor JSON path")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for data.val.batch_size",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(cfg: Dict[str, Any], batch_size_override: int | None) -> DataLoader:
    data_cfgs = cfg.get("data")
    if data_cfgs is None or "val" not in data_cfgs:
        raise KeyError("Config must define a 'data.val' section for export")
    data_cfg = data_cfgs["val"]

    split_value = data_cfg.get("split", "val")
    if split_value != "val":
        raise ValueError("Expected validation split for export; set data.val.split='val'")

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

    batch_size = batch_size_override or data_cfg.get("batch_size", 1)
    if batch_size < 1:
        raise ValueError("data.val.batch_size must be >= 1")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=False,
    )


def wxyz_to_matrix(quat_wxyz: Iterable[float]) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    quat_tensor = torch.from_numpy(quat_xyzw).float().unsqueeze(0)
    return quat_to_mat(quat_tensor).squeeze(0).cpu().numpy()


def matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
    rot_tensor = torch.from_numpy(rot).float().unsqueeze(0)
    quat_xyzw = mat_to_quat(rot_tensor).squeeze(0).cpu().numpy()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def build_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def extract_trailer_transforms(dataset: TruckTrailerDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fixed transforms from CAM_BACK to CAM_BACK_LEFT / CAM_BACK_RIGHT."""

    def camera_transform(sample_data: Dict[str, Any]) -> np.ndarray:
        calib = dataset.calib_sensor_dict[sample_data["calibrated_sensor_token"]]
        rotation = wxyz_to_matrix(calib["rotation"])
        translation = np.asarray(calib["translation"], dtype=np.float64)
        return build_transform(rotation, translation)

    rel_left: List[np.ndarray] = []
    rel_right: List[np.ndarray] = []

    for sample_info in dataset.samples:
        camera_data = sample_info["camera_data"]
        t_back = camera_transform(camera_data[CAM_BACK])
        t_left = camera_transform(camera_data[CAM_BACK_LEFT])
        t_right = camera_transform(camera_data[CAM_BACK_RIGHT])

        rel_left.append(np.linalg.inv(t_back) @ t_left)
        rel_right.append(np.linalg.inv(t_back) @ t_right)

    rel_left_stack = np.stack(rel_left)
    rel_right_stack = np.stack(rel_right)

    left_ref = rel_left_stack[0]
    right_ref = rel_right_stack[0]

    if not (np.allclose(rel_left_stack, left_ref, atol=1e-4) and np.allclose(rel_right_stack, right_ref, atol=1e-4)):
        raise RuntimeError("Trailer relative transforms vary across samples; cannot treat them as fixed.")

    return left_ref, right_ref


def collect_trailer_tokens(dataset: TruckTrailerDataset) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for sample_info in dataset.samples:
        sample_token = sample_info["sample_token"]
        camera_data = sample_info["camera_data"]
        mapping[sample_token] = {
            CAM_BACK: camera_data[CAM_BACK]["calibrated_sensor_token"],
            CAM_BACK_LEFT: camera_data[CAM_BACK_LEFT]["calibrated_sensor_token"],
            CAM_BACK_RIGHT: camera_data[CAM_BACK_RIGHT]["calibrated_sensor_token"],
        }
    return mapping


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    translation = pose[:3]
    quat_xyzw = pose[3:7]
    quat_tensor = torch.from_numpy(quat_xyzw).float().unsqueeze(0)
    rotation = quat_to_mat(quat_tensor).squeeze(0).cpu().numpy()
    return build_transform(rotation, translation)


def transform_to_pose(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    quat_wxyz = matrix_to_wxyz(rotation)
    return translation, quat_wxyz


def load_calibrated_sensor(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data, {entry["token"]: entry for entry in data}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; no compatible GPU was found.")

    device = torch.device(args.device)

    dataloader = build_dataloader(cfg, args.batch_size)
    dataset: TruckTrailerDataset = dataloader.dataset  # type: ignore[assignment]

    left_transform, right_transform = extract_trailer_transforms(dataset)
    trailer_tokens = collect_trailer_tokens(dataset)

    model_cfg = cfg["model"].copy()
    checkpoint_path = Path(args.checkpoint)

    model = build_model(model_cfg["backbone"], model_cfg["head"], checkpoint=str(checkpoint_path))
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    predictions: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        seq_tokens = list(batch["seq_name"])

        back_pose = model.predict_back_pose(batch)[:, 0, :7]
        back_pose_np = back_pose.detach().cpu().numpy()

        for token, pose in zip(seq_tokens, back_pose_np):
            t_back = pose_to_transform(pose)
            t_left = t_back @ left_transform
            t_right = t_back @ right_transform

            back_translation, back_quat = transform_to_pose(t_back)
            left_translation, left_quat = transform_to_pose(t_left)
            right_translation, right_quat = transform_to_pose(t_right)

            predictions[token] = {
                CAM_BACK: (back_translation, back_quat),
                CAM_BACK_LEFT: (left_translation, left_quat),
                CAM_BACK_RIGHT: (right_translation, right_quat),
            }

    data_root = Path(dataset.data_root) / dataset.version
    original_calib_path = data_root / "calibrated_sensor.json"
    original_data, calib_by_token = load_calibrated_sensor(original_calib_path)

    updated = 0
    for sample_token, camera_tokens in trailer_tokens.items():
        if sample_token not in predictions:
            continue
        sample_preds = predictions[sample_token]
        for cam_name, calib_token in camera_tokens.items():
            translation, quat_wxyz = sample_preds[cam_name]
            calib_entry = calib_by_token[calib_token]
            calib_entry["translation"] = [float(x) for x in translation.tolist()]
            calib_entry["rotation"] = [float(x) for x in quat_wxyz.tolist()]
            updated += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(original_data, handle, indent=2)

    print(f"Wrote predicted calibrations to {output_path} (updated {updated} camera entries).")


if __name__ == "__main__":
    main()

