# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Losses for predicting the trailer back camera pose."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dcap.camera_pose.utils.pose_enc import extri_intri_to_pose_encoding


def compute_back_camera_pose_loss(
    pred_dict: Dict[str, List[torch.Tensor]],
    batch_data: Dict[str, torch.Tensor],
    loss_type: str = "l1",
    gamma: float = 0.6,
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    **_: object,
) -> Dict[str, torch.Tensor]:
    """Compute staged losses for the trailer back camera pose prediction.

    Args:
        pred_dict: Model outputs containing ``back_pose_list``.
        batch_data: Batch dictionary produced by the dataset/dataloader.
        loss_type: ``"l1"`` (default) or ``"l2"`` for the regression metric.
        gamma: Exponential decay for multi-iteration supervision.
        weight_trans: Weight applied to the translation loss term.
        weight_rot: Weight applied to the quaternion loss term.

    Returns:
        Dictionary with per-component losses and the combined objective.
    """

    if "back_pose_list" not in pred_dict:
        raise KeyError("Model predictions must contain 'back_pose_list'.")

    pred_pose_list = pred_dict["back_pose_list"]
    if len(pred_pose_list) == 0:
        raise ValueError("'back_pose_list' is empty; expected at least one stage.")

    gt_extrinsics = batch_data["extrinsics"]
    gt_intrinsics = batch_data["intrinsics"]
    image_hw = batch_data["images"].shape[-2:]

    if gt_extrinsics.dim() >= 5:
        gt_extrinsics = gt_extrinsics[:, -1]
    if gt_intrinsics.dim() >= 5:
        gt_intrinsics = gt_intrinsics[:, -1]

    pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type="absT_quaR_FoV"
    )  # [B, 6, 9]

    # Index 3 corresponds to CAM_BACK in the dataset ordering. (x,y,z, qx, qy, qz, qw)
    gt_back_pose = pose_encoding[:, 3:4, :7]  # [B, 1, 7]

    total_trans = gt_back_pose.new_zeros(())
    total_rot = gt_back_pose.new_zeros(())
    weight_sum = gt_back_pose.new_zeros(())
    stage_trans_losses: List[torch.Tensor] = []
    stage_rot_losses: List[torch.Tensor] = []

    num_stages = len(pred_pose_list)
    for stage_idx, stage_pred in enumerate(pred_pose_list):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (num_stages - 1 - stage_idx)
        loss_T, loss_R = back_pose_loss_single(stage_pred, gt_back_pose, loss_type=loss_type)
        stage_trans_losses.append(loss_T.detach())
        stage_rot_losses.append(loss_R.detach())
        total_trans = total_trans + stage_weight * loss_T
        total_rot = total_rot + stage_weight * loss_R
        weight_sum = weight_sum + stage_weight

    weight_sum = weight_sum.clamp_min(1e-6)
    total_trans = total_trans / weight_sum
    total_rot = total_rot / weight_sum

    total_loss = weight_trans * total_trans + weight_rot * total_rot

    loss_dict: Dict[str, torch.Tensor] = {
        "total_loss": total_loss,
    }
    for idx, (stage_T, stage_R) in enumerate(zip(stage_trans_losses, stage_rot_losses)):
        loss_dict[f"loss_back_translation_stage{idx}"] = stage_T
        loss_dict[f"loss_back_rotation_stage{idx}"] = stage_R

    return loss_dict


def back_pose_loss_single(
    pred_pose_enc: torch.Tensor,
    gt_pose_enc: torch.Tensor,
    loss_type: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute translation and quaternion loss for a single prediction stage."""

    pred_T = pred_pose_enc[..., :3]
    pred_quat = pred_pose_enc[..., 3:7]

    gt_T = gt_pose_enc[..., :3]
    gt_quat = gt_pose_enc[..., 3:7]

    if loss_type == "l1":
        loss_T = (pred_T - gt_T).abs().mean()
        loss_R = (pred_quat - gt_quat).abs().mean()
    elif loss_type == "l2":
        loss_T = (pred_T - gt_T).norm(dim=-1).mean()
        loss_R = (pred_quat - gt_quat).norm(dim=-1).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss_T, loss_R
