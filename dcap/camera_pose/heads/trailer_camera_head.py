# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import List, Optional, Tuple

from dcap.camera_pose.backbone.layers import Mlp
from dcap.camera_pose.backbone.layers.block import Block
from dcap.camera_pose.heads.head_act import base_pose_act


class TrailerCameraHead(nn.Module):
    """Predict the articulated trailer back camera pose from backbone tokens."""

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
    ):
        super().__init__()
        
        # Target dimension: 3 translation + 4 quaternion = 7
        self.target_dim = 7
        
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.trunk_depth = trunk_depth

        self.cam_back_index = 3

        # Build transformer trunk for iterative refinement
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # Normalization layers
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable seed pose token
        self.initial_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Cross-attention components for back camera aggregation
        self.back_query = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.cam_embed = nn.Embedding(6, dim_in)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=num_heads,
            batch_first=True,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=num_heads,
            batch_first=True,
        )

        nn.init.normal_(self.back_query, std=1e-4)
        nn.init.zeros_(self.cam_embed.weight)

        self.delta_proj = nn.Linear(3, dim_in)

        self.temporal_dropout = nn.Dropout(p=0.1)

        self.temporal_scale = nn.Parameter(torch.tensor(0.2))
        self.back_residual_scale = nn.Parameter(torch.tensor(1.0))

        # Adaptive modulation for iterative refinement
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        # Output branch for back camera pose prediction
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)
        

    def forward(
        self,
        aggregated_tokens_list: list,
        *,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        cam_ids: Optional[torch.Tensor] = None,
        image_hw: Optional[Tuple[int, int]] = None,
        ego_poses: Optional[torch.Tensor] = None,
        num_iterations: int = 4,
    ) -> dict:
        """Predict the trailer back camera pose with temporal recursion over the queue."""

        tokens = aggregated_tokens_list[-1]
        if tokens.dim() == 4:
            tokens = tokens.unsqueeze(1)

        if tokens.dim() != 5:
            raise ValueError(
                f"Expected final backbone tokens with shape [B, T, N, S, C], got {tokens.shape}"
            )

        B, T, N, _, _ = tokens.shape
        pose_tokens = self.token_norm(tokens[..., 0, :])

        cam_ids_tensor = self._expand_cam_ids(cam_ids, B, N, pose_tokens.device)
        frame_mask_tensor = None
        if frame_mask is not None:
            frame_mask_tensor = frame_mask.to(device=pose_tokens.device, dtype=torch.bool)
            if frame_mask_tensor.dim() == 1:
                frame_mask_tensor = frame_mask_tensor.unsqueeze(0).expand(B, -1)

        prev_global: Optional[torch.Tensor] = None
        prev_ego_pose: Optional[torch.Tensor] = None
        state_mask = torch.zeros(B, dtype=torch.bool, device=pose_tokens.device)
        ego_pose_tensor = None
        if ego_poses is not None:
            ego_pose_tensor = ego_poses.to(pose_tokens.device)
            if ego_pose_tensor.dim() == 2:
                ego_pose_tensor = ego_pose_tensor.unsqueeze(0)

        pred_back_pose_storage: List[Optional[torch.Tensor]] = [None for _ in range(num_iterations)]

        orig_training = self.training

        for t in range(T):
            is_last = t == T - 1
            if orig_training and not is_last:
                self.eval()
                context = torch.no_grad()
            else:
                if orig_training:
                    self.train()
                context = nullcontext()

            with context:
                current_tokens = pose_tokens[:, t]
                current_ego_pose = ego_pose_tensor[:, t] if ego_pose_tensor is not None else None

                valid_prev_mask = state_mask.clone()
                if frame_mask_tensor is not None:
                    valid_prev_mask = valid_prev_mask & frame_mask_tensor[:, t]

                query = current_tokens.mean(dim=1, keepdim=True)
                kv_curr = self.build_kv(current_tokens, cam_ids_tensor)
                global_curr = self.cross_attend(query, kv_curr)

                temporal_residual = None
                if (
                    prev_global is not None
                    and current_ego_pose is not None
                    and prev_ego_pose is not None
                    and valid_prev_mask.any()
                ):
                    ego_delta = self._compute_ego_delta(prev_ego_pose, current_ego_pose)
                    ego_delta = ego_delta * valid_prev_mask.to(ego_delta.dtype).unsqueeze(-1)
                    aligned_prev = prev_global + self.delta_proj(ego_delta).unsqueeze(1)
                    temporal_out, _ = self.temporal_attn(global_curr, aligned_prev, aligned_prev)
                    temporal_residual = self.temporal_dropout(temporal_out)

                global_tokens = global_curr
                if temporal_residual is not None:
                    global_tokens = global_tokens + self.temporal_scale * temporal_residual

                global_tokens = global_tokens + self.back_residual_scale * current_tokens[:, self.cam_back_index : self.cam_back_index + 1, :]
                iter_outputs = self._iterative_refine(global_tokens, num_iterations)

                update_mask = torch.ones(B, dtype=torch.bool, device=pose_tokens.device)
                if frame_mask_tensor is not None:
                    update_mask = frame_mask_tensor[:, t] | (~state_mask)

            if orig_training and not is_last:
                self.train()

            if is_last:
                for idx, pose in enumerate(iter_outputs):
                    pred_back_pose_storage[idx] = pose
            else:
                state_mask = state_mask | update_mask

                current_global_detached = global_tokens.detach()
                if prev_global is None:
                    prev_global = current_global_detached
                else:
                    mask = update_mask.view(B, 1, 1)
                    prev_global = torch.where(mask, current_global_detached, prev_global)

                if current_ego_pose is not None:
                    detached_ego = current_ego_pose.detach()
                    if prev_ego_pose is None:
                        prev_ego_pose = detached_ego
                    else:
                        mask = update_mask.view(B, 1)
                        prev_ego_pose = torch.where(mask, detached_ego, prev_ego_pose)

        if orig_training:
            self.train()

        pred_back_pose_list: List[torch.Tensor] = []
        for pose in pred_back_pose_storage:
            if pose is None:
                raise RuntimeError("Missing pose prediction for one of the refinement stages.")
            pred_back_pose_list.append(pose)

        return {
            "back_pose_list": pred_back_pose_list,
        }

    def cross_attend(self, query: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross_attn(query, kv_tokens, kv_tokens, need_weights=False)
        return out

    def build_kv(self, pose_tokens: torch.Tensor, cam_ids: torch.Tensor) -> torch.Tensor:
        kv = pose_tokens + self.cam_embed(cam_ids)
        return kv

    def _iterative_refine(self, global_tokens: torch.Tensor, num_iterations: int) -> list:
        B = global_tokens.shape[0]
        pred_pose_enc = None
        pred_pose_list = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.initial_pose_tokens.expand(B, 1, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            tokens_modulated = gate_msa * modulate(self.adaln_norm(global_tokens), shift_msa, scale_msa)
            tokens_modulated = tokens_modulated + global_tokens
            tokens_modulated = self.trunk(tokens_modulated)
            pred_pose_delta = self.pose_branch(self.trunk_norm(tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_delta

            activated_pose = activate_back_pose(pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act)
            pred_pose_list.append(activated_pose)

        return pred_pose_list

    def _expand_cam_ids(self, cam_ids: Optional[torch.Tensor], batch: int, num_cams: int, device: torch.device) -> torch.Tensor:
        if cam_ids is None:
            base = torch.arange(num_cams, device=device)
            return base.unsqueeze(0).expand(batch, -1)
        cam_ids = cam_ids.to(device=device)
        if cam_ids.dim() == 1:
            cam_ids = cam_ids.unsqueeze(0).expand(batch, -1)
        return cam_ids

    def _select_prev_indices(
        self,
        frame_mask: Optional[torch.Tensor],
        length: int,
        device: torch.device,
        batch: int,
    ) -> torch.Tensor:
        if frame_mask is None:
            return torch.full((batch,), length - 1, dtype=torch.long, device=device)

        if frame_mask.dim() == 1:
            frame_mask = frame_mask.unsqueeze(0)
        frame_mask = frame_mask.to(device=device, dtype=torch.bool)
        prev_indices = torch.full((batch,), length - 1, dtype=torch.long, device=device)
        for b in range(batch):
            valid = torch.nonzero(frame_mask[b], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            if valid[-1].item() == length - 1 and valid.numel() >= 2:
                prev_indices[b] = valid[-2]
            else:
                prev_indices[b] = valid[-1]
        return prev_indices

    def _compute_ego_delta(
        self,
        prev_ego: torch.Tensor,
        curr_ego: torch.Tensor,
    ) -> torch.Tensor:
        delta_xy = curr_ego[:, :2] - prev_ego[:, :2]
        delta_yaw = curr_ego[:, 3] - prev_ego[:, 3]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
        return torch.stack([delta_xy[:, 0], delta_xy[:, 1], delta_yaw], dim=-1)
    

def activate_back_pose(pred_pose_enc, trans_act="linear", quat_act="linear"):
    """Apply activation to the predicted back camera pose."""
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)

    return torch.cat([T, quat], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate input tensor using scaling and shifting parameters.
    """
    return x * (1 + scale) + shift
