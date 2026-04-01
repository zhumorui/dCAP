"""Truck-trailer pose prediction model components."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn

from dcap.camera_pose.backbone import VGGTBackbone
from dcap.camera_pose.heads.trailer_camera_head import TrailerCameraHead
from dcap.camera_pose.losses.trailer_delta_loss import compute_back_camera_pose_loss


class TruckTrailerModel(nn.Module):
    """Wrapper that wires the VGGT backbone with the trailer camera head."""

    def __init__(self, backbone: VGGTBackbone, trailer_head: TrailerCameraHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.trailer_head = trailer_head

    def forward(self, inputs: Any, num_iterations: int = 4) -> Dict[str, Any]:
        batch: Optional[Dict[str, Any]]
        if isinstance(inputs, dict):
            batch = inputs
            images = batch["images"]
        else:
            batch = None
            images = inputs

        if images.dim() == 5:
            images = images.unsqueeze(1)
        if images.dim() != 6:
            raise ValueError(f"Expected input images with shape [B, T, 6, 3, H, W], got {images.shape}")

        B, T, S, C_in, H, W = images.shape

        token_buffers: list[list[torch.Tensor]] | None = None
        for t in range(T):
            step_tokens_list, _ = self.backbone(images[:, t])
            if token_buffers is None:
                token_buffers = [[tokens] for tokens in step_tokens_list]
            else:
                for idx, tokens in enumerate(step_tokens_list):
                    token_buffers[idx].append(tokens)

        if token_buffers is None:
            raise RuntimeError("Backbone produced no tokens; check input dimensions.")

        tokens_list = [torch.stack(buffer, dim=1) for buffer in token_buffers]

        head_kwargs: Dict[str, Any] = {"image_hw": (H, W)}
        if batch is not None:
            for key in ("extrinsics", "intrinsics", "frame_mask", "cam_ids", "ego_poses"):
                if key in batch:
                    head_kwargs[key] = batch[key]

        predictions = self.trailer_head(tokens_list, num_iterations=num_iterations, **head_kwargs)
        return predictions

    @torch.no_grad()
    def predict_back_pose(self, inputs: Any, num_iterations: int = 4) -> torch.Tensor:
        self.eval()
        predictions = self.forward(inputs, num_iterations=num_iterations)
        back_pose = predictions["back_pose_list"][-1]
        return back_pose


@dataclass
class LossConfig:
    loss_type: str = "l1"
    gamma: float = 0.6
    translation_weight: float = 1.0
    rotation_weight: float = 1.0


class TruckTrailerCriterion(nn.Module):
    """Composable loss that mirrors the training script from VGGT."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        pose_losses = compute_back_camera_pose_loss(
            predictions,
            batch,
            loss_type=self.config.loss_type,
            gamma=self.config.gamma,
            weight_trans=self.config.translation_weight,
            weight_rot=self.config.rotation_weight,
        )
        total_loss = pose_losses.pop("total_loss")
        losses.update(pose_losses)
        losses["objective"] = total_loss
        return losses


def build_model(
    backbone_cfg: Dict[str, Any],
    head_cfg: Dict[str, Any],
    checkpoint: Optional[str] = None,
    strict: bool = False,
) -> TruckTrailerModel:
    """Instantiate the model and optionally load VGGT pretrained weights."""
    backbone = VGGTBackbone(**backbone_cfg)
    head = TrailerCameraHead(**head_cfg)

    model = TruckTrailerModel(backbone, head)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        if not isinstance(state, dict):
            raise TypeError(
                f"Unsupported checkpoint payload type {type(state).__name__}; "
                "expected a state dict or a container with 'model'/'state_dict'."
            )

        keys = list(state.keys())
        has_backbone_or_head = any(
            key.startswith("backbone.") or key.startswith("trailer_head.") for key in keys
        )
        has_aggregator = any(
            key.startswith("aggregator.") or key.startswith("backbone.aggregator.") for key in keys
        )

        if has_backbone_or_head:
            load_info = model.load_state_dict(state, strict=False)
            missing_keys = list(getattr(load_info, "missing_keys", []))
            unexpected_keys = list(getattr(load_info, "unexpected_keys", []))
            if missing_keys or unexpected_keys:
                message = (
                    f"Checkpoint {checkpoint} is missing parameters {missing_keys} "
                    f"or contains unexpected entries {unexpected_keys}."
                )
                if strict:
                    raise RuntimeError(message)
                warnings.warn(message, UserWarning)
        elif has_aggregator and not has_backbone_or_head:
            unsupported = [
                key
                for key in keys
                if not (key.startswith("aggregator.") or key.startswith("backbone.aggregator."))
            ]
            if unsupported:
                message = (
                    "Checkpoint contains unsupported parameter groups when loading VGGT weights: "
                    f"{unsupported}"
                )
                if strict:
                    raise RuntimeError(message)
                warnings.warn(message, UserWarning)

            aggregator_state: Dict[str, Any] = {}
            for key, value in state.items():
                if key.startswith("aggregator."):
                    aggregator_state[key.partition(".")[2]] = value
                elif key.startswith("backbone.aggregator."):
                    aggregator_state[key.split(".", 2)[2]] = value

            load_info = backbone.aggregator.load_state_dict(aggregator_state, strict=False)
            missing_keys = list(getattr(load_info, "missing_keys", []))
            unexpected_keys = list(getattr(load_info, "unexpected_keys", []))
            if missing_keys or unexpected_keys:
                message = (
                    "VGGT aggregator checkpoint is incompatible: "
                    f"missing {missing_keys} or unexpected {unexpected_keys}"
                )
                if strict:
                    raise RuntimeError(message)
                warnings.warn(message, UserWarning)
        else:
            raise RuntimeError(
                "Checkpoint must contain either full model parameters "
                "(prefixed with 'backbone.' and 'trailer_head.') or VGGT aggregator weights."
            )

    return model
