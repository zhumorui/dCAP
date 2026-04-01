# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Light-weight wrapper around the VGGT aggregator for feature extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .aggregator import Aggregator


class VGGTBackbone(nn.Module):
    """VGGT backbone that exposes the aggregator features only.

    This module keeps the original VGGT aggregator architecture so that
    pretrained weights can be reused without pulling in the full VGGT package.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        aggregator_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        aggregator_kwargs = aggregator_kwargs or {}
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **aggregator_kwargs,
        )

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """Runs the VGGT aggregator and returns token lists.

        Args:
            images: Either ``[B, S, 3, H, W]`` or ``[S, 3, H, W]`` tensor and values in [0, 1].

        Returns:
            tuple: ``(tokens, patch_start_idx)`` where ``tokens`` is the list of
            feature tensors from alternating-attention blocks and
            ``patch_start_idx`` marks where patch tokens start in each tensor.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        return aggregated_tokens_list, patch_start_idx

    def load_pretrained(self, checkpoint_path: str, strict: bool = False) -> Dict[str, List[str]]:
        """Loads aggregator weights from a VGGT checkpoint.

        Args:
            checkpoint_path: Path to a serialized VGGT checkpoint (``.pth``).
            strict: Whether to enforce that all aggregator weights are present.

        Returns:
            Dict[str, List[str]]: ``{"missing": missing_keys, "unexpected": unexpected_keys}``.
        """
        state = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        aggregator_state = {}
        for key, value in state.items():
            if key.startswith("aggregator."):
                aggregator_state[key.partition(".")[2]] = value

        missing, unexpected = self.aggregator.load_state_dict(aggregator_state, strict=strict)
        return {"missing": missing, "unexpected": unexpected}


def build_vggt_backbone(config: Dict[str, Any]) -> VGGTBackbone:
    """Factory helper to build a VGGT backbone from a config dict."""
    kwargs = config.copy()
    aggregator_kwargs = kwargs.pop("aggregator_kwargs", None)
    backbone = VGGTBackbone(aggregator_kwargs=aggregator_kwargs, **kwargs)
    return backbone
