"""Minimal geometry helpers required for camera prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def closed_form_inverse_se3(
    se3: np.ndarray | torch.Tensor,
    R: Optional[np.ndarray | torch.Tensor] = None,
    T: Optional[np.ndarray | torch.Tensor] = None,
) -> np.ndarray | torch.Tensor:
    """Invert batched SE(3) matrices.

    Accepts either numpy arrays or torch tensors with shape ``(N, 4, 4)`` or ``(N, 3, 4)``.
    When ``(N, 3, 4)`` is provided, the last row ``[0, 0, 0, 1]`` is assumed.

    Args:
        se3: Batched transformation matrices.
        R: Optional rotation matrices corresponding to ``se3``. If ``None`` the
           rotation is extracted from ``se3``.
        T: Optional translation vectors corresponding to ``se3``. If ``None`` the
           translation is extracted from ``se3``.

    Returns:
        Batched inverse transformations with the same type (numpy or torch) as ``se3``.
    """
    is_numpy = isinstance(se3, np.ndarray)
    backend = np if is_numpy else torch

    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"Expected se3 of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    if R is None:
        R = se3[..., :3, :3]
    if T is None:
        T = se3[..., :3, 3:4]

    R_t = np.swapaxes(R, -1, -2) if is_numpy else R.transpose(-1, -2)
    minus_Rt_T = -R_t @ T

    batch = se3.shape[0]
    if is_numpy:
        eye_row = np.zeros((batch, 1, 4), dtype=se3.dtype)
        inv = np.zeros((batch, 4, 4), dtype=se3.dtype)
    else:
        eye_row = torch.zeros((batch, 1, 4), dtype=se3.dtype, device=se3.device)
        inv = torch.zeros((batch, 4, 4), dtype=se3.dtype, device=se3.device)
    eye_row[..., 0, 3] = 1.0

    inv[..., :3, :3] = R_t
    inv[..., :3, 3:4] = minus_Rt_T
    inv[..., 3:, :] = eye_row
    return inv


def se3_to_rt(se3: np.ndarray | torch.Tensor) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Split an SE(3) matrix into rotation and translation components."""
    R = se3[..., :3, :3]
    T = se3[..., :3, 3:4]
    return R, T
