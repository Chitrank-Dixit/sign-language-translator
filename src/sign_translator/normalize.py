"""Spatial landmark normalization for sign language and gesture recognition.

Provides scale- and translation-invariant transformations for 3D hand keypoints,
ensuring distance invariance, wrist-relative origin alignment, and numerical
stability across missing or occluded hands.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def normalize_single_hand(hand: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalize a single hand array of shape (21, 3).

    Translates all landmarks relative to the wrist (landmark 0) and scales
    by the maximum Euclidean distance from the wrist into [-1.0, 1.0].
    If the hand is all-zeros or degenerate (max distance <= eps), returns zeros.

    Args:
        hand: NumPy array of shape (21, 3) representing (x, y, z) landmarks.
        eps: Small epsilon guard to prevent ZeroDivisionError.

    Returns:
        Normalized NumPy array of shape (21, 3) bounded within [-1.0, 1.0].

    Raises:
        ValueError: If input shape is not (21, 3).
    """
    if hand.shape != (21, 3):
        raise ValueError(f"Expected single hand array of shape (21, 3), got {hand.shape}")

    clean = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0)
    wrist = clean[0:1, :]
    translated = clean - wrist
    dists = np.linalg.norm(translated, axis=-1)
    max_dist = float(np.max(dists))

    if max_dist > eps:
        return (translated / max_dist).astype(hand.dtype)
    return np.zeros_like(hand)


def _normalize_numpy(arr: np.ndarray, eps: float) -> np.ndarray:
    """Vectorized normalization implementation for NumPy arrays."""
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = clean.shape

    if orig_shape in ((126,), (2, 21, 3)):
        hands = clean.reshape(1, 2, 21, 3)
    elif orig_shape == (21, 3):
        return normalize_single_hand(arr, eps=eps)
    elif len(orig_shape) == 2 and orig_shape[1] == 126:
        # (T, 126)
        hands = clean.reshape(orig_shape[0], 2, 21, 3)
    elif len(orig_shape) == 4 and orig_shape[1:] == (2, 21, 3):
        # (T, 2, 21, 3)
        hands = clean
    elif len(orig_shape) == 3 and orig_shape[1:] == (2, 21, 3):
        # (T, 2, 21, 3) when rank is 3 or (B, 2, 63)
        hands = clean
    elif len(orig_shape) == 3 and orig_shape[2] == 126:
        # (B, T, 126)
        hands = clean.reshape(orig_shape[0], orig_shape[1], 2, 21, 3)
    elif len(orig_shape) == 5 and orig_shape[2:] == (2, 21, 3):
        # (B, T, 2, 21, 3)
        hands = clean
    else:
        raise ValueError(
            f"Unsupported landmarks shape: {orig_shape}. "
            "Supported shapes: (126,), (2, 21, 3), (21, 3), (T, 126), "
            "(T, 2, 21, 3), (B, T, 126), (B, T, 2, 21, 3)."
        )

    # hands shape is (..., 21, 3)
    wrist = hands[..., 0:1, :]
    translated = hands - wrist
    dists = np.linalg.norm(translated, axis=-1)
    max_dist = np.max(dists, axis=-1, keepdims=True)

    inv_dist = np.zeros_like(max_dist)
    valid_mask = max_dist > eps
    inv_dist[valid_mask] = 1.0 / max_dist[valid_mask]
    scale = inv_dist[..., np.newaxis]
    normalized = translated * scale

    return normalized.reshape(orig_shape).astype(arr.dtype)


def _normalize_torch(t: torch.Tensor, eps: float) -> torch.Tensor:
    """Vectorized normalization implementation for PyTorch tensors."""
    orig_shape = t.shape
    orig_device = t.device
    orig_dtype = t.dtype

    clean = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    if orig_shape in ((126,), (2, 21, 3)):
        hands = clean.reshape(1, 2, 21, 3)
    elif orig_shape == (21, 3):
        hands = clean.reshape(1, 1, 21, 3)
    elif len(orig_shape) == 2 and orig_shape[1] == 126:
        # (T, 126)
        hands = clean.reshape(orig_shape[0], 2, 21, 3)
    elif len(orig_shape) == 4 and orig_shape[1:] == (2, 21, 3):
        # (T, 2, 21, 3)
        hands = clean
    elif len(orig_shape) == 3 and orig_shape[2] == 126:
        # (B, T, 126)
        hands = clean.reshape(orig_shape[0], orig_shape[1], 2, 21, 3)
    elif len(orig_shape) == 5 and orig_shape[2:] == (2, 21, 3):
        # (B, T, 2, 21, 3)
        hands = clean
    else:
        raise ValueError(
            f"Unsupported landmarks shape: {orig_shape}. "
            "Supported shapes: (126,), (2, 21, 3), (21, 3), (T, 126), "
            "(T, 2, 21, 3), (B, T, 126), (B, T, 2, 21, 3)."
        )

    wrist = hands[..., 0:1, :]
    translated = hands - wrist
    dists = torch.linalg.vector_norm(translated, dim=-1)
    max_dist = torch.max(dists, dim=-1, keepdim=True).values

    inv_dist = torch.zeros_like(max_dist)
    valid_mask = max_dist > eps
    inv_dist[valid_mask] = 1.0 / max_dist[valid_mask]
    scale = inv_dist.unsqueeze(-1)
    normalized = translated * scale

    return normalized.reshape(orig_shape).to(device=orig_device, dtype=orig_dtype)


def normalize_landmarks(
    landmarks: np.ndarray | torch.Tensor,
    eps: float = 1e-7,
) -> Any:
    """Normalize 3D hand keypoints across translation and scale.

    Translates all 21 keypoints per hand relative to the wrist origin and
    scales coordinates by the maximum Euclidean distance from the wrist to
    normalize coordinates into [-1.0, 1.0].

    Zero-padded or missing hands are preserved as zeros without raising
    ZeroDivisionError or producing NaNs.

    Supported shapes:
        - Single frame vector: (126,)
        - Single frame hands: (2, 21, 3) or (21, 3)
        - Sequence tensor: (T, 126) or (T, 2, 21, 3)
        - Batched sequence tensor: (B, T, 126) or (B, T, 2, 21, 3)

    Args:
        landmarks: NumPy array or PyTorch Tensor containing coordinates.
        eps: Small threshold to protect against division by zero.

    Returns:
        Normalized array/tensor with identical shape, rank, and type.

    Raises:
        ValueError: If input tensor does not match supported shapes.
    """
    if isinstance(landmarks, torch.Tensor):
        return _normalize_torch(landmarks, eps=eps)
    if isinstance(landmarks, np.ndarray):
        return _normalize_numpy(landmarks, eps=eps)
    raise TypeError(
        f"Unsupported landmarks type: {type(landmarks)}. Expected np.ndarray or torch.Tensor."
    )
