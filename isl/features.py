"""Pure feature extraction and normalization functions shared by extract.py and matcher.py."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from isl.schema import COORDS, NUM_HANDS, NUM_LANDMARKS, SEQ_LEN, SUMMARY_DIM

_EPS: float = 1e-8
_WRIST_IDX: int = 0
_MIDDLE_TIP_IDX: int = 12


def normalize_hand(
    landmarks: NDArray[np.floating],
    align_y: bool = False,
) -> NDArray[np.float32]:
    """Translate wrist to origin, scale by wrist-to-middle-fingertip distance, and optionally align +Y."""
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, COORDS):
        raise ValueError(f"Expected shape ({NUM_LANDMARKS}, {COORDS}), got {pts.shape}")

    if np.allclose(pts, 0.0, atol=_EPS):
        return np.zeros((NUM_LANDMARKS, COORDS), dtype=np.float32)

    wrist = pts[_WRIST_IDX]
    centered = pts - wrist
    scale = float(np.linalg.norm(centered[_MIDDLE_TIP_IDX]))
    if scale < _EPS:
        return np.zeros((NUM_LANDMARKS, COORDS), dtype=np.float32)

    normed = (centered / scale).astype(np.float32)

    if align_y:
        tip = normed[_MIDDLE_TIP_IDX]
        angle = float(np.arctan2(tip[0], tip[1]))
        cos_a = float(np.cos(-angle))
        sin_a = float(np.sin(-angle))
        x = normed[:, 0].copy()
        y = normed[:, 1].copy()
        normed[:, 0] = cos_a * x - sin_a * y
        normed[:, 1] = sin_a * x + cos_a * y

    return normed


def resample(sequence: NDArray[np.floating], target_len: int = SEQ_LEN) -> NDArray[np.float32]:
    """Linearly interpolate a temporal landmark sequence to a fixed length T."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim < 2:
        raise ValueError(f"Expected sequence with >=2 dims, got shape {seq.shape}")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    in_len = seq.shape[0]
    if in_len == 0:
        out_shape = (target_len, *seq.shape[1:])
        return np.zeros(out_shape, dtype=np.float32)

    if in_len == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)

    if in_len == target_len:
        return seq.copy()

    src_x = np.linspace(0.0, 1.0, num=in_len, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    flat = seq.reshape(in_len, -1)
    out_flat = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out_flat[:, col] = np.interp(dst_x, src_x, flat[:, col])

    return out_flat.reshape((target_len, *seq.shape[1:])).astype(np.float32)


def resample_mask(mask: NDArray[np.bool_], target_len: int = SEQ_LEN) -> NDArray[np.bool_]:
    """Nearest-neighbor resample a boolean hand presence mask (T, 2) to target_len."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.shape[1] != NUM_HANDS:
        raise ValueError(f"Expected mask shape (T, {NUM_HANDS}), got {m.shape}")
    if m.shape[0] == 0:
        return np.zeros((target_len, NUM_HANDS), dtype=bool)
    indices = np.round(np.linspace(0, m.shape[0] - 1, num=target_len)).astype(int)
    return m[indices]


def summary_vector(
    sequence: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Compute fixed-size summary vector [mean, std, min, max, velocity] over sequence."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 4 or seq.shape[1:] != (NUM_HANDS, NUM_LANDMARKS, COORDS):
        raise ValueError(
            f"Expected sequence shape (T, {NUM_HANDS}, {NUM_LANDMARKS}, {COORDS}), got {seq.shape}"
        )

    t_steps = seq.shape[0]
    if t_steps == 0:
        return np.zeros((SUMMARY_DIM,), dtype=np.float32)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (t_steps, NUM_HANDS):
            raise ValueError(f"Expected mask shape {(t_steps, NUM_HANDS)}, got {m.shape}")
        masked_seq = np.where(m[:, :, None, None], seq, 0.0)
    else:
        masked_seq = seq

    flat = masked_seq.reshape(t_steps, -1)

    mean_feat = np.mean(flat, axis=0)
    std_feat = np.std(flat, axis=0)
    min_feat = np.min(flat, axis=0)
    max_feat = np.max(flat, axis=0)

    if t_steps > 1:
        diffs = np.diff(flat, axis=0)
        vel_feat = np.mean(np.abs(diffs), axis=0)
    else:
        vel_feat = np.zeros_like(mean_feat)

    summary = np.concatenate([mean_feat, std_feat, min_feat, max_feat, vel_feat], axis=0)
    return summary.astype(np.float32)