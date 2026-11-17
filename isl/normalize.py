"""
isl/normalize.py — Normalization of MediaPipe hand landmarks.

Convention (ISL handoff blueprint):
  1. Wrist-origin:  translate every landmark so wrist (index 0) is at (0,0,0).
  2. Scale:         divide by the Euclidean distance from wrist (0) to
                    middle-MCP (landmark 9).  This is roughly palm height and
                    makes the hand size view-invariant.
  3. Align Y (optional): rotate the XY plane so that landmark 12
                    (middle-finger-tip) points toward +Y.

Reference indices (MediaPipe Hands):
  0  = wrist
  9  = middle finger MCP  (scale reference, as in blueprint)
  12 = middle finger tip  (Y-alignment reference)
"""

from __future__ import annotations

import numpy as np

from isl.schema import NUM_LANDMARKS, NUM_COORDS, RESAMPLE_FRAMES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_sequence(
    pose_seq: np.ndarray,
    *,
    align_y: bool = True,
) -> np.ndarray:
    """Normalise a full hand-landmark sequence ``(T, 21, 3) → (T, 21, 3)``.

    Parameters
    ----------
    pose_seq:
        Raw landmark sequence, shape ``(T, 21, 3)``, dtype float32.
    align_y:
        If ``True``, rotate each frame so landmark-12 points toward +Y.

    Returns
    -------
    np.ndarray
        Normalised sequence, same shape ``(T, 21, 3)``, dtype float32.
        Frames that cannot be normalised (e.g. landmark-9 is at the wrist)
        are returned as zeros rather than NaN so downstream code is safe.
    """
    if pose_seq.ndim != 3 or pose_seq.shape[1:] != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(
            f"pose_seq must be shape (T, {NUM_LANDMARKS}, {NUM_COORDS}), "
            f"got {pose_seq.shape}"
        )

    out = np.empty_like(pose_seq, dtype=np.float32)

    for t, frame in enumerate(pose_seq):
        out[t] = _normalize_frame(frame, align_y=align_y)

    return out


def build_vec(norm_seq: np.ndarray) -> np.ndarray:
    """Global feature vector: mean, std, min, max across T frames.

    Parameters
    ----------
    norm_seq:
        Normalised sequence ``(T, 21, 3)`` float32.

    Returns
    -------
    np.ndarray
        Shape ``(252,)`` float32  — 21 × 3 × 4 statistics.
    """
    flat = norm_seq.reshape(len(norm_seq), -1)   # (T, 63)
    vec = np.concatenate(
        [
            flat.mean(axis=0),
            flat.std(axis=0),
            flat.min(axis=0),
            flat.max(axis=0),
        ]
    ).astype(np.float32)
    return vec   # (252,)


def build_flat_vec(norm_seq: np.ndarray, n_frames: int = RESAMPLE_FRAMES) -> np.ndarray:
    """Resample sequence to ``n_frames`` frames, then flatten.

    Parameters
    ----------
    norm_seq:
        Normalised sequence ``(T, 21, 3)`` float32.
    n_frames:
        Target number of frames (default: ``RESAMPLE_FRAMES = 30``).

    Returns
    -------
    np.ndarray
        Shape ``(n_frames * 21 * 3,)`` = ``(1890,)`` float32.
    """
    T = len(norm_seq)
    if T == 0:
        return np.zeros(n_frames * NUM_LANDMARKS * NUM_COORDS, dtype=np.float32)

    # Linear interpolation along the time axis using numpy
    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, n_frames)
    flat_original = norm_seq.reshape(T, -1)   # (T, 63)

    resampled = np.empty((n_frames, flat_original.shape[1]), dtype=np.float32)
    for col in range(flat_original.shape[1]):
        resampled[:, col] = np.interp(new_indices, old_indices, flat_original[:, col])

    return resampled.flatten().astype(np.float32)   # (1890,)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_frame(frame: np.ndarray, *, align_y: bool) -> np.ndarray:
    """Normalise a single ``(21, 3)`` landmark array."""
    frame = frame.astype(np.float32).copy()

    # 1. Wrist-origin translation
    wrist = frame[0].copy()
    frame -= wrist

    # 2. Scale by wrist → middle-MCP distance
    scale = float(np.linalg.norm(frame[9]))
    if scale < 1e-6:
        # Degenerate frame (all landmarks collapsed); return zeros
        return np.zeros_like(frame)
    frame /= scale

    # 3. Optional Y-alignment: rotate XY plane so landmark-12 → +Y
    if align_y:
        tip = frame[12]
        angle = np.arctan2(tip[0], tip[1])  # angle from +Y axis in XY plane
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        # Rotate x and y; leave z unchanged
        x = frame[:, 0].copy()
        y = frame[:, 1].copy()
        frame[:, 0] = cos_a * x - sin_a * y
        frame[:, 1] = sin_a * x + cos_a * y

    return frame
