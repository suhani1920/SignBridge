"""
isl/schema.py — P1 shared data contract for SignBridge.

This file is a shared contract. Once the team has agreed on its contents
and merged them into main, do not change field names, array shapes, or
units without opening a PR and getting team approval first.

Defines the data types used by every P1 pipeline module:
  PoseFrame   — landmarks for a single video frame (one hand only)
  SignSample  — the full sequence of frames for one video clip

Normalization convention (wrist-origin, landmark-12 scale):
  1. Translate: subtract wrist (landmark 0) so wrist is at origin.
  2. Scale:     divide by the distance from wrist (0) to middle-finger-MCP
                (landmark 9), which is roughly palm-height.
  3. Align Y:   rotate in the XY plane until landmark 12
                (middle-finger-tip) points toward +Y (optional; enabled by
                normalize.py when align_y=True).

Output arrays are float32 throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Constants agreed by the team (edit via PR only)
# ---------------------------------------------------------------------------

#: MediaPipe Hands produces 21 landmarks per hand.
NUM_LANDMARKS: int = 21

#: Each landmark has x, y, z coordinates.
NUM_COORDS: int = 3

#: Pipeline resamples every clip to this many frames before flattening.
RESAMPLE_FRAMES: int = 30

#: Minimum number of frames with a detected hand for a clip to be accepted.
MIN_HAND_FRAMES: int = 5

#: Supported raw-video file extensions (lowercase, with leading dot).
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".webm")

#: Global feature vector length: mean+std+min+max per channel.
VEC_LEN: int = NUM_LANDMARKS * NUM_COORDS * 4          # 252

#: Flattened sequence vector length: RESAMPLE_FRAMES × landmarks × coords.
FLAT_VEC_LEN: int = RESAMPLE_FRAMES * NUM_LANDMARKS * NUM_COORDS  # 1890

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PoseFrame:
    """Landmark data for a single video frame (one hand).

    Attributes
    ----------
    frame_idx:
        Zero-based index of the frame within the source video.
    landmarks:
        Raw (un-normalised) landmark array from MediaPipe Hands,
        shape ``(21, 3)`` in image-normalised coordinates
        (x and y in [0, 1], z relative to wrist depth).
    detected:
        ``True`` if MediaPipe reported a hand in this frame; ``False``
        means the entry contains zeros (interpolated or placeholder).
    """

    frame_idx: int
    landmarks: np.ndarray          # shape (NUM_LANDMARKS, NUM_COORDS) float32
    detected: bool = True

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32)
        if self.landmarks.shape != (NUM_LANDMARKS, NUM_COORDS):
            raise ValueError(
                f"landmarks must be shape ({NUM_LANDMARKS}, {NUM_COORDS}), "
                f"got {self.landmarks.shape}"
            )


@dataclass
class SignSample:
    """All data associated with one sign-language video clip.

    Attributes
    ----------
    sample_id:
        Video filename without its extension, e.g. ``"hello_001"``.
    gloss:
        Sign label (the name of the label folder), e.g. ``"hello"``.
    video_path:
        Absolute or repo-relative path to the source ``.mp4`` file.
    source_fps:
        Frame-rate of the source video (frames per second).
    frames:
        List of ``PoseFrame`` objects, one per video frame.
    """

    sample_id: str
    gloss: str
    video_path: str
    source_fps: float
    frames: list[PoseFrame] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of frames in the source video."""
        return len(self.frames)

    @property
    def detected_frames(self) -> int:
        """Number of frames where MediaPipe detected a hand."""
        return sum(1 for f in self.frames if f.detected)

    @property
    def pose_seq(self) -> np.ndarray:
        """Return the full landmark sequence as ``(T, 21, 3)`` float32.

        Only frames where a hand was detected are included (detected=True).
        """
        arrays = [f.landmarks for f in self.frames if f.detected]
        if not arrays:
            return np.zeros((0, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        return np.stack(arrays, axis=0)
