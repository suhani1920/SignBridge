"""Shared data contract constants and structured dataclasses for SignBridge (P1 + P2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
import numpy as np

# ===========================================================================
# 1. P2 Recognition Engine Contract Constants & Types
# ===========================================================================

SEQ_LEN: int = 32
NUM_LANDMARKS: int = 21
COORDS: int = 3
NUM_HANDS: int = 2  # [0: left / primary single-hand MVP slot, 1: right]

# Summary vector layout: mean, std, min, max, velocity across flattened hand coords
# Per frame flattened dim = 2 * 21 * 3 = 126 -> 5 stats * 126 = 630
FRAME_FEATURE_DIM: int = NUM_HANDS * NUM_LANDMARKS * COORDS
SUMMARY_STATS_COUNT: int = 5
SUMMARY_DIM: int = FRAME_FEATURE_DIM * SUMMARY_STATS_COUNT

# Required keys in per-video .npz files
NPZ_KEYS: tuple[str, ...] = (
    "sequence",   # (T, 2, 21, 3) float32
    "body_pos",   # (T, 2, 3) float32
    "mask",       # (T, 2) bool
    "summary",    # (D,) float32
    "label",      # str
)

METADATA_COLUMNS: tuple[str, ...] = (
    "video_path",
    "label",
    "signer_id",
    "fps",
    "total_frames",
    "valid_frames",
    "detection_rate",
    "status",
)


@dataclass(frozen=True)
class Candidate:
    """Represents a single candidate match from the pose database."""

    label: str
    score: float
    sample_index: int


@dataclass(frozen=True)
class MatchResult:
    """Result returned by match_pose evaluation."""

    label: str | None
    score: float
    status: Literal["confident", "uncertain", "reject"]
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class RecognizerConfig:
    """Configurable thresholds and runtime parameters for matching and detection."""

    accept_threshold: float = 0.85
    uncertain_threshold: float = 0.70
    min_margin: float = 0.03
    top_k: int = 5
    dtw_band: float = 0.10
    dtw_threshold: float = 0.35
    steady_velocity_threshold: float = 0.015
    steady_frames: int = 15
    ema_alpha: float = 0.4
    ring_buffer_size: int = 64
    cooldown_seconds: float = 1.5
    align_y: bool | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> RecognizerConfig:
        """Load configuration overrides from a JSON or YAML file."""
        file_path = Path(path)
        raw_text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data: dict[str, Any] = yaml.safe_load(raw_text) or {}
        elif file_path.suffix.lower() == ".json":
            data = json.loads(raw_text)
        else:
            raise ValueError(f"Unsupported config format: {file_path.suffix}")
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ===========================================================================
# 2. P1 Video-to-Pose Pipeline Contract Constants & Types
# ===========================================================================

#: Each landmark has x, y, z coordinates.
NUM_COORDS: int = 3

#: Pipeline resamples every clip to this many frames before flattening.
RESAMPLE_FRAMES: int = 30

#: Minimum number of frames with a detected hand for a clip to be accepted.
MIN_HAND_FRAMES: int = 5

#: Supported raw-video file extensions (lowercase, with leading dot).
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi", ".webm")

#: Global feature vector length: mean+std+min+max per channel.
VEC_LEN: int = NUM_LANDMARKS * NUM_COORDS * 4           # 252

#: Flattened sequence vector length: RESAMPLE_FRAMES × landmarks × coords.
FLAT_VEC_LEN: int = RESAMPLE_FRAMES * NUM_LANDMARKS * NUM_COORDS  # 1890


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
