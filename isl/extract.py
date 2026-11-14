"""
isl/extract.py — MediaPipe Hands extraction from a single video file.

What this file does
-------------------
Given a path to one video file, this module:
  1. Opens the video with OpenCV.
  2. Runs MediaPipe Hands on every frame.
  3. Returns a SignSample containing one PoseFrame per video frame.
     Frames with no detected hand are stored with detected=False and
     zero landmark arrays (they are excluded from pose_seq later).

If fewer than MIN_HAND_FRAMES have a detected hand the function raises
SkipVideo so the caller can log the reason and continue to the next clip.

First-time note
---------------
MediaPipe downloads a small model bundle (~10 MB) from Google the first
time it is imported on a machine that has not used it before. After that
first run everything works offline. The download needs internet access
only once.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from isl.schema import (
    NUM_LANDMARKS,
    NUM_COORDS,
    MIN_HAND_FRAMES,
    VIDEO_EXTENSIONS,
    PoseFrame,
    SignSample,
)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class SkipVideo(Exception):
    """Raised when a video is intentionally skipped (not an unexpected error)."""


# ---------------------------------------------------------------------------
# MediaPipe setup (module-level so it is initialised once per process)
# ---------------------------------------------------------------------------

_mp_hands = mp.solutions.hands


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sample(
    video_path: pathlib.Path,
    gloss: str,
    *,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> SignSample:
    """Extract hand landmarks from *video_path* and return a SignSample.

    Parameters
    ----------
    video_path:
        Path to a supported video file (``.mp4``, ``.mov``, ``.avi``,
        ``.webm``).
    gloss:
        Sign label (the name of the label folder).
    max_num_hands:
        How many hands MediaPipe tracks simultaneously. P1 uses 1 (signing
        hand only). Change only if the team agrees.
    min_detection_confidence, min_tracking_confidence:
        MediaPipe confidence thresholds.

    Returns
    -------
    SignSample
        Populated sample.  ``sample.detected_frames`` is the number of
        frames where MediaPipe found a hand.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    ValueError
        If the file extension is not supported, or if OpenCV cannot open
        the video (bad codec, truncated file, permission error).
    SkipVideo
        If fewer than ``MIN_HAND_FRAMES`` frames had a detected hand.
    """
    video_path = pathlib.Path(video_path)

    # --- guard: file exists ---
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- guard: supported extension ---
    ext = video_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{ext}' for {video_path.name}. "
            f"Supported: {', '.join(VIDEO_EXTENSIONS)}"
        )

    # --- open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(
            f"OpenCV could not open '{video_path}'. "
            "Check the codec, file integrity, and read permissions."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_id = video_path.stem

    sample = SignSample(
        sample_id=sample_id,
        gloss=gloss,
        video_path=str(video_path),
        source_fps=float(fps),
    )

    _zero = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    # --- run MediaPipe frame by frame ---
    with _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm_list = result.multi_hand_landmarks[0].landmark
                arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lm_list],
                    dtype=np.float32,
                )
                sample.frames.append(PoseFrame(frame_idx, arr, detected=True))
            else:
                sample.frames.append(PoseFrame(frame_idx, _zero.copy(), detected=False))

            frame_idx += 1

    cap.release()

    # --- guard: enough detected frames ---
    if sample.detected_frames < MIN_HAND_FRAMES:
        raise SkipVideo(
            f"Only {sample.detected_frames} frame(s) had a detected hand "
            f"(minimum required: {MIN_HAND_FRAMES}). "
            "Check lighting, hand visibility, or try a different recording."
        )

    return sample


def check_video_placement(raw_root: pathlib.Path) -> list[pathlib.Path]:
    """Find video files placed directly in *raw_root* (missing label folder).

    Parameters
    ----------
    raw_root:
        The ``isl/raw_videos/`` directory.

    Returns
    -------
    list[pathlib.Path]
        Paths of misplaced video files.  An empty list means all videos
        are correctly nested under a label subfolder.
    """
    misplaced = [
        p for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return misplaced
