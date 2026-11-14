# ISL Recognition Data Contract

This document defines the fixed data contract shared across dataset extraction (`extract.py`), feature processing (`isl/features.py`), and live recognition (`isl/matcher.py`). **Do not alter these shapes or keys.**

## 1. Core Constants (`isl/schema.py`)
- `SEQ_LEN = 32` (Fixed temporal frames per sign clip)
- `NUM_LANDMARKS = 21` (MediaPipe hand keypoints per hand)
- `COORDS = 3` (`x, y, z` coordinates)
- `NUM_HANDS = 2` (Index `0` = Left hand, Index `1` = Right hand)
- Missing hands are represented as all `0.0` with `mask = False`.

## 2. Per-Video `.npz` File Schema
Each extracted sign clip `.npz` file must contain:
- `sequence`: `float32` array of shape `(T, 2, 21, 3)` — hand-local normalized landmarks (wrist at origin, scaled by wrist-to-middle-fingertip distance).
- `body_pos`: `float32` array of shape `(T, 2, 3)` — raw wrist position relative to shoulder midpoint, scaled by shoulder width.
- `mask`: `bool` array of shape `(T, 2)` — `True` where hand is detected, `False` where missing.
- `summary`: `float32` array of shape `(630,)` — concatenated `[mean, std, min, max, velocity]` over flattened `(T, 126)` masked sequence.
- `label`: `str` — target sign class name.

## 3. Metadata & Labels
- `metadata.csv` columns: `video_path, label, signer_id, fps, total_frames, valid_frames, detection_rate, status`
- `labels.json`: JSON dictionary mapping `label (str) -> class_id (int)`