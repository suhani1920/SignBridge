# SignBridge — Shared Data Contract (P1 & P2)

> **Status: IMPLEMENTED — One-Hand MVP Contract (P1 + P2).**
> This file is a shared contract across P1 extraction (`isl/pipeline.py`), the P1-to-P2 adapter (`isl/export_p2.py`), feature processing (`isl/features.py`), and live recognition (`isl/matcher.py`). Once merged into main, do not change field names, array shapes, or units without opening a PR and getting team approval first.

---

## Overview

The team has agreed that the **MVP supports one hand**:
1. **P1 (`isl/pipeline.py`)** extracts a single hand (`max_num_hands=1`), normalizes landmarks, and writes per-clip `.npz` archives under `isl/extracted/<gloss>/<sample_id>.npz` alongside `isl/metadata.csv`.
2. **P1-to-P2 Bridge (`isl/export_p2.py`)** converts P1 `.npz` archives into P2 `PoseDatabase` archives under `data/pose_db/` by placing the single-hand sequence into hand slot `0` of P2's `(32, 2, 21, 3)` format.
3. **P2 (`isl/matcher.py`, `isl/adapter.py`)** consumes the exported `PoseDatabase` archives and supports both single-hand `(T, 21, 3)` and two-hand `(T, 2, 21, 3)` live inference inputs.

---

## Part 1 — P1 Video-to-Pose Data Contract (One-Hand MVP)

### Normalization Convention (`isl/normalize.py`)

All hand-landmark coordinates are normalized using the **wrist-origin, landmark-9 scale, landmark-12 Y-alignment** method:

| Step | Operation |
|---|---|
| 1. Translate | Subtract wrist (landmark 0) so wrist sits at `(0, 0, 0)` |
| 2. Scale | Divide all coordinates by `‖landmark 9 − wrist‖` (wrist → middle-MCP distance, ≈ palm height) |
| 3. Align Y | Rotate XY plane so landmark 12 (middle-finger tip) points toward `+Y` (enabled by default) |

After normalization (for non-degenerate frames):
- `landmarks[0]` is `(0, 0, 0)` (wrist at origin)
- `‖landmarks[9]‖ ≈ 1.0` (scale reference distance = 1)
- Landmark 12 points toward `+Y` (Y-aligned)

### MediaPipe Landmark Reference

MediaPipe Hands outputs **21 landmarks per hand**, indexed 0–20:

| Index | Name |
|---|---|
| 0 | Wrist (translation origin) |
| 9 | Middle finger MCP (scale reference) |
| 12 | Middle finger tip (Y-alignment reference) |
| 1–4 | Thumb |
| 5–8 | Index finger |
| 10–12 | Middle finger (cont.) |
| 13–16 | Ring finger |
| 17–20 | Pinky |

### P1 Output File: `isl/extracted/<gloss>/<sample_id>.npz`

One compressed NumPy archive per accepted video clip.

| Array key | dtype | Shape | Description |
|---|---|---|---|
| `pose_seq` | `float32` | `(T, 21, 3)` | Normalised per-frame landmark sequence (one hand). `T` = number of frames where a hand was detected. |
| `vec` | `float32` | `(252,)` | Global feature vector: `[mean, std, min, max]` per channel across `T` frames (`21 × 3 × 4 = 252`). |
| `flat_vec` | `float32` | `(1890,)` | Sequence resampled to 30 frames, then flattened (`30 × 21 × 3 = 1890`). |
| `gloss` | `str` (0-d Unicode array) | scalar `()` | The sign label (name of the label folder). |

### P1 Output File: `isl/metadata.csv`

One row per accepted video clip, uniquely keyed by `(gloss, sample_id)`. Updated by each pipeline run.

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Video filename without extension, e.g. `hello_001` |
| `gloss` | string | Sign label (label folder name), e.g. `hello` |
| `meaning` | string | Human-readable English gloss (blank until human annotation; nonblank existing values are preserved on pipeline reruns) |
| `video_path` | string | Relative path to source video |
| `npz_path` | string | Relative path to generated `.npz` file |
| `num_frames` | integer | Number of detected-hand frames used in `pose_seq` |

### P1 Constants (`isl/schema.py`)

| Constant | Value | Meaning |
|---|---|---|
| `NUM_LANDMARKS` | 21 | MediaPipe Hands landmarks per frame |
| `NUM_COORDS` | 3 | `x, y, z` per landmark |
| `RESAMPLE_FRAMES` | 30 | Frames used for `flat_vec` |
| `MIN_HAND_FRAMES` | 5 | Minimum detected frames to accept a video |
| `VEC_LEN` | 252 | Length of `vec` |
| `FLAT_VEC_LEN` | 1890 | Length of `flat_vec` |

---

## Part 2 — P2 Recognition Engine Contract & P1-to-P2 Bridge

### 1. P2 Core Constants (`isl/schema.py`)
- `SEQ_LEN = 32` (Fixed temporal frames per sign clip in P2)
- `NUM_LANDMARKS = 21` (MediaPipe hand keypoints per hand)
- `COORDS = 3` (`x, y, z` coordinates)
- `NUM_HANDS = 2` (Slot `0` = primary/single-hand MVP slot or Left hand; Slot `1` = Right hand, zeroed in one-hand MVP)
- Missing hands are represented as all `0.0` with `mask = False`.

### 2. P1-to-P2 Export Bridge (`isl/export_p2.py`)
`python -m isl.export_p2 --src isl/extracted --dst data/pose_db` converts each P1 `.npz` archive into a P2 `PoseDatabase` archive (`data/pose_db/<label>__<sample_id>.npz`):
- Reads P1 `pose_seq` `(T, 21, 3)` and `gloss`.
- Places the single-hand sequence into slot `0` of a `(T, 2, 21, 3)` array (`two_hand_seq[:, 0, :, :] = pose_seq`) to match `RecognitionService.recognize()`.
- Sets `mask[:, 0] = True` for non-zero frames and `mask[:, 1] = False`.
- Resamples `sequence` and `mask` to `SEQ_LEN = 32`, computes `summary` `(630,)`, zeros `body_pos` `(32, 2, 3)`, and sets `align_y = True` so P2 inference automatically applies matching XY Y-axis rotation.

### 3. P2 Per-Video `.npz` File Schema (`data/pose_db/*.npz`)
Each P2 database `.npz` file contains:
- `sequence`: `float32` array of shape `(T, 2, 21, 3)` — hand-local normalized landmarks.
- `body_pos`: `float32` array of shape `(T, 2, 3)` — wrist position relative to shoulder midpoint (zeros when exported from P1).
- `mask`: `bool` array of shape `(T, 2)` — `True` where hand is detected, `False` where missing.
- `summary`: `float32` array of shape `(630,)` — concatenated `[mean, std, min, max, velocity]` over flattened `(T, 126)` masked sequence.
- `label`: `str` — target sign class name.
- `align_y`: `bool` scalar (optional, `True` for P1-exported archives).

### 4. P2 Metadata & Labels
- `METADATA_COLUMNS`: `video_path, label, signer_id, fps, total_frames, valid_frames, detection_rate, status`
- `labels.json`: JSON dictionary mapping `label (str) -> class_id (int)`

---

## ChromaDB Integration

**[NOT IMPLEMENTED — future step]**

ChromaDB integration (embedding storage and similarity search) is planned as a later workstream.

---

## Team Decisions & Open Questions

| Topic | Status / Notes |
|---|---|
| Hand tracking scope | **Agreed:** MVP supports one hand (`max_num_hands=1` in P1, bridged to slot `0` in P2 via `isl/export_p2.py`) |
| Sign vocabulary (word list) | Determines label folder names |
| Train/val/test split | P1 or P2 assigns splits? |
| `meaning` column population | Blank until human annotation; preserved across pipeline reruns |
