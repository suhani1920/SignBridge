# SignBridge — P1 Data Contract

> **Status: IMPLEMENTED — P1 (Piyush Singh), 2026-11-13.**
> This file is a shared contract. Once merged into main, do not change
> field names, array shapes, or units without opening a PR and getting
> team approval first.

---

## Overview

This contract defines the data types and file formats that the P1 pipeline
produces and that all downstream workstreams (P2, P3, Backend) must consume.

---

## Normalization Convention

All hand-landmark coordinates are normalized using the **wrist-origin,
landmark-9 scale, landmark-12 Y-alignment** method from the ISL handoff blueprint:

| Step | Operation |
|---|---|
| 1. Translate | Subtract wrist (landmark 0) so wrist sits at (0, 0, 0) |
| 2. Scale | Divide all coordinates by ‖landmark 9 − wrist‖ (wrist → middle-MCP distance, ≈ palm height) |
| 3. Align Y | Rotate XY plane so landmark 12 (middle-finger tip) points toward +Y (enabled by default) |

After normalization:
- `landmarks[0]` is always `(0, 0, 0)` (wrist at origin)
- `‖landmarks[9]‖ ≈ 1.0` (scale reference distance = 1)
- Landmark 12 points roughly toward `+Y` (Y-aligned)

---

## MediaPipe Landmark Reference

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

---

## Output File: `isl/extracted/<gloss>/<sample_id>.npz`

One compressed NumPy archive per accepted video clip.

| Array key | dtype | Shape | Description |
|---|---|---|---|
| `pose_seq` | float32 | `(T, 21, 3)` | Normalised per-frame landmark sequence. T = number of frames where a hand was detected. |
| `vec` | float32 | `(252,)` | Global feature vector: `[mean, std, min, max]` per channel across T frames. 21 × 3 × 4 = 252. |
| `flat_vec` | float32 | `(1890,)` | Sequence resampled to 30 frames, then flattened. 30 × 21 × 3 = 1890. |
| `gloss` | str (object array) | scalar | The sign label (name of the label folder). |

---

## Output File: `isl/metadata.csv`

One row per accepted video clip. Updated by each pipeline run.

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Video filename without extension, e.g. `hello_001` |
| `gloss` | string | Sign label (label folder name), e.g. `hello` |
| `meaning` | string | Human-readable English gloss (filled in by team; blank until then) |
| `video_path` | string | Relative path to source video |
| `npz_path` | string | Relative path to generated `.npz` file |
| `num_frames` | integer | Number of detected-hand frames used in `pose_seq` |

---

## Constants (do not change without a PR)

| Constant | Value | Meaning |
|---|---|---|
| `NUM_LANDMARKS` | 21 | MediaPipe Hands landmarks per frame |
| `NUM_COORDS` | 3 | x, y, z per landmark |
| `RESAMPLE_FRAMES` | 30 | Frames used for `flat_vec` |
| `MIN_HAND_FRAMES` | 5 | Minimum detected frames to accept a video |
| `VEC_LEN` | 252 | Length of `vec` |
| `FLAT_VEC_LEN` | 1890 | Length of `flat_vec` |

---

## ChromaDB Integration

**[NOT IMPLEMENTED — future step]**

ChromaDB integration (embedding storage and similarity search) is planned
as a later workstream. P1 does not currently produce ChromaDB-compatible
embeddings. The `isl/db/chroma/` folder will be created when this step
is designed by the team.

---

## Open Questions for the Team

| Question | Why it matters |
|---|---|
| Which hand to track (dominant / both)? | Pipeline currently tracks the first detected hand only |
| Sign vocabulary (word list) | Determines label folder names |
| Train/val/test split | P1 or P2 assigns splits? |
| `meaning` column population | Who fills in the English gloss in metadata.csv? |

---

*Last updated by: P1 (Piyush Singh) — 2026-11-13*
