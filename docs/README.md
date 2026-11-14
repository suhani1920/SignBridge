# SignBridge — P1 Data Preparation

> **Status: P1 pipeline implemented — 2026-11-13.**
> This document describes only behavior that exists in the code.
> Items not yet implemented are clearly marked **[NOT IMPLEMENTED]**.

---

## Table of Contents
1. [What is SignBridge?](#1-what-is-signbridge)
2. [What P1 Does](#2-what-p1-does)
3. [Folder Layout](#3-folder-layout)
4. [How Teammates Add Videos](#4-how-teammates-add-videos)
5. [What P1 Produces](#5-what-p1-produces)
6. [Setup](#6-setup)
7. [Run the Pipeline](#7-run-the-pipeline)
8. [Run the Tests](#8-run-the-tests)
9. [Common Errors and Fixes](#9-common-errors-and-fixes)
10. [Verify the Results](#10-verify-the-results)
11. [Git Rules](#11-git-rules)
12. [Division of Next Steps](#12-division-of-next-steps)

---

## 1. What is SignBridge?

SignBridge is a college project that translates **Indian Sign Language (ISL)**
video clips into text. A person signs in front of a camera; the system returns
the word or phrase they signed.

| Workstream | Responsibility | Status |
|---|---|---|
| **P1** | Turn raw signing videos into clean numerical data | ✅ Implemented |
| P2 | Train a model on P1's data | 🔲 Not started |
| P3 | *(not yet assigned)* | 🔲 Not started |
| Backend | Serve predictions via an API | 🔲 Not started |
| Web | Show results in a browser | 🔲 Not started |

---

## 2. What P1 Does

P1 sits at the very start of the pipeline. It:

1. Reads `.mp4` (or `.mov`, `.avi`, `.webm`) video files from
   `isl/raw_videos/<sign_label>/`.
2. Runs **MediaPipe Hands** on every frame to get 21-point hand-landmark
   coordinates (x, y, z per point).
3. **Normalises** each frame using a wrist-origin, palm-height scale,
   and Y-axis alignment so clips filmed at different distances or
   resolutions are comparable.
4. Saves one compressed `.npz` file per video under `isl/extracted/`.
5. Updates `isl/metadata.csv` with one row per processed video.

P2 and downstream workstreams consume `.npz` files and `metadata.csv`.
They never touch raw videos.

---

## 3. Folder Layout

```
SignBridge/
├── AGENTS.md               ← rules for all contributors; read this first
├── .gitignore              ← keeps videos and generated data out of git
├── docs/
│   ├── README.md           ← this file
│   ├── CONTRACT.md         ← data schema (implemented — read before coding)
│   └── requirements.txt    ← Python dependencies
├── isl/
│   ├── __init__.py         ← makes isl/ a Python package
│   ├── schema.py           ← shared data types (PoseFrame, SignSample)
│   ├── extract.py          ← MediaPipe extraction from one video
│   ├── normalize.py        ← coordinate normalization
│   ├── metadata.py         ← read/write isl/metadata.csv
│   ├── pipeline.py         ← batch orchestrator (run this)
│   ├── raw_videos/
│   │   └── README.md       ← explains how to add videos (placeholder)
│   └── extracted/
│       └── README.md       ← explains output format (placeholder)
├── tests/
│   ├── __init__.py
│   └── test_features.py    ← P1 unit tests (synthetic data only)
├── backend/
│   └── sample.py           ← NOT IMPLEMENTED (owner TBD)
└── web/
    └── package-lock.json   ← NOT IMPLEMENTED (no frontend yet)
```

**Data folders — created locally, never in git:**
```
isl/raw_videos/<gloss>/     ← teammates drop videos here (see §4)
isl/extracted/<gloss>/      ← pipeline writes .npz files here
isl/metadata.csv            ← pipeline updates this CSV
```

---

## 4. How Teammates Add Videos

> **Summary: Pull → Add videos → Run pipeline. You do not need to send
> videos to anyone.**

### Step 1 — Pull the latest code

```bash
git pull origin main
```

### Step 2 — Create a label folder

Inside `isl/raw_videos/`, create a folder named after the sign (the
*gloss*). Use lowercase, no spaces:

```bash
mkdir -p isl/raw_videos/hello
mkdir -p isl/raw_videos/thankyou
```

### Step 3 — Drop videos into the folder

Copy or move your `.mp4` files into the label folder. Any filename works;
the filename without its extension becomes the `sample_id`.

```
isl/raw_videos/
├── hello/
│   ├── hello_001.mp4     ← sample_id = "hello_001"
│   └── hello_002.mp4     ← sample_id = "hello_002"
└── thankyou/
    └── thankyou_001.mp4
```

**Rules:**
- One subfolder per sign. The folder name is the class label.
- Supported formats: `.mp4`, `.mov`, `.avi`, `.webm`.
- Do NOT place videos directly in `isl/raw_videos/` without a label
  subfolder — the pipeline will warn and skip them.
- Never commit videos to git. The `.gitignore` file excludes video
  extensions, but always check with `git status` before committing.

### Step 4 — Run the pipeline

```bash
python -m isl.pipeline
```

---

## 5. What P1 Produces

### 5a. `isl/extracted/<gloss>/<sample_id>.npz`

One compressed NumPy archive per accepted video.

| Array key | Shape | dtype | What it contains |
|---|---|---|---|
| `pose_seq` | `(T, 21, 3)` | float32 | Per-frame normalised hand landmarks. T = frames where a hand was detected. |
| `vec` | `(252,)` | float32 | Global feature vector: mean, std, min, max per coordinate across T frames. 21 × 3 × 4 = 252 values. |
| `flat_vec` | `(1890,)` | float32 | Sequence resampled to 30 frames, then flattened. 30 × 21 × 3 = 1890 values. |
| `gloss` | scalar | str | The sign label (label folder name). |

**Normalization applied to `pose_seq`:**
1. Wrist (landmark 0) translated to `(0, 0, 0)`.
2. All coordinates divided by wrist → middle-MCP (landmark 9) distance.
3. XY plane rotated so landmark 12 (middle-finger tip) points toward +Y.

### 5b. `isl/metadata.csv`

One row per successfully processed video. Updated on every pipeline run.
Existing rows are updated in place — no duplicates.

| Column | Description |
|---|---|
| `sample_id` | Video filename without extension, e.g. `hello_001` |
| `gloss` | Sign label (label folder name) |
| `meaning` | English gloss — blank until the team fills it in |
| `video_path` | Relative path to the source video |
| `npz_path` | Relative path to the generated `.npz` file |
| `num_frames` | Number of detected-hand frames used in `pose_seq` |

---

## 6. Setup

### Requirements
- **Python 3.10, 3.11, or 3.12** (Python 3.13 also works with the installed package versions)
- A terminal open at the repo root

### Step 1 — Check your Python version

```bash
python3 --version
```

You should see `Python 3.10.x`, `3.11.x`, or later. If you see 2.x,
use `python3` explicitly everywhere.

### Step 2 — Create a virtual environment

A virtual environment keeps this project's packages separate from
your system Python. Run this once, from the repo root:

```bash
python3 -m venv .venv
```

### Step 3 — Activate the virtual environment

Run this every time you open a new terminal:

```bash
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

You should see `(.venv)` at the start of your prompt.

### Step 4 — Install dependencies

This installs mediapipe, opencv, numpy, and pytest:

```bash
pip install -r docs/requirements.txt
```

> **First-time note:** MediaPipe downloads a hand-landmark model bundle
> (~10 MB) from Google the **first time you run the pipeline** on a new
> machine. After that, everything works offline. The download requires
> internet access only once.

---

## 7. Run the Pipeline

This processes every video under `isl/raw_videos/` and writes results to
`isl/extracted/` and `isl/metadata.csv`:

```bash
python -m isl.pipeline
```

**What you will see:**

```
[hello] 2 video(s)
  [extract]  hello_001.mp4 ... done (47 frames, 2.3s)
  [extract]  hello_002.mp4 ... done (39 frames, 1.9s)

[thankyou] 1 video(s)
  [extract]  thankyou_001.mp4 ... done (55 frames, 2.7s)

==================================================
Done.  Processed: 3  |  Skipped/failed: 0
Output: isl/extracted/
Index:  isl/metadata.csv
```

**Re-running is safe:** videos whose `.npz` file is already newer than
the source video are skipped automatically.

---

## 8. Run the Tests

This runs all P1 unit tests. No video files or internet access are needed.
Tests use synthetic numpy arrays only:

```bash
pytest tests/test_features.py -v
```

You should see all tests pass with output like:

```
tests/test_features.py::TestSchema::test_poseframe_correct_shape_accepted PASSED
tests/test_features.py::TestSchema::test_constants_match_expected_values  PASSED
...
======================== 25 passed in X.Xs ========================
```

---

## 9. Common Errors and Fixes

| Error message | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'mediapipe'` | Dependencies not installed | Run `pip install -r docs/requirements.txt` with `.venv` active |
| `OpenCV could not open '...'` | Bad codec, truncated file, or permission | Re-export the video as plain H.264 `.mp4`; check file permissions |
| `Only N frame(s) had a detected hand (minimum required: 5)` | Hand not visible, bad lighting, or very short clip | Re-record with better lighting; ensure the signing hand is in frame |
| `No label subfolders found under raw_videos/` | `raw_videos/` is empty | Create a label subfolder and add videos (see §4) |
| `[WARNING] The following videos are directly inside raw_videos/` | Video not in a label subfolder | Move the video into `isl/raw_videos/<gloss>/` |
| `FileNotFoundError: Video not found` | Wrong path | Check that the file exists and the path is correct |
| `Unsupported video extension '...'` | File format not supported | Convert to `.mp4`, `.mov`, `.avi`, or `.webm` |

---

## 10. Verify the Results

After running the pipeline, check that outputs are correct:

**Check that metadata.csv is valid:**
```bash
python -c "
import csv
rows = list(csv.DictReader(open('isl/metadata.csv')))
print(f'{len(rows)} row(s) in metadata.csv')
for r in rows:
    print(r['sample_id'], r['gloss'], r['num_frames'], 'frames')
"
```

**Check a .npz file's array shapes:**
```bash
python -c "
import numpy as np, sys
p = sys.argv[1]
d = np.load(p, allow_pickle=True)
for k, v in d.items():
    shape = getattr(v, 'shape', '(scalar)')
    dtype = getattr(v, 'dtype', '')
    print(k, shape, dtype)
" isl/extracted/hello/hello_001.npz
```

Expected output:
```
pose_seq (T, 21, 3) float32
vec (252,) float32
flat_vec (1890,) float32
gloss () object
```

**Run the test suite:**
```bash
pytest tests/test_features.py -v
```

All tests must pass with 0 failures and 0 errors.

---

## 11. Git Rules

- **Never commit** video files (`.mp4`, `.mov`, `.avi`, `.webm`).
- **Never commit** generated `.npz` files or `isl/metadata.csv`.
- **Never commit** the `.venv/` folder.
- Always work on a **feature branch**, not directly on `main`.
- Stage files explicitly by name — do not use `git add .`.

```bash
# Create a feature branch for your work
git checkout -b feature/p1-extraction

# Stage only the files you intentionally changed
git add isl/pipeline.py isl/extract.py isl/normalize.py

# Write a clear commit message
git commit -m "P1: add extraction and normalization pipeline"

# Push and open a pull request
git push origin feature/p1-extraction
```

---

## 12. Division of Next Steps

| Workstream | What they need from P1 | Status |
|---|---|---|
| **P1** | Agree sign vocabulary; fill `meaning` column in metadata.csv; decide which hand (dominant only vs both) | In progress |
| **P2** | Load `isl/extracted/` `.npz` files; use `flat_vec` (shape 1890) or `pose_seq` (shape T×21×3) as model input | Waiting on P1 to finish |
| **P3** | *(not yet assigned)* | — |
| **Backend** | Load a trained model; accept a video, run P1 extraction, return predicted gloss | Waiting on P2 |
| **Web** | Show video upload UI and display prediction result | Waiting on Backend |
| **ChromaDB integration** | Embedding storage and similarity search | **[NOT IMPLEMENTED]** — planned future step; `isl/db/chroma/` will be created when designed |

---

*Last updated by: P1 (Piyush Singh) — 2026-11-13*
*Reflects code committed on branch `feature/video-pipeline`.*
