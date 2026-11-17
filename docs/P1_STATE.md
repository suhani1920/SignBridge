# P1 State & Architecture Status

> **Document purpose:** Concise record of verified codebase facts, active architectural conflicts, recommended canonical paths/formats, and required team decisions for the P1 video-data pipeline.
> **Status:** Working draft / state report.
> **Last updated:** 2026-11-16 (Section 9.16 added: P1 pipeline reliability fixes and test verification).

---

## 1. Confirmed Facts (Verified in Repository)

- **Git Branch & Status:** Current branch is `feature/video-pipeline`. Commit `e668516` ("Add shared project folder structure") created empty files in `data/raw_videos/`, `data/extracted/`, and `data/db/chroma/`. 15 files are currently staged in the index (including `AGENTS.md`, `docs/CONTRACT.md`, `docs/README.md`, `docs/requirements.txt`, `isl/*.py`, and `tests/test_features.py`).
- **Dependencies (`docs/requirements.txt`):** Specifies `mediapipe>=0.10.9`, `opencv-python>=4.9.0`, `numpy>=1.26.0`, and `pytest>=8.0.0`.
- **Existing Code Implementation:**
  - `isl/schema.py`: Implements `PoseFrame` and `SignSample` dataclasses. Defines constants `NUM_LANDMARKS=21`, `NUM_COORDS=3`, `RESAMPLE_FRAMES=30`, `MIN_HAND_FRAMES=5`, `VEC_LEN=252`, `FLAT_VEC_LEN=1890`.
  - `isl/extract.py`: MediaPipe Hands extractor with `max_num_hands=1` (tracks single hand).
  - `isl/normalize.py`: Implements wrist-origin translation, landmark-9 (middle MCP) distance scaling, and optional landmark-12 Y-alignment. Generates `vec` (252,) and resampled `flat_vec` (1890,).
  - `isl/metadata.py`: Reads/upserts `metadata.csv` atomically with columns: `sample_id`, `gloss`, `meaning`, `video_path`, `npz_path`, `num_frames`.
  - `isl/pipeline.py`: Batch processor scanning input videos, running extraction and normalization, outputting `.npz` files and updating metadata.
  - `tests/test_features.py`: Unit tests covering schema, normalization math, metadata upsert, and output shapes using synthetic data.
- **Verification Status:** Full test suite verified passing (46 passed, Section 9.16). Two-class pipeline smoke test completed across 38 samples (36 hello, 2 help) with 100% reported for both labels (Section 9.15).

---

## 2. Path & Architectural Conflicts Identified

1. **Input / Output Directory Divergence:**
   - **`AGENTS.md` & `e668516`:** Define top-level `data/` layout: `data/raw/` (or `data/raw_videos/`) and `data/processed/` (or `data/extracted/`).
   - **`isl/pipeline.py`, `docs/CONTRACT.md`, `docs/README.md`:** Hardcoded to `isl/raw_videos/<gloss>/` and `isl/extracted/<gloss>/`, with index at `isl/metadata.csv`.
   - **`.gitignore`:** Ignores `isl/extracted/**/*.npz`, `isl/metadata.csv`, and video extensions, but does not explicitly ignore `data/raw*` or `data/extracted*`.
2. **Tracked Placeholder Files vs Root `.gitignore`:**
   - Commit `e668516` committed empty `data/**/README.md` files while `AGENTS.md` explicitly noted that `data/` should not be committed to Git.
3. **Single Hand vs Two Hands:**
   - `extract.py` and `schema.py` assume a single hand (`max_num_hands=1`), whereas many Indian Sign Language (ISL) signs require two hands (bimanual).
4. **ChromaDB / Vector DB Scope:**
   - Mentioned in `CONTRACT.md` and `README.md` as planned future work, but `e668516` added `data/db/chroma/README.md`.
5. **Landmark Normalization Reference Inversion (Landmark 9 vs 12):**
   - Current P1 (`isl/normalize.py`) scales by landmark 9 (middle MCP) and Y-aligns by landmark 12 (middle tip). P2 (`isl/features.py`) scales by landmark 12 and Y-aligns by landmark 12. This creates a mathematical scale mismatch between P1 database poses and P2 live inference. See Section 9.14.

---

## 3. P1 Scope

- **In Scope:**
  - Ingestion and validation of raw sign video files across supported formats (`.mp4`, `.mov`, `.avi`, `.webm`).
  - Per-frame MediaPipe landmark extraction (coordinates, detection status).
  - Landmark coordinate normalization (translation, scaling, alignment).
  - Feature derivation (`pose_seq`, summary statistics `vec`, fixed-length temporal interpolation `flat_vec`).
  - Output serialization (`.npz`) and dataset catalog indexing (`metadata.csv`).
  - Unit testing with synthetic datasets.
- **Out of Scope (Downstream / Other Teams):**
  - Model architecture, training, and evaluation (P2).
  - Embeddings generation & Vector DB / ChromaDB storage (Future / P2 / P3).
  - Inference backend / HTTP API (`backend/`).
  - Frontend web application (`web/`).

---

## 4. Path & Format Alignment

### A. Approved / Established in Current Code
- **Data File Format:** Compressed NumPy archive (`.npz`) containing:
  - `pose_seq`: `float32`, shape `(T, 21, 3)`
  - `vec`: `float32`, shape `(252,)` (mean, std, min, max across time)
  - `flat_vec`: `float32`, shape `(1890,)` (linear temporal interpolation to 30 frames, flattened)
  - `gloss`: scalar string
- **Index Format:** `metadata.csv` (columns: `sample_id,gloss,meaning,video_path,npz_path,num_frames`).

### B. Decision: Canonical P1 Paths (user's current choice, pending team confirmation)
- **Current choice (2026-11-14):** Use the paths the existing P1 code and docs already use:
  - Raw input videos: `isl/raw_videos/<gloss>/<sample_id>.<ext>`
  - Extracted features: `isl/extracted/<gloss>/<sample_id>.npz`
  - Dataset index: `isl/metadata.csv`
- **Status:** Not team-confirmed. No files were moved or deleted; no application code was changed.
- **Layout mismatch to reconcile (documentation):** Top-level `data/raw_videos/`, `data/extracted/` and `data/db/chroma/` exist in git (placeholder `README.md` files from commit `e668516`) but are **not** used by `isl/pipeline.py`. `AGENTS.md` also describes `data/raw/` and `data/processed/`. These docs and folders should be reconciled once the team confirms the layout. Repository check: `git ls-files` shows `README.md` placeholders in both `data/` and `isl/` copies of these folders.

### C. Earlier Recommendations [SUPERSEDED by the choice above unless the team decides otherwise]
- **Recommendation 1 (Alternative: top-level `data/` structure):**
  - **Proposed Canonical Paths:**
    - Raw input videos: `data/raw_videos/<gloss>/<sample_id>.<ext>`
    - Extracted features: `data/extracted/<gloss>/<sample_id>.npz`
    - Dataset index: `data/metadata.csv`
  - **Rationale:** Keeps large data and generated artifacts strictly separated from Python source packages (`isl/`), aligning with standard ML project structure and `AGENTS.md`.
  - **Migration Required if Approved:** Update default paths in `isl/pipeline.py`, `isl/metadata.py`, `docs/CONTRACT.md`, `docs/README.md`, and update root `.gitignore` to ignore `data/extracted/` and `data/metadata.csv`.
- **Recommendation 2 (Bimanual Support Investigation):**
  - **Proposal:** Keep `max_num_hands=1` for initial baseline validation, but schedule schema extension to `(T, 2, 21, 3)` or `(T, 42, 3)` for full ISL compatibility before freezing contract with P2.

---

## 5. Decisions Still Needed from the Team / Lead

1. **Directory Canonicalization:** User's current choice is to keep the `isl/{raw_videos,extracted,metadata.csv}` layout. Team must confirm, and the `data/` folders and `AGENTS.md` layout must be reconciled.
2. **Sign Vocabulary:** Define initial target vocabulary list for gloss folder naming.
3. **One Hand vs Two Hands:** Confirm whether P1 MVP only supports one-handed signs or if two-hand extraction is required before P2 model training begins.
4. **Train / Val / Test Partitioning:** Decide whether dataset splitting (`split` column) is managed by P1 in `metadata.csv` or handled dynamically by P2.
5. **English Meaning Annotation:** Confirm workflow for populating the `meaning` column in `metadata.csv`.
6. **Scale & Y-Alignment Landmark Consensus:** Team must confirm whether to align P1 with blueprint/P2 on landmark-12 scaling (wrist-to-fingertip) or update P2 to landmark-9 scaling (palm height).

---

## 6. Next Immediate Steps

1. Get team confirmation of the `isl/` path choice, then reconcile `data/` placeholders and `AGENTS.md` layout in documentation.
2. Set up local virtual environment (`.venv312` with Python 3.12) and install dependencies from `docs/requirements.txt`.
3. Run test suite (`pytest tests/test_features.py -v`) to officially verify implementation status.

---

## 7. Contract Review Findings (2026-11-14)

> Scope: read-only review of `isl/schema.py`, `extract.py`, `normalize.py`, `metadata.py`, `pipeline.py`, `docs/CONTRACT.md`, `docs/README.md`. No code or paths were changed and no tests were run. Paths remain `isl/raw_videos/`, `isl/extracted/`, `isl/metadata.csv` (user's current choice, pending team confirmation).

### 7.1 What each saved field actually contains (per code)

`isl/extracted/<gloss>/<sample_id>.npz`:

| Key | Actual content (from code) | Not captured / unclear |
|---|---|---|
| `pose_seq` `(T,21,3)` f32 | Wrist-origin, palm-scaled, Y-aligned landmarks, **only for frames where a hand was detected**. Degenerate frames (landmark 9 ≈ wrist) are stored as **all zeros**. | Frame indices, gaps, source FPS and the raw (pre-normalisation) landmarks are not saved. Real timing is lost. |
| `vec` `(252,)` f32 | `[mean(63), std(63), min(63), max(63)]` over T frames. Each 63-block is landmark-major (`l0x,l0y,l0z,l1x,…`). | Layout order is not documented. Landmark 0 is always 0 so its 3 values × 4 stats are constant. |
| `flat_vec` `(1890,)` f32 | `pose_seq` linearly resampled by **frame index** to 30 frames, flattened time-major `(30,21,3)` in C-order. | Layout order is not documented. Interpolation ignores FPS and bridges detection gaps. |
| `gloss` | 0-d unicode array (`np.array(gloss)`), not an object array. | `CONTRACT.md` says "str (object array)". |

`isl/metadata.csv`: `sample_id` = video file stem. `gloss` = folder name. `meaning` = always blank on write. `video_path` / `npz_path` = `str(Path)` relative to the working directory the pipeline was run from. `num_frames` = **detected-hand frame count** (not total frames in the video).

### 7.2 One-hand setting (`max_num_hands=1`)

- `extract.py` takes `multi_hand_landmarks[0]`. If two hands are visible, MediaPipe may return a different hand on different frames, so **one sequence can mix left and right hand**.
- Handedness (Left/Right) is not stored, and nothing mirrors left-handed signers to a common orientation.
- The second hand is dropped. Many ISL signs are two-handed, so these would be encoded incompletely or wrongly.
- `schema.py` and `CONTRACT.md` fix the shape at 21 landmarks, so two hands would be a **shape/contract change** (e.g. `(T,2,21,3)` or `(T,42,3)`) plus a rule for the missing hand.
- Wrist-origin normalisation removes **where the hand is** in the frame, and Y-alignment removes hand rotation. Hand position, motion path and relative location (face/body) are lost. This is a significant limit for ISL.
- Open decision, already listed in `CONTRACT.md`: dominant hand, both hands, or one-handed signs only for the MVP.

### 7.3 Mismatches and issues found

| # | Where | Finding | Severity |
|---|---|---|---|
| 1 | `CONTRACT.md`, `README.md` | Both are marked "IMPLEMENTED" / ✅. `AGENTS.md` says nothing is complete until a teammate confirms it. The tests have never been run and no real video has been processed. | High (wording) |
| 2 | `CONTRACT.md` | It states `landmarks[0]==(0,0,0)` and `‖l9‖≈1` "always". That is false for the all-zero degenerate frames that `normalize.py` can emit. Zero frames also feed into `vec` / `flat_vec` statistics. | Medium |
| 3 | `extract.py` | MediaPipe landmarks are normalised by image width and height separately. Non-square video distorts distances (palm scale) and the rotation angle. Aspect ratio is not corrected. | Medium |
| 4 | `pipeline.py` + `metadata.py` | `upsert_row` rebuilds the whole row with `meaning=""`. **Reprocessing a video wipes a `meaning` that was entered by hand.** | Medium |
| 5 | `metadata.py` | Rows are keyed only on `sample_id`. Two glosses with the same file name (e.g. `001.mp4`) overwrite each other's row, while the `.npz` files stay separate. | Medium |
| 6 | `pipeline.py` | Up-to-date videos are skipped and no row is upserted. If `metadata.csv` is deleted (it is gitignored), a re-run will not rebuild it. | Medium |
| 7 | `pipeline.py` | `run(out_root=...)` is ignored because `_process_video` uses the global `OUT_ROOT`. `errors` is never incremented, so the summary counts are inaccurate. Paths are relative to the working directory, so the pipeline must be run from the repo root. | Low |
| 8 | `schema.py` | The header says "landmark-12 scale", but the code and `CONTRACT.md` scale by landmark 9. `PoseFrame` says a non-detected frame is "interpolated", but nothing interpolates. `video_path` says `.mp4` but four extensions are supported. | Low (docs) |
| 9 | `CONTRACT.md` | The landmark table lists the middle finger as "10–12" and 9 separately. The finger groups are 9–12. The `gloss` dtype is wrong (see 7.1). The ChromaDB path is `isl/db/chroma/`, but the repo has `data/db/chroma/`. | Low |
| 10 | `README.md` / `CONTRACT.md` | `num_frames` means detected frames, which is easy to misread as the clip length. | Low |

### 7.4 Questions needing a decision (user / team)

1. **Hands:** Dominant hand only, both hands, or one-handed MVP? If two hands, what shape, ordering (left/right vs dominant/non-dominant) and handling of a missing hand? Should handedness be saved? Should left-handed signers be mirrored?
2. **Position and motion:** Should the `.npz` also keep un-normalised wrist position (hand location in the frame) so P2 can use location and motion? Or is wrist-origin only acceptable?
3. **Missing frames and timing:** Keep only detected frames (current) or keep all T frames with a `detected` mask? Should `fps` and `frame_idx` be saved? Should `num_frames` be total or detected frames (or both)?
4. **Degenerate frames:** Drop them or keep them as zeros (current)? Should the contract allow them?
5. **Raw landmarks:** Save the un-normalised sequence as well, so normalisation can be changed without re-running MediaPipe?
6. **Aspect ratio:** Should the contract require x/y to be scaled by image width/height before normalisation?
7. **Y-alignment:** Keep it on by default? It changes the meaning of orientation for P2.
8. **Metadata keys and `meaning`:** Key on `(gloss, sample_id)` or a unique id? Must a re-run preserve a hand-entered `meaning`? Should `video_path` / `npz_path` be repo-root-relative with forward slashes?
9. **Field layout docs:** Confirm and document the layouts of `vec` (stat-major, landmark-major) and `flat_vec` (time-major) before P2 depends on them.
10. **Status wording:** Change "IMPLEMENTED" to "DRAFT — pending team confirmation" in `CONTRACT.md` and `README.md` until tests pass and a teammate confirms.
11. **Open from section 5:** vocabulary list, train/val/test split owner, and who fills in `meaning`.

### 7.5 Suggested order (nothing started)

1. Install dependencies and run `pytest tests/test_features.py -v`, then record the real result here.
2. Decide questions 1–3. They change the contract shape and are cheapest to settle before P2 builds on it.
3. After decisions, update `schema.py`, `normalize.py`, `pipeline.py`, `metadata.py` and the docs together in one reviewed change.
4. Fix the docs-only items (7.3 #1, #8, #9, #10) once the status wording is agreed.

---

## 8. Readiness for Hello Videos (2026-11-14)

> Read-only review. No code, paths or videos were changed; **no videos processed, no tests run**. Paths unchanged: `isl/raw_videos/`, `isl/extracted/`, `isl/metadata.csv`.

### 8.1 Confirmed from code / repo

- `isl/raw_videos/hello/` has 36 `.mp4` files (21 `hello__INCLUDE__*`, 15 `hello__ISL500__*`). Stems are unique, so `sample_id` (= file stem) will not collide. Stems contain only letters, digits, `_` and `-`.
- `pipeline.run()` scans `isl/raw_videos/<gloss>/`, gloss = `hello`. Extensions `.mp4/.mov/.avi/.webm` are accepted. Videos are git-ignored (`*.mp4`), `.npz` and `isl/metadata.csv` are too.
- The pipeline has **no CLI arguments**. `python -m isl.pipeline` calls `run()` with default paths. Paths are relative to the working directory, so run it from the repo root. It processes every gloss folder (only `hello` exists now).
- `.venv/` does not exist. `mediapipe`, `cv2`, `numpy` and `pytest` imported without error under system Python 3.13.3. Whether `mp.solutions.hands` works with the installed mediapipe version is **not verified**.

### 8.2 Smallest safe steps before extraction

1. Create `.venv` and install `docs/requirements.txt`, so the run is reproducible.
2. Run the synthetic tests and record the real result here.
3. Dry check of the video files (no extraction): open each file with `cv2.VideoCapture` and note fps, frame count and resolution. This catches unreadable files, odd aspect ratios (open issue 7.3 #3) and mixed sources (INCLUDE vs ISL500).
4. Decide the one-hand question (7.4 #1) before treating output as final. Run a **trial on 1–2 videos only** if needed, to test that the extractor works. Re-extraction is required if the contract changes.
5. Decide 7.4 #3 and #4 (frame mask, degenerate frames) at the same time as the hands decision.

### 8.3 Unresolved decisions that affect these videos

- Hands: one vs two (7.4 #1). Whether "Hello" in these clips is one-handed is unchecked.
- Mixed sources: the two groups may differ in resolution, fps, signer and handedness. No code handles this.
- `meaning` for `hello` (suggest `Hello`) is left blank by the pipeline. A re-run wipes any manual edit (7.3 #4).
- Whether `raw_videos/hello/` filenames should stay as they are (the long names go into `metadata.csv` as `sample_id`).

### 8.4 Commands (confirmed by code; not yet run)

Setup, from the repo root. Creates an isolated environment and installs the dependencies:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r docs/requirements.txt
```

Tests. Runs the synthetic-data tests only, no videos needed:

```bash
pytest tests/test_features.py -v
```

Extraction (single video trial — processes only the selected video under `isl/raw_videos/<gloss>/` and updates `isl/metadata.csv`):

```bash
python -m isl.pipeline --video isl/raw_videos/hello/<video_filename>.mp4
```

Extraction (full batch — processes all videos, writes `isl/extracted/<gloss>/<sample_id>.npz`, and updates `isl/metadata.csv`):

```bash
python -m isl.pipeline
```

Unchanged videos are skipped on re-run (stale check on file mtime).


---

## 9. Environment Setup & Test Run (2026-11-14)

> No videos were processed, the pipeline was not run, and no application code or `docs/requirements.txt` was changed. `.venv/` is git-ignored (`git check-ignore .venv` → `.venv`).

### 9.1 Commands run (from repo root) and actual results

| # | Command | Result |
|---|---|---|
| 1 | `python3.13 -m venv .venv` | OK. Python 3.13.3. |
| 2 | `.venv/bin/python -m pip install -r docs/requirements.txt` | First attempt failed (no network access in the restricted shell: `No matching distribution found for mediapipe`). Re-run with network access: **OK**. |
| 3 | `.venv/bin/python -m pytest tests/test_features.py -v` | **FAILED at collection. 0 tests ran, 1 error** (51 s). |

Installed versions (from pip output): `mediapipe 1.2.3`, `opencv-python 5.0.0.93`, `numpy 2.5.4`, `pytest 9.2.0`. Also pulled in by mediapipe: `opencv-contrib-python 5.0.0.93`, `matplotlib 3.11.2`, `sounddevice 0.5.6`, `absl-py`, `flatbuffers`, and others.

### 9.2 Failure

```
tests/test_features.py:41: from isl.extract import check_video_placement, SkipVideo
isl/extract.py:56: _mp_hands = mp.solutions.hands
E   AttributeError: module 'mediapipe' has no attribute 'solutions'
```

- **Cause (confirmed):** `isl/extract.py` uses the legacy `mp.solutions.hands` API. `docs/requirements.txt` has only a lower bound (`mediapipe>=0.10.9`), so pip installed `mediapipe 1.2.3`, which has no `mp.solutions`. This answers the open question in section 8.1.
- **Not yet verified (hypothesis):** pinning an older `mediapipe` (a 0.10.x release that still ships `mp.solutions`) would fix it. Not tried, because changing requirements or code was out of scope.
- **Consequence:** the synthetic tests cannot be collected, so none of them has been verified. Section 1 "Verification Status" still stands: tests have not passed.

### 9.3 Decision needed

Either pin `mediapipe` to a compatible 0.10.x in `docs/requirements.txt` (P1-owned), or port `isl/extract.py` to the newer MediaPipe Tasks API. Pinning is the smaller change. Re-run step 3 after the choice.

### 9.4 Python 3.12 + MediaPipe 0.10.21 Setup & Rerun (2026-11-14)

- **Decision:** Keep `mp.solutions.hands` in `isl/extract.py` by using Python 3.12 (`3.12.3`) in `.venv312` (leaving `.venv` intact) and pinning `mediapipe==0.10.21` in `docs/requirements.txt` (MediaPipe `<=0.10.21` has no `cp313` wheels on macOS).
- **Commands & Results:**

| # | Command | Result |
|---|---|---|
| 1 | `python3.12 --version` | `Python 3.12.3` available at `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`. |
| 2 | `python3.12 -m pip install --dry-run mediapipe==0.10.21 "opencv-python>=4.9.0" "numpy>=1.26.0" "pytest>=8.0.0"` | OK. Resolved `mediapipe-0.10.21`, `numpy-1.26.4`, `opencv-python-4.11.0.86`, `pytest-9.2.0`. |
| 3 | `python3.12 -m venv .venv312` | OK. Created `.venv312` without modifying `.venv`. |
| 4 | `.venv312/bin/python -m pip install -r docs/requirements.txt` | OK (with network access after `docs/requirements.txt` pinned `mediapipe==0.10.21`). |
| 5 | `.venv312/bin/python -m pytest tests/test_features.py -v` | **30 passed, 2 warnings in 46.13s** (`google._upb._message` Python 3.14 deprecation warnings). |

- **Remaining issues:**
  1. [RESOLVED] `.gitignore` updated to ignore `.venv312/`.
  2. [RESOLVED] `docs/README.md` and `AGENTS.md` setup instructions updated to specify Python 3.12 and `.venv312` consistently.
  3. Real-video extraction verified on 1 trial clip (see Section 9.7); full dataset batch extraction deferred pending team decisions (Section 7.4).

### 9.5 Environment & Documentation Alignment (2026-11-14)

- **Files changed:**
  - `.gitignore`: Added `.venv312/` to ignore list.
  - `docs/README.md`: Updated setup prerequisites and instructions to Python 3.12 and `.venv312`.
  - `AGENTS.md`: Updated setup section to Python 3.12 and `.venv312`.
  - `docs/P1_STATE.md`: Synchronized setup commands, marked resolved setup issues, and recorded file changes.

### 9.6 Single-Video Pipeline Option & Test (2026-11-14)

- **Files changed:**
  - `isl/pipeline.py`: Added `--video <path>` CLI argument (`main()`) and `video_path` parameter to `run()`, with `_validate_single_video()` validating that the target file exists, has a supported extension, and sits directly inside `<raw_root>/<gloss>/`. Preserved default full-batch behavior when `--video` is omitted.
  - `tests/test_features.py`: Added `TestPipelineHelpers::test_single_video_option_and_validation` using `tmp_path` and mocked `_process_video`/`upsert_row` (no real videos or MediaPipe execution).
  - `docs/P1_STATE.md`: Recorded changed files and single-video command.
- **Single-video command (not yet run on real videos):**
  ```bash
  .venv312/bin/python -m isl.pipeline --video isl/raw_videos/hello/<video_filename>.mp4
  ```
- **Test verification:** `.venv312/bin/python -m pytest tests/test_features.py -k test_single_video_option_and_validation -v` → **1 passed in 14.02s**.

### 9.7 Successful Single-Video Extraction Test (2026-11-14)

- **Command executed:**
  ```bash
  python -m isl.pipeline --video isl/raw_videos/hello/hello__ISL500__00087__Hello__session101__clip021.mp4
  ```
- **Results:**
  - Videos processed: 1 (43 detected hand frames).
  - Outputs generated:
    - Feature file: `isl/extracted/hello/hello__ISL500__00087__Hello__session101__clip021.npz`
    - Dataset index: `isl/metadata.csv`
  - `.npz` archive contents:
    - `pose_seq`: shape `(43, 21, 3)`, dtype `float32`
    - `vec`: shape `(252,)`, dtype `float32`
    - `flat_vec`: shape `(1890,)`, dtype `float32`
    - `gloss`: shape `()`, dtype `<U5` (NumPy Unicode string; differs from draft contract's object array specification)
  - `metadata.csv` entry:
    - `sample_id`: `hello__ISL500__00087__Hello__session101__clip021`
    - `gloss`: `hello`
    - `meaning`: `` (blank field; unpopulated)
    - `video_path`: `isl/raw_videos/hello/hello__ISL500__00087__Hello__session101__clip021.mp4`
    - `npz_path`: `isl/extracted/hello/hello__ISL500__00087__Hello__session101__clip021.npz`
    - `num_frames`: `43`

### 9.8 Meaning Field Analysis & Preservation Recommendation (2026-11-14)

- **Why `meaning` is blank:**
  - `isl/pipeline.py` (`_process_video`) hardcodes `"meaning": ""` on extraction.
  - `docs/CONTRACT.md` and `isl/metadata.py` designate `meaning` as a human-curated field intentionally left empty until annotated.
  - `isl/metadata.py` (`upsert_row`) removes existing rows matching `sample_id` without carrying over existing fields, so re-running the pipeline unconditionally overwrites manual annotations.
- **Recommended smallest safe rule:**
  - In `upsert_row`: if an existing row has a non-blank `meaning` and the incoming row has `meaning == ""`, retain the existing value (`existing["meaning"]`).
  - Fallback default: if no manual meaning exists, default to `gloss.replace("_", " ").title()` (e.g. `"Hello"`).
- **Decisions needed:**
  1. Default fallback: Should unannotated samples default to `gloss.title()` or remain empty (`""`) until team curation?
  2. Keying: Should `upsert_row` match on `sample_id` alone or composite `(gloss, sample_id)` to prevent collisions between signs sharing identical video file names?

### 9.9 Proposed Metadata MVP Decisions (Pending Team Confirmation)

- **Decisions adopted for proposal:**
  1. **Meaning lifecycle:** `meaning` remains blank (`""`) until populated by human annotation (no synthetic/title-cased fallback).
  2. **Preservation on re-run:** Re-running the pipeline preserves any existing non-blank `meaning` for that sample.
  3. **Unique keying:** Metadata rows are keyed by composite `(gloss, sample_id)` instead of `sample_id` alone, preventing cross-gloss filename collisions (e.g. `hello/001.mp4` vs `thankyou/001.mp4`).
- **Required Contract Changes (`docs/CONTRACT.md`):**
  - Clarify in the `isl/metadata.csv` table that the primary key is the composite `(gloss, sample_id)` pair.
  - Explicitly specify that `meaning` is initialized as empty string `""` and preserved across pipeline runs once manually edited.
  - (No CSV column names or order change; `isl/metadata.py` docstring and `upsert_row` implementation will require alignment when approved).

### 9.10 Metadata Implementation & Test Verification (2026-11-14)

- **Files changed:**
  - `isl/metadata.py`: Updated `upsert_row` to key on `(gloss, sample_id)`, keep `meaning` blank for new samples, and preserve an existing nonblank `meaning` when incoming `meaning` is blank.
  - `docs/CONTRACT.md`: Documented `(gloss, sample_id)` composite key and `meaning` preservation on pipeline reruns.
  - `tests/test_features.py`: Added `test_upsert_preserves_existing_nonblank_meaning` and `test_upsert_same_sample_id_different_glosses_kept_separate` in `TestMetadata`.
  - `docs/P1_STATE.md`: Recorded the metadata implementation and test results.
- **Test command:**
  ```bash
  .venv312/bin/python -m pytest tests/test_features.py -k TestMetadata -v
  ```
- **Actual result:** `7 passed, 26 deselected, 2 warnings in 13.42s`.

### 9.11 Merge Conflict Resolution with `origin/main` (2026-11-15)

- **Team Decision Confirmed:** MVP supports one hand (`max_num_hands=1` in P1), with `isl/export_p2.py` serving as the P1-to-P2 adapter into hand slot `0` of P2's `(32, 2, 21, 3)` format.
- **Files changed (resolved conflicts + state log):**
  - `.gitignore`: Combined P1 ignore rules with `*.npz` and `data/pose_db/` so both P1 (`isl/extracted/**/*.npz`, `isl/metadata.csv`) and P2 (`data/pose_db/`) generated data remain ignored.
  - `docs/CONTRACT.md`: Unified P1 one-hand extraction contract and P2 recognition engine contract, documenting the `isl/export_p2.py` one-hand bridge (`pose_seq` into slot `0`, `mask[:, 1] = False`, `align_y = True`).
  - `docs/requirements.txt`: Combined P1 and P2 dependencies (`mediapipe==0.10.21`, `opencv-python>=4.9.0`, `numpy>=1.26.0`, `pyyaml>=6.0`, `pytest>=8.0.0`).
  - `isl/__init__.py`: Preserved P2 public API exports (`RecognitionService`, `PoseDatabase`, `SignRecognizer`, `SteadyDetector`, `match_pose`, etc.) alongside the package docstring.
  - `isl/schema.py`: Combined P2 recognition types/constants and P1 pipeline types/constants with full `PoseFrame` and `SignSample` docstrings (and lazy `yaml` import in `RecognizerConfig.from_file`).
  - `tests/test_features.py`: Preserved all P1 unit tests (`TestSchema`, `TestNormalize`, `TestMetadata`, `TestPipelineHelpers`, `TestNpzOutputShape`) and P2 `isl.features` unit tests (`normalize_hand`, `resample`, `summary_vector`).
  - `docs/P1_STATE.md`: Recorded the resolved files and full test suite outcome.
- **Test command:**
  ```bash
  .venv312/bin/python -m pytest -q
  ```
- **Actual result:** `44 passed, 2 warnings in 14.47s`.

### 9.12 Real-Data Smoke Test & P2 Bridge Verification (2026-11-15)

- **Video Extraction:**
  - Processed two additional clips across both source formats:
    - `hello__INCLUDE__00566__MVI_0029.mp4`: 76 detected hand frames.
    - `hello__ISL500__00114__Hello__session159__clip015.mp4`: 38 detected hand frames.
  - Combined with the previously extracted clip (43 frames), total extracted samples in `isl/extracted/hello/` is 3.
- **P2 Export Bridge:**
  - Ran `isl.export_p2` (`--src isl/extracted --dst data/pose_db`); successfully converted and exported 3 archives into `data/pose_db/` with single-hand data in slot 0 and `align_y=True`.
- **Evaluator Outcome:**
  - Ran `backend.scripts.evaluate` (`--data-dir data/pose_db`); loaded all 3 samples and reported 100.00% Top-1 / Top-3 accuracy on the single `hello` class.
- **Important Distinction:**
  - This smoke test verifies **pipeline and bridge compatibility only** (codec handling, MediaPipe extraction, normalization, `.npz` format conversion, database indexing, and evaluator execution).
  - It does **not** demonstrate meaningful multi-class recognition accuracy, discrimination, or generalization, as all evaluated samples share the same class (`hello`).

### 9.13 Full Hello Batch Extraction & P1-to-P2 Pipeline Smoke Test (2026-11-15)

- **Batch Video Extraction:**
  - Full batch run completed on all 36 raw video clips under `isl/raw_videos/hello/`:
    - Processed: 33 videos.
    - Skipped: 3 previously extracted, up-to-date clips.
    - Errors/Failures: 0.
  - Generated outputs: 36 archives in `isl/extracted/hello/*.npz` and 36 indexed rows in `isl/metadata.csv`.
- **P2 Export Bridge:**
  - Ran `isl.export_p2` (`--src isl/extracted --dst data/pose_db`); successfully converted and exported all 36 archives into `data/pose_db/` with single-hand data in slot 0 and `align_y=True`.
- **Evaluator Outcome (LOOCV):**
  - Ran `backend.scripts.evaluate` (`--data-dir data/pose_db`); loaded 36 samples and reported 100.00% Top-1 / Top-3 accuracy on the single `hello` class.
- **Important Caveat:**
  - The 100% metric validates end-to-end **batch processing mechanics, bridge translation, and evaluation stability** at dataset scale.
  - It does **not** measure multi-class recognition accuracy or class separability, as no negative/distractor classes exist in the dataset yet.

### 9.14 Landmark 9 vs 12 Normalization Reference Review (2026-11-16)

> Read-only audit across blueprint, contracts, P1 normalization, and P2 recognition/bridge code. No code modified, no data reprocessed.

#### 1. What Each Component Implements & Expects

| Component | Scale Reference | Y-Alignment Reference | Details / Verification |
|---|---|---|---|
| **Original Blueprint** | Landmark 12 (middle fingertip) | Landmark 9 (middle MCP) | Original handoff specification. |
| **Current P1 (`isl/normalize.py`)** | Landmark 9 (`‖lm9‖ ≈ 1.0`) | Landmark 12 (rotates tip to +Y) | Palm height scaling, tip Y-alignment (`docs/CONTRACT.md` Part 1). |
| **Current P2 (`isl/features.py`)** | Landmark 12 (`‖lm12‖ = 1.0`) | Landmark 12 (rotates tip to +Y) | `_MIDDLE_TIP_IDX = 12` used for both scale and alignment. Verified in `test_normalize_hand_wrist_origin_and_unit_scale` (`assert ‖normed[12]‖ == 1.0`). |
| **P2 Live Inference (`isl/adapter.py`, `isl/matcher.py`)** | Landmark 12 | Landmark 12 | `RecognitionService.recognize()` and `SignRecognizer.process_frame()` normalize live inputs using `isl.features.normalize_hand()`, expecting landmark-12 scaling. |
| **P1-to-P2 Bridge (`isl/export_p2.py`)** | Landmark 9 (inherited from P1) | Landmark 12 (inherited from P1) | Passes P1 `pose_seq` directly into slot 0 without re-normalizing. |

#### 2. Findings & Conclusions

1. **Which rule P2 actually expects:**
   P2 expects **Landmark 12 for scaling** (and Landmark 12 for Y-alignment). Live recognition normalizes live inputs via `isl.features.normalize_hand()` where scale is `‖wrist − landmark 12‖`. Because P1 archives were scaled by Landmark 9 (`‖wrist − landmark 9‖ ≈ 1.0`), database poses in `data/pose_db/` have `‖lm12‖ ≈ 1.8–2.2`, whereas live query poses have `‖lm12‖ = 1.0`. LOOCV passed only because evaluation compared database samples against database samples.
2. **Whether this reversal was an intentional team decision:**
   **No.** Git history and repository documentation show no team decision or discussion approving a reversal. In commit `71fd80d`, `isl/schema.py`'s docstring header still read `(wrist-origin, landmark-12 scale)` while the function body scaled by 9. `docs/P1_STATE.md` (issue 7.3 #8) had flagged this as an unresolved inconsistency. P2 authored its contract (`567d254`) explicitly stating "scaled by wrist-to-middle-fingertip distance" (landmark 12). The inversion was an unintended discrepancy in P1 implementation.
3. **Outputs requiring regeneration if P1 is updated to Landmark 12 scale:**
   - `isl/extracted/**/*.npz` (all 36 `hello` archives in P1, as `pose_seq`, `vec`, and `flat_vec` embed the scale).
   - `data/pose_db/*.npz` (all 36 P2 database archives, as `sequence` and `summary` inherit coordinates from P1).
   - `data/pose_db/.posedb_*.npz` (PoseDatabase disk index cache).
   - Downstream benchmark / evaluation metrics.
   *(Note: Raw video recordings in `isl/raw_videos/` are unaffected; `isl/metadata.csv` requires no schema changes).*
4. **Team Decision Needed:**
   The team must formally decide whether to:
   - **Option A (Align with Blueprint & P2):** Update P1 (`isl/normalize.py`) to scale by Landmark 12 (and align by Landmark 12 or 9 per agreed convention) and regenerate `isl/extracted/` and `data/pose_db/`.
   - **Option B (Standardize on Landmark 9 Palm-Scale):** Update P2 (`isl/features.py:normalize_hand`) and its unit tests to scale by Landmark 9, keeping existing P1 extractions intact.

### 9.15 Two-Class Pipeline Smoke Test: Hello & Help (2026-11-16)

- **Video Extraction:**
  - Processed second ISL500 `help` clip (`help__ISL500__00115__Help__session155__clip011.mp4`) with 50 detected hand frames (joining the first `help` clip with 56 detected frames).
  - Total extracted samples across two glosses: 38 (36 `hello`, 2 `help`) indexed in `isl/metadata.csv`.
- **P2 Export Bridge:**
  - Ran `isl.export_p2` (`--src isl/extracted --dst data/pose_db`); successfully converted and exported all 38 archives (36 `hello`, 2 `help`) into `data/pose_db/` with single-hand data in slot 0 and `align_y=True`.
- **Evaluator Outcome (LOOCV):**
  - Ran `backend.scripts.evaluate` (`--data-dir data/pose_db`); loaded all 38 samples across both classes and completed with 100.00% Top-1 / Top-3 accuracy for both `hello` and `help` labels.
- **Critical Caveat:**
  - This is a small pipeline smoke test verifying end-to-end multi-label mechanics, directory indexing, and evaluator execution across classes.
  - It is **not** reliable evidence of true recognition accuracy or generalizable discrimination, given the minimal sample size (only 2 `help` clips) and lack of a distinct test split.

### 9.16 P1 Pipeline Reliability Fixes: Outcome Counts & Cached Metadata Restoration (2026-11-16)

- **Issues resolved:**
  1. **Distinct outcome tracking and reporting (7.3 #7):** Updated `isl/pipeline.py` (`_process_video` and `run`) to distinguish `"processed"`, `"cached"` (up-to-date `.npz`), `"skipped"` (`SkipVideo`), and `"error"` (`Exception`) outcomes. Both single-video and batch runs now report `Processed`, `Cached`, `Skipped`, and `Errors` separately instead of grouping normal cached skips under `Skipped/failed`.
  2. **Metadata restoration on cached `.npz` (7.3 #4, #6):** When an up-to-date `.npz` is encountered, `_process_video` reads `pose_seq.shape[0]` from the existing archive without re-extracting the video, and `run()` upserts the metadata row via `upsert_row`, restoring missing rows if `metadata.csv` was deleted or out of sync while preserving any existing manually entered `meaning`.
- **Files changed:**
  - `isl/pipeline.py`: Updated `_process_video` to return `(status, metadata_or_none)`, added `_unpack_outcome`, restored metadata rows for cached `.npz` archives without re-extraction, and separated `processed`, `cached`, `skipped`, and `errors` counts in `run()`.
  - `tests/test_features.py`: Added `TestPipelineHelpers::test_pipeline_distinct_outcome_counts_and_reporting` and `TestPipelineHelpers::test_cached_npz_restores_missing_metadata_and_preserves_meaning` using `tmp_path` and mocks (no real videos processed).
  - `docs/P1_STATE.md`: Recorded the reliability fixes, commands executed, and test outcomes.
- **Test commands and actual results:**
  - Focused pipeline helper tests:
    ```bash
    .venv312/bin/python -m pytest tests/test_features.py -k TestPipelineHelpers -v
    ```
    Result: `6 passed, 33 deselected, 2 warnings in 13.51s`.
  - Full test suite:
    ```bash
    .venv312/bin/python -m pytest -q
    ```
    Result: `46 passed, 2 warnings in 13.03s`.
