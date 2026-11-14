# AGENTS.md — SignBridge

> Instructions for AI coding assistants and human teammates working in this repository.
> Read this file before making any changes.

---

## Project Overview

**SignBridge** is a college project building an Indian Sign Language (ISL) video
translator. It takes sign-language video clips as input and produces text translations
as output.

The project is split into parallel workstreams (P1, P2, P3, …). Each workstream owns
specific files and pipeline stages. Do not modify files outside your assigned scope
without coordinating with the relevant owner first.

**Current status:** Early scaffold. Most files are empty stubs. Only P1 (data
preparation) work has begun. No pipeline stage, schema, or contract should be
described as complete unless a teammate has explicitly confirmed it and committed it.

---

## General Rules for All Contributors

### 1. Inspect Before You Edit

Before changing any file, read it in full. Check the git log for recent changes:

```bash
# See what changed recently
git log --oneline -10

# See the current state of a specific file
cat isl/schema.py
```

Never overwrite work you have not read. If a file looks empty, verify it is not a
stub that another agent will fill.

### 2. Stay in Your Lane

Each task (P1, P2, …) owns specific files. If your task is P1, do not edit backend,
web, or other pipeline files. If you are unsure whether a file belongs to you, ask
before editing it.

### 3. Preserve Shared Contracts Once Agreed

The files below are **shared contracts** — once the team has agreed on their content
and merged them, treat them as stable interfaces. Do not silently change field names,
file paths, or data formats:

| File | What it defines |
|---|---|
| `docs/CONTRACT.md` | Feature schema: field names, dtypes, units, file format |
| `isl/schema.py` | Python implementation of the schema (Pydantic / dataclass) |
| `AGENTS.md` | Agent ownership, pipeline commands, working conventions |

If you need to change a shared contract, propose the change in a pull request and
get a teammate's approval before merging.

### 4. Keep Changes Focused

Make the smallest change that fulfills your task. Do not refactor files outside your
scope, rename shared identifiers, or reorganise the directory structure without
discussing it with the team first.

### 5. Never Use Blender

This project does not use Blender for any stage. Do not install it, reference it in
scripts, or suggest it as a tool.

### 6. Never Commit Sensitive or Large Binary Data

The following must **never** be committed to git unless the full team explicitly
agrees and a strategy (Git LFS, DVC, cloud storage) has been decided:

- Passwords, API keys, tokens, or any private credentials
- Raw video files (`*.mp4`, `*.webm`, `*.avi`, …)
- Generated datasets (`*.npy`, `*.pkl`, `*.h5`, …)
- Model weights or checkpoints

Add such files to `.gitignore` before you create them. If you are unsure, check
`.gitignore` first and err on the side of exclusion.

### 7. Explain Commands Before Running Them

When a script or shell command is part of your work, write a one-line explanation of
what it does and what output to expect before listing it. This helps beginner
teammates run it safely. Example:

> The command below installs all Python dependencies listed in `requirements.txt`
> into your active virtual environment.
>
> ```bash
> pip install -r docs/requirements.txt
> ```

### 8. Report Truthfully

After completing any task, list:
- **Files changed** — exact paths, what was added or modified
- **Files left unchanged** — confirm you did not touch out-of-scope files
- **Checks run** — commands executed (lint, tests) and their outcomes

Do not claim a check passed if you did not run it. Do not claim a contract or
pipeline stage is complete if it is still a stub.

---

## Repository Layout

```
SignBridge/
├── AGENTS.md               ← this file; read first
├── docs/
│   ├── CONTRACT.md         ← shared data/API contract (stub — not yet defined)
│   ├── README.md           ← project overview and setup guide (stub)
│   ├── requirements.txt    ← Python dependencies (stub)
│   └── .gitignore          ← gitignore rules for the docs folder (stub)
├── isl/
│   └── schema.py           ← Python schema matching CONTRACT.md (stub)
├── backend/
│   └── sample.py           ← backend sample/demo (stub; owner TBD)
├── tests/
│   └── test_features.py    ← feature tests (stub)
├── web/
│   └── package-lock.json   ← frontend package lock (stub; no deps yet)
└── data/                   ← NOT IN GIT — create locally
    ├── raw/                ← input videos (never commit)
    └── processed/          ← generated keypoints / features (never commit)
```

> **Note:** The `data/` directory does not exist in the repo yet. Create it locally.
> It and its contents must remain in `.gitignore`.

---

## Workstream Ownership

| Workstream | Focus | Key files |
|---|---|---|
| **P1** | Sign-video data preparation | `isl/`, `tests/test_features.py`, `docs/CONTRACT.md`, `docs/requirements.txt` |
| P2 | *(not yet assigned)* | TBD |
| P3 | *(not yet assigned)* | TBD |
| Backend | Server / inference API | `backend/` |
| Web / Frontend | User interface | `web/` |

Update this table as roles are confirmed by the team.

---

## P1 — Data Preparation (In Progress)

**Goal:** Produce clean, normalised feature vectors from raw ISL video clips so that
downstream model training (P2) has a stable, versioned input.

**Pipeline (to be implemented):**

```
data/raw/isl/          →  isl/extract_keypoints.py  →  data/processed/keypoints/
data/processed/keypoints/  →  isl/build_features.py  →  data/processed/features/
```

**Planned files (not yet created):**

| File | Purpose |
|---|---|
| `isl/extract_keypoints.py` | Read raw videos, run MediaPipe, write per-frame landmark arrays |
| `isl/build_features.py` | Convert landmark arrays → normalised feature vectors per schema |
| `isl/video_utils.py` | Shared helpers: frame sampler, resizer, FPS normaliser |

**Status:** Schema and contract are not yet defined. See `docs/CONTRACT.md` and
`isl/schema.py` — both are empty stubs pending team agreement.

---

## Setup (Provisional — Not Yet Complete)

> These commands will work once `docs/requirements.txt` is filled in by P1.
> Do not run them yet; they are listed here as the intended workflow.

```bash
# 1. Create and activate a Python virtual environment
#    This keeps project dependencies isolated from your system Python.
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Install Python dependencies
#    Installs everything listed in docs/requirements.txt.
pip install -r docs/requirements.txt
```

---

## Running Tests

> Tests are defined in `tests/test_features.py`, which is currently an empty stub.
> The command below will work once tests are written.

```bash
# Run all feature tests with verbose output
pytest tests/test_features.py -v
```

---

## Git Workflow

```bash
# Always work on a feature branch, not directly on main
git checkout -b feature/<your-task>

# Stage only the files you intentionally changed
git add <specific-file>

# Write a clear commit message
git commit -m "P1: add keypoint extraction script"

# Push and open a pull request for review
git push origin feature/<your-task>
```

Do not `git add .` blindly — it may accidentally stage video files, generated data,
or editor config files. Stage files explicitly by name.

---

## What Is Not Done Yet

The following are **stubs or not started**. Do not describe them as complete:

- `docs/CONTRACT.md` — no schema agreed yet
- `isl/schema.py` — no implementation yet
- `docs/README.md` — no setup instructions yet
- `docs/requirements.txt` — no dependencies listed yet
- `backend/sample.py` — owner and purpose TBD
- `tests/test_features.py` — no tests written yet
- P2, P3, Backend, and Web workstreams — not yet started

---

*Last updated by: P1 (Piyush Singh) — 2026-11-13*
