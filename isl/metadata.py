"""
isl/metadata.py — Read and write isl/metadata.csv.

What this file does
-------------------
Every processed video gets one row in ``isl/metadata.csv``.  This module
keeps that file consistent: no duplicate rows, predictable column order,
and atomic writes (a crash mid-write does not corrupt the existing file).

Columns (do not rename without a team PR):
  sample_id   — video filename without extension, e.g. "hello_001"
  gloss       — sign label (label folder name), e.g. "hello"
  meaning     — human-readable English gloss; left blank until team fills it
  video_path  — relative path to the source video
  npz_path    — relative path to the generated .npz file
  num_frames  — number of detected-hand frames used in the .npz
"""

from __future__ import annotations

import csv
import pathlib
import tempfile
import os

METADATA_PATH = pathlib.Path("isl") / "metadata.csv"

COLUMNS = ["sample_id", "gloss", "meaning", "video_path", "npz_path", "num_frames"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_metadata(path: pathlib.Path = METADATA_PATH) -> list[dict]:
    """Return all rows from *path* as a list of dicts.

    Returns an empty list if the file does not exist yet.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def upsert_row(row: dict, path: pathlib.Path = METADATA_PATH) -> None:
    """Add or replace a metadata row keyed on ``(gloss, sample_id)``.

    - If ``(gloss, sample_id)`` already exists in the file, its row is
      updated while preserving any existing nonblank ``meaning`` when the
      incoming ``meaning`` is blank.
    - If it does not exist, a new row is appended.
    - Columns not in ``COLUMNS`` are silently ignored.
    - The file is written atomically so a crash mid-write leaves the
      existing file untouched.

    Parameters
    ----------
    row:
        A dict with at least ``gloss`` and ``sample_id`` set. Missing
        columns are written as empty strings.
    path:
        CSV file path (defaults to ``isl/metadata.csv``).
    """
    rows = load_metadata(path)
    key = (str(row.get("gloss", "")), str(row.get("sample_id", "")))

    existing_meaning = ""
    kept_rows = []
    for r in rows:
        if (r.get("gloss", ""), r.get("sample_id", "")) == key:
            if r.get("meaning", "").strip():
                existing_meaning = r["meaning"]
        else:
            kept_rows.append(r)

    # Build a clean row with all required columns
    clean = {col: str(row.get(col, "")) for col in COLUMNS}
    if not clean["meaning"].strip() and existing_meaning:
        clean["meaning"] = existing_meaning
    kept_rows.append(clean)

    # Sort by gloss then sample_id for stable output
    kept_rows.sort(key=lambda r: (r.get("gloss", ""), r.get("sample_id", "")))

    _write_csv(kept_rows, path)


def _write_csv(rows: list[dict], path: pathlib.Path) -> None:
    """Write *rows* to *path* atomically using a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        # Atomic replace
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
