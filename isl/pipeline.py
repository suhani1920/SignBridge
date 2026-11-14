"""
isl/pipeline.py — Batch orchestrator for the P1 data-preparation pipeline.

What this file does
-------------------
Run this module from the repo root to process every video under
``isl/raw_videos/`` and write normalised pose data to ``isl/extracted/``.

Usage (run from repo root after activating .venv312):
    python -m isl.pipeline
    python -m isl.pipeline --video isl/raw_videos/hello/hello_001.mp4

What it does, step by step:
  1. Scans ``isl/raw_videos/`` for label subfolders and video files.
  2. Warns if any videos are placed directly in raw_videos/ (no label folder).
  3. For each video:
       a. Extracts hand landmarks with MediaPipe (isl/extract.py).
       b. Normalises the sequence (isl/normalize.py).
       c. Saves a .npz file to isl/extracted/<gloss>/<sample_id>.npz.
       d. Upserts one row in isl/metadata.csv.
  4. Skips already-processed videos whose .npz is newer than the source
     video (re-run is safe and fast).
  5. Continues on error/skip; prints a summary at the end.

Output files
------------
  isl/extracted/<gloss>/<sample_id>.npz  — compressed numpy archive
  isl/metadata.csv                       — one row per processed sample
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

from isl.extract import extract_sample, check_video_placement, SkipVideo
from isl.normalize import normalize_sequence, build_vec, build_flat_vec
from isl.metadata import upsert_row
from isl.schema import VIDEO_EXTENSIONS

RAW_ROOT = pathlib.Path("isl") / "raw_videos"
OUT_ROOT = pathlib.Path("isl") / "extracted"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_stale(video_path: pathlib.Path, npz_path: pathlib.Path) -> bool:
    """Return True if the .npz does not exist or is older than the video."""
    if not npz_path.exists():
        return True
    return video_path.stat().st_mtime > npz_path.stat().st_mtime


def _validate_single_video(
    video_path: pathlib.Path, raw_root: pathlib.Path = RAW_ROOT
) -> tuple[pathlib.Path, str]:
    """Validate that *video_path* is a supported video inside *raw_root*/<gloss>/."""
    vp = pathlib.Path(video_path)
    if not vp.is_file():
        raise FileNotFoundError(f"Video not found: {vp}")
    if vp.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{vp.suffix}' for {vp.name}. "
            f"Supported: {', '.join(VIDEO_EXTENSIONS)}"
        )
    try:
        rel = vp.resolve().relative_to(raw_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Video must be inside {raw_root}/<gloss>/: {vp}"
        ) from exc
    if len(rel.parts) != 2 or rel.parts[0].startswith("."):
        raise ValueError(
            f"Video must be directly inside a label folder {raw_root}/<gloss>/: {vp}"
        )
    gloss = rel.parts[0]
    return raw_root / gloss / vp.name, gloss


def _process_video(
    video_path: pathlib.Path, gloss: str, out_root: pathlib.Path = OUT_ROOT
) -> dict | None:
    """Process one video. Returns the metadata dict on success, None on failure."""
    sample_id = video_path.stem
    npz_path = out_root / gloss / f"{sample_id}.npz"

    if not _is_stale(video_path, npz_path):
        print(f"  [skip — up to date] {video_path.name}")
        return None

    print(f"  [extract]  {video_path.name} ...", end="", flush=True)
    t0 = time.monotonic()

    try:
        sample = extract_sample(video_path, gloss)
    except SkipVideo as exc:
        print(f"\n  [SKIP] {video_path.name}: {exc}")
        return None
    except Exception as exc:
        print(f"\n  [ERROR] {video_path.name}: {exc}")
        return None

    # Normalise
    raw_seq = sample.pose_seq                           # (T, 21, 3)
    norm_seq = normalize_sequence(raw_seq, align_y=True)
    vec = build_vec(norm_seq)                           # (252,)
    flat_vec = build_flat_vec(norm_seq)                 # (1890,)

    # Save .npz
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        pose_seq=norm_seq.astype(np.float32),
        vec=vec.astype(np.float32),
        flat_vec=flat_vec.astype(np.float32),
        gloss=np.array(gloss),
    )

    elapsed = time.monotonic() - t0
    print(f" done ({sample.detected_frames} frames, {elapsed:.1f}s)")

    return {
        "sample_id": sample_id,
        "gloss": gloss,
        "meaning": "",
        "video_path": str(video_path),
        "npz_path": str(npz_path),
        "num_frames": sample.detected_frames,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    raw_root: pathlib.Path = RAW_ROOT,
    out_root: pathlib.Path = OUT_ROOT,
    video_path: pathlib.Path | None = None,
) -> None:
    """Run the full batch pipeline (or a single video when *video_path* is given)."""

    if not raw_root.exists():
        print(
            f"[ERROR] Input folder not found: {raw_root}\n"
            "Create it and add your sign-label subfolders first.\n"
            "Example: isl/raw_videos/hello/hello_001.mp4"
        )
        sys.exit(1)

    if video_path is not None:
        vp, gloss = _validate_single_video(video_path, raw_root)
        print(f"\n[{gloss}] 1 video(s)")
        meta = _process_video(vp, gloss, out_root)
        processed, skipped = (1, 0) if meta is not None else (0, 1)
        if meta is not None:
            upsert_row(meta)
        print(
            f"\n{'='*50}\n"
            f"Done.  Processed: {processed}  |  Skipped/failed: {skipped}\n"
            f"Output: {out_root}/\n"
            f"Index:  isl/metadata.csv\n"
        )
        return

    # --- warn about misplaced videos ---
    misplaced = check_video_placement(raw_root)
    if misplaced:
        print("\n[WARNING] The following videos are directly inside raw_videos/")
        print("          and will be SKIPPED because they have no sign-label folder.\n")
        for p in misplaced:
            print(f"  {p.name}")
        print(
            "\nFix: move each video into a subfolder named after its sign.\n"
            "Example:\n"
            "  isl/raw_videos/hello/hello_001.mp4   ← correct\n"
            "  isl/raw_videos/hello_001.mp4          ← WRONG (no label folder)\n"
        )

    # --- collect label folders ---
    label_dirs = sorted(
        d for d in raw_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not label_dirs:
        print(
            "[INFO] No label subfolders found under raw_videos/.\n"
            "Add at least one folder (e.g. isl/raw_videos/hello/) with video files."
        )
        return

    processed, skipped, errors = 0, 0, 0

    for label_dir in label_dirs:
        gloss = label_dir.name
        videos = sorted(
            p for p in label_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        if not videos:
            print(f"\n[{gloss}] No video files found — skipping folder.")
            continue

        print(f"\n[{gloss}] {len(videos)} video(s)")

        for vp in videos:
            meta = _process_video(vp, gloss, out_root)
            if meta is None:
                skipped += 1
            else:
                upsert_row(meta)
                processed += 1

    # --- summary ---
    print(
        f"\n{'='*50}\n"
        f"Done.  Processed: {processed}  |  Skipped/failed: {skipped + errors}\n"
        f"Output: {out_root}/\n"
        f"Index:  isl/metadata.csv\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the P1 ISL video pipeline.")
    parser.add_argument(
        "--video",
        type=pathlib.Path,
        default=None,
        help="Process only this single video inside isl/raw_videos/<gloss>/.",
    )
    args = parser.parse_args(argv)
    run(video_path=args.video)


if __name__ == "__main__":
    main()
