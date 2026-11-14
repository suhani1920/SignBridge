from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from isl.features import resample, resample_mask, summary_vector
from isl.schema import COORDS, NUM_HANDS, NUM_LANDMARKS, SEQ_LEN

P1_EXTRACTED_DIR = Path("isl") / "extracted"
P2_DB_DIR = Path("data") / "pose_db"


def convert_p1_archive_to_p2(p1_npz: Path, out_dir: Path) -> Path:
    with np.load(p1_npz, allow_pickle=False) as data:
        if "pose_seq" not in data or "gloss" not in data:
            raise KeyError(f"{p1_npz} is missing required P1 keys ('pose_seq', 'gloss')")

        pose_seq = np.asarray(data["pose_seq"], dtype=np.float32)  # (T, 21, 3)
        label = str(data["gloss"].item())

    if pose_seq.ndim != 3 or pose_seq.shape[1:] != (NUM_LANDMARKS, COORDS):
        raise ValueError(f"Expected pose_seq shape (T, 21, 3), got {pose_seq.shape}")

    t_frames = pose_seq.shape[0]
    two_hand_seq = np.zeros((t_frames, NUM_HANDS, NUM_LANDMARKS, COORDS), dtype=np.float32)
    mask = np.zeros((t_frames, NUM_HANDS), dtype=bool)

    # Place P1 single-hand sequence into slot 0 to match RecognitionService.recognize()
    if t_frames > 0:
        two_hand_seq[:, 0, :, :] = pose_seq
        mask[:, 0] = np.linalg.norm(pose_seq.reshape(t_frames, -1), axis=1) > 1e-8

    resampled_seq = resample(two_hand_seq, target_len=SEQ_LEN)
    resampled_mask = resample_mask(mask, target_len=SEQ_LEN)
    summary = summary_vector(resampled_seq, resampled_mask)
    body_pos = np.zeros((SEQ_LEN, NUM_HANDS, COORDS), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}__{p1_npz.stem}.npz"
    np.savez_compressed(
        out_path,
        sequence=resampled_seq.astype(np.float32),
        body_pos=body_pos,
        mask=resampled_mask.astype(bool),
        summary=summary.astype(np.float32),
        label=np.array(label),
        align_y=np.array(True),
    )
    return out_path


def export_all(p1_root: Path = P1_EXTRACTED_DIR, p2_root: Path = P2_DB_DIR) -> int:
    if not p1_root.exists():
        raise FileNotFoundError(f"P1 extracted directory not found: {p1_root}")

    count = 0
    for npz_file in sorted(p1_root.rglob("*.npz")):
        convert_p1_archive_to_p2(npz_file, p2_root)
        count += 1
    print(f"Exported {count} P2-compatible archive(s) to {p2_root}/")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert P1 extracted .npz files to P2 PoseDatabase format.")
    parser.add_argument("--src", type=Path, default=P1_EXTRACTED_DIR, help="P1 extracted directory")
    parser.add_argument("--dst", type=Path, default=P2_DB_DIR, help="Target P2 PoseDatabase directory")
    args = parser.parse_args()
    export_all(args.src, args.dst)


if __name__ == "__main__":
    main()