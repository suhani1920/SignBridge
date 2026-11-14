"""
tests/test_features.py — P1 focused tests for SignBridge.

All tests use synthetic (fake) data: numpy arrays, temporary files, and
short in-memory videos built with OpenCV. No real sign-language videos
are needed. No internet access is required.

Run the tests from the repo root:
    pytest tests/test_features.py -v

What each test verifies:
  TestSchema            — PoseFrame and SignSample data types
  TestNormalize         — normalization math and helpers
  TestMetadata          — CSV read / upsert / deduplication
  TestPipelineHelpers   — misplaced-video detection
  TestNpzOutputShape    — final .npz array dimensions
"""

from __future__ import annotations

import csv
import pathlib
import tempfile

import numpy as np
import pytest

from isl.schema import (
    NUM_LANDMARKS,
    NUM_COORDS,
    RESAMPLE_FRAMES,
    VEC_LEN,
    FLAT_VEC_LEN,
    MIN_HAND_FRAMES,
    VIDEO_EXTENSIONS,
    PoseFrame,
    SignSample,
)
from isl.normalize import normalize_sequence, build_vec, build_flat_vec, _normalize_frame
from isl.metadata import load_metadata, upsert_row, COLUMNS
from isl.extract import check_video_placement, SkipVideo
from isl import pipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _random_frame(seed: int = 0) -> np.ndarray:
    """Return a reproducible (21, 3) float32 landmark array."""
    rng = np.random.default_rng(seed)
    return rng.random((NUM_LANDMARKS, NUM_COORDS)).astype(np.float32)


def _random_seq(T: int = 10, seed: int = 42) -> np.ndarray:
    """Return a reproducible (T, 21, 3) float32 sequence."""
    rng = np.random.default_rng(seed)
    return rng.random((T, NUM_LANDMARKS, NUM_COORDS)).astype(np.float32)


# ---------------------------------------------------------------------------
# TestSchema
# ---------------------------------------------------------------------------


class TestSchema:
    """PoseFrame and SignSample data types behave as specified."""

    def test_poseframe_correct_shape_accepted(self):
        """PoseFrame accepts a correctly shaped (21, 3) array."""
        arr = _random_frame()
        pf = PoseFrame(frame_idx=0, landmarks=arr)
        assert pf.landmarks.shape == (NUM_LANDMARKS, NUM_COORDS)
        assert pf.landmarks.dtype == np.float32

    def test_poseframe_wrong_shape_raises(self):
        """PoseFrame raises ValueError for a wrong-shaped array."""
        bad = np.zeros((20, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            PoseFrame(frame_idx=0, landmarks=bad)

    def test_poseframe_coerces_to_float32(self):
        """PoseFrame converts float64 input to float32."""
        arr = np.ones((NUM_LANDMARKS, NUM_COORDS), dtype=np.float64)
        pf = PoseFrame(frame_idx=0, landmarks=arr)
        assert pf.landmarks.dtype == np.float32

    def test_signsample_detected_frames_count(self):
        """SignSample.detected_frames counts only frames with detected=True."""
        frames = [
            PoseFrame(0, _random_frame(0), detected=True),
            PoseFrame(1, _random_frame(1), detected=False),
            PoseFrame(2, _random_frame(2), detected=True),
        ]
        s = SignSample("id", "gloss", "/path/v.mp4", 25.0, frames)
        assert s.detected_frames == 2
        assert s.total_frames == 3

    def test_signsample_pose_seq_excludes_undetected(self):
        """SignSample.pose_seq returns only detected-frame arrays."""
        frames = [
            PoseFrame(0, _random_frame(0), detected=True),
            PoseFrame(1, np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32), detected=False),
            PoseFrame(2, _random_frame(2), detected=True),
        ]
        s = SignSample("id", "gloss", "/path/v.mp4", 25.0, frames)
        seq = s.pose_seq
        assert seq.shape == (2, NUM_LANDMARKS, NUM_COORDS)

    def test_signsample_empty_pose_seq(self):
        """SignSample.pose_seq with no detected frames returns shape (0, 21, 3)."""
        s = SignSample("id", "gloss", "/path/v.mp4", 25.0, [])
        assert s.pose_seq.shape == (0, NUM_LANDMARKS, NUM_COORDS)

    def test_constants_match_expected_values(self):
        """Schema constants have the values the blueprint specifies."""
        assert NUM_LANDMARKS == 21
        assert NUM_COORDS == 3
        assert RESAMPLE_FRAMES == 30
        assert VEC_LEN == 252       # 21 × 3 × 4
        assert FLAT_VEC_LEN == 1890 # 30 × 21 × 3
        assert MIN_HAND_FRAMES == 5


# ---------------------------------------------------------------------------
# TestNormalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Normalization functions produce correct shapes and values."""

    def test_normalize_sequence_shape_preserved(self):
        """normalize_sequence returns the same (T, 21, 3) shape."""
        seq = _random_seq(T=12)
        out = normalize_sequence(seq)
        assert out.shape == seq.shape

    def test_normalize_sequence_dtype_float32(self):
        """normalize_sequence output is always float32."""
        seq = _random_seq(T=8)
        out = normalize_sequence(seq)
        assert out.dtype == np.float32

    def test_normalize_sequence_no_nan(self):
        """Normalized output contains no NaN values."""
        seq = _random_seq(T=15)
        out = normalize_sequence(seq)
        assert not np.any(np.isnan(out))

    def test_normalize_sequence_no_inf(self):
        """Normalized output contains no infinite values."""
        seq = _random_seq(T=15)
        out = normalize_sequence(seq)
        assert not np.any(np.isinf(out))

    def test_wrist_at_origin_after_normalization(self):
        """After normalization, the wrist landmark (index 0) is at the origin."""
        seq = _random_seq(T=10)
        out = normalize_sequence(seq, align_y=False)
        # Wrist (landmark 0) should be (0, 0, 0) in every frame
        np.testing.assert_allclose(out[:, 0, :], 0.0, atol=1e-5)

    def test_scale_reference_distance_is_one(self):
        """After normalization (no Y-align), ‖landmark_9‖ ≈ 1.0 for each frame."""
        seq = _random_seq(T=10)
        out = normalize_sequence(seq, align_y=False)
        for frame in out:
            dist = np.linalg.norm(frame[9])
            assert abs(dist - 1.0) < 1e-5, f"Expected ‖lm9‖=1.0, got {dist:.6f}"

    def test_degenerate_frame_returns_zeros(self):
        """A frame where all landmarks are at the wrist returns zeros."""
        frame = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        out = _normalize_frame(frame, align_y=False)
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_normalize_sequence_wrong_shape_raises(self):
        """normalize_sequence raises ValueError for bad input shape."""
        bad = np.zeros((5, 20, 3), dtype=np.float32)  # 20 landmarks, not 21
        with pytest.raises(ValueError, match="shape"):
            normalize_sequence(bad)

    def test_build_vec_shape(self):
        """build_vec returns shape (252,)."""
        seq = _random_seq(T=10)
        norm = normalize_sequence(seq)
        v = build_vec(norm)
        assert v.shape == (VEC_LEN,)
        assert v.dtype == np.float32

    def test_build_flat_vec_shape(self):
        """build_flat_vec returns shape (1890,) by default."""
        seq = _random_seq(T=10)
        norm = normalize_sequence(seq)
        fv = build_flat_vec(norm)
        assert fv.shape == (FLAT_VEC_LEN,)
        assert fv.dtype == np.float32

    def test_build_flat_vec_empty_seq(self):
        """build_flat_vec on an empty sequence returns a zero vector."""
        empty = np.zeros((0, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        fv = build_flat_vec(empty)
        assert fv.shape == (FLAT_VEC_LEN,)
        np.testing.assert_array_equal(fv, 0.0)

    def test_build_flat_vec_custom_frames(self):
        """build_flat_vec respects a custom n_frames argument."""
        seq = _random_seq(T=10)
        norm = normalize_sequence(seq)
        fv = build_flat_vec(norm, n_frames=15)
        assert fv.shape == (15 * NUM_LANDMARKS * NUM_COORDS,)


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """metadata.csv read / write / upsert operations."""

    @pytest.fixture
    def tmp_csv(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / "metadata.csv"

    def test_load_missing_file_returns_empty(self, tmp_csv):
        """load_metadata returns [] when the CSV does not exist."""
        assert load_metadata(tmp_csv) == []

    def test_upsert_creates_file(self, tmp_csv):
        """upsert_row creates the CSV if it does not exist."""
        row = {
            "sample_id": "hello_001",
            "gloss": "hello",
            "meaning": "",
            "video_path": "isl/raw_videos/hello/hello_001.mp4",
            "npz_path": "isl/extracted/hello/hello_001.npz",
            "num_frames": 20,
        }
        upsert_row(row, tmp_csv)
        rows = load_metadata(tmp_csv)
        assert len(rows) == 1
        assert rows[0]["sample_id"] == "hello_001"

    def test_upsert_no_duplicate(self, tmp_csv):
        """Upserting the same sample_id twice results in exactly one row."""
        row = {
            "sample_id": "hello_001", "gloss": "hello", "meaning": "",
            "video_path": "v.mp4", "npz_path": "h.npz", "num_frames": 10,
        }
        upsert_row(row, tmp_csv)
        row["num_frames"] = 20   # update
        upsert_row(row, tmp_csv)
        rows = load_metadata(tmp_csv)
        assert len(rows) == 1
        assert rows[0]["num_frames"] == "20"

    def test_upsert_multiple_rows(self, tmp_csv):
        """Multiple different sample_ids create multiple rows."""
        for i in range(3):
            upsert_row({
                "sample_id": f"hello_00{i+1}", "gloss": "hello", "meaning": "",
                "video_path": f"v{i}.mp4", "npz_path": f"h{i}.npz",
                "num_frames": i * 5 + 5,
            }, tmp_csv)
        rows = load_metadata(tmp_csv)
        assert len(rows) == 3

    def test_csv_has_correct_columns(self, tmp_csv):
        """Written CSV has exactly the agreed column names."""
        upsert_row({
            "sample_id": "x", "gloss": "g", "meaning": "",
            "video_path": "v", "npz_path": "n", "num_frames": 1,
        }, tmp_csv)
        with tmp_csv.open() as fh:
            header = next(csv.reader(fh))
        assert header == COLUMNS

    def test_upsert_preserves_existing_nonblank_meaning(self, tmp_csv):
        """Rerunning upsert with blank meaning preserves existing nonblank meaning."""
        upsert_row({
            "sample_id": "hello_001", "gloss": "hello", "meaning": "Hello",
            "video_path": "v.mp4", "npz_path": "h.npz", "num_frames": 10,
        }, tmp_csv)
        upsert_row({
            "sample_id": "hello_001", "gloss": "hello", "meaning": "",
            "video_path": "v.mp4", "npz_path": "h.npz", "num_frames": 25,
        }, tmp_csv)
        rows = load_metadata(tmp_csv)
        assert len(rows) == 1
        assert rows[0]["meaning"] == "Hello"
        assert rows[0]["num_frames"] == "25"

    def test_upsert_same_sample_id_different_glosses_kept_separate(self, tmp_csv):
        """Same sample_id under different glosses creates separate rows."""
        upsert_row({
            "sample_id": "001", "gloss": "hello", "meaning": "",
            "video_path": "isl/raw_videos/hello/001.mp4",
            "npz_path": "isl/extracted/hello/001.npz", "num_frames": 10,
        }, tmp_csv)
        upsert_row({
            "sample_id": "001", "gloss": "thankyou", "meaning": "",
            "video_path": "isl/raw_videos/thankyou/001.mp4",
            "npz_path": "isl/extracted/thankyou/001.npz", "num_frames": 15,
        }, tmp_csv)
        rows = load_metadata(tmp_csv)
        assert len(rows) == 2
        assert [(r["gloss"], r["sample_id"]) for r in rows] == [
            ("hello", "001"),
            ("thankyou", "001"),
        ]


# ---------------------------------------------------------------------------
# TestPipelineHelpers
# ---------------------------------------------------------------------------


class TestPipelineHelpers:
    """check_video_placement detects misplaced video files."""

    def test_no_misplaced_videos(self, tmp_path: pathlib.Path):
        """Returns empty list when all videos are inside label subfolders."""
        label = tmp_path / "hello"
        label.mkdir()
        (label / "hello_001.mp4").write_bytes(b"fake")
        result = check_video_placement(tmp_path)
        assert result == []

    def test_detects_misplaced_mp4(self, tmp_path: pathlib.Path):
        """Returns the misplaced file when a .mp4 is in raw_videos/ directly."""
        misfile = tmp_path / "oops.mp4"
        misfile.write_bytes(b"fake")
        result = check_video_placement(tmp_path)
        assert misfile in result

    def test_readme_not_flagged_as_misplaced(self, tmp_path: pathlib.Path):
        """The README.md placeholder inside raw_videos/ is not flagged."""
        (tmp_path / "README.md").write_text("placeholder")
        result = check_video_placement(tmp_path)
        assert result == []

    def test_single_video_option_and_validation(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        """run(video_path=...) processes only the selected video and validates its path."""
        raw_root = tmp_path / "raw_videos"
        out_root = tmp_path / "extracted"
        hello_dir = raw_root / "hello"
        hello_dir.mkdir(parents=True)
        v1 = hello_dir / "hello_001.mp4"
        v2 = hello_dir / "hello_002.mp4"
        v1.write_bytes(b"fake1")
        v2.write_bytes(b"fake2")

        processed_calls: list[tuple[pathlib.Path, str, pathlib.Path]] = []
        upserted_rows: list[dict] = []

        def fake_process_video(
            vp: pathlib.Path, gloss: str, out: pathlib.Path = pipeline.OUT_ROOT
        ) -> dict:
            processed_calls.append((vp, gloss, out))
            return {
                "sample_id": vp.stem,
                "gloss": gloss,
                "meaning": "",
                "video_path": str(vp),
                "npz_path": str(out / gloss / f"{vp.stem}.npz"),
                "num_frames": 10,
            }

        monkeypatch.setattr(pipeline, "_process_video", fake_process_video)
        monkeypatch.setattr(pipeline, "upsert_row", lambda row: upserted_rows.append(row))

        # Single-video mode processes only v1
        pipeline.run(raw_root=raw_root, out_root=out_root, video_path=v1)
        assert processed_calls == [(v1, "hello", out_root)]
        assert len(upserted_rows) == 1
        assert upserted_rows[0]["sample_id"] == "hello_001"

        # Full-batch mode (video_path omitted) processes both v1 and v2
        processed_calls.clear()
        upserted_rows.clear()
        pipeline.run(raw_root=raw_root, out_root=out_root)
        assert [c[0] for c in processed_calls] == [v1, v2]
        assert len(upserted_rows) == 2

        # Validation rejects missing, unsupported, misplaced, or outside files
        with pytest.raises(FileNotFoundError):
            pipeline.run(raw_root=raw_root, out_root=out_root, video_path=hello_dir / "missing.mp4")

        bad_ext = hello_dir / "note.txt"
        bad_ext.write_text("x")
        with pytest.raises(ValueError, match="Unsupported video extension"):
            pipeline.run(raw_root=raw_root, out_root=out_root, video_path=bad_ext)

        misplaced = raw_root / "root_video.mp4"
        misplaced.write_bytes(b"fake")
        with pytest.raises(ValueError, match="label folder"):
            pipeline.run(raw_root=raw_root, out_root=out_root, video_path=misplaced)

        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"fake")
        with pytest.raises(ValueError, match="inside"):
            pipeline.run(raw_root=raw_root, out_root=out_root, video_path=outside)


# ---------------------------------------------------------------------------
# TestNpzOutputShape
# ---------------------------------------------------------------------------


class TestNpzOutputShape:
    """The final .npz output arrays have the shapes the contract specifies."""

    def test_npz_arrays_round_trip(self, tmp_path: pathlib.Path):
        """Save and reload a .npz; verify all keys and shapes survive."""
        T = 18
        seq = _random_seq(T=T)
        norm = normalize_sequence(seq)
        vec = build_vec(norm)
        flat_vec = build_flat_vec(norm)
        gloss = "hello"

        npz_path = tmp_path / "hello_001.npz"
        np.savez_compressed(
            npz_path,
            pose_seq=norm.astype(np.float32),
            vec=vec.astype(np.float32),
            flat_vec=flat_vec.astype(np.float32),
            gloss=np.array(gloss),
        )

        data = np.load(npz_path, allow_pickle=True)

        assert data["pose_seq"].shape == (T, NUM_LANDMARKS, NUM_COORDS)
        assert data["vec"].shape == (VEC_LEN,)
        assert data["flat_vec"].shape == (FLAT_VEC_LEN,)
        assert str(data["gloss"]) == gloss

    def test_npz_pose_seq_dtype(self, tmp_path: pathlib.Path):
        """pose_seq stored in .npz is float32."""
        norm = normalize_sequence(_random_seq(T=5))
        path = tmp_path / "t.npz"
        np.savez_compressed(path, pose_seq=norm.astype(np.float32),
                            vec=build_vec(norm), flat_vec=build_flat_vec(norm),
                            gloss=np.array("test"))
        data = np.load(path, allow_pickle=True)
        assert data["pose_seq"].dtype == np.float32

    def test_npz_no_nan_in_outputs(self, tmp_path: pathlib.Path):
        """Normalized .npz arrays contain no NaN values."""
        norm = normalize_sequence(_random_seq(T=8))
        vec = build_vec(norm)
        flat_vec = build_flat_vec(norm)
        assert not np.any(np.isnan(norm))
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isnan(flat_vec))
