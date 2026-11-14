"""Unit tests for isl.matcher (database, cosine, DTW, SteadyDetector, SignRecognizer)."""

from pathlib import Path
import numpy as np
import pytest

from isl.features import normalize_hand, summary_vector
from isl.matcher import (
    PoseDatabase,
    SignRecognizer,
    SteadyDetector,
    cosine_sim,
    dtw_distance,
    match_pose,
)
from isl.schema import COORDS, NUM_HANDS, NUM_LANDMARKS, RecognizerConfig, SEQ_LEN


def _make_dummy_sample(
    folder: Path,
    filename: str,
    label: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seq = rng.normal(size=(SEQ_LEN, NUM_HANDS, NUM_LANDMARKS, COORDS)).astype(np.float32)
    for t in range(SEQ_LEN):
        for h in range(NUM_HANDS):
            seq[t, h] = normalize_hand(seq[t, h])
    mask = np.ones((SEQ_LEN, NUM_HANDS), dtype=bool)
    body_pos = np.zeros((SEQ_LEN, NUM_HANDS, COORDS), dtype=np.float32)
    summ = summary_vector(seq, mask)
    np.savez(
        folder / filename,
        sequence=seq,
        body_pos=body_pos,
        mask=mask,
        summary=summ,
        label=label,
    )
    return seq, mask, summ


def test_cosine_sim_identical_and_orthogonal() -> None:
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    zeros = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mat = np.vstack([v1, v2, zeros])

    scores = cosine_sim(v1, mat)
    assert np.isclose(scores[0], 1.0, atol=1e-6)
    assert np.isclose(scores[1], 0.0, atol=1e-6)
    assert np.isclose(scores[2], 0.0, atol=1e-6)


def test_pose_db_and_match_pose(tmp_path: Path) -> None:
    _, _, summ_hello = _make_dummy_sample(tmp_path, "hello_1.npz", "HELLO", seed=1)
    _make_dummy_sample(tmp_path, "thanks_1.npz", "THANKS", seed=2)

    db = PoseDatabase(tmp_path)
    assert len(db) == 2

    res = match_pose(summ_hello, db)
    assert res.label == "HELLO"
    assert res.status == "confident"
    assert res.score >= 0.99


def test_dtw_distance_identical_is_zero() -> None:
    rng = np.random.default_rng(7)
    seq = rng.normal(size=(SEQ_LEN, NUM_HANDS, NUM_LANDMARKS, COORDS)).astype(np.float32)
    dist = dtw_distance(seq, seq, band=0.1)
    assert np.isclose(dist, 0.0, atol=1e-5)


def test_steady_detector_triggers_after_n_frames() -> None:
    detector = SteadyDetector(threshold=0.02, frames=5, ema_alpha=0.5)
    pos = np.array([[0.2, 0.3, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    # First frame initializes position
    assert detector.update(pos) is False
    # Missing frame resets safely without crashing
    assert detector.update(None) is False

    triggered = False
    for _ in range(7):
        triggered = detector.update(pos)
    assert triggered is True


def test_sign_recognizer_end_to_end(tmp_path: Path) -> None:
    seq_hello, mask_hello, _ = _make_dummy_sample(tmp_path, "hello_1.npz", "HELLO", seed=10)
    _make_dummy_sample(tmp_path, "bye_1.npz", "BYE", seed=20)

    db = PoseDatabase(tmp_path)
    cfg = RecognizerConfig(steady_frames=5, cooldown_seconds=2.0)
    recognizer = SignRecognizer(db, config=cfg)

    steady_body = np.array([[0.1, 0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=np.float32)
    event = None
    for t in range(SEQ_LEN):
        out = recognizer.push_frame(
            landmarks=seq_hello[t],
            body_pos=steady_body,
            mask=mask_hello[t],
            timestamp=100.0 + t * 0.03,
        )
        if out is not None:
            event = out
            break

    assert event is not None
    assert event["label"] == "HELLO"
    assert event["method"] in ("cosine", "dtw")


def test_backend_matcher_and_evaluate_cli(tmp_path: Path) -> None:
    import subprocess
    import sys

    from backend.core.matcher import (
        SignRecognizer as BridgeRecognizer,
        SteadyDetector as BridgeDetector,
        dtw_distance as bridge_dtw_distance,
    )

    assert BridgeRecognizer is SignRecognizer
    assert BridgeDetector is SteadyDetector
    assert bridge_dtw_distance is dtw_distance
    seq, _, _ = _make_dummy_sample(tmp_path, "hello_1.npz", "HELLO", seed=10)
    _make_dummy_sample(tmp_path, "hello_2.npz", "HELLO", seed=10)
    _make_dummy_sample(tmp_path, "bye_1.npz", "BYE", seed=20)
    assert bridge_dtw_distance(seq[:, 0], seq[:, 0]) == 0.0

    result = subprocess.run(
        [sys.executable, "-m", "backend.scripts.evaluate", "--data-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Cosine only | Top-1 accuracy: 66.67% | Top-3 accuracy: 66.67%" in result.stderr
    assert "Cosine + DTW fallback | Top-1 accuracy: 66.67%" in result.stderr
    assert "BYE | Top-1: 0.00% | Top-3: 0.00%" in result.stderr


def test_recognition_service_accepts_both_json_frame_shapes(tmp_path: Path) -> None:
    import json

    from backend.core.matcher import RecognitionService

    seq, mask, _ = _make_dummy_sample(tmp_path, "hello.npz", "HELLO", seed=10)
    seq[:, 1] = 0
    mask[:, 1] = False
    np.savez(
        tmp_path / "hello.npz",
        sequence=seq,
        mask=mask,
        body_pos=np.zeros((SEQ_LEN, NUM_HANDS, COORDS), dtype=np.float32),
        summary=summary_vector(seq, mask),
        label="HELLO",
    )
    service = RecognitionService(PoseDatabase(tmp_path, use_cache=False))
    single = service.recognize(seq[:, 0].tolist())
    assert json.loads(json.dumps(single)) == service.recognize(seq.tolist())
    assert single["label"] == "HELLO"
    assert single["status"] == "confident"
    assert single["method"] == "cosine"
    assert single["confidence"] == pytest.approx(1.0)
    assert single["alternatives"][0]["label"] == "HELLO"
