"""Unit tests for isl.features."""

import numpy as np
import pytest

from isl.features import normalize_hand, resample, summary_vector
from isl.schema import COORDS, NUM_HANDS, NUM_LANDMARKS, SEQ_LEN, SUMMARY_DIM


def test_normalize_hand_wrist_origin_and_unit_scale() -> None:
    rng = np.random.default_rng(42)
    hand = rng.normal(loc=5.0, scale=2.0, size=(NUM_LANDMARKS, COORDS)).astype(np.float32)
    hand[0] = np.array([10.0, 20.0, 30.0], dtype=np.float32)  # wrist
    hand[12] = np.array([13.0, 24.0, 30.0], dtype=np.float32)  # distance = 5.0

    normed = normalize_hand(hand)
    assert normed.shape == (NUM_LANDMARKS, COORDS)
    assert np.allclose(normed[0], 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(normed[12]), 1.0, atol=1e-6)


def test_normalize_hand_zero_guard() -> None:
    zeros = np.zeros((NUM_LANDMARKS, COORDS), dtype=np.float32)
    out = normalize_hand(zeros)
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_resample_fixed_length() -> None:
    seq = np.linspace(0.0, 10.0, num=20 * 2 * 21 * 3, dtype=np.float32).reshape(20, 2, 21, 3)
    out = resample(seq, target_len=SEQ_LEN)
    assert out.shape == (SEQ_LEN, NUM_HANDS, NUM_LANDMARKS, COORDS)
    assert np.allclose(out[0], seq[0])
    assert np.allclose(out[-1], seq[-1])


def test_summary_vector_shape_and_masking() -> None:
    seq = np.ones((SEQ_LEN, NUM_HANDS, NUM_LANDMARKS, COORDS), dtype=np.float32)
    mask = np.zeros((SEQ_LEN, NUM_HANDS), dtype=bool)
    vec = summary_vector(seq, mask)
    assert vec.shape == (SUMMARY_DIM,)
    assert np.all(vec == 0.0)