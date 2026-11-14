"""Recognition engine: PoseDatabase, vectorized cosine matching, DTW, SteadyDetector, and pipeline."""

from __future__ import annotations

from collections import deque
import hashlib
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from isl.features import normalize_hand, resample, resample_mask, summary_vector
from isl.schema import (
    COORDS,
    Candidate,
    MatchResult,
    NUM_HANDS,
    NUM_LANDMARKS,
    RecognizerConfig,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)

_EPS: float = 1e-8

try:
    from numba import njit  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


class PoseDatabase:
    """In-memory database of normalized summary vectors with disk caching and lazy sequence loading."""

    def __init__(self, data_dir: str | Path, use_cache: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache
        self.summaries: NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)
        self.labels: list[str] = []
        self.file_paths: list[Path] = []
        self._sequence_cache: dict[int, tuple[NDArray[np.float32], NDArray[np.bool_]]] = {}
        self._load()

    def __len__(self) -> int:
        return len(self.labels)

    def _cache_path(self, files: list[Path]) -> Path:
        digest = hashlib.sha256()
        for f in files:
            stat = f.stat()
            digest.update(f"{f.name}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
        return self.data_dir / f".posedb_{digest.hexdigest()[:16]}.npz"

    def _load(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")

        npz_files = sorted(
            p for p in self.data_dir.glob("*.npz") if not p.name.startswith(".posedb_")
        )
        if not npz_files:
            logger.warning("No .npz files found in %s", self.data_dir)
            return

        cache_file = self._cache_path(npz_files)
        if self.use_cache and cache_file.exists():
            try:
                with np.load(cache_file, allow_pickle=False) as cached:
                    self.summaries = np.ascontiguousarray(cached["summaries"], dtype=np.float32)
                    self.labels = [str(x) for x in cached["labels"].tolist()]
                    self.file_paths = [self.data_dir / str(x) for x in cached["filenames"].tolist()]
                logger.info("Loaded %d samples from cache %s", len(self.labels), cache_file.name)
                return
            except Exception as exc:
                logger.warning("Cache load failed (%s), rebuilding index.", exc)

        raw_summaries: list[NDArray[np.float32]] = []
        labels: list[str] = []
        valid_paths: list[Path] = []

        for path in npz_files:
            with np.load(path, allow_pickle=False) as data:
                if "summary" not in data or "label" not in data:
                    logger.warning("Skipping %s: missing required keys", path.name)
                    continue
                vec = np.asarray(data["summary"], dtype=np.float32).ravel()
                norm = float(np.linalg.norm(vec))
                if norm > _EPS:
                    vec = vec / norm
                else:
                    vec = np.zeros_like(vec, dtype=np.float32)

                raw_summaries.append(vec)
                labels.append(str(data["label"].item()))
                valid_paths.append(path)

        if not raw_summaries:
            return

        self.summaries = np.ascontiguousarray(np.vstack(raw_summaries), dtype=np.float32)
        self.labels = labels
        self.file_paths = valid_paths

        if self.use_cache:
            try:
                np.savez(
                    cache_file,
                    summaries=self.summaries,
                    labels=np.array(self.labels, dtype=np.str_),
                    filenames=np.array([p.name for p in self.file_paths], dtype=np.str_),
                )
            except OSError as exc:
                logger.warning("Could not write database cache: %s", exc)

        logger.info("Indexed %d pose samples from %s", len(self.labels), self.data_dir)

    def get_sequence(self, index: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Lazy-load and cache (sequence, mask) for sample at index."""
        if index in self._sequence_cache:
            return self._sequence_cache[index]

        path = self.file_paths[index]
        with np.load(path, allow_pickle=False) as data:
            seq = np.ascontiguousarray(data["sequence"], dtype=np.float32)
            if "mask" in data:
                mask = np.ascontiguousarray(data["mask"], dtype=bool)
            else:
                mask = np.ones((seq.shape[0], NUM_HANDS), dtype=bool)

        self._sequence_cache[index] = (seq, mask)
        return seq, mask


def cosine_sim(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Compute vectorized cosine similarity between 1D query and 2D row-matrix.

    Safe against zero-norm vectors (returns 0.0 for zero norms).
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    mat = np.asarray(matrix, dtype=np.float32)

    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != q.shape[0]:
        raise ValueError(f"Shape mismatch: query {q.shape} vs matrix {mat.shape}")

    q_norm = float(np.linalg.norm(q))
    if q_norm < _EPS:
        return np.zeros((mat.shape[0],), dtype=np.float32)

    q_unit = q / q_norm
    row_norms = np.linalg.norm(mat, axis=1)
    safe_norms = np.where(row_norms > _EPS, row_norms, 1.0)
    dots = mat @ q_unit
    scores = np.where(row_norms > _EPS, dots / safe_norms, 0.0)
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def match_pose(
    query_summary: NDArray[np.floating],
    db: PoseDatabase,
    config: RecognizerConfig | None = None,
    accept: float | None = None,
    uncertain: float | None = None,
    top_k: int | None = None,
) -> MatchResult:
    """Match a summary vector against PoseDatabase using cosine similarity and margin rules."""
    cfg = config or RecognizerConfig()
    accept_th = accept if accept is not None else cfg.accept_threshold
    uncertain_th = uncertain if uncertain is not None else cfg.uncertain_threshold
    k = top_k if top_k is not None else cfg.top_k

    if len(db) == 0:
        return MatchResult(label=None, score=0.0, status="reject", candidates=[])

    scores = cosine_sim(query_summary, db.summaries)
    sorted_indices = np.argsort(scores)[::-1]

    # Collect top_k distinct labels (keeping best score per label)
    seen_labels: set[str] = set()
    candidates: list[Candidate] = []
    for idx in sorted_indices:
        idx_int = int(idx)
        lbl = db.labels[idx_int]
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            candidates.append(
                Candidate(label=lbl, score=float(scores[idx_int]), sample_index=idx_int)
            )
            if len(candidates) >= k:
                break

    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else -1.0
    margin = best.score - second_score

    if best.score >= accept_th and margin >= cfg.min_margin:
        return MatchResult(
            label=best.label,
            score=best.score,
            status="confident",
            candidates=candidates,
        )

    if best.score >= uncertain_th:
        return MatchResult(
            label=best.label,
            score=best.score,
            status="uncertain",
            candidates=candidates,
        )

    return MatchResult(
        label=None,
        score=best.score,
        status="reject",
        candidates=candidates,
    )


def _compute_cost_matrix(
    seq_a: NDArray[np.float32],
    seq_b: NDArray[np.float32],
    mask_a: NDArray[np.bool_] | None = None,
    mask_b: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Compute pairwise frame-to-frame Euclidean distance matrix respecting hand masks."""
    # Ensure shape (T, 2, 21, 3)
    a = seq_a[:, None, :, :] if seq_a.ndim == 3 else seq_a
    b = seq_b[:, None, :, :] if seq_b.ndim == 3 else seq_b

    ta, num_h = a.shape[0], a.shape[1]
    tb = b.shape[0]

    if mask_a is None:
        ma = np.ones((ta, num_h), dtype=bool)
    else:
        ma = np.asarray(mask_a, dtype=bool).reshape(ta, num_h)

    if mask_b is None:
        mb = np.ones((tb, num_h), dtype=bool)
    else:
        mb = np.asarray(mask_b, dtype=bool).reshape(tb, num_h)

    total_cost = np.zeros((ta, tb), dtype=np.float32)
    valid_hands = np.zeros((ta, tb), dtype=np.float32)

    for h in range(num_h):
        ha = a[:, h].reshape(ta, -1)  # (Ta, 63)
        hb = b[:, h].reshape(tb, -1)  # (Tb, 63)
        diff = ha[:, None, :] - hb[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (Ta, Tb)

        both_present = ma[:, h][:, None] & mb[:, h][None, :]
        total_cost += np.where(both_present, dist, 0.0)
        valid_hands += both_present.astype(np.float32)

    # Where at least one hand overlaps, average over present hands; otherwise penalize
    fallback_penalty = float(np.max(total_cost)) + 1.0 if np.any(valid_hands > 0) else 1.0
    cost = np.where(valid_hands > 0, total_cost / np.maximum(valid_hands, 1.0), fallback_penalty)
    return cost.astype(np.float32)


def _dtw_dp_numpy(cost: NDArray[np.float32], radius: int) -> float:
    """Sakoe-Chiba banded DTW path distance normalized by warping path length."""
    ta, tb = cost.shape
    inf = np.float32(1e18)
    dp = np.full((ta + 1, tb + 1), inf, dtype=np.float32)
    steps = np.zeros((ta + 1, tb + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, ta + 1):
        center = int(round((i - 1) * (tb / max(ta, 1)))) + 1
        j_start = max(1, center - radius)
        j_end = min(tb, center + radius)
        for j in range(j_start, j_end + 1):
            c_diag = dp[i - 1, j - 1]
            c_up = dp[i - 1, j]
            c_left = dp[i, j - 1]

            if c_diag <= c_up and c_diag <= c_left:
                prev_cost = c_diag
                prev_steps = steps[i - 1, j - 1]
            elif c_up <= c_left:
                prev_cost = c_up
                prev_steps = steps[i - 1, j]
            else:
                prev_cost = c_left
                prev_steps = steps[i, j - 1]

            if prev_cost < inf * 0.5:
                dp[i, j] = cost[i - 1, j - 1] + prev_cost
                steps[i, j] = prev_steps + 1

    path_len = max(int(steps[ta, tb]), 1)
    return float(dp[ta, tb]) / float(path_len)


if _HAS_NUMBA:  # pragma: no cover

    @njit(cache=True)
    def _dtw_dp_numba(cost: NDArray[np.float32], radius: int) -> float:
        ta, tb = cost.shape
        inf = 1e18
        dp = np.full((ta + 1, tb + 1), inf, dtype=np.float32)
        steps = np.zeros((ta + 1, tb + 1), dtype=np.int32)
        dp[0, 0] = 0.0

        for i in range(1, ta + 1):
            center = int(round((i - 1) * (tb / max(ta, 1)))) + 1
            j_start = max(1, center - radius)
            j_end = min(tb, center + radius)
            for j in range(j_start, j_end + 1):
                c_diag = dp[i - 1, j - 1]
                c_up = dp[i - 1, j]
                c_left = dp[i, j - 1]
                if c_diag <= c_up and c_diag <= c_left:
                    prev_cost = c_diag
                    prev_steps = steps[i - 1, j - 1]
                elif c_up <= c_left:
                    prev_cost = c_up
                    prev_steps = steps[i - 1, j]
                else:
                    prev_cost = c_left
                    prev_steps = steps[i, j - 1]

                if prev_cost < inf * 0.5:
                    dp[i, j] = cost[i - 1, j - 1] + prev_cost
                    steps[i, j] = prev_steps + 1

        path_len = max(int(steps[ta, tb]), 1)
        return float(dp[ta, tb]) / float(path_len)


def dtw_distance(
    seq_a: NDArray[np.floating],
    seq_b: NDArray[np.floating],
    band: float = 0.1,
    mask_a: NDArray[np.bool_] | None = None,
    mask_b: NDArray[np.bool_] | None = None,
) -> float:
    """Compute Sakoe-Chiba banded DTW distance normalized by path length.

    Supports sequences of shape (T, 2, 21, 3) or (T, 21, 3) and optional hand masks.
    """
    a = np.asarray(seq_a, dtype=np.float32)
    b = np.asarray(seq_b, dtype=np.float32)

    if a.ndim not in (3, 4) or b.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D sequences, got {a.shape} and {b.shape}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")

    cost = _compute_cost_matrix(a, b, mask_a=mask_a, mask_b=mask_b)
    max_len = max(a.shape[0], b.shape[0])
    radius = max(1, int(np.ceil(band * max_len)))

    if _HAS_NUMBA:  # pragma: no cover
        return _dtw_dp_numba(cost, radius)
    return _dtw_dp_numpy(cost, radius)


def resolve_uncertain(
    query_seq: NDArray[np.floating],
    db: PoseDatabase,
    candidates: list[Candidate],
    config: RecognizerConfig | None = None,
    query_mask: NDArray[np.bool_] | None = None,
) -> tuple[str | None, float]:
    """Refine uncertain top_k candidates using DTW distance against stored sequences."""
    cfg = config or RecognizerConfig()
    if not candidates or len(db) == 0:
        return None, float("inf")

    target_labels = {c.label for c in candidates[: cfg.top_k]}
    best_label: str | None = None
    best_dist: float = float("inf")

    for idx, lbl in enumerate(db.labels):
        if lbl not in target_labels:
            continue
        ref_seq, ref_mask = db.get_sequence(idx)
        dist = dtw_distance(
            query_seq,
            ref_seq,
            band=cfg.dtw_band,
            mask_a=query_mask,
            mask_b=ref_mask,
        )
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    if best_label is not None and best_dist <= cfg.dtw_threshold:
        return best_label, best_dist
    return None, best_dist


class SteadyDetector:
    """Detects when raw body-relative wrist motion settles below threshold for N frames."""

    def __init__(
        self,
        threshold: float = 0.015,
        frames: int = 15,
        ema_alpha: float = 0.4,
    ) -> None:
        self.threshold = threshold
        self.frames = frames
        self.ema_alpha = ema_alpha
        self._prev_pos: NDArray[np.float32] | None = None
        self._ema_vel: float | None = None
        self._steady_count: int = 0

    def reset(self) -> None:
        """Reset velocity state and consecutive frame counter."""
        self._prev_pos = None
        self._ema_vel = None
        self._steady_count = 0

    def update(self, body_pos_frame: NDArray[np.floating] | None) -> bool:
        """Update detector with a single frame of body-relative wrist coords (2, 3)."""
        if body_pos_frame is None:
            self._steady_count = 0
            return False

        pos = np.asarray(body_pos_frame, dtype=np.float32)
        if pos.shape != (NUM_HANDS, COORDS) or not np.isfinite(pos).all():
            self._steady_count = 0
            return False

        # Missing hands are represented as all zeros
        hand_present = np.linalg.norm(pos, axis=1) > _EPS
        if not np.any(hand_present):
            self._prev_pos = None
            self._steady_count = 0
            return False

        if self._prev_pos is None:
            self._prev_pos = pos.copy()
            self._steady_count = 0
            return False

        prev_present = np.linalg.norm(self._prev_pos, axis=1) > _EPS
        both_valid = hand_present & prev_present

        if not np.any(both_valid):
            self._prev_pos = pos.copy()
            self._steady_count = 0
            return False

        diffs = np.linalg.norm(pos[both_valid] - self._prev_pos[both_valid], axis=1)
        inst_vel = float(np.mean(diffs))
        self._prev_pos = pos.copy()

        if self._ema_vel is None:
            self._ema_vel = inst_vel
        else:
            self._ema_vel = self.ema_alpha * inst_vel + (1.0 - self.ema_alpha) * self._ema_vel

        if self._ema_vel < self.threshold:
            self._steady_count += 1
        else:
            self._steady_count = 0

        return self._steady_count >= self.frames


class SignRecognizer:
    """Real-time sign recognition pipeline tying buffering, SteadyDetector, cosine, and DTW."""

    def __init__(self, db: PoseDatabase, config: RecognizerConfig | None = None) -> None:
        self.db = db
        self.config = config or RecognizerConfig()
        self.detector = SteadyDetector(
            threshold=self.config.steady_velocity_threshold,
            frames=self.config.steady_frames,
            ema_alpha=self.config.ema_alpha,
        )
        self._seq_buffer: deque[NDArray[np.float32]] = deque(maxlen=self.config.ring_buffer_size)
        self._mask_buffer: deque[NDArray[np.bool_]] = deque(maxlen=self.config.ring_buffer_size)
        self._last_emit_time: float = -float("inf")
        self._last_emitted_label: str | None = None

    def reset(self) -> None:
        """Clear frame buffers and detector state."""
        self._seq_buffer.clear()
        self._mask_buffer.clear()
        self.detector.reset()

    def push_frame(
        self,
        landmarks: NDArray[np.floating] | None,
        body_pos: NDArray[np.floating] | None = None,
        mask: NDArray[np.bool_] | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any] | None:
        """Ingest one frame of hand landmarks and emit recognition event when steady."""
        now = timestamp if timestamp is not None else time.monotonic()

        if landmarks is None:
            norm_frame = np.zeros((NUM_HANDS, NUM_LANDMARKS, COORDS), dtype=np.float32)
            frame_mask = np.zeros((NUM_HANDS,), dtype=bool)
        else:
            raw = np.asarray(landmarks, dtype=np.float32)
            if raw.shape != (NUM_HANDS, NUM_LANDMARKS, COORDS):
                raise ValueError(
                    f"Expected landmarks shape {(NUM_HANDS, NUM_LANDMARKS, COORDS)}, got {raw.shape}"
                )
            norm_frame = np.zeros_like(raw)
            inferred_mask = np.zeros((NUM_HANDS,), dtype=bool)
            for h in range(NUM_HANDS):
                if mask is not None and not bool(mask[h]):
                    continue
                if np.allclose(raw[h], 0.0, atol=_EPS):
                    continue
                norm_frame[h] = normalize_hand(raw[h])
                inferred_mask[h] = True
            frame_mask = np.asarray(mask, dtype=bool) if mask is not None else inferred_mask

        self._seq_buffer.append(norm_frame)
        self._mask_buffer.append(frame_mask)

        # Fallback body_pos from wrists if caller did not supply shoulder-scaled body_pos
        if body_pos is None and landmarks is not None:
            raw_arr = np.asarray(landmarks, dtype=np.float32)
            body_pos = np.stack([raw_arr[0, 0], raw_arr[1, 0]], axis=0)

        is_steady = self.detector.update(body_pos)
        if not is_steady or len(self._seq_buffer) < 2:
            return None

        seq_arr = np.stack(list(self._seq_buffer), axis=0)
        mask_arr = np.stack(list(self._mask_buffer), axis=0)

        resampled_seq = resample(seq_arr, target_len=SEQ_LEN)
        resampled_mask = resample_mask(mask_arr, target_len=SEQ_LEN)
        query_summary = summary_vector(resampled_seq, resampled_mask)

        match = match_pose(query_summary, self.db, config=self.config)
        method: str | None = None
        chosen_label: str | None = None
        chosen_score: float = match.score

        if match.status == "confident" and match.label is not None:
            chosen_label = match.label
            method = "cosine"
        elif match.status == "uncertain":
            dtw_label, dtw_dist = resolve_uncertain(
                resampled_seq,
                self.db,
                match.candidates,
                config=self.config,
                query_mask=resampled_mask,
            )
            if dtw_label is not None:
                chosen_label = dtw_label
                chosen_score = float(1.0 / (1.0 + dtw_dist))
                method = "dtw"

        if chosen_label is None or method is None:
            return None

        # Cooldown check to prevent duplicate emissions of the same sign
        if (
            chosen_label == self._last_emitted_label
            and (now - self._last_emit_time) < self.config.cooldown_seconds
        ):
            return None

        self._last_emit_time = now
        self._last_emitted_label = chosen_label
        self.detector.reset()

        return {
            "label": chosen_label,
            "score": round(chosen_score, 4),
            "method": method,
            "timestamp": now,
        }