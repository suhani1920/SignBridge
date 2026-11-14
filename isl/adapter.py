"""JSON-friendly recognition of complete landmark clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from isl.features import normalize_hand, resample, resample_mask, summary_vector
from isl.matcher import PoseDatabase, SignRecognizer, match_pose, resolve_uncertain
from isl.schema import COORDS, NUM_HANDS, NUM_LANDMARKS, RecognizerConfig


class RecognitionService:
    """Share a database between clip recognition and independent live streams."""

    def __init__(
        self,
        data_dir: str | Path | PoseDatabase,
        config: RecognizerConfig | None = None,
    ) -> None:
        self.db = data_dir if isinstance(data_dir, PoseDatabase) else PoseDatabase(data_dir)
        self.config = config or RecognizerConfig()

    def new_recognizer(self) -> SignRecognizer:
        """Create a stateful recognizer for one live stream."""
        return SignRecognizer(self.db, config=self.config)

    def recognize(self, frames: ArrayLike) -> dict:
        """Recognize one clip, placing single-hand input in the left-hand slot."""
        sequence = np.asarray(frames, dtype=np.float32)
        if sequence.shape == (0,):
            sequence = sequence.reshape(0, NUM_LANDMARKS, COORDS)
        if sequence.ndim == 3 and sequence.shape[1:] == (NUM_LANDMARKS, COORDS):
            sequence = np.stack((sequence, np.zeros_like(sequence)), axis=1)
        if sequence.ndim != 4 or sequence.shape[1:] != (NUM_HANDS, NUM_LANDMARKS, COORDS):
            raise ValueError("Expected frames with shape (T, 21, 3) or (T, 2, 21, 3)")
        if not np.isfinite(sequence).all():
            raise ValueError("Frame coordinates must be finite")

        mask = np.any(np.abs(sequence) > 1e-8, axis=(2, 3))
        response = {
            "label": None,
            "confidence": 0.0,
            "status": "reject",
            "method": "cosine",
            "alternatives": [],
        }
        if not mask.any():
            return response

        normalized = np.zeros_like(sequence)
        for frame, hand in zip(*np.nonzero(mask)):
            normalized[frame, hand] = normalize_hand(sequence[frame, hand])
        query = resample(normalized)
        query_mask = resample_mask(mask)
        match = match_pose(summary_vector(query, query_mask), self.db, config=self.config)
        response.update(
            confidence=float(match.score),
            status=match.status,
            alternatives=[{"label": c.label, "score": float(c.score)} for c in match.candidates],
        )
        if match.status == "confident":
            response["label"] = match.label
        elif match.status == "uncertain":
            label, distance = resolve_uncertain(
                query, self.db, match.candidates, config=self.config, query_mask=query_mask
            )
            if label is not None:
                response.update(
                    label=label, confidence=1.0 / (1.0 + distance), status="confident", method="dtw"
                )
        return response
