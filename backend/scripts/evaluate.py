"""Compare cosine rankings with and without DTW fallback using LOOCV."""

from __future__ import annotations

import argparse
from collections import Counter
import logging
from pathlib import Path

from backend.core.matcher import PoseDatabase, match_pose, resolve_uncertain
from isl.schema import RecognizerConfig

logger = logging.getLogger(__name__)


class _TrainingFold(PoseDatabase):
    """Present only training samples while sharing the database's sequence cache."""

    def __init__(self, db: PoseDatabase, held_out: int) -> None:
        self._db = db
        self._indices = [i for i in range(len(db)) if i != held_out]
        self.summaries = db.summaries[self._indices]
        self.labels = [db.labels[i] for i in self._indices]

    def get_sequence(self, index: int):
        return self._db.get_sequence(self._indices[index])


def evaluate(data_dir: Path) -> None:
    """Log ranked accuracy overall and for each true sign."""
    if not data_dir.is_dir():
        logger.warning("Dataset directory is missing or is not a directory: %s", data_dir)
        return

    db = PoseDatabase(data_dir, use_cache=False)
    if len(db) < 2:
        logger.warning("LOOCV requires at least two samples; found %d in %s", len(db), data_dir)
        return

    config = RecognizerConfig()
    totals = Counter(db.labels)
    methods = ("Cosine only", "Cosine + DTW fallback")
    top1 = {method: Counter() for method in methods}
    top3 = {method: Counter() for method in methods}

    for held_out, actual in enumerate(db.labels):
        training = _TrainingFold(db, held_out)
        match = match_pose(db.summaries[held_out], training, config=config)
        cosine = [candidate.label for candidate in match.candidates]
        refined = cosine
        if match.status == "uncertain":
            sequence, mask = db.get_sequence(held_out)
            label, _ = resolve_uncertain(
                sequence, training, match.candidates, config=config, query_mask=mask
            )
            if label is not None:
                refined = [label, *(candidate for candidate in cosine if candidate != label)]

        for method, ranking in zip(methods, (cosine, refined)):
            top1[method][actual] += actual in ranking[:1]
            top3[method][actual] += actual in ranking[:3]

    for method in methods:
        logger.info(
            "%s | Top-1 accuracy: %.2f%% | Top-3 accuracy: %.2f%% (%d samples)",
            method, 100 * top1[method].total() / len(db), 100 * top3[method].total() / len(db),
            len(db),
        )
        for label, count in sorted(totals.items()):
            logger.info(
                "%s | %s | Top-1: %.2f%% | Top-3: %.2f%% (%d samples)",
                method, label, 100 * top1[method][label] / count,
                100 * top3[method][label] / count, count,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/extracted"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(args.data_dir)


if __name__ == "__main__":
    main()
