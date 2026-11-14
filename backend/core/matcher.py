"""Compatibility imports for the backend recognition code."""

from isl import (
    PoseDatabase,
    RecognitionService,
    SignRecognizer,
    SteadyDetector,
    classify_candidates,
    cosine_sim,
    dtw_distance,
    match_pose,
    resolve_uncertain,
)

__all__ = [
    "PoseDatabase",
    "cosine_sim",
    "match_pose",
    "classify_candidates",
    "dtw_distance",
    "resolve_uncertain",
    "SteadyDetector",
    "SignRecognizer",
    "RecognitionService",
]
