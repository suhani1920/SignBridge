"""
isl/__init__.py — Public API for the SignBridge ISL package (P1 pipeline + P2 recognition engine).
"""

from isl.adapter import RecognitionService
from isl.matcher import (
    PoseDatabase,
    SignRecognizer,
    SteadyDetector,
    cosine_sim,
    dtw_distance,
    match_pose,
    resolve_uncertain,
)

classify_candidates = match_pose

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
