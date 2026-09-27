"""Style-aware postprocessing for CSN-V4 categorical predictions.

Runs *after* model inference on an independent copy of the raw prediction.
Never mutates the raw array in place. Never uses ground-truth targets to
decide editable regions or to tune decisions at test time.
"""
from __future__ import annotations

from csn_v4.postprocess.api import PostprocessConfig, run_postprocess
from csn_v4.postprocess.types import PostprocessReport, PostprocessResult
from csn_v4.postprocess.profiles import list_profiles, profile_status

__all__ = [
    "PostprocessConfig",
    "PostprocessReport",
    "PostprocessResult",
    "run_postprocess",
    "list_profiles",
    "profile_status",
]
