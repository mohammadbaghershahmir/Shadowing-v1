"""Datatypes for CSN-V4 postprocessing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PostprocessReport:
    profile: str
    available: bool
    ran: bool
    reason: str = ""
    changed_pixels: int = 0
    regions_total: int = 0
    regions_edited: list[str] = field(default_factory=list)
    regions_skipped: list[str] = field(default_factory=list)
    style_influence: float = 0.0
    confidence_threshold: float = 0.0
    required_editable_mask: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "available": self.available,
            "ran": self.ran,
            "reason": self.reason,
            "changed_pixels": int(self.changed_pixels),
            "regions_total": int(self.regions_total),
            "regions_edited": list(self.regions_edited),
            "regions_skipped": list(self.regions_skipped),
            "style_influence": float(self.style_influence),
            "confidence_threshold": float(self.confidence_threshold),
            "required_editable_mask": bool(self.required_editable_mask),
            "extras": dict(self.extras),
        }

    def summary_line(self) -> str:
        status = "ran" if self.ran else ("unavailable" if not self.available else "skipped")
        return (
            f"style={self.profile} ({status}) changed={self.changed_pixels} "
            f"edited={len(self.regions_edited)} skipped={len(self.regions_skipped)}"
            + (f" | {self.reason}" if self.reason else "")
        )


@dataclass
class PostprocessResult:
    """Repaired categorical gray image + change map + report.

    ``raw_gray`` is an independent copy of the input prediction (never aliased
    to the caller's array). ``repaired_gray`` is a separate array.
    """

    repaired_gray: Any  # np.ndarray uint8
    change_map: Any  # np.ndarray bool HxW
    report: PostprocessReport
    raw_gray: Any  # np.ndarray uint8 — untouched copy of the raw prediction
