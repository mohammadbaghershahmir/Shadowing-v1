"""Style profile registry for CSN-V4 postprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from csn_v4.postprocess.decide import apply_style_proposal, conservative_cleanup_edits
from csn_v4.postprocess.regions import MotifRegion
from csn_v4.postprocess.styles_io import load_miakhi_assets, propose_miakhi_for_regions


@dataclass(frozen=True)
class ProfileInfo:
    name: str
    available: bool
    requires_editable_mask: bool
    reason: str = ""
    decorative: bool = False


ProfileFn = Callable[..., tuple[np.ndarray, np.ndarray, list[str], list[str]]]


def list_profiles() -> list[str]:
    return ["off", "conservative_cleanup", "miakhi"]


def profile_status(name: str, *, styles_root: str | Path | None = None) -> ProfileInfo:
    key = (name or "off").strip().lower()
    if key in ("off", "none", "disabled"):
        return ProfileInfo("off", available=True, requires_editable_mask=False, reason="passthrough")
    if key in ("conservative_cleanup", "cleanup", "conservative"):
        return ProfileInfo(
            "conservative_cleanup",
            available=True,
            requires_editable_mask=False,
            reason="Isolated low-confidence cleanup; not a decorative style.",
            decorative=False,
        )
    if key == "miakhi":
        assets = load_miakhi_assets(styles_root)
        if assets.available:
            return ProfileInfo(
                "miakhi",
                available=True,
                requires_editable_mask=True,
                reason=assets.reason or "Approved miakhi examples/rules loaded.",
                decorative=True,
            )
        return ProfileInfo(
            "miakhi",
            available=False,
            requires_editable_mask=True,
            reason=assets.reason
            or (
                "miakhi is unavailable: no approved examples or rules found. "
                "See Transfer/styles/miakhi/README.md — do not treat cleanup as miakhi."
            ),
            decorative=True,
        )
    return ProfileInfo(
        key,
        available=False,
        requires_editable_mask=True,
        reason=f"Unknown style profile {key!r}. Known: {list_profiles()}",
        decorative=True,
    )


def run_profile(
    name: str,
    *,
    raw_gray: np.ndarray,
    scores: np.ndarray | None,
    regions: list[MotifRegion],
    source_bw: np.ndarray,
    eligible: np.ndarray,
    styles_root: str | Path | None,
    confidence_threshold: float,
    style_influence: float,
    user_style_selected: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], ProfileInfo]:
    info = profile_status(name, styles_root=styles_root)
    if info.name == "off":
        z = np.zeros(raw_gray.shape, dtype=bool)
        return raw_gray.copy(), z, [], [], info
    if not info.available:
        z = np.zeros(raw_gray.shape, dtype=bool)
        return raw_gray.copy(), z, [], [f"profile_unavailable:{info.reason}"], info

    if info.name == "conservative_cleanup":
        g, ch, ed, sk = conservative_cleanup_edits(
            raw_gray,
            scores,
            regions,
            confidence_threshold=confidence_threshold,
        )
        return g, ch, ed, sk, info

    if info.name == "miakhi":
        assets = load_miakhi_assets(styles_root)
        proposal = propose_miakhi_for_regions(
            source_bw, eligible, regions, assets, raw_gray=raw_gray,
        )
        g, ch, ed, sk = apply_style_proposal(
            raw_gray,
            scores,
            regions,
            proposal,
            confidence_threshold=confidence_threshold,
            style_influence=style_influence,
            user_style_selected=user_style_selected,
        )
        return g, ch, ed, sk, info

    z = np.zeros(raw_gray.shape, dtype=bool)
    return raw_gray.copy(), z, [], [f"unhandled:{info.name}"], info
