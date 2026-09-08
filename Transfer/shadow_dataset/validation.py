"""Strict pair validation against the pixel contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shadow_dataset.constants import (
    IGNORE_INDEX,
    INDEX_100,
    INDEX_150,
    INDEX_200,
    INPUT_ALLOWED_COLORS,
    OUTLINE,
    RGB_VALUE_TO_CLASS,
    STATUS_INVALID,
    STATUS_VALID,
    STATUS_WARNING,
    TARGET_ALLOWED_COLORS,
    TARGET_REGION,
    UNSHADED,
)
from shadow_dataset.image_io import ImageLoadError, load_rgb_and_validate_alpha
from shadow_dataset.types import ColorHistogram, PairPaths, ValidationResult, path_to_str

LOGGER = logging.getLogger(__name__)


def _histogram(rgb: np.ndarray) -> ColorHistogram:
    """Build a sparse colour histogram for an HxWx3 uint8 image."""
    flat = rgb.reshape(-1, 3)
    packed = (
        flat[:, 0].astype(np.uint32) << 16
        | flat[:, 1].astype(np.uint32) << 8
        | flat[:, 2].astype(np.uint32)
    )
    unique, counts = np.unique(packed, return_counts=True)
    hist = ColorHistogram()
    for value, count in zip(unique.tolist(), counts.tolist()):
        r = (value >> 16) & 0xFF
        g = (value >> 8) & 0xFF
        b = value & 0xFF
        hist.add((r, g, b), int(count))
    return hist


def _unexpected_colors(
    hist: ColorHistogram,
    allowed: frozenset[tuple[int, int, int]],
) -> dict[str, int]:
    unexpected: dict[str, int] = {}
    for key, count in hist.counts.items():
        parts = key.split(",")
        rgb = (int(parts[0]), int(parts[1]), int(parts[2]))
        if rgb not in allowed:
            unexpected[key] = int(count)
    return unexpected


def encode_target_labels(
    input_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    """Map target RGB to categorical labels {0,1,2,-100} using TARGET_REGION.

    Labels are assigned only where input is magenta (TARGET_REGION).
    Outside that mask the label is IGNORE_INDEX (-100).
    """
    if input_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Shape mismatch for label encoding: input {input_rgb.shape} "
            f"vs target {target_rgb.shape}"
        )
    h, w, _ = input_rgb.shape
    labels = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
    candidate = (
        (input_rgb[:, :, 0] == TARGET_REGION[0])
        & (input_rgb[:, :, 1] == TARGET_REGION[1])
        & (input_rgb[:, :, 2] == TARGET_REGION[2])
    )
    equal_rgb = (
        (target_rgb[:, :, 0] == target_rgb[:, :, 1])
        & (target_rgb[:, :, 1] == target_rgb[:, :, 2])
    )
    index_vals = target_rgb[:, :, 0]
    for value, label in RGB_VALUE_TO_CLASS.items():
        mask = candidate & equal_rgb & (index_vals == value)
        labels[mask] = label
    return labels


def validate_pair(pair: PairPaths) -> ValidationResult:
    """Validate one input/target pair under the full pixel contract."""
    assert pair.target_path is not None
    input_path = path_to_str(pair.input_path)
    target_path = path_to_str(pair.target_path)

    reasons: list[str] = []
    status = STATUS_VALID
    alpha_problem = False
    dimension_mismatch = False
    width = 0
    height = 0
    input_hist: dict[str, int] = {}
    target_hist: dict[str, int] = {}
    unexpected_input: dict[str, int] = {}
    unexpected_target: dict[str, int] = {}
    candidate_count = 0
    class_200 = 0
    class_150 = 0
    class_100 = 0
    mismatch_count = 0
    bw_mismatch = 0
    no_candidates = False

    try:
        input_rgb, input_alpha_bad = load_rgb_and_validate_alpha(pair.input_path)
        target_rgb, target_alpha_bad = load_rgb_and_validate_alpha(pair.target_path)
    except ImageLoadError as exc:
        return _invalid_load_result(pair.stem, input_path, target_path, f"Failed to load image: {exc}")
    except OSError as exc:
        return _invalid_load_result(
            pair.stem, input_path, target_path, f"I/O error loading image: {exc}"
        )

    if input_alpha_bad or target_alpha_bad:
        alpha_problem = True
        reasons.append("Non-opaque alpha channel detected (all alpha must be 255)")
        status = STATUS_INVALID

    ih, iw = input_rgb.shape[:2]
    th, tw = target_rgb.shape[:2]
    if (ih, iw) != (th, tw):
        dimension_mismatch = True
        reasons.append(
            f"Dimension mismatch: input {iw}x{ih} vs target {tw}x{th}"
        )
        status = STATUS_INVALID
        width, height = iw, ih
    else:
        width, height = iw, ih

    in_hist_obj = _histogram(input_rgb)
    tg_hist_obj = _histogram(target_rgb)
    input_hist = in_hist_obj.to_dict()
    target_hist = tg_hist_obj.to_dict()

    unexpected_input = _unexpected_colors(in_hist_obj, INPUT_ALLOWED_COLORS)
    unexpected_target = _unexpected_colors(tg_hist_obj, TARGET_ALLOWED_COLORS)

    if unexpected_input:
        reasons.append(
            f"Unexpected input colours: {_format_unexpected(unexpected_input)}"
        )
        status = STATUS_INVALID
    if unexpected_target:
        reasons.append(
            f"Unexpected target colours: {_format_unexpected(unexpected_target)}"
        )
        status = STATUS_INVALID

    if not dimension_mismatch:
        candidate = _match_color(input_rgb, TARGET_REGION)
        outline_in = _match_color(input_rgb, OUTLINE)
        white_in = _match_color(input_rgb, UNSHADED)

        outline_tg = _match_color(target_rgb, OUTLINE)
        white_tg = _match_color(target_rgb, UNSHADED)
        idx_200 = _match_color(target_rgb, INDEX_200)
        idx_150 = _match_color(target_rgb, INDEX_150)
        idx_100 = _match_color(target_rgb, INDEX_100)
        index_any = idx_200 | idx_150 | idx_100

        candidate_count = int(candidate.sum())
        class_200 = int(idx_200.sum())
        class_150 = int(idx_150.sum())
        class_100 = int(idx_100.sum())

        cand_not_index = candidate & ~index_any
        index_not_cand = index_any & ~candidate
        mismatch_count = int(cand_not_index.sum() + index_not_cand.sum())
        if mismatch_count > 0:
            reasons.append(
                f"TARGET_REGION/class-index spatial mismatch on {mismatch_count} pixels"
            )
            status = STATUS_INVALID

        black_bad = outline_in & ~outline_tg
        white_bad = white_in & ~white_tg
        bw_mismatch = int(black_bad.sum() + white_bad.sum())
        if bw_mismatch > 0:
            reasons.append(
                f"Black/white consistency failed on {bw_mismatch} pixels"
            )
            status = STATUS_INVALID

        if candidate_count == 0 and status != STATUS_INVALID:
            no_candidates = True
            reasons.append("No TARGET_REGION (magenta) pixels")
            status = STATUS_WARNING
        elif candidate_count == 0:
            no_candidates = True
            reasons.append("No TARGET_REGION (magenta) pixels")

    reason = "; ".join(reasons) if reasons else "OK"
    return ValidationResult(
        stem=pair.stem,
        input_path=input_path,
        target_path=target_path,
        width=int(width),
        height=int(height),
        input_histogram=input_hist,
        target_histogram=target_hist,
        candidate_pixel_count=int(candidate_count),
        class_200_count=int(class_200),
        class_150_count=int(class_150),
        class_100_count=int(class_100),
        unexpected_input_colors=unexpected_input,
        unexpected_target_colors=unexpected_target,
        dimension_mismatch=bool(dimension_mismatch),
        candidate_target_mismatch_count=int(mismatch_count),
        black_white_mismatch_count=int(bw_mismatch),
        alpha_problem=bool(alpha_problem),
        status=status,
        reason=reason,
        no_candidate_pixels=bool(no_candidates),
    )


def validate_pairs(
    pairs: list[PairPaths],
    *,
    progress: bool = True,
) -> list[ValidationResult]:
    """Validate all pairs; optionally show a progress bar."""
    results: list[ValidationResult] = []
    iterator: Any = pairs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(pairs, desc="Validating pairs", unit="pair")
        except ImportError:
            iterator = pairs

    for pair in iterator:
        result = validate_pair(pair)
        if result.status == STATUS_INVALID:
            LOGGER.warning("INVALID %s: %s", result.stem, result.reason)
        elif result.status == STATUS_WARNING:
            LOGGER.warning("WARNING %s: %s", result.stem, result.reason)
        results.append(result)
    return results


def _invalid_load_result(
    stem: str, input_path: str, target_path: str, reason: str
) -> ValidationResult:
    return ValidationResult(
        stem=stem,
        input_path=input_path,
        target_path=target_path,
        width=0,
        height=0,
        input_histogram={},
        target_histogram={},
        candidate_pixel_count=0,
        class_200_count=0,
        class_150_count=0,
        class_100_count=0,
        unexpected_input_colors={},
        unexpected_target_colors={},
        dimension_mismatch=False,
        candidate_target_mismatch_count=0,
        black_white_mismatch_count=0,
        alpha_problem=False,
        status=STATUS_INVALID,
        reason=reason,
        no_candidate_pixels=False,
    )


def _match_color(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return (
        (rgb[:, :, 0] == color[0])
        & (rgb[:, :, 1] == color[1])
        & (rgb[:, :, 2] == color[2])
    )


def _format_unexpected(unexpected: dict[str, int], limit: int = 8) -> str:
    items = sorted(unexpected.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"({k})={v}" for k, v in items[:limit]]
    if len(items) > limit:
        parts.append(f"...+{len(items) - limit} more")
    return ", ".join(parts)


def is_usable_for_dataset(result: ValidationResult) -> bool:
    """Valid pairs enter splits; warnings (no TARGET_REGION) are excluded."""
    return result.status == STATUS_VALID
