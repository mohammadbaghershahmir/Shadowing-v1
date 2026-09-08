"""Typed structures for validation, splits, and crop manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _path_str(path: Path | str) -> str:
    """Serialize paths with forward slashes for JSON portability on Windows."""
    return Path(path).as_posix()


@dataclass(frozen=True)
class DiscoveredFile:
    stem: str
    path: Path


@dataclass
class PairPaths:
    stem: str
    input_path: Path
    target_path: Path | None = None


@dataclass
class ColorHistogram:
    """Sparse RGB histogram as colour -> count."""

    counts: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def color_key(rgb: tuple[int, int, int]) -> str:
        return f"{rgb[0]},{rgb[1]},{rgb[2]}"

    def add(self, rgb: tuple[int, int, int], count: int = 1) -> None:
        key = self.color_key(rgb)
        self.counts[key] = self.counts.get(key, 0) + int(count)

    def to_dict(self) -> dict[str, int]:
        return dict(self.counts)


@dataclass
class ValidationResult:
    stem: str
    input_path: str
    target_path: str
    width: int
    height: int
    input_histogram: dict[str, int]
    target_histogram: dict[str, int]
    candidate_pixel_count: int
    class_200_count: int
    class_150_count: int
    class_100_count: int
    unexpected_input_colors: dict[str, int]
    unexpected_target_colors: dict[str, int]
    dimension_mismatch: bool
    candidate_target_mismatch_count: int
    black_white_mismatch_count: int
    alpha_problem: bool
    status: str
    reason: str
    no_candidate_pixels: bool = False

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "input_path": self.input_path,
            "target_path": self.target_path,
            "width": self.width,
            "height": self.height,
            "input_histogram": _dict_to_compact(self.input_histogram),
            "target_histogram": _dict_to_compact(self.target_histogram),
            "candidate_pixel_count": self.candidate_pixel_count,
            "class_200_count": self.class_200_count,
            "class_150_count": self.class_150_count,
            "class_100_count": self.class_100_count,
            "unexpected_input_colors": _dict_to_compact(self.unexpected_input_colors),
            "unexpected_target_colors": _dict_to_compact(
                self.unexpected_target_colors
            ),
            "dimension_mismatch": self.dimension_mismatch,
            "candidate_target_mismatch_count": self.candidate_target_mismatch_count,
            "black_white_mismatch_count": self.black_white_mismatch_count,
            "alpha_problem": self.alpha_problem,
            "no_candidate_pixels": self.no_candidate_pixels,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class ImageRecord:
    """Full-image entry for train/val/test image manifests."""

    schema_version: int
    source_id: str
    stem: str
    input_path: str
    target_path: str
    width: int
    height: int
    split: str
    class_counts: dict[str, int]
    candidate_pixel_count: int
    group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = int(self.schema_version)
        data["width"] = int(self.width)
        data["height"] = int(self.height)
        data["candidate_pixel_count"] = int(self.candidate_pixel_count)
        data["class_counts"] = {k: int(v) for k, v in self.class_counts.items()}
        return data


@dataclass
class CropRecord:
    """Crop manifest entry (lazy loading coordinates)."""

    schema_version: int
    sample_id: str
    source_id: str
    input_path: str
    target_path: str
    split: str
    image_width: int
    image_height: int
    x: int
    y: int
    crop_size: int
    valid_margin: int
    sampling_strategy: str
    candidate_pixels: int
    candidate_pixels_in_valid_center: int
    class_counts: dict[str, int]
    boundary_pixels: int
    component_count: int
    normalized_crop_box: list[float]
    is_tiny_component: bool = False
    group_id: str | None = None
    transform_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "input_path": self.input_path,
            "target_path": self.target_path,
            "split": self.split,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "x": int(self.x),
            "y": int(self.y),
            "crop_size": int(self.crop_size),
            "valid_margin": int(self.valid_margin),
            "sampling_strategy": self.sampling_strategy,
            "candidate_pixels": int(self.candidate_pixels),
            "candidate_pixels_in_valid_center": int(
                self.candidate_pixels_in_valid_center
            ),
            "class_counts": {k: int(v) for k, v in self.class_counts.items()},
            "boundary_pixels": int(self.boundary_pixels),
            "component_count": int(self.component_count),
            "normalized_crop_box": [float(v) for v in self.normalized_crop_box],
            "is_tiny_component": bool(self.is_tiny_component),
            "group_id": self.group_id,
            "transform_id": int(self.transform_id),
        }


@dataclass
class CropSamplingStats:
    """Counters for crop generation diagnostics."""

    generated: int = 0
    accepted: int = 0
    rejected_no_candidate: int = 0
    rejected_duplicate: int = 0
    rejected_high_iou: int = 0
    tiny_component_crops: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated": int(self.generated),
            "accepted": int(self.accepted),
            "rejected_no_candidate": int(self.rejected_no_candidate),
            "rejected_duplicate": int(self.rejected_duplicate),
            "rejected_high_iou": int(self.rejected_high_iou),
            "tiny_component_crops": int(self.tiny_component_crops),
            "by_strategy": {k: int(v) for k, v in self.by_strategy.items()},
        }


def _dict_to_compact(d: dict[str, int]) -> str:
    if not d:
        return ""
    parts = [f"{k}:{v}" for k, v in sorted(d.items())]
    return ";".join(parts)


def path_to_str(path: Path | str) -> str:
    return _path_str(path)
