"""End-to-end dataset preparation pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadow_dataset.constants import (
    CROP_IOU_DEDUP_THRESHOLD_V2,
    DEFAULT_CROP_SIZE,
    DEFAULT_EDGE_BAND,
    DEFAULT_GRID_STRIDE,
    DEFAULT_INFERENCE_STRIDE_HINT,
    DEFAULT_MAX_CROPS_PER_IMAGE,
    DEFAULT_MIN_CROPS_PER_IMAGE,
    DEFAULT_PREVIEW_COUNT,
    DEFAULT_SEED,
    DEFAULT_TARGET_TRAIN_CROPS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALID_MARGIN,
    DEFAULT_VAL_RATIO,
    SCHEMA_VERSION,
    STATUS_INVALID,
    STATUS_WARNING,
    STRATEGY_WEIGHTS_V2,
    class_to_rgb_config,
)
from shadow_dataset.crops import (
    compute_train_rare_class_order,
    desired_crop_count,
    sample_crops_for_image,
)
from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    clear_generated_outputs,
    ensure_output_dir,
)
from shadow_dataset.pairing import discover_images, pair_by_stem
from shadow_dataset.previews import write_previews
from shadow_dataset.splitting import assign_splits, build_image_records, splits_summary
from shadow_dataset.stats import compute_dataset_stats, format_summary
from shadow_dataset.types import CropRecord, CropSamplingStats, ImageRecord
from shadow_dataset.validation import is_usable_for_dataset, validate_pairs

LOGGER = logging.getLogger(__name__)

GENERATED_FILES = [
    "dataset_config.json",
    "dataset_stats.json",
    "pair_readiness_audit.json",
    "validation_report.csv",
    "invalid_pairs.csv",
    "missing_pairs.csv",
    "duplicate_stems.csv",
    "splits.json",
    "train_images.jsonl",
    "val_images.jsonl",
    "test_images.jsonl",
    "train_crops.jsonl",
    "val_crops.jsonl",
    "previews",
]

VALIDATION_CSV_FIELDS = [
    "stem",
    "input_path",
    "target_path",
    "width",
    "height",
    "input_histogram",
    "target_histogram",
    "candidate_pixel_count",
    "class_200_count",
    "class_150_count",
    "class_100_count",
    "unexpected_input_colors",
    "unexpected_target_colors",
    "dimension_mismatch",
    "candidate_target_mismatch_count",
    "black_white_mismatch_count",
    "alpha_problem",
    "no_candidate_pixels",
    "status",
    "reason",
]


@dataclass
class PipelineConfig:
    input_dir: Path
    target_dir: Path
    output_dir: Path
    crop_size: int = DEFAULT_CROP_SIZE
    valid_margin: int = DEFAULT_VALID_MARGIN
    inference_stride_hint: int = DEFAULT_INFERENCE_STRIDE_HINT
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    min_crops_per_image: int = DEFAULT_MIN_CROPS_PER_IMAGE
    max_crops_per_image: int = DEFAULT_MAX_CROPS_PER_IMAGE
    preview_count: int = DEFAULT_PREVIEW_COUNT
    seed: int = DEFAULT_SEED
    group_regex: str | None = None
    overwrite: bool = False
    val_crops_per_image: int | None = None
    include_spatial_grid: bool = False
    allow_padded_origins: bool = False
    bake_dihedral: bool = False
    merge_train_val_crops: bool = False
    grid_stride: int = DEFAULT_GRID_STRIDE
    edge_band: int = DEFAULT_EDGE_BAND
    target_train_crops: int | None = None
    iou_dedup_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "input_dir": Path(self.input_dir).as_posix(),
            "target_dir": Path(self.target_dir).as_posix(),
            "output_dir": Path(self.output_dir).as_posix(),
            "class_to_rgb": class_to_rgb_config(),
            "ignore_index": -100,
            "semantic_note": (
                "Target RGB values INDEX_200/INDEX_150/INDEX_100 are fixed categorical "
                "colour identifiers. Numeric magnitudes have no ordinal, lighting, "
                "opacity, geometric, or distance meaning."
            ),
            "crop_size": int(self.crop_size),
            "valid_margin": int(self.valid_margin),
            "valid_center": int(self.crop_size - 2 * self.valid_margin),
            "inference_stride_hint": int(self.inference_stride_hint),
            "train_ratio": float(self.train_ratio),
            "val_ratio": float(self.val_ratio),
            "test_ratio": float(self.test_ratio),
            "min_crops_per_image": int(self.min_crops_per_image),
            "max_crops_per_image": int(self.max_crops_per_image),
            "preview_count": int(self.preview_count),
            "seed": int(self.seed),
            "group_regex": self.group_regex,
            "overwrite": bool(self.overwrite),
            "val_crops_per_image": self.val_crops_per_image,
            "include_spatial_grid": bool(self.include_spatial_grid),
            "allow_padded_origins": bool(self.allow_padded_origins),
            "bake_dihedral": bool(self.bake_dihedral),
            "merge_train_val_crops": bool(self.merge_train_val_crops),
            "grid_stride": int(self.grid_stride),
            "edge_band": int(self.edge_band),
            "target_train_crops": self.target_train_crops,
            "iou_dedup_threshold": self.iou_dedup_threshold,
        }


def _write_pair_readiness_audit(
    output_dir: Path,
    *,
    n_input: int,
    n_target: int,
    n_pairs: int,
    missing_rows: list,
    duplicate_rows: list,
    results: list,
    usable: list,
    invalid: list,
    warnings: list,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for r in invalid + warnings:
        key = str(getattr(r, "reason", "") or "unknown")
        reason_counts[key] = reason_counts.get(key, 0) + 1

    audit = {
        "discovered_input_files": n_input,
        "discovered_target_files": n_target,
        "paired_stems": n_pairs,
        "missing_pair_rows": len(missing_rows),
        "duplicate_stem_rows": len(duplicate_rows),
        "valid_pairs": len(usable),
        "invalid_pairs": len(invalid),
        "warning_pairs": len(warnings),
        "ready_for_training": len(usable) > 0 and len(invalid) == 0,
        "notes": [
            "Only status=valid pairs enter train/val/test splits.",
            "Input must be exact black/white/magenta; target greys only on magenta.",
            "Test split remains locked out of overfit merge.",
        ],
        "failure_reason_counts": reason_counts,
        "invalid_stems": [r.stem for r in invalid],
        "warning_stems": [r.stem for r in warnings],
        "missing_stems": [row.get("stem") for row in missing_rows],
    }
    atomic_write_json(output_dir / "pair_readiness_audit.json", audit)
    LOGGER.info(
        "Pair readiness: valid=%d invalid=%d warning=%d missing=%d duplicates=%d ready=%s",
        audit["valid_pairs"],
        audit["invalid_pairs"],
        audit["warning_pairs"],
        audit["missing_pair_rows"],
        audit["duplicate_stem_rows"],
        audit["ready_for_training"],
    )
    return audit


def _per_image_budget(
    images: list[ImageRecord],
    config: PipelineConfig,
    *,
    is_train: bool,
) -> dict[str, int]:
    """Compute per-image spatial crop budgets (before dihedral bake)."""
    if not images:
        return {}
    if not is_train:
        fixed = (
            config.val_crops_per_image
            if config.val_crops_per_image is not None
            else min(
                8,
                desired_crop_count(
                    images[0].height,
                    images[0].width,
                    stride=config.grid_stride if config.include_spatial_grid else 384,
                    min_crops=config.min_crops_per_image,
                    max_crops=config.max_crops_per_image,
                ),
            )
        )
        return {img.source_id: fixed for img in images}

    raw = {
        img.source_id: desired_crop_count(
            img.height,
            img.width,
            stride=config.grid_stride if config.include_spatial_grid else 384,
            min_crops=config.min_crops_per_image,
            max_crops=config.max_crops_per_image,
        )
        for img in images
    }
    target = config.target_train_crops
    if target is None:
        return raw

    # Budget is for spatial boxes; bake multiplies later.
    bake_factor = 8 if config.bake_dihedral else 1
    spatial_target = max(len(images), int(math.ceil(target / bake_factor)))
    total_raw = sum(raw.values())
    if total_raw <= 0:
        return {sid: config.min_crops_per_image for sid in raw}

    scaled = {
        sid: max(config.min_crops_per_image, int(round(v * spatial_target / total_raw)))
        for sid, v in raw.items()
    }
    # Cap individual images but allow exceeding legacy max when targeting 10k.
    hard_cap = max(config.max_crops_per_image, spatial_target)
    scaled = {sid: min(v, hard_cap) for sid, v in scaled.items()}

    # Fix rounding drift toward spatial_target.
    while sum(scaled.values()) > spatial_target:
        sid = max(scaled, key=scaled.get)
        if scaled[sid] <= config.min_crops_per_image:
            break
        scaled[sid] -= 1
    while sum(scaled.values()) < spatial_target:
        sid = min(scaled, key=scaled.get)
        if scaled[sid] >= hard_cap:
            break
        scaled[sid] += 1
    return scaled


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full preparation pipeline and write all artefacts."""
    if config.valid_margin < 0:
        raise ValueError("valid_margin must be >= 0")
    if config.valid_margin * 2 >= config.crop_size:
        raise ValueError(
            f"valid_margin ({config.valid_margin}) must leave a positive central "
            f"region inside crop_size ({config.crop_size})"
        )

    output_dir = Path(config.output_dir)
    ensure_output_dir(output_dir, overwrite=config.overwrite)
    if config.overwrite:
        clear_generated_outputs(output_dir, GENERATED_FILES)

    input_files = discover_images(config.input_dir)
    target_files = discover_images(config.target_dir)
    LOGGER.info(
        "Discovered %d input and %d target images",
        len(input_files),
        len(target_files),
    )
    pairs, missing_rows, duplicate_rows = pair_by_stem(input_files, target_files)
    LOGGER.info(
        "Paired %d stems (%d missing, %d duplicate-stem rows)",
        len(pairs),
        len(missing_rows),
        len(duplicate_rows),
    )

    results = validate_pairs(pairs, progress=True)
    usable = [r for r in results if is_usable_for_dataset(r)]
    invalid = [r for r in results if r.status == STATUS_INVALID]
    warnings = [r for r in results if r.status == STATUS_WARNING]
    LOGGER.info(
        "Validation: %d valid, %d invalid, %d warning",
        len(usable),
        len(invalid),
        len(warnings),
    )

    audit = _write_pair_readiness_audit(
        output_dir,
        n_input=len(input_files),
        n_target=len(target_files),
        n_pairs=len(pairs),
        missing_rows=missing_rows,
        duplicate_rows=duplicate_rows,
        results=results,
        usable=usable,
        invalid=invalid,
        warnings=warnings,
    )

    stem_to_split = assign_splits(
        usable,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        group_regex=config.group_regex,
    )
    image_records = build_image_records(
        usable, stem_to_split, group_regex=config.group_regex
    )
    split_map = splits_summary(image_records)

    train_images = [r for r in image_records if r.split == "train"]
    val_images = [r for r in image_records if r.split == "val"]
    test_images = [r for r in image_records if r.split == "test"]
    rare_order = compute_train_rare_class_order(train_images)

    # Overfit V2: train crops come from train+val images; test stays locked.
    crop_train_images = train_images + val_images if config.merge_train_val_crops else train_images
    crop_val_images = [] if config.merge_train_val_crops else val_images

    train_budgets = _per_image_budget(crop_train_images, config, is_train=True)
    val_budgets = _per_image_budget(crop_val_images, config, is_train=False)

    train_crops: list[CropRecord] = []
    val_crops: list[CropRecord] = []
    combined_stats = CropSamplingStats()
    component_sizes: list[int] = []
    boundary_total = 0
    images_with_corner = 0

    sample_kwargs = dict(
        crop_size=config.crop_size,
        valid_margin=config.valid_margin,
        min_crops=config.min_crops_per_image,
        max_crops=config.max_crops_per_image,
        include_spatial_grid=config.include_spatial_grid,
        allow_padded_origins=config.allow_padded_origins,
        grid_stride=config.grid_stride,
        edge_band=config.edge_band,
        iou_dedup_threshold=(
            config.iou_dedup_threshold
            if config.iou_dedup_threshold is not None
            else (CROP_IOU_DEDUP_THRESHOLD_V2 if config.include_spatial_grid else None)
        ),
        strategy_weights=STRATEGY_WEIGHTS_V2 if config.include_spatial_grid else None,
        bake_dihedral=config.bake_dihedral,
        dihedral_variants=list(range(8)) if config.bake_dihedral else [0],
    )

    def process_split(
        images: list[ImageRecord],
        budgets: dict[str, int],
        *,
        is_train: bool,
    ) -> list[CropRecord]:
        nonlocal boundary_total, images_with_corner
        out: list[CropRecord] = []
        try:
            from tqdm import tqdm

            iterator = tqdm(images, desc="Crops " + ("train" if is_train else "val"))
        except ImportError:
            iterator = images

        for image in iterator:
            input_rgb, _ = load_rgb_exact(image.input_path)
            target_rgb, _ = load_rgb_exact(image.target_path)
            fixed = budgets.get(image.source_id)
            crops, stats, extras = sample_crops_for_image(
                image,
                input_rgb,
                target_rgb,
                global_seed=config.seed,
                rare_class_keys=rare_order if is_train else None,
                fixed_count=fixed,
                **sample_kwargs,
            )
            out.extend(crops)
            _merge_stats(combined_stats, stats)
            component_sizes.extend(extras.get("component_sizes", []))  # type: ignore[arg-type]
            boundary_total += int(extras.get("boundary_pixels", 0))
            if extras.get("has_corner_crop"):
                images_with_corner += 1
        return out

    train_crops = process_split(crop_train_images, train_budgets, is_train=True)
    val_crops = process_split(crop_val_images, val_budgets, is_train=False)

    preview_dir = output_dir / "previews"
    preview_source = train_crops + val_crops
    write_previews(
        preview_source,
        preview_dir,
        count=config.preview_count,
        seed=config.seed,
    )

    stats = compute_dataset_stats(
        n_input_files=len(input_files),
        n_target_files=len(target_files),
        validation_results=results,
        image_records=image_records,
        train_crops=train_crops,
        val_crops=val_crops,
        crop_sampling_stats=combined_stats,
        component_sizes=component_sizes,
        boundary_pixel_total=boundary_total,
    )
    stats["missing_pairs"] = len(missing_rows)
    stats["duplicate_stem_rows"] = len(duplicate_rows)
    stats["pair_readiness"] = audit
    stats["images_with_corner_or_edge_crop"] = images_with_corner
    stats["merge_train_val_crops"] = bool(config.merge_train_val_crops)
    stats["bake_dihedral"] = bool(config.bake_dihedral)
    stats["target_train_crops"] = config.target_train_crops or DEFAULT_TARGET_TRAIN_CROPS

    atomic_write_json(output_dir / "dataset_config.json", config.to_dict())
    atomic_write_json(output_dir / "dataset_stats.json", stats)
    atomic_write_json(
        output_dir / "splits.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "group_regex": config.group_regex,
            "splits": split_map,
            "merge_train_val_crops": bool(config.merge_train_val_crops),
        },
    )

    atomic_write_csv(
        output_dir / "validation_report.csv",
        VALIDATION_CSV_FIELDS,
        [r.to_csv_row() for r in results],
    )
    atomic_write_csv(
        output_dir / "invalid_pairs.csv",
        VALIDATION_CSV_FIELDS,
        [r.to_csv_row() for r in invalid],
    )
    atomic_write_csv(
        output_dir / "missing_pairs.csv",
        ["stem", "side", "present_path", "missing_side"],
        missing_rows,
    )
    atomic_write_csv(
        output_dir / "duplicate_stems.csv",
        ["stem", "side", "paths", "count"],
        duplicate_rows,
    )

    atomic_write_jsonl(
        output_dir / "train_images.jsonl",
        [r.to_dict() for r in train_images],
    )
    atomic_write_jsonl(
        output_dir / "val_images.jsonl",
        [r.to_dict() for r in val_images],
    )
    atomic_write_jsonl(
        output_dir / "test_images.jsonl",
        [r.to_dict() for r in test_images],
    )
    atomic_write_jsonl(
        output_dir / "train_crops.jsonl",
        [r.to_dict() for r in train_crops],
    )
    atomic_write_jsonl(
        output_dir / "val_crops.jsonl",
        [r.to_dict() for r in val_crops],
    )

    summary = format_summary(stats)
    LOGGER.info("\n%s", summary)
    print(summary)
    print(
        f"Train crops written: {len(train_crops)} "
        f"(target~={config.target_train_crops or 'n/a'}, "
        f"corner/edge images={images_with_corner})"
    )
    return stats


def _merge_stats(dst: CropSamplingStats, src: CropSamplingStats) -> None:
    dst.generated += src.generated
    dst.accepted += src.accepted
    dst.rejected_no_candidate += src.rejected_no_candidate
    dst.rejected_duplicate += src.rejected_duplicate
    dst.rejected_high_iou += src.rejected_high_iou
    dst.tiny_component_crops += src.tiny_component_crops
    for k, v in src.by_strategy.items():
        dst.by_strategy[k] = dst.by_strategy.get(k, 0) + v
