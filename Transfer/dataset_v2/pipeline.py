"""Dataset V2 pilot pipeline orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dataset_v2.artifacts import SampleArtifacts, build_all_artifacts
from dataset_v2.augment import (
    AUG_ROT0,
    apply_op,
    list_augmentation_ops,
    op_from_name,
    verify_inverse_exact,
)
from dataset_v2.bmp_io import (
    BmpInspectResult,
    copy_original_bmp,
    count_unique_rgb,
    make_unexpected_colors_diagnostic,
    resolve_bmp_rgb,
    save_gray_bmp,
    save_rgb_bmp,
    validate_five_colors,
)
from dataset_v2.constants import (
    BLACK,
    DATASET_NAME,
    DATASET_VERSION,
    SHADE_DARK,
    SHADE_LIGHT,
    SHADE_MEDIUM,
    WHITE,
)
from dataset_v2.crops import CropBox, select_crops
from dataset_v2.ids import sample_id_from_path
from dataset_v2.io_bundle import ArtifactBundle, crop_artifacts, save_artifact_bundle
from dataset_v2.manifests import (
    AUGMENTATION_FIELDS,
    CANONICAL_SAMPLE_FIELDS,
    CROP_MANIFEST_FIELDS,
    write_csv,
)
from dataset_v2.reports import (
    write_pilot_report_html,
    write_pilot_summary_csv,
    write_validation_report,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PilotConfig:
    input_dir: Path
    output_dir: Path
    # 0 or negative → use all eligible independent scenes
    sample_count: int = 0
    seed: int = 42
    crop_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    recursive: bool = True
    sample_files: list[str] | None = None
    rotations: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    include_flips: bool = False
    # When capping, sample across path-derived families (parent folder name).
    stratify_by_family: bool = True


@dataclass
class PilotResult:
    selected: list[str]
    rejected: list[dict[str, Any]]
    canonical_rows: list[dict[str, Any]]
    crop_rows: list[dict[str, Any]]
    aug_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    validation_report: dict[str, Any]


def discover_bmps(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.bmp" if recursive else "*.bmp"
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def _class_stats(shade_level_id: np.ndarray) -> dict[str, Any]:
    total = shade_level_id.size
    c0 = int((shade_level_id == 0).sum())
    c1 = int((shade_level_id == 1).sum())
    c2 = int((shade_level_id == 2).sum())
    def pct(n: int) -> float:
        return 100.0 * n / max(total, 1)
    return {
        "class_0_count": c0,
        "class_1_count": c1,
        "class_2_count": c2,
        "class_0_pct": round(pct(c0), 4),
        "class_1_pct": round(pct(c1), 4),
        "class_2_pct": round(pct(c2), 4),
    }


def _inspect_file(path: Path, input_root: Path) -> BmpInspectResult:
    rgb, meta = resolve_bmp_rgb(path)
    meta.relative_path = str(path.relative_to(input_root)).replace("\\", "/")
    meta.unique_rgb = count_unique_rgb(rgb)
    valid_mask, unexpected, errors = validate_five_colors(rgb)
    if meta.had_alpha:
        errors.append("non-opaque alpha or transparency chunk present")
    if unexpected:
        meta.status = "failed"
        meta.unexpected_rgb = unexpected
        meta.errors = errors
    else:
        meta.status = "valid"
        meta.errors = []
    meta._rgb = rgb  # type: ignore[attr-defined]
    meta._valid_mask = valid_mask  # type: ignore[attr-defined]
    return meta


def _family_key(path: Path, input_root: Path) -> str:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return path.parent.name or "root"
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return "root"


def _select_pilot_paths(
    all_paths: list[Path],
    input_root: Path,
    cfg: PilotConfig,
) -> tuple[list[BmpInspectResult], list[BmpInspectResult]]:
    if cfg.sample_files:
        wanted = {Path(s).name for s in cfg.sample_files}
        all_paths = [p for p in all_paths if p.name in wanted]

    rng = np.random.default_rng(cfg.seed)
    use_all = cfg.sample_count is None or int(cfg.sample_count) <= 0

    # Inspect all candidates first so reports cover the full pool.
    selected: list[BmpInspectResult] = []
    rejected: list[BmpInspectResult] = []
    valid_pool: list[BmpInspectResult] = []

    order = np.arange(len(all_paths))
    rng.shuffle(order)
    shuffled = [all_paths[i] for i in order.tolist()]

    for path in shuffled:
        try:
            meta = _inspect_file(path, input_root)
        except Exception as exc:
            rejected.append(
                BmpInspectResult(
                    path=path,
                    relative_path=str(path.relative_to(input_root)),
                    width=0,
                    height=0,
                    pillow_mode="",
                    bit_depth=None,
                    color_type="unknown",
                    unique_rgb={},
                    sha256="",
                    had_alpha=False,
                    status="failed",
                    errors=[str(exc)],
                )
            )
            continue
        if meta.status == "valid":
            valid_pool.append(meta)
        else:
            rejected.append(meta)
            _write_failure_diagnostic(cfg.output_dir, meta)

    if use_all:
        selected = list(valid_pool)
    else:
        cap = int(cfg.sample_count)
        if not cfg.stratify_by_family or cap >= len(valid_pool):
            selected = valid_pool[:cap]
        else:
            # Round-robin across families so a cap still covers design diversity.
            by_fam: dict[str, list[BmpInspectResult]] = {}
            for m in valid_pool:
                by_fam.setdefault(_family_key(m.path, input_root), []).append(m)
            fam_keys = list(by_fam.keys())
            rng.shuffle(fam_keys)
            for fam in fam_keys:
                rng.shuffle(by_fam[fam])
            selected = []
            idx = 0
            while len(selected) < cap:
                progressed = False
                for fam in fam_keys:
                    bucket = by_fam[fam]
                    if idx < len(bucket):
                        selected.append(bucket[idx])
                        progressed = True
                        if len(selected) >= cap:
                            break
                if not progressed:
                    break
                idx += 1

    # Report counts by family / shade class (from inspected RGB if available).
    fam_counts: dict[str, int] = {}
    for m in selected:
        fam = _family_key(m.path, input_root)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    LOGGER.info(
        "Selected %d / %d eligible scenes (cap=%s). By family: %s",
        len(selected),
        len(valid_pool),
        "all" if use_all else cfg.sample_count,
        dict(sorted(fam_counts.items())),
    )
    return selected, rejected


def _write_failure_diagnostic(output_dir: Path, meta: BmpInspectResult) -> None:
    if not hasattr(meta, "_rgb"):
        return
    rgb = meta._rgb  # type: ignore[attr-defined]
    valid = meta._valid_mask  # type: ignore[attr-defined]
    diag = make_unexpected_colors_diagnostic(rgb, valid, meta.unexpected_rgb)
    fail_dir = output_dir / "reports" / "failures" / Path(meta.relative_path).stem
    fail_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_bmp(fail_dir / "unexpected_colors.bmp", diag)
    (fail_dir / "errors.json").write_text(
        json.dumps(
            {
                "relative_path": meta.relative_path,
                "unexpected_rgb": {str(k): v for k, v in meta.unexpected_rgb.items()},
                "errors": meta.errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _roundtrip_count(artifacts) -> int:
    return int((artifacts.roundtrip_diff > 0).sum())


def run_pilot(cfg: PilotConfig) -> PilotResult:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        LOGGER.warning("Output dir %s is not empty; writing alongside existing files", cfg.output_dir)

    input_root = cfg.input_dir.resolve()
    all_paths = discover_bmps(input_root, cfg.recursive)
    LOGGER.info("Discovered %d BMP file(s) under %s", len(all_paths), input_root)

    selected_meta, rejected_meta = _select_pilot_paths(all_paths, input_root, cfg)
    used_ids: set[str] = set()
    rng = np.random.default_rng(cfg.seed)

    canonical_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    aug_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    html_samples: list[dict[str, Any]] = []

    aug_ops = list_augmentation_ops(cfg.rotations, cfg.include_flips)

    for meta in selected_meta:
        rgb = meta._rgb  # type: ignore[attr-defined]
        valid_mask = meta._valid_mask  # type: ignore[attr-defined]
        sample_id = sample_id_from_path(meta.path, input_root, used_ids)
        group_id = sample_id
        full_dir = cfg.output_dir / "samples" / sample_id / "full"
        copy_original_bmp(meta.path, full_dir / "target_original.bmp")

        artifacts = build_all_artifacts(rgb, valid_mask)
        rt_mismatch = _roundtrip_count(artifacts)
        if rt_mismatch != 0:
            meta.status = "failed"
            meta.errors.append(f"roundtrip mismatch count={rt_mismatch}")
            rejected_meta.append(meta)
            _write_failure_diagnostic(cfg.output_dir, meta)
            continue

        color_counts = {
            "black_pixels": meta.unique_rgb.get(BLACK, 0),
            "white_pixels": meta.unique_rgb.get(WHITE, 0),
            "shade_light_pixels": meta.unique_rgb.get(SHADE_LIGHT, 0),
            "shade_medium_pixels": meta.unique_rgb.get(SHADE_MEDIUM, 0),
            "shade_dark_pixels": meta.unique_rgb.get(SHADE_DARK, 0),
        }

        full_meta = {
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "sample_id": sample_id,
            "group_id": group_id,
            "original_filename": meta.path.name,
            "relative_path": meta.relative_path,
            "width": meta.width,
            "height": meta.height,
            "pillow_mode": meta.pillow_mode,
            "bit_depth": meta.bit_depth,
            "color_type": meta.color_type,
            "sha256": meta.sha256,
            "unique_rgb": {str(k): v for k, v in meta.unique_rgb.items()},
            "roundtrip_mismatch_count": rt_mismatch,
            "validation_status": "PASS",
        }
        save_artifact_bundle(
            full_dir,
            ArtifactBundle(artifacts=artifacts, metadata=full_meta),
        )

        canonical_rows.append(
            {
                "dataset_version": DATASET_VERSION,
                "sample_id": sample_id,
                "group_id": group_id,
                "original_filename": meta.path.name,
                "relative_path": meta.relative_path,
                "width": meta.width,
                "height": meta.height,
                "sha256": meta.sha256,
                "validation_status": "PASS",
                **color_counts,
                "roundtrip_mismatch_count": rt_mismatch,
                "full_artifact_dir": str(full_dir.relative_to(cfg.output_dir)).replace("\\", "/"),
            }
        )

        crops = select_crops(
            sample_id,
            meta.width,
            meta.height,
            cfg.crop_sizes,
            transition=artifacts.transition_mask,
            shade=artifacts.shade_mask,
            shade_level_id=artifacts.shade_level_id,
            source_bw=artifacts.source_bw,
            rng=rng,
        )

        sample_crop_html: list[dict[str, Any]] = []
        canonical_crop_count = 0
        aug_count = 0

        for box in crops:
            crop_art = crop_artifacts(artifacts, box.x, box.y, box.width)
            crop_rt = _roundtrip_count(crop_art)
            if crop_rt != 0:
                LOGGER.error("Crop roundtrip failed %s mismatches=%d", box.crop_id, crop_rt)
                continue

            canonical_dir = (
                cfg.output_dir / "samples" / sample_id / "crops" / box.crop_id / "canonical"
            )
            stats = _class_stats(crop_art.shade_level_id)
            shade_cov = 100.0 * (crop_art.shade_mask > 0).sum() / max(crop_art.shade_mask.size, 1)
            trans_den = (crop_art.transition_mask > 0).sum() / max(crop_art.transition_mask.size, 1)
            content_hash = hashlib.sha256(
                crop_art.target_semantic.tobytes() + crop_art.shade_mask.tobytes()
            ).hexdigest()[:16]

            crop_meta = {
                **full_meta,
                "crop_id": box.crop_id,
                "base_crop_id": box.crop_id,
                "crop_category": box.category,
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
                "shade_coverage_pct": round(shade_cov, 4),
                "transition_density": round(float(trans_den), 6),
                "content_hash": content_hash,
                "roundtrip_mismatch_count": crop_rt,
            }
            save_artifact_bundle(canonical_dir, ArtifactBundle(artifacts=crop_art, metadata=crop_meta))
            canonical_crop_count += 1

            crop_rows.append(
                {
                    "dataset_version": DATASET_VERSION,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "parent_sample_id": sample_id,
                    "original_filename": meta.path.name,
                    "crop_id": box.crop_id,
                    "base_crop_id": box.crop_id,
                    "crop_category": box.category,
                    "x": box.x,
                    "y": box.y,
                    "width": box.width,
                    "height": box.height,
                    **stats,
                    "shade_coverage_pct": round(shade_cov, 4),
                    "transition_density": round(float(trans_den), 6),
                    "content_hash": content_hash,
                    "validation_status": "PASS",
                    "artifact_dir": str(canonical_dir.relative_to(cfg.output_dir)).replace("\\", "/"),
                }
            )

            sample_crop_html.append(
                {
                    "crop_id": box.crop_id,
                    "category": box.category,
                    "x": box.x,
                    "y": box.y,
                    "width": box.width,
                    "height": box.height,
                    "contact_sheet": str(
                        (canonical_dir / "contact_sheet.bmp").relative_to(cfg.output_dir)
                    ).replace("\\", "/"),
                }
            )

            for op in aug_ops:
                if op.name == AUG_ROT0:
                    continue
                aug_dir = (
                    cfg.output_dir
                    / "samples"
                    / sample_id
                    / "crops"
                    / box.crop_id
                    / "augmentation_candidates"
                    / op.name
                )
                aug_art = SampleArtifacts(
                    target_semantic=apply_op(crop_art.target_semantic, op),
                    source_bw=apply_op(crop_art.source_bw, op),
                    source_oracle=apply_op(crop_art.source_oracle, op),
                    shade_mask=apply_op(crop_art.shade_mask, op),
                    shade_level_id=apply_op(crop_art.shade_level_id, op),
                    shade_level_preview=apply_op(crop_art.shade_level_preview, op),
                    transition_mask=apply_op(crop_art.transition_mask, op),
                    black_lock=apply_op(crop_art.black_lock, op),
                    valid_mask=apply_op(crop_art.valid_mask, op),
                    reconstructed_target=apply_op(crop_art.reconstructed_target, op),
                    roundtrip_diff=apply_op(crop_art.roundtrip_diff, op),
                )
                inv_ok = verify_inverse_exact(crop_art.target_semantic, op)
                rt_ok = _roundtrip_count(aug_art) == 0
                aug_id = f"{box.crop_id}__{op.name}"
                aug_meta = {
                    **crop_meta,
                    "augmentation_id": aug_id,
                    "operation": op.name,
                    "inverse_transform_exact": inv_ok,
                    "roundtrip_exact": rt_ok,
                    "approved_for_training": False,
                }
                save_artifact_bundle(aug_dir, ArtifactBundle(artifacts=aug_art, metadata=aug_meta))
                aug_count += 1
                aug_rows.append(
                    {
                        "dataset_version": DATASET_VERSION,
                        "group_id": group_id,
                        "parent_sample_id": sample_id,
                        "base_crop_id": box.crop_id,
                        "augmentation_id": aug_id,
                        "operation": op.name,
                        "crop_size": box.width,
                        "inverse_transform_exact": inv_ok,
                        "roundtrip_exact": rt_ok,
                        "approved_for_training": False,
                        "artifact_dir": str(aug_dir.relative_to(cfg.output_dir)).replace("\\", "/"),
                    }
                )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "status": "PASS",
                "original_filename": meta.path.name,
                "width": meta.width,
                "height": meta.height,
                "canonical_crops": canonical_crop_count,
                "augmentation_candidates": aug_count,
                "roundtrip_mismatch_count": rt_mismatch,
                "unexpected_colors": 0,
            }
        )
        html_samples.append(
            {
                "sample_id": sample_id,
                "status": "PASS",
                "original_filename": meta.path.name,
                "width": meta.width,
                "height": meta.height,
                "roundtrip_mismatch_count": rt_mismatch,
                "full_contact_sheet": str(
                    (full_dir / "contact_sheet.bmp").relative_to(cfg.output_dir)
                ).replace("\\", "/"),
                "crops": sample_crop_html,
            }
        )

    rejected_payload = []
    for meta in rejected_meta:
        rejected_payload.append(
            {
                "relative_path": meta.relative_path,
                "status": meta.status,
                "errors": meta.errors,
                "unexpected_rgb": {str(k): v for k, v in meta.unexpected_rgb.items()},
                "unique_rgb": {str(k): v for k, v in meta.unique_rgb.items()},
            }
        )
        summary_rows.append(
            {
                "sample_id": Path(meta.relative_path).stem,
                "status": "FAIL",
                "original_filename": meta.path.name if meta.path else "",
                "width": meta.width,
                "height": meta.height,
                "canonical_crops": 0,
                "augmentation_candidates": 0,
                "roundtrip_mismatch_count": "",
                "unexpected_colors": len(meta.unexpected_rgb),
            }
        )

    all_inspections: list[dict[str, Any]] = []
    for path in all_paths:
        try:
            inspected = _inspect_file(path, input_root)
            all_inspections.append(
                {
                    "relative_path": inspected.relative_path,
                    "status": inspected.status,
                    "width": inspected.width,
                    "height": inspected.height,
                    "unique_rgb": {str(k): v for k, v in inspected.unique_rgb.items()},
                    "unexpected_rgb": {str(k): v for k, v in inspected.unexpected_rgb.items()},
                    "errors": inspected.errors,
                }
            )
            if inspected.status == "failed" and not any(r.path == path for r in rejected_meta):
                _write_failure_diagnostic(cfg.output_dir, inspected)
        except Exception as exc:
            all_inspections.append(
                {
                    "relative_path": str(path.relative_to(input_root)).replace("\\", "/"),
                    "status": "failed",
                    "errors": [str(exc)],
                }
            )

    validation_report = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "mode": "pilot",
        "input_dir": str(input_root),
        "output_dir": str(cfg.output_dir.resolve()),
        "seed": cfg.seed,
        "sample_count_requested": cfg.sample_count,
        "summary": {
            "discovered_bmp_files": len(all_paths),
            "selected_valid_samples": len(canonical_rows),
            "rejected_files": len(rejected_payload),
            "canonical_crops": len(crop_rows),
            "augmentation_candidates": len(aug_rows),
            "all_roundtrip_zero": all(
                r.get("roundtrip_mismatch_count") == 0 for r in canonical_rows
            ),
            "all_inverse_exact": all(r.get("inverse_transform_exact") for r in aug_rows) if aug_rows else True,
        },
        "selected_sample_ids": [r["sample_id"] for r in canonical_rows],
        "rejected_files": rejected_payload,
        "all_files": all_inspections,
    }

    manifest_dir = cfg.output_dir / "manifests"
    report_dir = cfg.output_dir / "reports"
    write_csv(manifest_dir / "canonical_samples.csv", CANONICAL_SAMPLE_FIELDS, canonical_rows)
    write_csv(manifest_dir / "crop_manifest.csv", CROP_MANIFEST_FIELDS, crop_rows)
    write_csv(manifest_dir / "augmentation_candidates.csv", AUGMENTATION_FIELDS, aug_rows)
    write_validation_report(report_dir / "validation_report.json", validation_report)
    write_pilot_summary_csv(report_dir / "pilot_summary.csv", summary_rows)
    write_pilot_report_html(
        report_dir / "pilot_report.html",
        output_dir=cfg.output_dir,
        validation=validation_report,
        samples=html_samples,
    )

    return PilotResult(
        selected=[r["sample_id"] for r in canonical_rows],
        rejected=rejected_payload,
        canonical_rows=canonical_rows,
        crop_rows=crop_rows,
        aug_rows=aug_rows,
        summary_rows=summary_rows,
        validation_report=validation_report,
    )
