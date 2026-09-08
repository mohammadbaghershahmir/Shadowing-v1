"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from shdowing.ai2bmp.bmp.raster import index_to_rgb
from shdowing.ai2bmp.bmp.validate import validate_bmp_file
from shdowing.ai2bmp.canonical.artifacts import generate_canonical_artifacts
from shdowing.ai2bmp.constants import DATASET_VERSION, SCHEMA_VERSION
from shdowing.ai2bmp.crops.materialize import materialize_crop
from shdowing.ai2bmp.crops.selection import select_crops_for_design
from shdowing.ai2bmp.duplicates import assign_design_group_id, build_duplicate_report, content_hash_index_and_palette
from shdowing.ai2bmp.invariants import run_invariants
from shdowing.ai2bmp.io_utils import ensure_dir, output_dir_nonempty, read_json, rel_path, write_json
from shdowing.ai2bmp.labels.pipeline import generate_exact_labels
from shdowing.ai2bmp.manifests import write_all_manifests
from shdowing.ai2bmp.report.contact_sheet import build_contact_sheet
from shdowing.ai2bmp.report.pilot_html import build_pilot_html
from shdowing.ai2bmp.report.summary import write_pilot_summary
from shdowing.ai2bmp.synthetic.io import generate_variant
from shdowing.ai2bmp.synthetic.profiles import full_variants, pilot_variants


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    mode: str = "pilot"
    sample_count: int = 1
    sample_file: str | None = "1-111-6B.bmp"
    seed: int = 42
    crop_sizes: tuple[int, ...] = (256, 512, 1024)
    variants_profile: str = "pilot"
    recursive: bool = True
    materialize_crops: bool = True
    resume: bool = False
    verify_hashes: bool = True


def discover_bmps(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.bmp" if recursive else "*.bmp"
    return sorted(input_dir.glob(pattern))


def select_pilot_files(all_bmps: list[Path], cfg: PipelineConfig) -> list[Path]:
    if cfg.sample_file:
        matches = [p for p in all_bmps if p.name.lower() == cfg.sample_file.lower()]
        if matches:
            rec = validate_bmp_file(matches[0], relative_to=cfg.input_dir)
            if rec.status != "invalid":
                return [matches[0]]
    valid = []
    for p in all_bmps:
        rec = validate_bmp_file(p, relative_to=cfg.input_dir)
        if rec.status == "invalid":
            continue
        if rec.used_index_count >= 8 and rec.width >= 1024 and rec.absolute_height >= 1024:
            valid.append(p)
    valid.sort(key=lambda x: x.name)
    return valid[: cfg.sample_count]


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    cfg.input_dir = cfg.input_dir.resolve()
    cfg.output_dir = cfg.output_dir.resolve()

    try:
        cfg.output_dir.resolve().relative_to(cfg.input_dir.resolve())
        raise ValueError("Output directory must not be inside source directory")
    except ValueError as exc:
        if "must not" in str(exc):
            raise

    if output_dir_nonempty(cfg.output_dir) and not cfg.resume:
        raise FileExistsError(
            f"Output directory {cfg.output_dir} is nonempty. Pass --resume to continue/overwrite outputs."
        )

    ensure_dir(cfg.output_dir)
    write_json(
        cfg.output_dir / "dataset_config.json",
        {
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "seed": cfg.seed,
            "mode": cfg.mode,
            "crop_sizes": list(cfg.crop_sizes),
            "variants_profile": cfg.variants_profile,
        },
    )

    all_bmps = discover_bmps(cfg.input_dir, cfg.recursive)
    if cfg.mode == "pilot":
        selected = select_pilot_files(all_bmps, cfg)
        audit_paths = selected
    else:
        selected = []
        audit_paths = all_bmps
        for p in tqdm(all_bmps, desc="validate"):
            rec = validate_bmp_file(p, relative_to=cfg.input_dir)
            if rec.status != "invalid":
                selected.append(p)

    rejected_rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for p in tqdm(audit_paths if cfg.mode == "pilot" else all_bmps, desc="audit"):
        rec = validate_bmp_file(p, relative_to=cfg.input_dir)
        row = rec.to_dict()
        audit_records.append(row)
        if rec.status == "invalid":
            rejected_rows.append({"filename": rec.filename, "reason": rec.reason, "errors": "; ".join(rec.validation_errors)})

    duplicate_rows = build_duplicate_report(
        [{**r, "content_hash": r.get("content_hash", r.get("sha256", ""))} for r in audit_records if r["status"] != "invalid"]
    )

    canonical_rows: list[dict[str, Any]] = []
    palette_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    augmentation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    ctx_variants: list[dict[str, Any]] = []
    ctx_crops: list[dict[str, Any]] = []
    contact_images: list[tuple[str, np.ndarray]] = []

    last_ctx: dict[str, Any] = {}

    for src in selected:
        canonical = generate_canonical_artifacts(src, cfg.output_dir, input_root=cfg.input_dir)
        labels = generate_exact_labels(canonical)
        boundary = labels["boundary_mask"]
        thin = np.array(Image.open(canonical.design_dir / "exact_labels" / "thin_structure_mask.bmp"))

        group_id = assign_design_group_id(canonical.audit.filename, canonical.audit.sha256, duplicate_rows)
        content_hash = content_hash_index_and_palette(canonical.index_map, canonical.palette)

        canonical_rows.append(
            {
                "sample_id": canonical.sample_id,
                "design_group_id": group_id,
                "original_filename": canonical.audit.filename,
                "source_bmp_path": str(src),
                "width": canonical.audit.width,
                "height": canonical.audit.absolute_height,
                "used_index_count": canonical.audit.used_index_count,
                "unique_used_rgb_count": canonical.audit.unique_used_rgb_count,
                "sha256": canonical.audit.sha256,
                "content_hash": content_hash,
                "validation_status": canonical.audit.status,
            }
        )

        used_pal = read_json(canonical.canonical_dir / "palette_used_indices.json")
        for entry in used_pal.get("entries", []):
            palette_rows.append({"sample_id": canonical.sample_id, **entry})

        variants = pilot_variants(cfg.seed, canonical.sample_id) if cfg.variants_profile == "pilot" else full_variants(cfg.seed, canonical.sample_id)
        for recipe in tqdm(variants, desc=f"variants:{canonical.sample_id}"):
            result = generate_variant(canonical, recipe, boundary, labels["label_metadata"])
            ctx_variants.append(result)
            variant_rows.append(
                {
                    "variant_id": recipe.variant_id,
                    "sample_id": canonical.sample_id,
                    "design_group_id": group_id,
                    "family": recipe.family,
                    "seed": recipe.seed,
                    "status": result.get("status"),
                    "default_dense_supervision": recipe.default_dense_supervision,
                    "severe_ood": recipe.use_for_confidence_or_robustness,
                    "recipe_path": rel_path(
                        canonical.design_dir / "inputs" / "synthetic" / recipe.variant_id / "corruption_recipe.json",
                        cfg.output_dir,
                    )
                    if result.get("status") == "generated"
                    else "",
                }
            )
            if result.get("status") == "generated":
                vdir = Path(result["variant_dir"])
                aligned = np.array(Image.open(vdir / "input_aligned.png").convert("RGB"))
                contact_images.append((recipe.variant_id, aligned[::8, ::8]))

        crops, rejected_crops = select_crops_for_design(
            sample_id=canonical.sample_id,
            source_id=canonical.sample_id,
            split="train",
            input_path=rel_path(canonical.canonical_dir / "target_rgb.png", cfg.output_dir),
            target_path=rel_path(canonical.canonical_dir / "target_rgb.png", cfg.output_dir),
            index_map=canonical.index_map,
            contiguous_map=canonical.contiguous_used_index_map,
            boundary_mask=boundary,
            thin_mask=thin,
            image_width=canonical.audit.width,
            image_height=canonical.audit.absolute_height,
            crop_sizes=list(cfg.crop_sizes),
            global_seed=cfg.seed,
        )
        ctx_crops.extend(crops)

        clean_aligned = rel_path(canonical.design_dir / "inputs" / "clean" / "input_aligned.png", cfg.output_dir)
        if not (canonical.design_dir / "inputs" / "clean" / "input_aligned.png").exists():
            clean_aligned = rel_path(
                canonical.design_dir / "inputs" / "synthetic" / "clean_identity" / "input_aligned.png",
                cfg.output_dir,
            )

        for crop in crops:
            crop_rows.append(crop)
            aligned_path = clean_aligned
            for var in ctx_variants:
                if var.get("status") == "generated" and var.get("family") != "clean":
                    aligned_path = rel_path(Path(var["variant_dir"]) / "input_aligned.png", cfg.output_dir)
                    break

            training_rows.append(
                {
                    **crop,
                    "dataset_version": DATASET_VERSION,
                    "parent_design_id": canonical.sample_id,
                    "design_group_id": group_id,
                    "original_filename": canonical.audit.filename,
                    "variant_id": "clean_identity",
                    "aligned_input_path": aligned_path,
                    "target_rgb_path": rel_path(canonical.canonical_dir / "target_rgb.png", cfg.output_dir),
                    "original_index_map_path": rel_path(canonical.canonical_dir / "original_index_map.npy", cfg.output_dir),
                    "contiguous_used_index_map_path": rel_path(
                        canonical.canonical_dir / "contiguous_used_index_map.npy", cfg.output_dir
                    ),
                    "contiguous_unique_rgb_map_path": rel_path(
                        canonical.canonical_dir / "contiguous_unique_rgb_map.npy", cfg.output_dir
                    ),
                    "palette_used_indices_path": rel_path(
                        canonical.canonical_dir / "palette_used_indices.json", cfg.output_dir
                    ),
                    "boundary_mask_path": rel_path(
                        canonical.design_dir / "exact_labels" / "region_boundary_mask.bmp", cfg.output_dir
                    ),
                    "thin_structure_mask_path": rel_path(
                        canonical.design_dir / "exact_labels" / "thin_structure_mask.bmp", cfg.output_dir
                    ),
                    "valid_supervision_mask_path": rel_path(
                        canonical.design_dir / "inputs" / "synthetic" / "clean_identity" / "valid_supervision_mask.bmp",
                        cfg.output_dir,
                    ),
                    "k_index": canonical.audit.used_index_count,
                    "k_rgb": canonical.audit.unique_used_rgb_count,
                    "task_availability": labels["provisional_meta"]["task_availability"],
                    "severe_ood": False,
                }
            )

            if cfg.materialize_crops:
                aligned = np.array(Image.open(cfg.output_dir / aligned_path.replace("/", "\\")).convert("RGB"))
                materialize_crop(
                    crop,
                    aligned,
                    canonical.rgb,
                    boundary,
                    ensure_dir(canonical.design_dir / "crops" / "materialized"),
                )

        last_ctx = {
            "audit": canonical.audit,
            "canonical": canonical,
            "variants": ctx_variants,
            "crops": ctx_crops,
            "output_root": cfg.output_dir,
        }

    write_all_manifests(
        cfg.output_dir,
        canonical_rows=canonical_rows,
        palette_rows=palette_rows,
        variant_rows=variant_rows,
        crop_rows=crop_rows,
        augmentation_rows=augmentation_rows,
        duplicate_rows=duplicate_rows,
        rejected_rows=rejected_rows,
        training_rows=training_rows,
    )

    stats = {
        "discovered_input_files": len(all_bmps),
        "valid_pairs_processed": len(selected),
        "invalid_files": len(rejected_rows),
        "variant_count": len(variant_rows),
        "crop_count": len(crop_rows),
        "duplicate_groups": len(duplicate_rows),
    }
    write_json(cfg.output_dir / "dataset_stats.json", stats)

    if cfg.mode == "pilot" and last_ctx:
        invariants = run_invariants(last_ctx)
        last_ctx["invariants"] = invariants
        html_path = build_pilot_html(cfg.output_dir, last_ctx)
        if contact_images:
            build_contact_sheet(contact_images[:16], cfg.output_dir / "reports" / "pilot_contact_sheet.png")
        write_pilot_summary(
            cfg.output_dir,
            [{"metric": k, "value": v} for k, v in stats.items()],
            {"invariants": invariants, "stats": stats},
        )
        stats["pilot_report_html"] = str(html_path)

    return stats
