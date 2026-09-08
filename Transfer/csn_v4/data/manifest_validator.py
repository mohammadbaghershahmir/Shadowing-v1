"""Manifest and dataset validation for CSN-V4."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from csn_v4.data.paths import resolve_dataset_path, sha256_file, sha256_manifest
from csn_v4.data.splits import assert_no_split_leakage
from csn_v4.data.targets import load_gray_bmp
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import assert_full_coverage, build_coverage_map, iter_core_tiles

ALLOWED_SEMANTIC = frozenset({0, 100, 150, 200, 255})
MANIFEST_FILES = (
    "manifests/scenes.jsonl",
    "manifests/train_capacity_tiles_bw.jsonl",
    "manifests/val_spatial_fixed_bw.jsonl",
    "manifests/val_spatial_subscenes_bw.jsonl",
    "manifests/val_scene_holdout_bw.jsonl",
    "manifests/eval_full_images.jsonl",
)
TILE_MANIFESTS = frozenset({
    "manifests/train_capacity_tiles_bw.jsonl",
    "manifests/val_spatial_fixed_bw.jsonl",
})
SUBSCENE_MANIFEST = "manifests/val_spatial_subscenes_bw.jsonl"
TILE_REQUIRED = frozenset({
    "sample_id", "scene_id", "transform_id", "core_origin_x", "core_origin_y", "tile_uid",
})


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _check_semantic(arr: np.ndarray, ctx: str, errors: list[str]) -> None:
    uniq = set(int(v) for v in np.unique(arr))
    bad = uniq - ALLOWED_SEMANTIC
    if bad:
        errors.append(f"{ctx}: invalid semantic values {sorted(bad)}")


def _audit_scene_record(scene: dict, dataset_root: Path, errors: list[str], warnings: list[str]) -> dict:
    info: dict[str, Any] = {"scene_id": scene["scene_id"]}
    src = Path(scene["source_bw_path"])
    sem = Path(scene["target_semantic_path"])
    if not src.is_file():
        src = resolve_dataset_path(scene.get("source_bw_path", ""), dataset_root)
    if not sem.is_file():
        sem = resolve_dataset_path(scene.get("target_semantic_path", ""), dataset_root)
    for label, path in (("source_bw", src), ("target_semantic", sem)):
        if not path.is_file():
            errors.append(f"scene {scene['scene_id']}: missing {label} at {path}")
            continue
        try:
            arr = load_gray_bmp(path)
        except Exception as exc:
            errors.append(f"scene {scene['scene_id']}: failed to decode {label}: {exc}")
            continue
        info[f"{label}_shape"] = list(arr.shape)
        file_hash = sha256_file(path)
        info[f"{label}_sha256"] = file_hash
        if label == "source_bw":
            meta_hash = scene.get("source_sha256")
            if meta_hash and meta_hash != file_hash:
                warnings.append(
                    f"scene {scene['scene_id']}: source_sha256 mismatch "
                    f"(manifest={meta_hash[:12]} file={file_hash[:12]})"
                )
        if label == "target_semantic":
            _check_semantic(arr, f"scene {scene['scene_id']}", errors)
            src_shape = info.get("source_bw_shape")
            if src_shape and list(arr.shape) != src_shape:
                errors.append(
                    f"scene {scene['scene_id']}: semantic shape {list(arr.shape)} != source {src_shape}"
                )
    meta_w = int(scene.get("image_width") or 0)
    meta_h = int(scene.get("image_height") or 0)
    if "source_bw_shape" in info and meta_w and meta_h:
        h, w = info["source_bw_shape"]
        if (w, h) != (meta_w, meta_h):
            warnings.append(
                f"scene {scene['scene_id']}: metadata size ({meta_w}x{meta_h}) != BMP ({w}x{h})"
            )
    return info


def _audit_d4_uniqueness(errors: list[str]) -> None:
    seen: dict[tuple[int, ...], int] = {}
    probe = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for k in range(8):
        out = D4Transform(k).apply(probe)
        key = tuple(out.flatten().tolist())
        if key in seen:
            errors.append(f"D4 collision: transform {k} equals transform {seen[key]}")
        seen[key] = k
        inv = D4Transform(k).inverse.apply(out)
        if not np.array_equal(inv, probe):
            errors.append(f"D4 inverse failed for transform {k}")


def _audit_tile_manifest(
    rows: list[dict],
    scenes: dict[str, dict],
    kind: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    sample_ids: set[str] = set()
    tile_uids: set[str] = set()
    transform_counts: Counter[int] = Counter()
    region_counts: Counter[str] = Counter()
    by_scene_d4: dict[str, set[int]] = {}

    for row in rows:
        sid = row.get("scene_id")
        if sid and sid not in scenes:
            errors.append(f"{kind}: row {row.get('sample_id')} references unknown scene {sid}")
        sample_id = row.get("sample_id")
        if not sample_id:
            errors.append(f"{kind}: row missing sample_id")
        elif sample_id in sample_ids:
            errors.append(f"{kind}: duplicate sample_id {sample_id}")
        else:
            sample_ids.add(sample_id)

        if kind in TILE_MANIFESTS:
            missing = TILE_REQUIRED - set(row.keys())
            if missing:
                errors.append(f"{kind}: {sample_id} missing fields {sorted(missing)}")
            if row.get("record_type") not in (None, "tile"):
                errors.append(f"{kind}: non-tile record_type in tile manifest: {sample_id}")
            uid = row.get("tile_uid")
            if uid:
                if uid in tile_uids:
                    errors.append(f"{kind}: duplicate tile_uid {uid}")
                tile_uids.add(uid)

        if "transform_id" in row:
            k = int(row["transform_id"])
            transform_counts[k] += 1
            if sid:
                by_scene_d4.setdefault(sid, set()).add(k)
        region_counts[row.get("sampling_region", "unknown")] += 1

    if kind in TILE_MANIFESTS and rows and len(rows) != len(tile_uids):
        errors.append(
            f"{kind}: row count ({len(rows)}) != unique tile_uid count ({len(tile_uids)})"
        )

    for sid, tids in by_scene_d4.items():
        missing = set(range(8)) - tids
        if missing:
            errors.append(f"{kind} scene {sid} missing D4 orientations {sorted(missing)}")

    return {
        "n_rows": len(rows),
        "n_unique_tile_uids": len(tile_uids),
        "transform_distribution": {str(k): v for k, v in sorted(transform_counts.items())},
        "region_distribution": dict(region_counts),
    }


def _audit_subscene_manifest(rows: list[dict], scenes: dict[str, dict], errors: list[str]) -> dict[str, Any]:
    sample_ids: set[str] = set()
    for row in rows:
        if row.get("record_type") != "subscene":
            errors.append(f"subscene manifest: expected record_type=subscene for {row.get('sample_id')}")
        if "core_origin_x" in row or "core_origin_y" in row:
            errors.append(f"subscene manifest: tile fields in subscene row {row.get('sample_id')}")
        sid = row.get("scene_id")
        if sid and sid not in scenes:
            errors.append(f"subscene: unknown scene {sid}")
        sample_id = row.get("sample_id")
        if sample_id in sample_ids:
            errors.append(f"subscene: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
    return {"n_rows": len(rows)}


def _audit_coverage_for_scene(scene: dict, errors: list[str]) -> None:
    src = load_gray_bmp(scene["source_bw_path"])
    fh, fw = src.shape
    for k in range(8):
        tfw, tfh = D4Transform(k).transformed_wh(fw, fh)
        specs = list(iter_core_tiles(tfw, tfh, transform_id=k, scene_id=scene["scene_id"]))
        try:
            assert_full_coverage(tfw, tfh, specs)
        except AssertionError as exc:
            errors.append(f"scene {scene['scene_id']} d4_{k}: {exc}")


def _audit_overwrite_from_tiles(scene: dict, tile_rows: list[dict], errors: list[str]) -> None:
    """Detect duplicate core coverage from manifest tile origins (not just canonical grid)."""
    src = load_gray_bmp(scene["source_bw_path"])
    fh, fw = src.shape
    sid = scene["scene_id"]
    scene_tiles = [r for r in tile_rows if r.get("scene_id") == sid]
    for k in range(8):
        tfw, tfh = D4Transform(k).transformed_wh(fw, fh)
        specs = []
        for r in scene_tiles:
            if int(r.get("transform_id", -1)) != k:
                continue
            from csn_v4.geometry.tile_spec import build_tile_spec
            specs.append(build_tile_spec(
                tfw, tfh, int(r["core_origin_x"]), int(r["core_origin_y"]),
                transform_id=k, scene_id=sid,
            ))
        if not specs:
            continue
        cov = build_coverage_map(tfw, tfh, specs)
        if (cov > 1).any():
            n = int((cov > 1).sum())
            errors.append(f"scene {sid} d4_{k}: manifest tile overwrite ({n} pixels cov>1)")


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "manifests": {},
        "scenes": [],
        "leakage": {},
        "fingerprints": {},
        "errors": errors,
        "warnings": warnings,
        "critical_count": 0,
    }

    _audit_d4_uniqueness(errors)

    scenes_path = dataset_root / "manifests/scenes.jsonl"
    if not scenes_path.is_file():
        errors.append(f"Missing {scenes_path}")
        report["critical_count"] = len(errors)
        return report

    scenes_list = _load_jsonl(scenes_path)
    scenes = {s["scene_id"]: s for s in scenes_list}
    train_scenes = [s for s in scenes_list if s.get("split") == "train"]
    val_scenes = [s for s in scenes_list if s.get("split") == "val"]

    try:
        assert_no_split_leakage(train_scenes, val_scenes)
        report["leakage"] = {
            "status": "ok",
            "n_train": len(train_scenes),
            "n_val": len(val_scenes),
            "train_groups": sorted({s.get("group_id") for s in train_scenes}),
            "val_groups": sorted({s.get("group_id") for s in val_scenes}),
        }
    except AssertionError as exc:
        errors.append(str(exc))
        report["leakage"] = {"status": "fail", "detail": str(exc)}

    for scene in scenes_list:
        info = _audit_scene_record(scene, dataset_root, errors, warnings)
        report["scenes"].append(info)
        try:
            _audit_coverage_for_scene(scene, errors)
        except FileNotFoundError as exc:
            errors.append(str(exc))

    cap_rows: list[dict] = []
    for rel in MANIFEST_FILES:
        path = dataset_root / rel
        if not path.is_file():
            if rel.endswith("val_scene_holdout_bw.jsonl") and not val_scenes:
                warnings.append(f"Optional manifest missing (no val scenes): {rel}")
                continue
            if rel == SUBSCENE_MANIFEST:
                warnings.append(f"Optional subscene manifest missing: {rel}")
                continue
            errors.append(f"Missing manifest: {rel}")
            continue
        report["fingerprints"][rel] = sha256_manifest(path)
        rows = _load_jsonl(path)
        if rel == SUBSCENE_MANIFEST:
            report["manifests"][rel] = _audit_subscene_manifest(rows, scenes, errors)
        else:
            report["manifests"][rel] = _audit_tile_manifest(rows, scenes, rel, errors, warnings)
        if rel == "manifests/train_capacity_tiles_bw.jsonl":
            cap_rows = rows

    for scene in train_scenes:
        try:
            _audit_overwrite_from_tiles(scene, cap_rows, errors)
        except FileNotFoundError as exc:
            errors.append(str(exc))

    report["critical_count"] = len(errors)
    report["status"] = "fail" if errors else "ok"
    return report


def format_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CSN-V4 Dataset Audit",
        "",
        f"**Dataset root:** `{report.get('dataset_root')}`",
        f"**Status:** {report.get('status', 'unknown')}",
        f"**Critical errors:** {report.get('critical_count', 0)}",
        "",
        "## Leakage",
        "",
        f"```json\n{json.dumps(report.get('leakage', {}), indent=2)}\n```",
        "",
        "## Manifest fingerprints",
        "",
    ]
    for name, fp in report.get("fingerprints", {}).items():
        lines.append(f"- `{name}`: `{fp[:16]}…`")
    lines.extend(["", "## Manifest summaries", ""])
    for name, summary in report.get("manifests", {}).items():
        lines.append(f"### `{name}`")
        lines.append(f"- rows: {summary.get('n_rows', 0)}")
        if "n_unique_tile_uids" in summary:
            lines.append(f"- unique tile_uids: {summary.get('n_unique_tile_uids', 0)}")
        lines.append(f"- transforms: {summary.get('transform_distribution', {})}")
        lines.append("")
    if report.get("errors"):
        lines.extend(["## Errors", ""])
        for err in report["errors"]:
            lines.append(f"- {err}")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warn in report["warnings"]:
            lines.append(f"- {warn}")
    return "\n".join(lines) + "\n"
