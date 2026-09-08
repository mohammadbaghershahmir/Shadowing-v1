#!/usr/bin/env python3
"""Build Dataset_V4 manifests: scenes + deterministic D4 x core tiles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import iter_core_tiles, tile_uid


def _scene_record(scene_id: str, full_dir: Path) -> dict | None:
    src = full_dir / "source_bw.bmp"
    sem = full_dir / "target_semantic.bmp"
    if not src.exists() or not sem.exists():
        return None
    return {
        "scene_id": scene_id,
        "source_bw_path": str(src.resolve()),
        "target_semantic_path": str(sem.resolve()),
        "artifact_dir": str(full_dir.parent.resolve()),
        "full_artifact_dir": str(full_dir.resolve()),
        "split": "train",
    }


def discover_scenes_from_manifest(manifest_path: Path) -> list[dict]:
    scenes: list[dict] = []
    seen: set[str] = set()
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            full_dir = Path(rec["full_artifact_dir"])
            scene_id = str(rec.get("sample_id") or full_dir.parent.name)
            if scene_id in seen:
                continue
            row = _scene_record(scene_id, full_dir)
            if row:
                seen.add(scene_id)
                scenes.append(row)
    return scenes


def discover_scenes_from_samples(samples_dir: Path) -> list[dict]:
    scenes: list[dict] = []
    if not samples_dir.is_dir():
        return scenes
    for art in sorted(samples_dir.iterdir()):
        if not art.is_dir():
            continue
        full_dir = art / "full"
        if full_dir.is_dir():
            row = _scene_record(art.name, full_dir)
        else:
            row = _scene_record(art.name, art)
        if row:
            scenes.append(row)
    return scenes


def discover_scenes(dataset_root: Path, source_dataset: Path | None) -> list[dict]:
    candidates: list[Path] = []
    for root in (dataset_root, source_dataset):
        if root is None:
            continue
        root = Path(root)
        for name in ("train_crops_bw.jsonl", "manifests/train_crops_bw.jsonl"):
            p = root / name
            if p.exists():
                candidates.append(p)

    for manifest in candidates:
        scenes = discover_scenes_from_manifest(manifest)
        if scenes:
            return scenes

    for root in (dataset_root, source_dataset):
        if root is None:
            continue
        scenes = discover_scenes_from_samples(Path(root) / "samples")
        if scenes:
            return scenes

    return []


def build_tile_manifest(scenes: list[dict], *, input_mode: str = "bw") -> list[dict]:
    rows: list[dict] = []
    from csn_v4.data.targets import load_gray_bmp

    for scene in scenes:
        bw = load_gray_bmp(scene["source_bw_path"])
        fh, fw = bw.shape[:2]
        for k in range(8):
            tfw, tfh = D4Transform(k).transformed_wh(fw, fh)
            for spec in iter_core_tiles(tfw, tfh, transform_id=k, scene_id=scene["scene_id"]):
                rows.append({
                    "scene_id": scene["scene_id"],
                    "transform_id": k,
                    "core_origin_x": spec.core_x,
                    "core_origin_y": spec.core_y,
                    "core_w": spec.core_w,
                    "core_h": spec.core_h,
                    "input_mode": input_mode,
                    "tile_uid": tile_uid(scene["scene_id"], k, spec.core_x, spec.core_y),
                    "split": scene.get("split", "train"),
                })
    return rows


def build_eval_full(scenes: list[dict], holdout: int = 3) -> list[dict]:
    out = []
    for scene in scenes[:holdout]:
        out.append({
            "sample_id": scene["scene_id"],
            "source_bw_path": scene["source_bw_path"],
            "target_semantic_path": scene["target_semantic_path"],
            "split": "eval_full",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Dataset V4 manifests")
    parser.add_argument("--dataset-root", type=Path, default=Path("E:/Shadowing/Dataset_V4"))
    parser.add_argument("--source-dataset", type=Path, default=Path("E:/Shadowing/Dataset_V3"))
    parser.add_argument("--eval-holdout", type=int, default=3)
    args = parser.parse_args()

    root = args.dataset_root
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(root, args.source_dataset)
    if not scenes:
        raise SystemExit(
            f"No scenes found under {root} or {args.source_dataset}. "
            "Expected V3 manifests/train_crops_bw.jsonl or samples/*/full/*.bmp"
        )

    tiles = build_tile_manifest(scenes)
    eval_full = build_eval_full(scenes, args.eval_holdout)

    with (manifest_dir / "scenes.jsonl").open("w", encoding="utf-8") as fh:
        for s in scenes:
            fh.write(json.dumps(s) + "\n")

    with (manifest_dir / "train_tiles_bw.jsonl").open("w", encoding="utf-8") as fh:
        for t in tiles:
            fh.write(json.dumps(t) + "\n")

    with (manifest_dir / "eval_full_images.jsonl").open("w", encoding="utf-8") as fh:
        for e in eval_full:
            fh.write(json.dumps(e) + "\n")

    summary = {"scenes": len(scenes), "tiles": len(tiles), "eval_full": len(eval_full)}
    with (manifest_dir / "build_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
