#!/usr/bin/env python3
"""Build Dataset V3 with D4-baked 512 crops and dual manifests."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_v2.bmp_io import save_gray_bmp, save_rgb_bmp
from dataset_v2.pipeline import PilotConfig, run_pilot
from shadow_model.augment import apply_dihedral_tensor

LOGGER = logging.getLogger(__name__)

GRAY_ARTIFACTS = (
    "target_semantic.bmp", "source_bw.bmp", "shade_mask.bmp", "shade_level_id.bmp",
    "transition_mask.bmp", "black_lock.bmp", "valid_mask.bmp",
)
RGB_ARTIFACTS = ("source_oracle.bmp",)


def _apply_d4_file(src: Path, dst: Path, k: int, rgb: bool = False) -> None:
    img = np.asarray(Image.open(src).convert("RGB" if rgb else "L"), dtype=np.uint8)
    if rgb:
        t = torch.from_numpy(img).permute(2, 0, 1)
        t = apply_dihedral_tensor(t, k)
        out = t.permute(1, 2, 0).numpy()
        save_rgb_bmp(dst, out)
    else:
        t = torch.from_numpy(img).unsqueeze(0)
        t = apply_dihedral_tensor(t, k)
        save_gray_bmp(dst, t.squeeze(0).numpy())


def bake_crop_d4(canonical_dir: Path, out_dir: Path, k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAY_ARTIFACTS:
        src = canonical_dir / name
        if src.exists():
            _apply_d4_file(src, out_dir / name, k, rgb=False)
    for name in RGB_ARTIFACTS:
        src = canonical_dir / name
        if src.exists():
            _apply_d4_file(src, out_dir / name, k, rgb=True)


def find_magenta_guide(sample_id: str, guide_dir: Path) -> str | None:
    if not guide_dir.is_dir():
        return None
    for p in sorted(guide_dir.glob("*.bmp")):
        if sample_id in p.stem or p.stem in sample_id:
            return str(p)
    return None


def build_manifests(
    dataset_root: Path,
    guide_dir: Path,
    eval_holdout: int,
    min_boundary_score: float,
) -> tuple[int, int]:
    crop_csv = dataset_root / "manifests" / "crop_manifest.csv"
    if not crop_csv.exists():
        raise FileNotFoundError(f"Missing {crop_csv}; run pilot build first.")

    rows_bw: list[dict] = []
    rows_mag: list[dict] = []
    eval_full: list[dict] = []
    sample_ids = set()

    with crop_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        crop_rows = [r for r in reader if int(r.get("width", 0)) == 512 and int(r.get("height", 0)) == 512]

    crop_rows.sort(key=lambda r: -float(r.get("transition_score", r.get("score", 0)) or 0))
    holdout_samples = sorted({r["sample_id"] for r in crop_rows})[:eval_holdout]

    for row in crop_rows:
        sample_id = row["sample_id"]
        sample_ids.add(sample_id)
        artifact_dir = Path(row["artifact_dir"])
        if not artifact_dir.is_absolute():
            artifact_dir = dataset_root / artifact_dir
        if not artifact_dir.is_dir():
            continue
        trans_score = float(row.get("transition_score", row.get("score", 0)) or 0)
        if trans_score < min_boundary_score and row.get("category") != "transition_rich":
            continue

        full_dir = dataset_root / "samples" / sample_id / "full"
        if not full_dir.is_dir():
            full_dir = artifact_dir.parent.parent.parent / "full"

        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])
        group_id = f"{sample_id}__{x}_{y}"

        for k in range(8):
            if k == 0:
                local_dir = artifact_dir
            else:
                local_dir = artifact_dir.parent / f"d4_k{k}"
                if not local_dir.exists():
                    bake_crop_d4(artifact_dir, local_dir, k)

            base = {
                "sample_id": sample_id,
                "full_artifact_dir": str(full_dir),
                "local_artifact_dir": str(local_dir),
                "x": x, "y": y, "w": w, "h": h,
                "split": "train",
                "group_id": group_id,
                "transform_id": k,
                "category": row.get("category", ""),
            }
            rows_bw.append({**base, "input_mode": "bw", "aug_id": f"d4_k{k}"})

            mag = {**base, "input_mode": "magenta", "aug_id": f"d4_k{k}"}
            guide = find_magenta_guide(sample_id, guide_dir)
            if guide:
                mag["guide_path"] = guide
            rows_mag.append(mag)

    for sid in holdout_samples:
        full_dir = dataset_root / "samples" / sid / "full"
        src_bw = full_dir / "source_bw.bmp"
        if src_bw.exists():
            eval_full.append({
                "sample_id": sid,
                "source_bw_path": str(src_bw),
                "target_semantic_path": str(full_dir / "target_semantic.bmp"),
                "split": "eval_full",
            })

    manifest_dir = dataset_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    write_jsonl(manifest_dir / "train_crops_bw.jsonl", rows_bw)
    write_jsonl(manifest_dir / "train_crops_magenta.jsonl", rows_mag)
    write_jsonl(manifest_dir / "eval_full_images.jsonl", eval_full)

    dihedral_index = {"group_format": "sample_id__x_y", "transform_ids": list(range(8))}
    (manifest_dir / "dihedral_index.json").write_text(json.dumps(dihedral_index, indent=2), encoding="utf-8")

    return len(rows_bw), len(eval_full)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Dataset V3")
    parser.add_argument("--input-dir", type=Path, default=Path("E:/Shadowing/converted_output"))
    parser.add_argument("--dataset-root", type=Path, default=Path("E:/Shadowing/Dataset_V3"))
    parser.add_argument("--magenta-guide-dir", type=Path, default=Path("E:/Shadowing/converted_ready"))
    parser.add_argument("--source-dataset", type=Path, default=Path("E:/Shadowing/Dataset_V2"))
    parser.add_argument("--reuse-source", action="store_true", help="Reuse Dataset_V2 samples/manifests")
    parser.add_argument("--skip-copy", action="store_true", help="Skip sample copy (manifests + D4 bake only)")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--eval-holdout", type=int, default=3)
    parser.add_argument("--min-boundary-score", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    root = args.dataset_root
    root.mkdir(parents=True, exist_ok=True)

    samples_dir = root / "samples"
    if args.reuse_source and args.source_dataset.is_dir() and not args.skip_copy:
        src_samples = args.source_dataset / "samples"
        src_manifests = args.source_dataset / "manifests"
        if src_manifests.is_dir() and not (root / "manifests" / "crop_manifest.csv").exists():
            LOGGER.info("Copying manifests from %s", args.source_dataset)
            shutil.copytree(src_manifests, root / "manifests", dirs_exist_ok=True)
        if src_samples.is_dir():
            if not samples_dir.exists():
                LOGGER.info("Copying samples from %s -> %s", args.source_dataset, root)
                shutil.copytree(src_samples, samples_dir, dirs_exist_ok=True)
            else:
                LOGGER.info("samples/ already exists at %s — skipping copy", samples_dir)
    elif args.skip_copy and samples_dir.is_dir():
        LOGGER.info("--skip-copy: using existing samples at %s", samples_dir)

    if not (root / "manifests" / "crop_manifest.csv").exists():
        LOGGER.info("Running Dataset V2 pilot build into %s", root)
        cfg = PilotConfig(
            input_dir=args.input_dir,
            output_dir=root,
            sample_count=args.sample_count,
            seed=args.seed,
            crop_sizes=args.crop_sizes,
            recursive=True,
            rotations=[0],
            include_flips=False,
        )
        run_pilot(cfg)

    n_bw, n_eval = build_manifests(root, args.magenta_guide_dir, args.eval_holdout, args.min_boundary_score)
    LOGGER.info("Wrote %d BW/magenta manifest rows, %d full-image eval entries", n_bw, n_eval)
    print(f"Dataset V3 ready at {root}")
    print(f"  train_crops_bw.jsonl rows: {n_bw}")
    print(f"  eval_full_images.jsonl: {n_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
