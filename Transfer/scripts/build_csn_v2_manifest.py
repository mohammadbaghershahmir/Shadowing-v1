#!/usr/bin/env python3
"""Build CSN-V2 train_crops.jsonl from Dataset V2 crop_manifest.csv."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CSN-V2 training manifest.")
    parser.add_argument("--dataset-root", type=Path, default=Path("E:/Shadowing/Dataset_V2"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=512)
    args = parser.parse_args(argv)

    root = args.dataset_root
    crop_csv = root / "manifests" / "crop_manifest.csv"
    if not crop_csv.is_file():
        print(f"Missing {crop_csv}")
        return 1

    out_path = args.out or (root / "manifests" / "train_crops.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with crop_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            w, h = int(row.get("width", 0)), int(row.get("height", 0))
            if w != args.crop_size or h != args.crop_size:
                continue
            artifact_dir = Path(row["artifact_dir"])
            if not artifact_dir.is_absolute():
                artifact_dir = root / artifact_dir
            sample_id = row.get("sample_id", "")
            full_dir = root / "samples" / sample_id / "full"
            if not full_dir.is_dir():
                full_dir = artifact_dir.parent.parent.parent / "full"
            rec = {
                "sample_id": sample_id,
                "full_artifact_dir": str(full_dir),
                "local_artifact_dir": str(artifact_dir),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "w": w,
                "h": h,
                "split": "train",
                "group_id": sample_id,
                "aug_id": "canonical",
            }
            if (artifact_dir / "source_bw.bmp").exists():
                records.append(rec)

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} crops to {out_path}")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
