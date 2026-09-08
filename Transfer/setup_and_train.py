#!/usr/bin/env python3
"""
CSN-V4 Transfer Setup — from raw converted_output to training
==============================================================
Starts from ground-truth BMPs (converted_output) and runs the full chain:

  converted_output  →  Dataset_V3  →  Dataset_V4  →  materialize  →  train

Usage:
    python setup_and_train.py ^
        --input-dir "F:/Shadowing/converted_output" ^
        --dino-root "D:/Dino/dinov3" ^
        --dino-checkpoint "D:/Dino/dinov3/.../dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ^
        --work-root "F:/Shadowing" ^
        --device cuda ^
        --steps 5000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG_TEMPLATE = _ROOT / "configs" / "csn_v4_generalization.yaml"
CONFIG_ACTIVE = _ROOT / "configs" / "csn_v4_generalization_local.yaml"


def patch_config(
    dataset_v4_root: Path,
    dino_root: Path,
    dino_checkpoint: Path,
) -> Path:
    """Rewrite absolute paths in the config template for this machine."""
    text = CONFIG_TEMPLATE.read_text(encoding="utf-8")

    replacements = {
        'dataset_root: "E:/Shadowing/Dataset_V4"': f'dataset_root: "{dataset_v4_root.as_posix()}"',
        'repo_root: "E:/Dino/dinov3"': f'repo_root: "{dino_root.as_posix()}"',
        'checkpoint: "E:/Dino/dinov3/dinov3_carpet_probe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"':
            f'checkpoint: "{dino_checkpoint.as_posix()}"',
        'cache_dir: "E:/Shadowing/Dataset_V4/global_tokens_csn_v4_bw"':
            f'cache_dir: "{(dataset_v4_root / "global_tokens_csn_v4_bw").as_posix()}"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    CONFIG_ACTIVE.write_text(text, encoding="utf-8")
    print(f"[OK] Config written: {CONFIG_ACTIVE}")
    return CONFIG_ACTIVE


def discover_bmps(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.bmp") if p.is_file())


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    rc = subprocess.call(cmd, cwd=_ROOT)
    if rc != 0:
        print(f"\n[ERROR] {desc} failed with exit code {rc}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CSN-V4: converted_output → V3 → V4 → train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python setup_and_train.py \\\n"
            '    --input-dir "F:/Shadowing/converted_output" \\\n'
            '    --work-root "F:/Shadowing" \\\n'
            '    --dino-root "D:/Dino/dinov3" \\\n'
            '    --dino-checkpoint "D:/Dino/dinov3/.../weights.pth"\n'
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Raw ground-truth folder (converted_output with 5-color BMP files)",
    )
    parser.add_argument(
        "--work-root", type=Path, default=None,
        help="Parent folder for Dataset_V3 and Dataset_V4 (default: parent of --input-dir)",
    )
    parser.add_argument("--dataset-v3-root", type=Path, default=None,
                        help="Override V3 output path (default: {work-root}/Dataset_V3)")
    parser.add_argument("--dataset-v4-root", type=Path, default=None,
                        help="Override V4 output path (default: {work-root}/Dataset_V4)")
    parser.add_argument("--dino-root", type=Path, required=True,
                        help="Path to dinov3 repository root")
    parser.add_argument("--dino-checkpoint", type=Path, required=True,
                        help="Path to dinov3 .pth weights file")
    parser.add_argument("--sample-count", type=int, default=24,
                        help="Number of scenes to build from converted_output")
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[512],
                        help="Crop sizes for V3 pilot build")
    parser.add_argument("--magenta-guide-dir", type=Path, default=None,
                        help="Optional magenta guide folder (converted_ready)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training optimizer steps")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-v3", action="store_true",
                        help="Skip V3 build (reuse existing Dataset_V3)")
    parser.add_argument("--skip-v4-manifests", action="store_true",
                        help="Skip V4 manifest build")
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from checkpoint .pt")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    work_root = (args.work_root or input_dir.parent).resolve()
    dataset_v3 = (args.dataset_v3_root or work_root / "Dataset_V3").resolve()
    dataset_v4 = (args.dataset_v4_root or work_root / "Dataset_V4").resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir not found: {input_dir}")
    bmps = discover_bmps(input_dir)
    if not bmps and not args.skip_v3:
        raise SystemExit(
            f"No .bmp files under {input_dir}\n"
            "converted_output must contain 5-color ground-truth BMP images "
            "(OUTLINE=black, UNSHADED=white, INDEX_200/150/100 grays)."
        )
    if not args.skip_v3:
        print(f"[OK] Found {len(bmps)} BMP file(s) in {input_dir}")
    if not args.dino_root.is_dir():
        raise SystemExit(f"--dino-root not found: {args.dino_root}")
    if not args.dino_checkpoint.is_file():
        raise SystemExit(f"--dino-checkpoint not found: {args.dino_checkpoint}")

    cfg = patch_config(dataset_v4, args.dino_root, args.dino_checkpoint)

    # Step 1: converted_output → Dataset_V3 (V2 pilot + V3 manifests + scene full/)
    if not args.skip_v3:
        v3_cmd = [
            PYTHON, str(_ROOT / "tools" / "build_dataset_v3.py"),
            "--input-dir", str(input_dir),
            "--dataset-root", str(dataset_v3),
            "--sample-count", str(args.sample_count),
            "--crop-sizes", *[str(s) for s in args.crop_sizes],
            "--seed", str(args.seed),
        ]
        if args.magenta_guide_dir is not None:
            v3_cmd += ["--magenta-guide-dir", str(args.magenta_guide_dir)]
        rc = run(v3_cmd, "Step 1: Build Dataset_V3 from converted_output")
        if rc != 0:
            return rc
    else:
        print("[SKIP] Dataset_V3 build")
        if not (dataset_v3 / "samples").is_dir():
            raise SystemExit(f"--skip-v3 but no samples/ in {dataset_v3}")

    # Step 2: Dataset_V3 → Dataset_V4 manifests
    if not args.skip_v4_manifests:
        rc = run([
            PYTHON, str(_ROOT / "scripts" / "prepare_csn_v4_dataset.py"),
            "--dataset-root", str(dataset_v4),
            "--source-dataset", str(dataset_v3),
            "--config", str(cfg),
            "--seed", str(args.seed),
        ], "Step 2: Build Dataset_V4 manifests from V3 scenes")
        if rc != 0:
            return rc
    else:
        print("[SKIP] V4 manifests")

    # Step 3: Materialize tiles
    if not args.skip_materialize:
        rc = run([
            PYTHON, str(_ROOT / "scripts" / "materialize_csn_v4_tiles.py"),
            "--config", str(cfg),
            "--dataset-root", str(dataset_v4),
        ], "Step 3: Materialize tiles to NPZ")
        if rc != 0:
            return rc
    else:
        print("[SKIP] Materialize")

    # Step 4: Train
    if not args.skip_train:
        train_cmd = [
            PYTHON, str(_ROOT / "scripts" / "train_csn_v4.py"),
            "--config", str(cfg),
            "--device", args.device,
            "--workers", str(args.workers),
            "--steps", str(args.steps),
            "--skip-initial-eval",
        ]
        if args.resume:
            train_cmd += ["--resume", str(args.resume)]
        rc = run(train_cmd, "Step 4: Train CSN-V4")
        if rc != 0:
            return rc
    else:
        print("[SKIP] Training")

    print(f"\n{'='*60}")
    print("  ALL DONE!")
    print(f"  Input (converted_output): {input_dir}")
    print(f"  Dataset_V3 (intermediate): {dataset_v3}")
    print(f"  Dataset_V4: {dataset_v4}")
    print(f"  Config: {cfg}")
    print(f"  Runs: {_ROOT / 'runs'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
