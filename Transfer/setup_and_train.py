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

import yaml

_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
CONFIG_TEMPLATE = _ROOT / "configs" / "csn_v4_generalization.yaml"
CONFIG_ACTIVE = _ROOT / "configs" / "csn_v4_generalization_local.yaml"


def _set_path(node: dict, keys: list[str], value: str) -> None:
    cur = node
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def patch_config(
    dataset_v4_root: Path,
    dino_root: Path,
    dino_checkpoint: Path,
    *,
    input_mode: str | None = None,
) -> Path:
    """Structurally rewrite paths in the config template and verify on reload."""
    if not CONFIG_TEMPLATE.is_file():
        raise FileNotFoundError(f"Missing config template: {CONFIG_TEMPLATE}")
    raw = yaml.safe_load(CONFIG_TEMPLATE.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw)}")

    cache_dir = (dataset_v4_root / "global_tokens_csn_v4_bw").as_posix()
    _set_path(raw, ["data", "dataset_root"], dataset_v4_root.as_posix())
    _set_path(raw, ["model", "dino", "repo_root"], dino_root.as_posix())
    _set_path(raw, ["model", "dino", "checkpoint"], dino_checkpoint.as_posix())
    _set_path(raw, ["model", "dino", "cache_dir"], cache_dir)
    if input_mode is not None:
        _set_path(raw, ["data", "input_mode"], input_mode)

    CONFIG_ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_ACTIVE.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    check = yaml.safe_load(CONFIG_ACTIVE.read_text(encoding="utf-8")) or {}
    errors = []
    got_root = ((check.get("data") or {}).get("dataset_root"))
    got_dino = ((check.get("model") or {}).get("dino") or {})
    if got_root != dataset_v4_root.as_posix():
        errors.append(f"dataset_root: got {got_root!r}")
    if got_dino.get("repo_root") != dino_root.as_posix():
        errors.append(f"dino.repo_root: got {got_dino.get('repo_root')!r}")
    if got_dino.get("checkpoint") != dino_checkpoint.as_posix():
        errors.append(f"dino.checkpoint: got {got_dino.get('checkpoint')!r}")
    if got_dino.get("cache_dir") != cache_dir:
        errors.append(f"dino.cache_dir: got {got_dino.get('cache_dir')!r}")
    if input_mode is not None and (check.get("data") or {}).get("input_mode") != input_mode:
        errors.append(f"input_mode: got {(check.get('data') or {}).get('input_mode')!r}")
    if errors:
        raise RuntimeError("Config patch verification failed:\n  - " + "\n  - ".join(errors))

    print(f"[OK] Config written and verified: {CONFIG_ACTIVE}")
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
                        help="Path to dinov3 repository root (parent of the dinov3 package)")
    parser.add_argument("--dino-checkpoint", type=Path, required=True,
                        help="Path to dinov3 .pth weights file")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Scenes to ingest (0 = all eligible; >0 = family-stratified cap)",
    )
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
    parser.add_argument(
        "--input-mode",
        type=str,
        default=None,
        choices=["bw", "indexed_guided"],
        help="Override data.input_mode in the generated config",
    )
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

    cfg = patch_config(
        dataset_v4,
        args.dino_root.resolve(),
        args.dino_checkpoint.resolve(),
        input_mode=args.input_mode,
    )

    # Validate resolved training paths before kicking off the pipeline.
    check = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    dino = ((check.get("model") or {}).get("dino") or {})
    for label, p in (
        ("dataset_root (will be created)", dataset_v4),
        ("dino.repo_root", Path(dino.get("repo_root", ""))),
        ("dino.checkpoint", Path(dino.get("checkpoint", ""))),
    ):
        if label.startswith("dataset_root"):
            continue
        if not p.exists():
            raise SystemExit(f"Resolved path missing for {label}: {p}")
    print(f"[OK] Resolved paths validated (dino + config)")

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
