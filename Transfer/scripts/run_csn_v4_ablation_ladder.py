#!/usr/bin/env python3
"""CSN-V4 controlled ablation + inference experiment runner.

Uses the same Dataset_V4 split, seed, and optimizer_steps for each fusion config.
Do NOT claim accuracy gains without comparing held-out scene metrics (full 8 D4).

Example:
  python scripts/run_csn_v4_ablation_ladder.py --print-only
  python scripts/run_csn_v4_ablation_ladder.py --run-dir D:/Shadowing_runs/ablations
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

ABLATIONS = [
    ("A0_stem_convnext", "configs/ablations/csn_v4_ablation_stem_convnext.yaml"),
    ("A1_plus_local", "configs/ablations/csn_v4_ablation_plus_local.yaml"),
    ("A2_plus_context", "configs/ablations/csn_v4_ablation_plus_context.yaml"),
    ("A3_full", "configs/ablations/csn_v4_ablation_full.yaml"),
]

INFER_VARIANT_FLAGS = [
    ("core_paste", []),
    ("soft_overlap", ["--blend-mode", "soft_overlap"]),
    ("d4_tta", ["--d4-tta"]),
    ("soft_overlap_d4_tta", ["--blend-mode", "soft_overlap", "--d4-tta"]),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V4 ablation ladder")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=_ROOT / "runs" / "ablations")
    parser.add_argument("--steps", type=int, default=None, help="Override optimizer_steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-checkpoint", type=Path, default=None,
                        help="If set, only run held-out eval variants on this checkpoint")
    args = parser.parse_args(argv)

    print("CSN-V4 Ablation Ladder (held-out comparison required before keeping a branch)\n")
    for code, cfg in ABLATIONS:
        cmd = [
            PYTHON, str(_ROOT / "scripts" / "train_csn_v4.py"),
            "--config", str(_ROOT / cfg),
            "--device", args.device,
            "--workers", "0",
            "--skip-initial-eval",
            "--run-dir", str(args.run_dir / code),
        ]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]
        print(f"{code}: {cfg}")
        print(f"  TRAIN: {' '.join(cmd)}")
        if not args.print_only and not args.skip_train and args.eval_checkpoint is None:
            rc = subprocess.call(cmd, cwd=_ROOT)
            if rc != 0:
                return rc
        # Final comparison must use full 8 orientations on holdout scenes
        eval_cmd = [
            PYTHON, str(_ROOT / "scripts" / "evaluate_csn_v4.py"),
            "--checkpoint", str(args.run_dir / code / "best.pt"),
            "--config", str(_ROOT / cfg),
            "--suite", "val_scene_holdout",
            "--output-dir", str(args.run_dir / code / "eval_holdout_d4x8"),
            "--device", args.device,
            "--all-orientations",
        ]
        print(f"  EVAL8: {' '.join(eval_cmd)}")
        print()

    print("Inference seam / TTA variants (measure accuracy + runtime; keep constraints):")
    for name, flags in INFER_VARIANT_FLAGS:
        print(f"  {name}: evaluate_csn_v4.py {' '.join(flags)}")

    print("\nAcceptance:")
    print("  - Keep a DINO branch only if val_scene_holdout (8 D4) improves global_miou_fp")
    print("  - Spatial / train-scene scores are diagnostic only — never best.pt in generalization")
    print("  - Do not report numerical gains without a real held-out comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
