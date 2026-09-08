#!/usr/bin/env python3
"""CSN-V4 experiment ladder runner (E0-E7). Does not train — documents commands."""
from __future__ import annotations

EXPERIMENTS = [
    ("E0", "Phase-0 baseline", "python scripts/eval_csn_phase0.py --output-dir eval_phase0"),
    ("E1", "V4 geometry + categorical loss", "python scripts/train_csn_v4.py --config configs/csn_v4_bw_overfit_full_crop.yaml --skip-audit"),
    ("E2", "Deterministic D4 (built into Dataset_V4)", "python tools/build_dataset_v4.py && python scripts/train_csn_v4.py"),
    ("E3", "Halo weight ablation", "Edit data.halo_loss_weight: 1.0 vs 0.35 in config"),
    ("E4", "Branch ablation", "Disable fusion branches via config flags (manual)"),
    ("E5", "Paired-tile consistency", "Enable when implemented in CSNV4Loss"),
    ("E6", "D4 TTA inference", "Run tiled inference 8x + inverse average"),
    ("E7", "ROI global attention", "Only if E4 shows global branch harm"),
]


def main() -> int:
    print("CSN-V4 Experiment Ladder\n")
    for code, name, cmd in EXPERIMENTS:
        print(f"{code}: {name}")
        print(f"  {cmd}\n")
    print("Acceptance gates:")
    print("  - valid-pixel accuracy >= 99%")
    print("  - presence-aware mIoU >= 0.98")
    print("  - worst D4 orientation within 1pp of identity")
    print("  - full-image within 1pp of direct trusted-core")
    print("  - small-component recall >= 0.95 when global acc >= 99%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
