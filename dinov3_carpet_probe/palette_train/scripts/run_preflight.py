"""Run raw-cardinality and JPG-observability preflight analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dinov3_carpet_probe.palette_train.src.preflight_analysis import run_preflight  # noqa: E402
from dinov3_carpet_probe.src.io_utils import load_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dinov3_carpet_probe" / "palette_train" / "configs" / "palette_train.yaml",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    report = run_preflight(cfg, limit_samples=args.limit_samples)
    out_path = Path(cfg["manifest_dir"]) / "preflight_report.json"
    write_json(out_path, report)
    print(f"preflight={out_path}")
    print(f"excluded_over_12_count={report['excluded_over_12_count']}")
    print(f"trainable_count={report['trainable_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
