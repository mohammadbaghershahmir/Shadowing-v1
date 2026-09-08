"""CLI for AI2BMP Synthetic Behavior Dataset V1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shdowing.ai2bmp.io_utils import clean_path  # noqa: E402
from shdowing.ai2bmp.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build AI2BMP Synthetic Behavior Dataset V1 from indexed carpet-map BMPs."
    )
    p.add_argument("--input-dir", type=str, required=True, help="Source indexed BMP folder.")
    p.add_argument("--output-dir", type=str, required=True, help="Output dataset root (must not be inside input-dir).")
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot", help="pilot=limited review run; full=entire folder.")
    p.add_argument("--sample-count", type=int, default=1, help="Number of designs in pilot mode.")
    p.add_argument("--sample-file", type=str, default="1-111-6B.bmp", help="Preferred pilot BMP filename.")
    p.add_argument("--seed", type=int, default=42, help="Global deterministic seed.")
    p.add_argument("--crop-sizes", type=int, nargs="+", default=[256, 512, 1024], help="Native crop sizes (must divide by 16).")
    p.add_argument("--variants-profile", choices=["pilot", "full"], default="pilot", help="Synthetic variant catalog size.")
    p.add_argument("--recursive", action="store_true", help="Recursively discover BMP files.")
    p.add_argument("--materialize-crops", action="store_true", help="Write representative crop folders for QA.")
    p.add_argument("--resume", action="store_true", help="Allow writing into an existing nonempty output directory.")
    p.add_argument("--verify-hashes", action="store_true", help="Verify file hashes when loading dataset.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = clean_path(args.input_dir)
    output_dir = clean_path(args.output_dir)
    if input_dir is None or output_dir is None:
        raise SystemExit("--input-dir and --output-dir are required")

    cfg = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=args.mode,
        sample_count=args.sample_count,
        sample_file=args.sample_file,
        seed=args.seed,
        crop_sizes=tuple(args.crop_sizes),
        variants_profile=args.variants_profile,
        recursive=bool(args.recursive),
        materialize_crops=bool(args.materialize_crops),
        resume=bool(args.resume),
        verify_hashes=bool(args.verify_hashes),
    )
    stats = run_pipeline(cfg)
    print("AI2BMP build complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
