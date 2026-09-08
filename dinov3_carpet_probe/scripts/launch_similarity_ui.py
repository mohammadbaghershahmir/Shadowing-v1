#!/usr/bin/env python
"""Launch Gradio cosine-similarity click UI for a completed probe run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image

from dinov3_carpet_probe.src.similarity import build_similarity_gradio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--image", required=True, type=str, help="image stem inside run-dir")
    p.add_argument("--model", type=str, default="vitl16_lvd")
    p.add_argument("--features", type=str, default="features_global_final.npy")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    image_dir = run_dir / args.image
    img_path = image_dir / "input.png"
    feat_path = image_dir / args.model / "raw" / args.features
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    image = Image.open(img_path).convert("RGB")
    feats = np.load(feat_path)
    if feats.ndim == 4:
        feats = feats[0]
    demo = build_similarity_gradio(
        image,
        feats,
        out_dir=image_dir / args.model / "viz" / "interactive",
    )
    demo.launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
