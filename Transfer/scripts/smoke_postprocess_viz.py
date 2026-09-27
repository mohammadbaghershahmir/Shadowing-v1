#!/usr/bin/env python3
"""Write one sample raw + postprocessed pred_viz pair (no model required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.constants import GRAY_BLACK, GRAY_SHADE_100, GRAY_SHADE_200, GRAY_WHITE
from csn_v4.postprocess import PostprocessConfig, run_postprocess
from csn_v4.training.pred_viz import compose_pred_with_fixed_input


def main() -> int:
    out = _ROOT / "runs" / "_postprocess_smoke" / "pred_viz" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    sid = "smoke_motif"

    h, w = 64, 96
    src = np.full((h, w), GRAY_WHITE, dtype=np.uint8)
    src[0, :] = src[-1, :] = src[:, 0] = src[:, -1] = GRAY_BLACK
    src[8:56, 8] = src[8:56, 88] = src[8, 8:89] = src[55, 8:89] = GRAY_BLACK

    raw = src.copy()
    raw[10:54, 10:88] = GRAY_SHADE_200
    raw[30, 40] = GRAY_SHADE_100

    from csn_v4.constants import NUM_CLASSES, SEMANTIC_GRAY_TO_CLASS

    scores = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    for gv, cls in SEMANTIC_GRAY_TO_CLASS.items():
        scores[cls] = (raw == gv).astype(np.float32) * 0.8 + 0.05
    scores[:, 30, 40] = 0.25
    scores /= scores.sum(axis=0, keepdims=True).clip(min=1e-6)

    raw_path = out / f"{sid}__t0_pred_full.png"
    Image.fromarray(compose_pred_with_fixed_input(src, raw)).save(raw_path)

    result = run_postprocess(
        src, raw, scores,
        config=PostprocessConfig(profile="conservative_cleanup", confidence_threshold=0.5),
    )
    assert np.array_equal(result.raw_gray, raw)

    post_path = out / f"{sid}__t0_pred_full__postprocessed.png"
    Image.fromarray(compose_pred_with_fixed_input(src, result.repaired_gray)).save(post_path)
    report_path = out / f"{sid}__t0_postprocess_report.json"
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")

    print("RAW_PATH=", raw_path.resolve())
    print("POST_PATH=", post_path.resolve())
    print("REPORT_PATH=", report_path.resolve())
    print("REPORT=", result.report.summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
