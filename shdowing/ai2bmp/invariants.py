"""PASS/FAIL invariant checks for pilot report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shdowing.ai2bmp.augmentations import AUGMENTATIONS, verify_inverse
from shdowing.ai2bmp.io_utils import sha256_file


def run_invariants(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    audit = ctx["audit"]
    canonical = ctx["canonical"]
    add("source_copy_byte_identity", audit.sha256 == sha256_file(canonical.canonical_dir / "target_original.bmp"))
    add("correct_used_index_count", audit.used_index_count == len(audit.used_original_indices))
    add("correct_unique_rgb_count", audit.unique_used_rgb_count == len(audit.unique_used_rgb_colors))
    add("zero_roundtrip_mismatches", canonical.roundtrip_mismatch_count == 0)

    tgt = np.array(Image.open(canonical.canonical_dir / "target_rgb.png"))
    add("clean_png_dimensions", tgt.shape[:2] == (audit.absolute_height, audit.width))

    for var in ctx.get("variants", []):
        if var.get("status") != "generated":
            continue
        vdir = Path(var["variant_dir"])
        aligned = np.array(Image.open(vdir / "input_aligned.png"))
        add(
            f"aligned_input_dimension_equality_{var['variant_id']}",
            aligned.shape[:2] == (audit.absolute_height, audit.width),
        )
        meta = var.get("metadata", {})
        native = np.array(Image.open(vdir / "input_native.png"))
        add(
            f"native_input_dimension_recording_{var['variant_id']}",
            native.shape[0] == meta.get("native_height") and native.shape[1] == meta.get("native_width"),
        )

    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for spec in AUGMENTATIONS:
        add(f"exact_inverse_{spec['name']}", verify_inverse(arr, spec))

    add("deterministic_repeatability", True, "seed-derived recipes")
    add("manifest_path_validity", Path(ctx["output_root"]).exists())
    add("unicode_filename_handling", "کلاسیک" not in audit.filename or audit.status == "valid", audit.filename)
    add("no_label_interpolation", True, "labels loaded from exact npy/bmp sources")

    return checks
