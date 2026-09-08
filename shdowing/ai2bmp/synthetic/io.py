"""Write synthetic variant artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shdowing.ai2bmp.canonical.artifacts import CanonicalArtifacts
from shdowing.ai2bmp.io_utils import ensure_dir, save_index_bmp, save_rgb_png, write_json
from shdowing.ai2bmp.synthetic.engine import apply_recipe
from shdowing.ai2bmp.synthetic.recipe import CorruptionRecipe, rng_for


def generate_variant(
    canonical: CanonicalArtifacts,
    recipe: CorruptionRecipe,
    boundary_mask: np.ndarray,
    labels_meta: dict[str, Any],
) -> dict[str, Any]:
    if recipe.status == "not_applicable":
        return {"variant_id": recipe.variant_id, "status": "not_applicable", "reason": "no defensible shade group"}

    variant_dir = ensure_dir(canonical.design_dir / "inputs" / "synthetic" / recipe.variant_id)
    rng = rng_for(recipe.seed)

    aligned, native, modified, op_meta = apply_recipe(
        canonical.rgb,
        canonical.index_map,
        canonical.palette,
        boundary_mask,
        recipe.operations,
        rng,
    )

    save_rgb_png(variant_dir / "input_native.png", native)
    save_rgb_png(variant_dir / "input_aligned.png", aligned)
    save_index_bmp(variant_dir / "modified_pixels_mask.bmp", (modified.astype(np.uint8) * 255))

    valid_supervision = np.ones(aligned.shape[:2], dtype=np.uint8) * 255
    if not recipe.default_dense_supervision:
        valid_supervision[:] = 0
    save_index_bmp(variant_dir / "valid_supervision_mask.bmp", valid_supervision)

    write_json(variant_dir / "corruption_recipe.json", recipe.to_dict())

    h, w = canonical.rgb.shape[:2]
    nh, nw = native.shape[:2]
    metadata = {
        "variant_id": recipe.variant_id,
        "family": recipe.family,
        "seed": recipe.seed,
        "source_width": w,
        "source_height": h,
        "native_width": nw,
        "native_height": nh,
        "aligned_width": w,
        "aligned_height": h,
        "default_dense_supervision": recipe.default_dense_supervision,
        "use_for_confidence_or_robustness": recipe.use_for_confidence_or_robustness,
        **op_meta,
    }
    write_json(variant_dir / "metadata.json", metadata)

    # clean identity copy
    if recipe.family == "clean":
        clean_dir = ensure_dir(canonical.design_dir / "inputs" / "clean")
        save_rgb_png(clean_dir / "input_native.png", canonical.rgb)
        save_rgb_png(clean_dir / "input_aligned.png", canonical.rgb)

    return {
        "variant_id": recipe.variant_id,
        "family": recipe.family,
        "status": "generated",
        "variant_dir": str(variant_dir),
        "metadata": metadata,
        "recipe": recipe.to_dict(),
    }
