"""Variant catalogs for pilot and full profiles."""

from __future__ import annotations

from shdowing.ai2bmp.synthetic.recipe import CorruptionRecipe


def pilot_variants(global_seed: int, sample_id: str) -> list[CorruptionRecipe]:
    from shdowing.ai2bmp.synthetic.recipe import derive_seed

    specs = [
        ("clean_identity", "clean", "clean_identity", "easy", True, []),
        ("palette_shift_mild", "palette_shift", "global_palette_shift_mild", "easy", True, [{"op": "palette_shift", "delta_e_band": "mild"}]),
        ("palette_shift_strong", "palette_shift", "global_palette_shift_strong", "hard", True, [{"op": "palette_shift", "delta_e_band": "strong"}]),
        ("intra_region_easy", "intra_region", "smooth_region_variation_easy", "easy", True, [{"op": "intra_region", "band": "easy"}]),
        ("intra_region_hard", "intra_region", "smooth_region_variation_hard", "hard", True, [{"op": "intra_region", "band": "hard"}]),
        ("boundary_antialias", "boundary_stress", "antialias_boundary", "medium", True, [{"op": "boundary_stress", "kind": "antialias"}]),
        ("boundary_bleed", "boundary_stress", "color_bleed", "medium", True, [{"op": "boundary_stress", "kind": "bleed"}]),
        ("shade_family_correlated", "shade_family", "correlated_family_recolor", "medium", True, [{"op": "shade_family", "kind": "correlated"}]),
        ("shade_family_collapse", "shade_family", "two_level_collapse", "hard", True, [{"op": "shade_family", "kind": "collapse"}]),
        ("resolution_445", "resolution_codec", "scale_0_445", "medium", True, [{"op": "resolution", "scale": 0.445}]),
        ("jpeg_artifacts", "resolution_codec", "jpeg_roundtrip_q50", "medium", True, [{"op": "jpeg", "quality": 50}]),
        ("mixed_palette_boundary", "mixed", "palette_shift_plus_boundary_softening", "medium", True, [{"op": "palette_shift", "delta_e_band": "moderate"}, {"op": "boundary_stress", "kind": "blur"}]),
        ("mixed_downscale_boundary", "mixed", "downscale_plus_boundary_softening", "hard", True, [{"op": "resolution", "scale": 0.445}, {"op": "boundary_stress", "kind": "blur"}]),
        ("severe_low_contrast", "severe", "very_low_contrast", "severe", False, [{"op": "palette_shift", "delta_e_band": "stress", "chroma_only": True}]),
        ("severe_inconsistency", "severe", "extreme_intra_region_inconsistency", "severe", False, [{"op": "intra_region", "band": "extreme"}]),
    ]
    out: list[CorruptionRecipe] = []
    for vid, family, name, diff, dense, ops in specs:
        seed = derive_seed(global_seed, sample_id, vid)
        out.append(
            CorruptionRecipe(
                variant_id=vid,
                family=family,
                name=name,
                seed=seed,
                operations=ops,
                difficulty=diff,
                default_dense_supervision=dense,
                use_for_confidence_or_robustness=not dense,
            )
        )
    return out


def full_variants(global_seed: int, sample_id: str) -> list[CorruptionRecipe]:
    # Full 48-variant catalog - extend pilot with additional named recipes
    base = pilot_variants(global_seed, sample_id)
    from shdowing.ai2bmp.synthetic.recipe import derive_seed

    extra_specs = [
        (f"full_extra_{i}", "mixed", f"full_recipe_{i}", "medium", True, [{"op": "intra_region", "band": "medium"}])
        for i in range(33)
    ]
    for vid, family, name, diff, dense, ops in extra_specs:
        base.append(
            CorruptionRecipe(
                variant_id=vid,
                family=family,
                name=name,
                seed=derive_seed(global_seed, sample_id, vid),
                operations=ops,
                difficulty=diff,
                default_dense_supervision=dense,
            )
        )
    return base[:48]
