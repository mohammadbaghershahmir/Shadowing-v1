"""Dihedral augmentation for CSN-V3 multiscale samples."""
from __future__ import annotations

from typing import Any

import torch

from shadow_model.augment import apply_dihedral_tensor


def apply_dihedral_csn_sample(sample: dict[str, Any], k: int) -> dict[str, Any]:
    if k == 0:
        return sample
    out = dict(sample)
    tensor_keys = (
        "local_bw", "context_bw", "full_bw",
        "local_onehot", "local_rgb",
        "target_class", "shade_mask", "transition_mask", "black_lock", "valid_mask",
        "where_target", "level_target", "ordinal_target", "center_supervision",
    )
    for key in tensor_keys:
        if key not in out:
            continue
        t = out[key]
        if t.ndim == 3 and t.shape[0] in (1, 2, 3, 4):
            out[key] = apply_dihedral_tensor(t, k)
        elif t.ndim == 2:
            out[key] = apply_dihedral_tensor(t.unsqueeze(0), k).squeeze(0)
        elif t.ndim == 4 and key == "ordinal_target":
            out[key] = apply_dihedral_tensor(t, k)
    out["dihedral_k"] = k
    return out
