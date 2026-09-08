"""Deterministic dihedral augmentation for semantic crops."""
from __future__ import annotations

from typing import Any

import torch


def apply_dihedral_tensor(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply D4 transform k in {0..7} to CHW or HW tensor."""
    if k == 0:
        return x
    if k == 1:
        return torch.flip(x, dims=[-1])
    if k == 2:
        return torch.flip(x, dims=[-2])
    if k == 3:
        return torch.flip(torch.flip(x, dims=[-1]), dims=[-2])
    if k == 4:
        return torch.rot90(x, k=1, dims=[-2, -1])
    if k == 5:
        return torch.rot90(x, k=2, dims=[-2, -1])
    if k == 6:
        return torch.rot90(x, k=3, dims=[-2, -1])
    if k == 7:
        return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])
    raise ValueError(f"Invalid dihedral k={k}")


def transform_crop_box(box: torch.Tensor, k: int) -> torch.Tensor:
    """Transform normalized crop_box [x,y,w,h] under D4."""
    x, y, w, h = box.unbind(-1)
    if k == 0:
        return box
    if k == 1:
        return torch.stack([1 - x - w, y, w, h], dim=-1)
    if k == 2:
        return torch.stack([x, 1 - y - h, w, h], dim=-1)
    if k == 3:
        return torch.stack([1 - x - w, 1 - y - h, w, h], dim=-1)
    if k == 4:
        return torch.stack([y, 1 - x - w, h, w], dim=-1)
    if k == 5:
        return torch.stack([1 - x - w, 1 - y - h, w, h], dim=-1)
    if k == 6:
        return torch.stack([1 - y - h, x, h, w], dim=-1)
    if k == 7:
        return torch.stack([1 - y - h, 1 - x - w, h, w], dim=-1)
    raise ValueError(k)


def apply_dihedral_sample(sample: dict[str, Any], k: int) -> dict[str, Any]:
    if k == 0:
        return sample
    out = dict(sample)
    for key in ("local_input_rgb", "local_input_onehot", "local_input_rgb_norm"):
        if key in out:
            out[key] = apply_dihedral_tensor(out[key], k)
    for key in ("target_label", "candidate_mask", "image_valid_mask", "central_valid_mask"):
        if key in out:
            t = out[key]
            if t.ndim == 2:
                out[key] = apply_dihedral_tensor(t.unsqueeze(0), k).squeeze(0)
            else:
                out[key] = apply_dihedral_tensor(t, k)
    if "crop_box_normalized" in out:
        out["crop_box_normalized"] = transform_crop_box(out["crop_box_normalized"], k)
    out["dihedral_k"] = k
    return out
