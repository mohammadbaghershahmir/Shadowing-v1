"""Exact 90-degree rotations and axis flips with inverse verification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AUG_ROT0 = "rot0"
AUG_ROT90 = "rot90"
AUG_ROT180 = "rot180"
AUG_ROT270 = "rot270"
AUG_FLIP_H = "flip_horizontal"
AUG_FLIP_V = "flip_vertical"

ROTATIONS = {
    0: AUG_ROT0,
    90: AUG_ROT90,
    180: AUG_ROT180,
    270: AUG_ROT270,
}


@dataclass(frozen=True)
class AugmentOp:
    name: str
    k_rot: int = 0
    flip_axis: int | None = None


def op_from_name(name: str) -> AugmentOp:
    mapping = {
        AUG_ROT0: AugmentOp(AUG_ROT0, 0, None),
        AUG_ROT90: AugmentOp(AUG_ROT90, 1, None),
        AUG_ROT180: AugmentOp(AUG_ROT180, 2, None),
        AUG_ROT270: AugmentOp(AUG_ROT270, 3, None),
        AUG_FLIP_H: AugmentOp(AUG_FLIP_H, 0, 1),
        AUG_FLIP_V: AugmentOp(AUG_FLIP_V, 0, 0),
    }
    if name not in mapping:
        raise ValueError(f"Unknown augmentation {name!r}")
    return mapping[name]


def apply_op(arr: np.ndarray, op: AugmentOp) -> np.ndarray:
    out = arr
    if op.k_rot:
        out = np.rot90(out, k=op.k_rot)
    if op.flip_axis is not None:
        out = np.flip(out, axis=op.flip_axis)
    return np.ascontiguousarray(out.copy())


def inverse_op(arr: np.ndarray, op: AugmentOp) -> np.ndarray:
    out = arr
    if op.flip_axis is not None:
        out = np.flip(out, axis=op.flip_axis)
    if op.k_rot:
        out = np.rot90(out, k=(4 - op.k_rot) % 4)
    return np.ascontiguousarray(out.copy())


def verify_inverse_exact(original: np.ndarray, op: AugmentOp) -> bool:
    transformed = apply_op(original, op)
    restored = inverse_op(transformed, op)
    return bool(np.array_equal(original, restored))


def list_augmentation_ops(
    rotations: list[int],
    include_flips: bool,
) -> list[AugmentOp]:
    ops: list[AugmentOp] = []
    for deg in rotations:
        name = ROTATIONS.get(deg)
        if name is None:
            raise ValueError(f"Unsupported rotation {deg}; use 0/90/180/270")
        ops.append(op_from_name(name))
    if include_flips:
        ops.append(op_from_name(AUG_FLIP_H))
        ops.append(op_from_name(AUG_FLIP_V))
    return ops
