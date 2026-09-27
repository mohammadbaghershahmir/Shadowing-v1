"""Pair validation, X←Y construction, lock-outside-M composition."""
from __future__ import annotations

import numpy as np

from csn_v4.indexed.schema import (
    INDEX_ALLOWED,
    INDEX_SHADOW_1,
    INDEX_SHADOW_2,
    INDEX_TO_INTERNAL,
    INDEXED_INPUT_INDICES,
    INDEXED_OUTPUT_INDICES,
    INTERNAL_TO_INDEX,
)


def validate_index_map(
    arr: np.ndarray,
    *,
    allowed: frozenset[int],
    name: str,
) -> np.ndarray:
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} must be integer index map, got dtype={a.dtype}")
    if a.ndim != 2:
        raise ValueError(f"{name} must be HxW, got shape={a.shape}")
    uniq = set(int(x) for x in np.unique(a))
    bad = uniq - set(allowed)
    if bad:
        raise ValueError(f"{name} has invalid index value(s) {sorted(bad)}; allowed={sorted(allowed)}")
    return a.astype(np.int64, copy=False)


def validate_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = validate_index_map(x, allowed=INDEXED_INPUT_INDICES, name="X")
    y = validate_index_map(y, allowed=INDEXED_OUTPUT_INDICES, name="Y")
    if x.shape != y.shape:
        raise ValueError(f"X/Y shape mismatch: X={x.shape} Y={y.shape}")
    m = allowed_mask(x)
    # Outside M, Y must equal X (background/outline locked)
    outside = ~m
    if outside.any() and not np.array_equal(y[outside], x[outside]):
        n = int(np.sum(y[outside] != x[outside]))
        raise ValueError(
            f"Y differs from X outside allowed mask M on {n} pixel(s); "
            "background/outline must be identical"
        )
    # Inside M, Y may only be 2/3/4
    if m.any():
        y_in = set(int(v) for v in np.unique(y[m]))
        bad = y_in - {INDEX_ALLOWED, INDEX_SHADOW_1, INDEX_SHADOW_2}
        if bad:
            raise ValueError(f"Y inside M has invalid indices {sorted(bad)}; expected {{2,3,4}}")
    return x, y


def allowed_mask(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) == INDEX_ALLOWED


def build_input_from_target(y: np.ndarray) -> np.ndarray:
    """Construct X from Y: map shadow indices 3,4 → 2; keep 0,1,2."""
    y = validate_index_map(y, allowed=INDEXED_OUTPUT_INDICES, name="Y")
    x = y.copy()
    x[(y == INDEX_SHADOW_1) | (y == INDEX_SHADOW_2)] = INDEX_ALLOWED
    validate_index_map(x, allowed=INDEXED_INPUT_INDICES, name="X(from Y)")
    return x


def target_to_internal(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Map Y indices inside M to {0,1,2}; outside → -100 ignore."""
    out = np.full(y.shape, -100, dtype=np.int64)
    for idx, internal in INDEX_TO_INTERNAL.items():
        sel = m & (y == idx)
        out[sel] = internal
    return out


def internal_to_index(internal: np.ndarray) -> np.ndarray:
    out = np.empty(internal.shape, dtype=np.int64)
    for i, idx in INTERNAL_TO_INDEX.items():
        out[internal == i] = idx
    return out


def compose_output(x: np.ndarray, pred_internal: np.ndarray) -> np.ndarray:
    """Copy X; replace only M=(X==2) with predicted indices 2/3/4."""
    x = validate_index_map(x, allowed=INDEXED_INPUT_INDICES, name="X")
    if pred_internal.shape != x.shape:
        raise ValueError(f"pred shape {pred_internal.shape} != X {x.shape}")
    m = allowed_mask(x)
    uniq = set(int(v) for v in np.unique(pred_internal[m])) if m.any() else set()
    bad = uniq - {0, 1, 2}
    if bad:
        raise ValueError(f"pred_internal has values {sorted(bad)} inside M; expected {{0,1,2}}")
    out = x.copy()
    mapped = internal_to_index(pred_internal)
    out[m] = mapped[m]
    # Hard guarantee
    out[~m] = x[~m]
    return out
