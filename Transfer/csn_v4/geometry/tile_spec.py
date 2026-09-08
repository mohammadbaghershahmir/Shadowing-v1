"""Shared tile/core geometry for CSN-V4 training, dataset, and inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

INPUT_SIZE = 512
HALO = 64
CORE_SIZE = INPUT_SIZE - 2 * HALO  # 384


def core_starts(length: int, core_size: int = CORE_SIZE) -> list[int]:
    if length <= 0:
        return [0]
    starts = list(range(0, length, core_size))
    if not starts:
        starts = [0]
    return starts


def _valid_input_mask(
    ix: int,
    iy: int,
    input_size: int,
    fw: int,
    fh: int,
) -> np.ndarray:
    mask = np.zeros((input_size, input_size), dtype=bool)
    for sy in range(input_size):
        for sx in range(input_size):
            gy, gx = iy + sy, ix + sx
            if 0 <= gy < fh and 0 <= gx < fw:
                mask[sy, sx] = True
    return mask


def _valid_core_mask(cw: int, ch: int, ix: int, iy: int, fw: int, fh: int) -> np.ndarray:
    mask = np.zeros((ch, cw), dtype=bool)
    for dy in range(ch):
        for dx in range(cw):
            gy, gx = iy + HALO + dy, ix + HALO + dx
            if 0 <= gy < fh and 0 <= gx < fw:
                mask[dy, dx] = True
    return mask


def normalize_bbox(x: int, y: int, w: int, h: int, fw: int, fh: int) -> np.ndarray:
    return np.array(
        [x / fw, y / fh, w / fw, h / fh, (x + w / 2) / fw, (y + h / 2) / fh],
        dtype=np.float32,
    )


def normalize_core_coords(cx: int, cy: int, cw: int, ch: int, fw: int, fh: int) -> np.ndarray:
    return normalize_bbox(cx, cy, cw, ch, fw, fh)


def normalize_input_coords(ix: int, iy: int, iw: int, ih: int, fw: int, fh: int) -> np.ndarray:
    return normalize_bbox(ix, iy, iw, ih, fw, fh)


@dataclass(frozen=True)
class TileSpec:
    image_wh: tuple[int, int]
    transform_id: int
    core_origin_xy: tuple[int, int]
    core_size_xy: tuple[int, int]
    input_origin_xy: tuple[int, int]
    input_size: int
    valid_input_mask: np.ndarray
    valid_core_mask: np.ndarray
    input_bbox_norm: np.ndarray
    core_bbox_norm: np.ndarray
    scene_id: str = ""

    @property
    def crop_coords_norm(self) -> np.ndarray:
        """Model coordinate embedding uses the 512×512 input window, not core-only box."""
        return self.input_bbox_norm

    @property
    def core_x(self) -> int:
        return self.core_origin_xy[0]

    @property
    def core_y(self) -> int:
        return self.core_origin_xy[1]

    @property
    def core_w(self) -> int:
        return self.core_size_xy[0]

    @property
    def core_h(self) -> int:
        return self.core_size_xy[1]

    @property
    def input_x(self) -> int:
        return self.input_origin_xy[0]

    @property
    def input_y(self) -> int:
        return self.input_origin_xy[1]

    def model_core_y_slice(self) -> slice:
        return slice(HALO, HALO + self.core_h)

    def model_core_x_slice(self) -> slice:
        return slice(HALO, HALO + self.core_w)

    def model_core_slices(self) -> tuple[slice, slice]:
        return self.model_core_y_slice(), self.model_core_x_slice()


def build_tile_spec(
    fw: int,
    fh: int,
    cx: int,
    cy: int,
    *,
    transform_id: int = 0,
    core_size: int = CORE_SIZE,
    halo: int = HALO,
    input_size: int = INPUT_SIZE,
    scene_id: str = "",
) -> TileSpec:
    cw = min(core_size, fw - cx)
    ch = min(core_size, fh - cy)
    ix, iy = cx - halo, cy - halo
    valid_in = _valid_input_mask(ix, iy, input_size, fw, fh)
    valid_core = _valid_core_mask(cw, ch, ix, iy, fw, fh)
    input_bbox_norm = normalize_input_coords(ix, iy, input_size, input_size, fw, fh)
    core_bbox_norm = normalize_core_coords(cx, cy, cw, ch, fw, fh)
    return TileSpec(
        image_wh=(fw, fh),
        transform_id=transform_id,
        core_origin_xy=(cx, cy),
        core_size_xy=(cw, ch),
        input_origin_xy=(ix, iy),
        input_size=input_size,
        valid_input_mask=valid_in,
        valid_core_mask=valid_core,
        input_bbox_norm=input_bbox_norm,
        core_bbox_norm=core_bbox_norm,
        scene_id=scene_id,
    )


def iter_core_tiles(
    fw: int,
    fh: int,
    *,
    transform_id: int = 0,
    core_size: int = CORE_SIZE,
    halo: int = HALO,
    input_size: int = INPUT_SIZE,
    scene_id: str = "",
) -> Iterator[TileSpec]:
    for cy in core_starts(fh, core_size):
        for cx in core_starts(fw, core_size):
            yield build_tile_spec(
                fw, fh, cx, cy,
                transform_id=transform_id,
                core_size=core_size,
                halo=halo,
                input_size=input_size,
                scene_id=scene_id,
            )


def model_core_slices(spec: TileSpec) -> tuple[slice, slice]:
    return spec.model_core_slices()


def tile_uid(scene_id: str, transform_id: int, cx: int, cy: int) -> str:
    return f"{scene_id}__t{transform_id}__{cx}_{cy}"


def build_coverage_map(fw: int, fh: int, specs: list[TileSpec]) -> np.ndarray:
    cov = np.zeros((fh, fw), dtype=np.int32)
    for spec in specs:
        cx, cy = spec.core_origin_xy
        cw, ch = spec.core_size_xy
        region = spec.valid_core_mask
        cov[cy : cy + ch, cx : cx + cw] += region.astype(np.int32)
    return cov


def assert_full_coverage(
    fw: int,
    fh: int,
    specs: list[TileSpec] | None = None,
    *,
    core_size: int = CORE_SIZE,
    allow_overwrite: bool = False,
) -> None:
    if specs is None:
        specs = list(iter_core_tiles(fw, fh, core_size=core_size))
    cov = build_coverage_map(fw, fh, specs)
    real = np.ones((fh, fw), dtype=bool)
    holes = real & (cov == 0)
    if holes.any():
        n = int(holes.sum())
        raise AssertionError(f"Coverage map has {n} uncovered real pixels for image {fw}x{fh}")
    if not allow_overwrite:
        over = cov > 1
        if over.any():
            n = int(over.sum())
            raise AssertionError(f"Coverage map has {n} overwrite pixels (cov>1) for image {fw}x{fh}")


def paste_core_predictions(
    canvas: np.ndarray,
    spec: TileSpec,
    core_pred: np.ndarray,
) -> None:
    """Write trusted core predictions into full-image canvas."""
    cx, cy = spec.core_origin_xy
    cw, ch = spec.core_size_xy
    assert core_pred.shape == (ch, cw), f"core_pred {core_pred.shape} != ({ch},{cw})"
    mask = spec.valid_core_mask
    region = canvas[cy : cy + ch, cx : cx + cw]
    region[mask] = core_pred[mask]
