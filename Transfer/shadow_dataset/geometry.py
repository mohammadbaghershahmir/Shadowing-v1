"""Internal categorical class-boundary and connected-component utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from shadow_dataset.constants import (
    CLASS_COUNT_KEYS,
    INDEX_100,
    INDEX_150,
    INDEX_200,
    RGB_VALUE_TO_CLASS,
    TARGET_REGION,
)


def candidate_mask_from_input(input_rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of TARGET_REGION (magenta) pixels."""
    return (
        (input_rgb[:, :, 0] == TARGET_REGION[0])
        & (input_rgb[:, :, 1] == TARGET_REGION[1])
        & (input_rgb[:, :, 2] == TARGET_REGION[2])
    )


def class_label_map(target_rgb: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return int16 map with categorical labels 0/1/2 inside TARGET_REGION, -1 elsewhere."""
    h, w = candidate.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    equal = (
        (target_rgb[:, :, 0] == target_rgb[:, :, 1])
        & (target_rgb[:, :, 1] == target_rgb[:, :, 2])
    )
    vals = target_rgb[:, :, 0]
    for value, label in RGB_VALUE_TO_CLASS.items():
        mask = candidate & equal & (vals == value)
        labels[mask] = np.int16(label)
    return labels


def detect_class_boundaries(class_labels: np.ndarray) -> np.ndarray:
    """Detect 4-connected categorical transitions between class labels 0/1/2.

    A pixel is a transition pixel if it has a valid class label and at least one
    4-neighbour with a different valid class label. Inequality is categorical;
    RGB magnitudes are irrelevant.
    """
    valid = class_labels >= 0
    boundary = np.zeros(class_labels.shape, dtype=bool)

    diff = (class_labels[:, :-1] != class_labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    boundary[:, :-1] |= diff
    boundary[:, 1:] |= diff

    diff = (class_labels[:-1, :] != class_labels[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[:-1, :] |= diff
    boundary[1:, :] |= diff

    return boundary


@dataclass(frozen=True)
class ComponentInfo:
    label: int
    pixel_count: int
    centroid_y: float
    centroid_x: float
    min_y: int
    min_x: int
    max_y: int
    max_x: int
    ys: np.ndarray
    xs: np.ndarray

    @property
    def is_tiny(self) -> bool:
        return self.pixel_count <= 64


def find_candidate_components(
    candidate: np.ndarray,
    *,
    connectivity: int = 8,
) -> list[ComponentInfo]:
    """Label connected components in the TARGET_REGION mask.

    Uses 8-connectivity by default via ``scipy.ndimage.label``. Returns
    components sorted by pixel count descending, then by centroid.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.int8)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)

    labeled, n_labels = ndimage.label(candidate.astype(np.uint8), structure=structure)
    if n_labels == 0:
        return []

    components: list[ComponentInfo] = []
    slices = ndimage.find_objects(labeled)
    for label_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        region = labeled[slc]
        mask = region == label_id
        ys_local, xs_local = np.nonzero(mask)
        ys = ys_local.astype(np.int32) + int(slc[0].start)
        xs = xs_local.astype(np.int32) + int(slc[1].start)
        components.append(
            ComponentInfo(
                label=label_id,
                pixel_count=int(ys.size),
                centroid_y=float(ys.mean()),
                centroid_x=float(xs.mean()),
                min_y=int(ys.min()),
                min_x=int(xs.min()),
                max_y=int(ys.max()),
                max_x=int(xs.max()),
                ys=ys,
                xs=xs,
            )
        )

    components.sort(
        key=lambda c: (-c.pixel_count, c.centroid_y, c.centroid_x, c.label)
    )
    return components


def class_index_masks(
    target_rgb: np.ndarray, candidate: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-class boolean masks keyed by CLASS_COUNT_KEYS in class-id order."""
    index_rgb = {
        "200": INDEX_200,
        "150": INDEX_150,
        "100": INDEX_100,
    }
    return {
        key: candidate & _match(target_rgb, index_rgb[key]) for key in CLASS_COUNT_KEYS
    }


def _match(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return (
        (rgb[:, :, 0] == color[0])
        & (rgb[:, :, 1] == color[1])
        & (rgb[:, :, 2] == color[2])
    )
