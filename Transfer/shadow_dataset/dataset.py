"""PyTorch Dataset for on-demand crop loading from JSONL manifests."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from shadow_dataset.constants import (
    IGNORE_INDEX,
    ONEHOT_CHANNEL_OUTLINE,
    ONEHOT_CHANNEL_TARGET_REGION,
    ONEHOT_CHANNEL_UNSHADED,
    ONEHOT_NUM_CHANNELS,
    OUTLINE,
    UNSHADED,
)
from shadow_dataset.geometry import candidate_mask_from_input
from shadow_dataset.image_io import load_rgb_exact
from shadow_dataset.io_utils import read_jsonl
from shadow_dataset.validation import encode_target_labels

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is a declared dependency
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


def extract_padded_crop(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    crop_size: int,
    pad_value: int | tuple[int, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a crop with constant padding; return (crop, image_valid_mask).

    ``image_valid_mask`` is True for pixels that come from the original image.
    """
    if image.ndim == 2:
        h, w = image.shape
        channels = None
    elif image.ndim == 3:
        h, w, channels = image.shape
    else:
        raise ValueError(f"Unsupported image ndim={image.ndim}")

    if channels is None:
        crop = np.full((crop_size, crop_size), pad_value, dtype=image.dtype)
    else:
        if isinstance(pad_value, tuple):
            fill = np.asarray(pad_value, dtype=image.dtype)
        else:
            fill = pad_value
        crop = np.empty((crop_size, crop_size, channels), dtype=image.dtype)
        crop[:, :] = fill

    valid = np.zeros((crop_size, crop_size), dtype=bool)

    src_x0 = max(0, x)
    src_y0 = max(0, y)
    src_x1 = min(w, x + crop_size)
    src_y1 = min(h, y + crop_size)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return crop, valid

    dst_x0 = src_x0 - x
    dst_y0 = src_y0 - y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if channels is None:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    else:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    valid[dst_y0:dst_y1, dst_x0:dst_x1] = True
    return crop, valid


def build_input_onehot(input_rgb: np.ndarray) -> np.ndarray:
    """Build float32 one-hot [3,H,W]: outline, unshaded, TARGET_REGION."""
    h, w, _ = input_rgb.shape
    onehot = np.zeros((ONEHOT_NUM_CHANNELS, h, w), dtype=np.float32)
    outline = (
        (input_rgb[:, :, 0] == OUTLINE[0])
        & (input_rgb[:, :, 1] == OUTLINE[1])
        & (input_rgb[:, :, 2] == OUTLINE[2])
    )
    unshaded = (
        (input_rgb[:, :, 0] == UNSHADED[0])
        & (input_rgb[:, :, 1] == UNSHADED[1])
        & (input_rgb[:, :, 2] == UNSHADED[2])
    )
    candidate = candidate_mask_from_input(input_rgb)
    onehot[ONEHOT_CHANNEL_OUTLINE][outline] = 1.0
    onehot[ONEHOT_CHANNEL_UNSHADED][unshaded] = 1.0
    onehot[ONEHOT_CHANNEL_TARGET_REGION][candidate] = 1.0
    return onehot


def central_valid_mask(crop_size: int, valid_margin: int) -> np.ndarray:
    mask = np.zeros((crop_size, crop_size), dtype=bool)
    m = valid_margin
    mask[m : crop_size - m, m : crop_size - m] = True
    return mask


class SourceImageLRU:
    """LRU cache for input+target RGB pairs keyed by source_id."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max(1, max_entries)
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, source_id: str, input_path: str | Path, target_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        if source_id in self._cache:
            self._cache.move_to_end(source_id)
            return self._cache[source_id]

        input_rgb, _ = load_rgb_exact(input_path)
        target_rgb, _ = load_rgb_exact(target_path)
        self._cache[source_id] = (input_rgb, target_rgb)
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return input_rgb, target_rgb

    def clear(self) -> None:
        self._cache.clear()


class ShadowCropDataset(Dataset):
    """Load crops on demand from a crop manifest JSONL.

    Returned dictionary keys
    ------------------------
    local_input_rgb : float32 [3,H,W] in [0,1]
    local_input_onehot : float32 [3,H,W]
        channel 0 = outline, 1 = unshaded, 2 = TARGET_REGION
    target_rgb : uint8 [3,H,W]
    target_label : int64 [H,W] with values {0,1,2,-100}
    candidate_mask : bool [H,W]
    image_valid_mask : bool [H,W]
    central_valid_mask : bool [H,W]
    crop_box_normalized : float32 [4]
    source_id, sample_id, sampling_strategy : str
    """

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        records: list[dict[str, Any]] | None = None,
        image_cache: SourceImageLRU | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for ShadowCropDataset")
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.image_cache = image_cache or SourceImageLRU(max_entries=8)
        if records is not None:
            self.records = list(records)
        else:
            if self.manifest_path is None:
                raise ValueError("manifest_path or records is required")
            self.records = read_jsonl(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return self.load_record(record)

    def load_record(self, record: dict[str, Any]) -> dict[str, Any]:
        crop_size = int(record["crop_size"])
        valid_margin = int(record["valid_margin"])
        x = int(record["x"])
        y = int(record["y"])

        source_id = str(record["source_id"])
        input_rgb_full, target_rgb_full = self.image_cache.get(
            source_id, record["input_path"], record["target_path"],
        )

        local_input, image_valid = extract_padded_crop(
            input_rgb_full,
            x=x,
            y=y,
            crop_size=crop_size,
            pad_value=UNSHADED,
        )
        local_target, _ = extract_padded_crop(
            target_rgb_full,
            x=x,
            y=y,
            crop_size=crop_size,
            pad_value=UNSHADED,
        )

        # Labels: encode on full images then crop with ignore padding, or
        # encode on crops (padding is white/non-candidate -> -100).
        labels_full = encode_target_labels(input_rgb_full, target_rgb_full)
        local_labels, _ = extract_padded_crop(
            labels_full,
            x=x,
            y=y,
            crop_size=crop_size,
            pad_value=IGNORE_INDEX,
        )
        # Outside image_valid_mask force ignore.
        local_labels = local_labels.copy()
        local_labels[~image_valid] = IGNORE_INDEX

        candidate = candidate_mask_from_input(local_input)
        onehot = build_input_onehot(local_input)
        center_mask = central_valid_mask(crop_size, valid_margin)

        # CHW float input in [0, 1]
        local_input_f = (local_input.astype(np.float32) / 255.0).transpose(2, 0, 1)
        target_chw = local_target.transpose(2, 0, 1)

        box = record.get("normalized_crop_box", [0.0, 0.0, 0.0, 0.0])
        box_arr = np.asarray([float(v) for v in box], dtype=np.float32)

        return {
            "local_input_rgb": torch.from_numpy(local_input_f.copy()),
            "local_input_onehot": torch.from_numpy(onehot.copy()),
            "target_rgb": torch.from_numpy(target_chw.copy()),
            "target_label": torch.from_numpy(local_labels.astype(np.int64, copy=True)),
            "candidate_mask": torch.from_numpy(candidate.copy()),
            "image_valid_mask": torch.from_numpy(image_valid.copy()),
            "central_valid_mask": torch.from_numpy(center_mask.copy()),
            "crop_box_normalized": torch.from_numpy(box_arr.copy()),
            "source_id": str(record["source_id"]),
            "sample_id": str(record["sample_id"]),
            "sampling_strategy": str(record["sampling_strategy"]),
            "transform_id": int(record.get("transform_id", 0)),
        }
