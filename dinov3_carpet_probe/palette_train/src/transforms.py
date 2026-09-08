"""JPG loading and letterbox preprocessing for palette training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinov3_carpet_probe.src.preprocessing import normalize_tensor, resize_long_side_preserve_aspect


def load_rgb_jpg(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class PaletteImageTransform:
    """Apply DINO-compatible aspect-preserving preprocessing to JPGs."""

    def __init__(
        self,
        *,
        long_side: int,
        pad_multiple: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        allow_hflip: bool = False,
        allow_vflip: bool = False,
    ) -> None:
        self.long_side = int(long_side)
        self.pad_multiple = int(pad_multiple)
        self.mean = mean
        self.std = std
        self.allow_hflip = bool(allow_hflip)
        self.allow_vflip = bool(allow_vflip)

    def _maybe_flip(self, image: Image.Image) -> tuple[Image.Image, dict[str, bool]]:
        flipped_h = self.allow_hflip and random.random() < 0.5
        flipped_v = self.allow_vflip and random.random() < 0.5
        if flipped_h:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flipped_v:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image, {"hflip": flipped_h, "vflip": flipped_v}

    def transform_image(self, image: Image.Image, *, apply_flips: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        image = image.convert("RGB")
        if apply_flips:
            image, flips = self._maybe_flip(image)
        else:
            flips = {"hflip": False, "vflip": False}
        resized, scale = resize_long_side_preserve_aspect(image, self.long_side)
        arr = np.asarray(resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        content_h, content_w = tensor.shape[1], tensor.shape[2]
        pad_h = self.long_side - content_h
        pad_w = self.long_side - content_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Letterbox target {self.long_side} is smaller than resized image {(content_h, content_w)}."
            )
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        valid_mask = torch.ones(1, content_h, content_w, dtype=torch.float32)
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            valid_mask = F.pad(valid_mask, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        if tensor.shape[1] % self.pad_multiple != 0 or tensor.shape[2] % self.pad_multiple != 0:
            raise ValueError(
                f"Processed tensor shape {tuple(tensor.shape)} is not divisible by pad_multiple={self.pad_multiple}."
            )
        raw_tensor = tensor.clone()
        tensor = normalize_tensor(tensor, self.mean, self.std)
        meta_dict = {
            "original_hw": (image.size[1], image.size[0]),
            "resized_hw": (content_h, content_w),
            "processed_hw": (tensor.shape[1], tensor.shape[2]),
            "pad_box": (pad_top, pad_left, pad_bottom, pad_right),
            "scale": scale,
            "pad_multiple": self.pad_multiple,
            "mean": self.mean,
            "std": self.std,
            "mode": "letterbox_square",
            "augment": flips,
            "valid_mask": valid_mask,
            "raw_tensor": raw_tensor,
        }
        return tensor, meta_dict

    def __call__(self, path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
        image = load_rgb_jpg(path)
        return self.transform_image(image, apply_flips=True)


def build_transform(cfg: dict[str, Any], *, training: bool) -> PaletteImageTransform:
    return PaletteImageTransform(
        long_side=int(cfg["input_long_side"]),
        pad_multiple=int(cfg["pad_multiple"]),
        mean=tuple(float(x) for x in cfg["mean"]),
        std=tuple(float(x) for x in cfg["std"]),
        allow_hflip=bool(cfg.get("allow_hflip", False)) if training else False,
        allow_vflip=bool(cfg.get("allow_vflip", False)) if training else False,
    )
