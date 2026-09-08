"""Color-space utilities for training and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    l = 0.4122214708 * rgb[..., 0] + 0.5363325363 * rgb[..., 1] + 0.0514459929 * rgb[..., 2]
    m = 0.2119034982 * rgb[..., 0] + 0.6806995451 * rgb[..., 1] + 0.1073969566 * rgb[..., 2]
    s = 0.0883024619 * rgb[..., 0] + 0.2817188376 * rgb[..., 1] + 0.6299787005 * rgb[..., 2]
    l_ = torch.clamp(l, min=1e-8).pow(1.0 / 3.0)
    m_ = torch.clamp(m, min=1e-8).pow(1.0 / 3.0)
    s_ = torch.clamp(s, min=1e-8).pow(1.0 / 3.0)
    return torch.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        dim=-1,
    )


def srgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    return linear_rgb_to_oklab(srgb_to_linear(rgb))


def _srgb_to_xyz_np(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    x = rgb[..., 0] * 0.4124564 + rgb[..., 1] * 0.3575761 + rgb[..., 2] * 0.1804375
    y = rgb[..., 0] * 0.2126729 + rgb[..., 1] * 0.7151522 + rgb[..., 2] * 0.0721750
    z = rgb[..., 0] * 0.0193339 + rgb[..., 1] * 0.1191920 + rgb[..., 2] * 0.9503041
    return np.stack([x, y, z], axis=-1)


def xyz_to_lab_np(xyz: np.ndarray) -> np.ndarray:
    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = xyz / ref
    delta = 6 / 29
    xyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta * delta) + 4 / 29)
    l = 116 * xyz[..., 1] - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([l, a, b], axis=-1)


def srgb_uint8_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float64) / 255.0
    return xyz_to_lab_np(_srgb_to_xyz_np(rgb))


def ciede2000_np(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt((c_bar**7) / (c_bar**7 + 25**7 + 1e-12)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    dh = np.where((c1p * c2p) == 0, 0, dh)
    delta_hp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2

    h_bar = (h1p + h2p) / 2
    h_bar = np.where(np.abs(h1p - h2p) > 180, h_bar + 180, h_bar) % 360
    h_bar = np.where((c1p * c2p) == 0, h1p + h2p, h_bar)

    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    delta_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25**7 + 1e-12))
    s_l = 1 + (0.015 * ((l_bar - 50) ** 2)) / np.sqrt(20 + ((l_bar - 50) ** 2))
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2 * delta_theta)) * r_c

    return np.sqrt(
        (delta_lp / s_l) ** 2
        + (delta_cp / s_c) ** 2
        + (delta_hp / s_h) ** 2
        + r_t * (delta_cp / s_c) * (delta_hp / s_h)
    )


def ciede2000_from_uint8(rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
    return ciede2000_np(srgb_uint8_to_lab(rgb1), srgb_uint8_to_lab(rgb2))
