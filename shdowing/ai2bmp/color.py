"""Color space utilities for synthetic corruption."""

from __future__ import annotations

from typing import Any

import numpy as np


def srgb_uint8_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float64) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    x = rgb[..., 0] * 0.4124564 + rgb[..., 1] * 0.3575761 + rgb[..., 2] * 0.1804375
    y = rgb[..., 0] * 0.2126729 + rgb[..., 1] * 0.7151522 + rgb[..., 2] * 0.0721750
    z = rgb[..., 0] * 0.0193339 + rgb[..., 1] * 0.1191920 + rgb[..., 2] * 0.9503041
    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=-1) / ref
    delta = 6 / 29
    xyz = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta * delta) + 4 / 29)
    l = 116 * xyz[..., 1] - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([l, a, b], axis=-1)


def lab_to_srgb_uint8(lab: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    delta = 6 / 29
    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

    def f_inv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3 * delta * delta * (t - 4 / 29))

    x = ref[0] * f_inv(fx)
    y = ref[1] * f_inv(fy)
    z = ref[2] * f_inv(fz)
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    bch = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
    rgb = np.stack([r, g, bch], axis=-1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    clipped = np.logical_or(rgb < 0, rgb > 1)
    clip_count = int(np.sum(clipped))
    rgb = np.clip(rgb, 0, 1)
    out = np.round(rgb * 255.0).astype(np.uint8)
    stats = {
        "out_of_gamut_pixel_channels": clip_count,
        "out_of_gamut_fraction": float(clip_count / max(1, rgb.size)),
    }
    return out, stats


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
    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((c1p * c2p) == 0, 0, dhp)
    dhp_rad = np.radians(dhp)
    dhp_term = 2 * np.sqrt(c1p * c2p) * np.sin(dhp_rad / 2)
    lp_bar = (l1 + l2) / 2
    cp_bar = (c1p + c2p) / 2
    hp_bar = np.where(
        np.abs(h1p - h2p) > 180,
        (h1p + h2p + 360) / 2,
        (h1p + h2p) / 2,
    )
    hp_bar = np.where((c1p * c2p) == 0, h1p + h2p, hp_bar)
    t = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    rc = 2 * np.sqrt((cp_bar**7) / (cp_bar**7 + 25**7 + 1e-12))
    sl = 1 + (0.015 * (lp_bar - 50) ** 2) / np.sqrt(20 + (lp_bar - 50) ** 2)
    sc = 1 + 0.045 * cp_bar
    sh = 1 + 0.015 * cp_bar * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dhp_term / sh) ** 2
        + rt * (dcp / sc) * (dhp_term / sh)
    )


def delta_e_stats(original: np.ndarray, modified: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    lab1 = srgb_uint8_to_lab(original)
    lab2 = srgb_uint8_to_lab(modified)
    de = ciede2000_np(lab1, lab2)
    if mask is not None:
        de = de[mask.astype(bool)]
    if de.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(de)),
        "p50": float(np.percentile(de, 50)),
        "p90": float(np.percentile(de, 90)),
        "p95": float(np.percentile(de, 95)),
        "max": float(np.max(de)),
    }
