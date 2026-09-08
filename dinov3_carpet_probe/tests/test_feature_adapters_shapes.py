"""Shape / API tests for feature adapters without requiring real weights."""

from __future__ import annotations

import torch

from dinov3_carpet_probe.src.feature_adapters import _l2_normalize_hwc, _nchw_to_bhwc


def test_nchw_to_bhwc():
    x = torch.randn(2, 8, 4, 5)
    y = _nchw_to_bhwc(x)
    assert y.shape == (2, 4, 5, 8)


def test_l2_normalize():
    x = torch.randn(1, 3, 3, 16)
    y = _l2_normalize_hwc(x)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
