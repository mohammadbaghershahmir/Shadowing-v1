from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.matching import hungarian_match_single


def test_hungarian_matching_hand_example():
    pred_rgb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    presence = torch.tensor([2.0, 2.0], dtype=torch.float32)
    target_rgb = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    match = hungarian_match_single(
        pred_rgb,
        presence,
        target_rgb,
        rgb_weight=1.0,
        perceptual_weight=0.0,
        presence_weight=0.0,
    )
    assert match.pred_indices.tolist() == [0, 1]
    assert match.target_indices.tolist() == [1, 0]
