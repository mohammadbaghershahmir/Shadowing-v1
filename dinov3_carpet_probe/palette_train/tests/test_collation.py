from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.dataset import collate_palette_batch


def test_mixed_cardinality_collation():
    batch = [
        {
            "stem": "000000",
            "sample_id": 0,
            "image": torch.zeros(3, 16, 16),
            "image_meta": {},
            "target_rgb": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "target_rgb_uint8": torch.tensor([[0, 0, 0], [255, 255, 255]], dtype=torch.uint8),
            "target_hex": ["#000000", "#FFFFFF"],
            "num_colors": 2,
        },
        {
            "stem": "000001",
            "sample_id": 1,
            "image": torch.zeros(3, 16, 16),
            "image_meta": {},
            "target_rgb": torch.tensor([[1.0, 0.0, 0.0]]),
            "target_rgb_uint8": torch.tensor([[255, 0, 0]], dtype=torch.uint8),
            "target_hex": ["#FF0000"],
            "num_colors": 1,
        },
    ]
    out = collate_palette_batch(batch)
    assert out["target_rgb"].shape == (2, 2, 3)
    assert out["target_mask"].tolist() == [[True, True], [True, False]]
