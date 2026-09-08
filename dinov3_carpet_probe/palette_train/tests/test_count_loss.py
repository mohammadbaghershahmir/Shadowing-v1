from __future__ import annotations

import torch
import torch.nn.functional as F

from dinov3_carpet_probe.palette_train.src.losses import compute_palette_losses


def test_count_loss_matches_cross_entropy():
    batch = {
        "target_rgb": torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
        "target_mask": torch.tensor([[True]]),
        "target_counts": torch.tensor([1], dtype=torch.int64),
        "stems": ["000000"],
    }
    outputs = {
        "pred_rgb": torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
        "presence_logits": torch.tensor([[2.0]], dtype=torch.float32),
        "count_logits": torch.tensor([[0.0, 3.0, -1.0]], dtype=torch.float32),
        "aux_outputs": [],
    }
    losses, _ = compute_palette_losses(
        outputs,
        batch,
        matcher_rgb_weight=1.0,
        matcher_perceptual_weight=0.0,
        matcher_presence_weight=0.0,
        loss_rgb_weight=0.0,
        loss_perceptual_weight=0.0,
        loss_presence_weight=0.0,
        loss_count_weight=1.0,
        loss_count_consistency_weight=0.0,
    )
    expected = F.cross_entropy(outputs["count_logits"], batch["target_counts"])
    assert torch.allclose(losses["loss_count"], expected)
