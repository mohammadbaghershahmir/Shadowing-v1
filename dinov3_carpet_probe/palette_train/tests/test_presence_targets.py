from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.losses import compute_palette_losses


def test_presence_loss_targets_matched_and_unmatched():
    batch = {
        "target_rgb": torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
        "target_mask": torch.tensor([[True]]),
        "target_counts": torch.tensor([1], dtype=torch.int64),
        "stems": ["000000"],
    }
    outputs = {
        "pred_rgb": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32),
        "presence_logits": torch.tensor([[4.0, -4.0]], dtype=torch.float32),
        "count_logits": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        "aux_outputs": [],
    }
    losses, matches = compute_palette_losses(
        outputs,
        batch,
        matcher_rgb_weight=1.0,
        matcher_perceptual_weight=0.0,
        matcher_presence_weight=0.0,
        loss_rgb_weight=1.0,
        loss_perceptual_weight=0.0,
        loss_presence_weight=1.0,
        loss_count_weight=1.0,
        loss_count_consistency_weight=0.0,
    )
    assert matches[0].pred_indices.tolist() == [0]
    assert matches[0].unmatched_pred_indices.tolist() == [1]
    assert losses["loss_presence"].item() < 0.1
