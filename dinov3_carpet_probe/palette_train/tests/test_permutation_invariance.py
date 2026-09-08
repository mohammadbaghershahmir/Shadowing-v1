from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.losses import compute_palette_losses


def test_shuffled_target_order_keeps_total_loss(dummy_batch):
    outputs = {
        "pred_rgb": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.3, 0.3, 0.3]],
                [[1.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]],
            ],
            dtype=torch.float32,
        ),
        "presence_logits": torch.tensor([[4.0, 4.0, -4.0], [4.0, -4.0, -4.0]], dtype=torch.float32),
        "count_logits": torch.tensor([[0.0, 0.0, 4.0], [0.0, 4.0, 0.0]], dtype=torch.float32),
        "aux_outputs": [],
    }
    losses_a, _ = compute_palette_losses(
        outputs,
        dummy_batch,
        matcher_rgb_weight=1.0,
        matcher_perceptual_weight=1.0,
        matcher_presence_weight=0.25,
        loss_rgb_weight=1.0,
        loss_perceptual_weight=1.0,
        loss_presence_weight=1.0,
        loss_count_weight=1.0,
        loss_count_consistency_weight=0.1,
    )
    shuffled = dict(dummy_batch)
    shuffled["target_rgb"] = dummy_batch["target_rgb"].clone()
    shuffled["target_rgb"][0] = shuffled["target_rgb"][0].flip(0)
    losses_b, _ = compute_palette_losses(
        outputs,
        shuffled,
        matcher_rgb_weight=1.0,
        matcher_perceptual_weight=1.0,
        matcher_presence_weight=0.25,
        loss_rgb_weight=1.0,
        loss_perceptual_weight=1.0,
        loss_presence_weight=1.0,
        loss_count_weight=1.0,
        loss_count_consistency_weight=0.1,
    )
    assert torch.allclose(losses_a["loss_total"], losses_b["loss_total"], atol=1e-5)
