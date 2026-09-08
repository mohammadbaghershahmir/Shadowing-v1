from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.metrics import evaluate_palette_batch


def test_set_error_penalizes_count_mismatch():
    batch = {
        "stems": ["000000"],
        "target_rgb": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32),
        "target_mask": torch.tensor([[True, True]]),
        "target_counts": torch.tensor([2], dtype=torch.int64),
    }
    outputs_good = {
        "pred_rgb": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]], dtype=torch.float32),
        "presence_logits": torch.tensor([[5.0, 5.0, -5.0]], dtype=torch.float32),
        "count_logits": torch.tensor([[0.0, 0.0, 4.0, -4.0]], dtype=torch.float32),
    }
    outputs_bad = {
        "pred_rgb": outputs_good["pred_rgb"],
        "presence_logits": torch.tensor([[5.0, -5.0, -5.0]], dtype=torch.float32),
        "count_logits": torch.tensor([[0.0, 4.0, 0.0, -4.0]], dtype=torch.float32),
    }
    good = evaluate_palette_batch(outputs_good, batch, matcher_rgb_weight=1.0, matcher_perceptual_weight=1.0, matcher_presence_weight=0.25)
    bad = evaluate_palette_batch(outputs_bad, batch, matcher_rgb_weight=1.0, matcher_perceptual_weight=1.0, matcher_presence_weight=0.25)
    assert bad["set_error"] > good["set_error"]
