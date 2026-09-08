from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.head import PaletteSetHead


def test_head_output_shapes_and_bounds():
    head = PaletteSetHead(
        fused_dim=4096,
        cls_dim=1024,
        max_num_colors=10,
        spare_queries=4,
        head_dim=256,
        decoder_layers=2,
        decoder_heads=8,
        decoder_ffn_dim=512,
        aux_loss=True,
    )
    patch_tokens = torch.randn(2, 16, 4096)
    cls_token = torch.randn(2, 1024)
    out = head(patch_tokens, cls_token, grid_hw=(4, 4))
    assert out["pred_rgb"].shape == (2, 14, 3)
    assert out["presence_logits"].shape == (2, 14)
    assert out["count_logits"].shape == (2, 11)
    assert torch.all(out["pred_rgb"] >= 0.0)
    assert torch.all(out["pred_rgb"] <= 1.0)
    assert len(out["aux_outputs"]) == 1
