from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.head import PaletteSetHead


def test_patch_mask_excludes_padding_from_count_pooling():
    head = PaletteSetHead(
        fused_dim=4,
        raw_color_dim=2,
        cls_dim=6,
        max_num_colors=3,
        spare_queries=1,
        head_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        decoder_ffn_dim=16,
    )
    patch_tokens = torch.zeros(1, 4, 4)
    raw_features = torch.zeros(1, 4, 2)
    raw_features[0, 0] = 1.0
    raw_features[0, 1] = 1.0
    raw_features[0, 2] = 99.0
    raw_features[0, 3] = 99.0
    cls_token = torch.zeros(1, 6)
    valid = torch.tensor([[True, True, False, False]])
    out_valid = head(patch_tokens, raw_features, cls_token, grid_hw=(2, 2), patch_valid_mask=valid)
    out_all = head(patch_tokens, raw_features, cls_token, grid_hw=(2, 2), patch_valid_mask=torch.ones_like(valid))
    assert not torch.allclose(out_valid["count_logits"], out_all["count_logits"])
