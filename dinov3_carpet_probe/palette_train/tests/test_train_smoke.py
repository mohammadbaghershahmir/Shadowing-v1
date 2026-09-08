from __future__ import annotations

import pytest
import torch

from dinov3_carpet_probe.palette_train.src.head import PaletteSetHead
from dinov3_carpet_probe.palette_train.src.losses import compute_palette_losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for smoke coverage.")
def test_one_batch_gpu_train_smoke():
    head = PaletteSetHead(
        fused_dim=4096,
        cls_dim=1024,
        max_num_colors=2,
        spare_queries=1,
        head_dim=256,
        decoder_layers=2,
        decoder_heads=8,
        decoder_ffn_dim=512,
    ).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    patch_tokens = torch.randn(2, 16, 4096, device="cuda")
    cls_token = torch.randn(2, 1024, device="cuda")
    outputs = head(patch_tokens, cls_token, grid_hw=(4, 4))
    batch = {
        "target_rgb": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "target_mask": torch.tensor([[True, True], [True, False]]),
        "target_counts": torch.tensor([2, 1], dtype=torch.int64),
        "stems": ["000000", "000001"],
    }
    losses, _ = compute_palette_losses(
        outputs,
        batch,
        matcher_rgb_weight=1.0,
        matcher_perceptual_weight=1.0,
        matcher_presence_weight=0.25,
        loss_rgb_weight=1.0,
        loss_perceptual_weight=1.0,
        loss_presence_weight=1.0,
        loss_count_weight=1.0,
        loss_count_consistency_weight=0.1,
    )
    optimizer.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    optimizer.step()
