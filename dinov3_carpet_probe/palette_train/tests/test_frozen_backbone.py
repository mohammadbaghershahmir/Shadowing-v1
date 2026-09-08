from __future__ import annotations

import torch
import torch.nn as nn


class DummyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(feat)


def test_frozen_backbone_has_no_grad_and_no_weight_change():
    model = DummyPredictor()
    before = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    for param in model.backbone.parameters():
        assert param.grad is None
    for key, tensor in model.backbone.state_dict().items():
        assert torch.equal(before[key], tensor)
