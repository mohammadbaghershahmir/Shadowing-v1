from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from dinov3_carpet_probe.palette_train.src.checkpointing import load_checkpoint, save_checkpoint


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 2)
        self.checkpoint_path = "dummy.pth"


def test_checkpoint_roundtrip(tmp_path: Path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        save_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        global_step=10,
        config={"model_key": "vitl16_lvd", "max_num_colors": 8},
        manifest={"dataset_id": "abc"},
        best_metric=1.23,
    )
    original = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    for param in model.head.parameters():
        param.data.zero_()
    checkpoint = load_checkpoint(
        save_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    for key, value in model.head.state_dict().items():
        assert torch.equal(value, original[key])
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 10
