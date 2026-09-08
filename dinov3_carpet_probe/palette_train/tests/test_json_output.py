from __future__ import annotations

import torch

from dinov3_carpet_probe.palette_train.src.decode import decode_palette_prediction


def test_json_decode_schema_and_hex():
    decoded = decode_palette_prediction(
        stem="000000",
        pred_rgb=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        presence_logits=torch.tensor([5.0, 4.0], dtype=torch.float32),
        count_logits=torch.tensor([0.0, 0.0, 3.0], dtype=torch.float32),
    )
    data = decoded.to_dict()
    assert data["sample_id"] == 0
    assert data["num_colors"] == len(data["rgb"]) == len(data["hex"])
    assert data["hex"] == [value.upper() for value in data["hex"]]
    for rgb, hex_value in zip(data["rgb"], data["hex"]):
        assert hex_value == "#{:02X}{:02X}{:02X}".format(*rgb)
