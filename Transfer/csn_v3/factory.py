"""Model factory and checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import torch

from csn_v3.config import CSNV3Config, load_config
from csn_v3.model import CarpetShadeNetV3


def build_model(cfg: CSNV3Config) -> CarpetShadeNetV3:
    return CarpetShadeNetV3(cfg)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    strict: bool = True,
) -> tuple[CarpetShadeNetV3, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path is None:
        config_path = ckpt.get("config_path")
    cfg = load_config(config_path) if config_path else CSNV3Config()
    model = build_model(cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    if device is not None:
        model = model.to(device)
    return model, ckpt
