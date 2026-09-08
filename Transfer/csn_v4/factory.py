"""Model factory and checkpoint loading for CSN-V4."""
from __future__ import annotations

from pathlib import Path

import torch

from csn_v4.config import CSNV4Config, load_config
from csn_v4.model import CarpetShadeNetV4


def build_model(cfg: CSNV4Config) -> CarpetShadeNetV4:
    return CarpetShadeNetV4(cfg)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    strict: bool = True,
) -> tuple[CarpetShadeNetV4, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path is None:
        config_path = ckpt.get("config_path")
    cfg = load_config(config_path) if config_path else CSNV4Config()
    model = build_model(cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    if device is not None:
        model = model.to(device)
    return model, ckpt
