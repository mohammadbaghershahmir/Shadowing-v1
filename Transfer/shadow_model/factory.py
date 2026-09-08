"""Model factory for training, validation, and inference."""
from __future__ import annotations

from typing import Any

import torch

from shadow_model.config import ShadowConfig
from shadow_model.model import ShadowNet
from shadow_model.profiles import ExperimentProfile, get_profile


def build_model(cfg: ShadowConfig, profile: ExperimentProfile | None = None) -> ShadowNet:
    profile = profile or get_profile(cfg.profile)
    model = ShadowNet(
        profile=profile,
        local_model_name=cfg.local_encoder.model_name,
        pretrained=cfg.local_encoder.pretrained,
        num_classes=cfg.model.num_classes,
        decoder_dim=cfg.model.decoder_dim,
        stem_channels=cfg.model.stem_channels,
        refine_channels=cfg.model.refine_channels,
        dino_dim=cfg.dino.token_dim,
        dino_cfg=cfg.dino,
        cross_attn_blocks=cfg.model.cross_attn_blocks,
        cross_attn_heads=cfg.model.cross_attn_heads,
        gate_init_local=cfg.model.gate_init_local,
        gate_init_global=cfg.model.gate_init_global,
        decoder_dropout=cfg.model.decoder_dropout,
    )
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> tuple[ShadowNet, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = state.get("resolved_config") or state.get("config")
    if resolved is None:
        raise ValueError("Checkpoint missing resolved_config")
    if isinstance(resolved, dict):
        from shadow_model.config import ShadowConfig, DataConfig, DinoConfig, LocalEncoderConfig, ModelConfig, LossConfig, TrainConfig, EvalConfig
        cfg = ShadowConfig(
            profile=resolved.get("profile", "BEST_V1"),
            data=DataConfig(**resolved.get("data", {})),
            local_encoder=LocalEncoderConfig(**resolved.get("local_encoder", {})),
            dino=DinoConfig(**resolved.get("dino", resolved.get("global_encoder", {}))),
            model=ModelConfig(**resolved.get("model", {})),
            loss=LossConfig(),
            train=TrainConfig(),
            eval=EvalConfig(),
        )
    else:
        cfg = resolved
    profile = get_profile(state.get("profile", cfg.profile))
    model = build_model(cfg, profile)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    return model, state
