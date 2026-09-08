"""Configuration loader for CSN-V4."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from csn_v3.config import (
    BoundaryBoostConfig,
    ConvNextConfig,
    DecoderConfig,
    DinoConfig,
    FusionConfig,
    HeadsConfig,
    InferenceConfig,
    ModelConfig,
    OptimizerConfig,
    RendererConfig,
    StructuralStemConfig,
)


@dataclass
class DataConfig:
    dataset_root: str = "E:/Shadowing/Dataset_V4"
    train_tiles: str = "manifests/train_tiles_bw.jsonl"
    scenes: str = "manifests/scenes.jsonl"
    eval_full_images: str = "manifests/eval_full_images.jsonl"
    input_mode: str = "bw"
    magenta_guide_dir: str = "E:/Shadowing/converted_ready"
    crop_size: int = 512
    context_size: int = 1024
    global_long_side: int = 1024
    halo: int = 64
    core_size: int = 384
    halo_loss_weight: float = 1.0
    supervision_mode: str = "overfit_full_crop"
    ignore_index: int = -100
    boundary_dilate: int = 2
    deterministic_d4: bool = True
    materialized_root: str = "materialized"
    prefer_materialized: bool = True
    dataloader_workers: int = 2


@dataclass
class LossConfig:
    final_ce: float = 1.0
    final_dice: float = 0.3
    where: float = 0.5
    level_ce: float = 0.5
    ordinal: float = 0.0
    use_ordinal_head: bool = False
    transition: float = 0.5
    affinity: float = 0.3
    transition_boundary_dice: float = 0.4
    base_refined_consistency: float = 0.1
    boundary_boost: BoundaryBoostConfig = field(default_factory=BoundaryBoostConfig)


@dataclass
class TrainConfig:
    micro_batch: int = 1
    grad_accum: int = 4
    amp: str = "auto_bf16_else_fp16"
    optimizer_steps: int = 3000
    total_micro_steps: int | None = None
    warmup_ratio: float = 0.02
    schedule: str = "cosine"
    min_lr: float = 1.0e-6
    seed: int = 42
    phase: int = 6
    teacher_forcing: float = 0.0
    teacher_forcing_schedule: list[float] = field(
        default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0]
    )
    checkpoint_every_steps: int = 200
    eval_every_steps: int = 400
    full_image_eval_every_steps: int = 1000
    plot_every_steps: int = 100
    score_on_train: bool = True
    eval_all_tiles: bool = False
    eval_max_tiles: int = 32
    initial_eval_max_tiles: int = 16
    skip_initial_eval: bool = False
    eval_all_orientations: bool = True
    eval_max_holdout_scenes: int = 1
    # During training scene holdout: how many D4 orients (1–8). Full 8 is very slow on large carpets.
    eval_holdout_max_orientations: int = 2
    weight_decay: float = 0.0
    mode: str = "overfit"


@dataclass
class CSNV4Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)

    def resolved_micro_steps(self) -> int:
        if self.train.total_micro_steps is not None:
            return self.train.total_micro_steps
        return self.train.optimizer_steps * self.train.grad_accum


def _dict_to_config(d: dict) -> CSNV4Config:
    model_d = dict(d.get("model", {}))
    dino_d = model_d.pop("dino", {})
    conv_d = model_d.pop("convnext", {})
    stem_d = model_d.pop("structural_stem", {})
    fusion_d = model_d.pop("fusion", {})
    dec_d = model_d.pop("decoder", {})
    heads_d = model_d.pop("heads", {})
    if model_d.get("name"):
        model_d["name"] = "CarpetShadeNetV4"

    loss_d = dict(d.get("loss", {}))
    boost_d = loss_d.pop("boundary_boost", {})
    opt_d = dict(d.get("optimizer", {}))
    train_d = dict(d.get("train", {}))
    if "weight_decay" in train_d:
        opt_d["weight_decay"] = train_d["weight_decay"]

    cfg = CSNV4Config(
        model=ModelConfig(
            **model_d,
            dino=DinoConfig(**dino_d),
            convnext=ConvNextConfig(**conv_d),
            structural_stem=StructuralStemConfig(**stem_d),
            fusion=FusionConfig(**fusion_d),
            decoder=DecoderConfig(**dec_d),
            heads=HeadsConfig(**heads_d),
        ),
        data=DataConfig(**d.get("data", {})),
        loss=LossConfig(**loss_d, boundary_boost=BoundaryBoostConfig(**boost_d)),
        optimizer=OptimizerConfig(**opt_d),
        train=TrainConfig(**train_d),
        inference=InferenceConfig(**d.get("inference", {})),
        renderer=RendererConfig(**d.get("renderer", {})),
    )
    if cfg.train.total_micro_steps is None:
        cfg.train.total_micro_steps = cfg.resolved_micro_steps()
    return cfg


def load_config(path: str | Path) -> CSNV4Config:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _dict_to_config(raw or {})


def config_to_dict(cfg: CSNV4Config) -> dict:
    return asdict(cfg)
