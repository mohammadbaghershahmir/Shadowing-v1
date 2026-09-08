"""Configuration loader for CSN-V3."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset_root: str = "E:/Shadowing/Dataset_V3"
    train_crops: str = "manifests/train_crops_bw.jsonl"
    eval_full_images: str = "manifests/eval_full_images.jsonl"
    input_mode: str = "bw"  # "bw" | "magenta"
    magenta_guide_dir: str = "E:/Shadowing/converted_ready"
    crop_size: int = 512
    context_size: int = 1024
    global_long_side: int = 1024
    center_halo: int = 64
    ignore_index: int = -100
    allow_padded_context: bool = True
    augment_dihedral: bool = True
    boundary_dilate: int = 2


@dataclass
class DinoConfig:
    repo_root: str = "E:/Dino/dinov3"
    checkpoint: str = (
        "E:/Dino/dinov3/dinov3_carpet_probe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    )
    architecture: str = "dinov3_vitl16"
    patch_size: int = 16
    embed_dim: int = 1024
    intermediate_layers: list[int] = field(default_factory=lambda: [-4, -3, -2, -1])
    frozen: bool = True
    strict_load: bool = True
    cache_dir: str = "E:/Shadowing/Dataset_V3/global_tokens_csn_v3"
    cache_dtype: str = "float16"


@dataclass
class ConvNextConfig:
    architecture: str = "convnextv2_base"
    model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    output_dims: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    output_strides: list[int] = field(default_factory=lambda: [4, 8, 16, 32])


@dataclass
class StructuralStemConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 96])


@dataclass
class FusionConfig:
    dim: int = 256
    cross_attention_heads: int = 4
    cross_attention_blocks: int = 1
    sources: list[str] = field(
        default_factory=lambda: ["convnext", "dino_local", "dino_context", "dino_global"]
    )
    dynamic_spatial_channel_gate: bool = True
    zero_init_residual: bool = True


@dataclass
class DecoderConfig:
    channels: list[int] = field(default_factory=lambda: [256, 256, 192, 128, 96, 64])
    learnable_pixelshuffle_upsampling: bool = True
    gated_lateral_fusion: bool = True


@dataclass
class HeadsConfig:
    where: bool = True
    transition: bool = True
    affinity_offsets: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    level_classes: int = 3
    ordinal_thresholds: int = 2
    joint_decision: bool = True
    line_aware_refiner: bool = True


@dataclass
class ModelConfig:
    name: str = "CarpetShadeNetV3"
    local_size: int = 512
    context_size: int = 1024
    global_long_side: int = 1024
    decoder_dim: int = 256
    dino: DinoConfig = field(default_factory=DinoConfig)
    convnext: ConvNextConfig = field(default_factory=ConvNextConfig)
    structural_stem: StructuralStemConfig = field(default_factory=StructuralStemConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)


@dataclass
class BoundaryBoostConfig:
    enabled: bool = True
    transition_dilate: int = 2
    where_boost: float = 2.0
    transition_boost: float = 3.0
    affinity_boost: float = 2.0
    level_ce_boost: float = 1.5


@dataclass
class LossConfig:
    final_ce: float = 1.0
    final_dice: float = 0.3
    where: float = 0.5
    level_ce: float = 0.5
    ordinal: float = 0.2
    transition: float = 0.5
    affinity: float = 0.3
    transition_boundary_dice: float = 0.4
    overlap_consistency: float = 0.1
    use_soft_ordinal_emd: bool = True
    boundary_boost: BoundaryBoostConfig = field(default_factory=BoundaryBoostConfig)


@dataclass
class OptimizerConfig:
    lr_new_modules: float = 3.0e-4
    lr_convnext_stage34: float = 2.0e-5
    lr_convnext_stage12: float = 0.0
    lr_dino: float = 0.0
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    micro_batch: int = 1
    grad_accum: int = 4
    amp: str = "auto_bf16_else_fp16"
    optimizer: str = "adamw"
    total_micro_steps: int = 15000
    optimizer_steps: int = 3000
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
    train_eval_max_batches: int = 8
    train_eval_max_samples: int = 16
    rotated_eval_k: list[int] = field(default_factory=lambda: [1, 2, 3])
    disable_val: bool = True
    mode: str = "overfit"


@dataclass
class InferenceConfig:
    tile_size: int = 512
    halo: int = 64
    stride: int = 384
    global_long_side: int = 1024


@dataclass
class RendererConfig:
    class_to_gray: dict[int, int] = field(
        default_factory=lambda: {0: 255, 1: 200, 2: 150, 3: 100}
    )
    hard_black_lock: bool = True
    output_format: str = "bmp"


@dataclass
class CSNV3Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)


def _dict_to_config(d: dict) -> CSNV3Config:
    model_d = dict(d.get("model", {}))
    dino_d = model_d.pop("dino", {})
    conv_d = model_d.pop("convnext", {})
    stem_d = model_d.pop("structural_stem", {})
    fusion_d = model_d.pop("fusion", {})
    dec_d = model_d.pop("decoder", {})
    heads_d = model_d.pop("heads", {})

    loss_d = dict(d.get("loss", {}))
    boost_d = loss_d.pop("boundary_boost", {})

    opt_d = dict(d.get("optimizer", {}))
    train_d = dict(d.get("train", {}))
    if "optimizer" in train_d:
        opt_d = {**opt_d, **train_d.pop("optimizer")}

    return CSNV3Config(
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


def load_config(path: str | Path) -> CSNV3Config:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _dict_to_config(raw or {})


def config_to_dict(cfg: CSNV3Config) -> dict:
    return asdict(cfg)
