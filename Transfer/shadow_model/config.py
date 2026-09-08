"""Dataclass-based configuration loader for shadow model training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class CeLossConfig:
    weight: float = 1.0
    class_weights: list[float] = field(default_factory=list)


@dataclass
class DiceLossConfig:
    weight: float = 0.3


@dataclass
class TransitionLossConfig:
    weight: float = 0.15
    ramp_start: float = 0.05
    ramp_end: float = 0.15
    neighbors: int = 4
    dilate: int = 1
    pos_weight_clip: float = 10.0


@dataclass
class AffinityLossConfig:
    weight: float = 0.05
    ramp_start: float = 0.10
    ramp_end: float = 0.20
    offsets: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    max_pairs: int = 4096
    same_component: bool = True


@dataclass
class AuxLossConfig:
    weight: float = 0.10


@dataclass
class ScoreConfig:
    miou: float = 0.55
    bf1_2px: float = 0.30
    small_comp_miou: float = 0.15


@dataclass
class DataConfig:
    metadata_dir: str = ""
    train_crops: str = "train_crops.jsonl"
    val_crops: str = "val_crops.jsonl"
    crop_size: int = 512
    valid_center: int = 384
    ignore_index: int = -100
    max_crops_per_source_in_batch: int = 2
    augment_dihedral: bool = False
    allowed_dihedral: list[int] = field(default_factory=lambda: [0])
    fold: int = 0
    group_manifest: str = "design_families.json"


@dataclass
class LocalEncoderConfig:
    family: str = "convnextv2"
    model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    trainable: bool = True
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    layerwise_lr_decay: float = 0.875
    drop_path: float = 0.10


@dataclass
class DinoConfig:
    model_name: str = "vitl16_lvd"
    hub_entry: str = "dinov3_vitl16"
    frozen: bool = True
    shared_weights: bool = True
    patch_size: int = 16
    token_dim: int = 1024
    input_size: int = 512
    feature_layer: int = 23
    repo_dir: str = ""
    checkpoint: str = ""
    cache_dir: str = ""
    cache_dtype: str = "float16"
    require_cache: bool = True
    use_local_view: bool = True
    use_global_view: bool = True


@dataclass
class ModelConfig:
    decoder_dim: int = 256
    stem_channels: list[int] = field(default_factory=lambda: [32, 64, 96])
    refine_channels: int = 64
    use_structural_stem: bool = True
    cross_attn_blocks: int = 1
    cross_attn_heads: int = 4
    gate_init_local: float = 0.05
    gate_init_global: float = 0.05
    decoder_dropout: float = 0.10
    num_classes: int = 3


@dataclass
class LossConfig:
    ce: CeLossConfig = field(default_factory=CeLossConfig)
    dice: DiceLossConfig = field(default_factory=DiceLossConfig)
    transition: TransitionLossConfig = field(default_factory=TransitionLossConfig)
    affinity: AffinityLossConfig = field(default_factory=AffinityLossConfig)
    aux: AuxLossConfig = field(default_factory=AuxLossConfig)
    main_logit_boundary: bool = True


@dataclass
class TrainConfig:
    micro_batch: int = 1
    grad_accum: int = 8
    amp: str = "auto_bf16_else_fp16"
    optimizer: str = "adamw"
    lr_new: float = 1.25e-4
    lr_backbone: float = 1.5e-5
    weight_decay: float = 0.05
    no_decay: list[str] = field(default_factory=lambda: ["bias", "norm"])
    warmup_ratio: float = 0.05
    schedule: str = "cosine"
    min_lr: float = 1.0e-6
    grad_clip: float = 1.0
    total_steps: int = 3000
    hard_cap_steps: int = 6500
    val_every_steps: int = 130
    full_image_val_every_steps: int = 260
    early_stop_patience_full_image: int = 5
    seed: int = 42
    activation_checkpointing: bool = True
    disable_val: bool = False
    score_on_train: bool = False
    train_eval_max_batches: int = 64
    checkpoint_every_steps: int = 130


@dataclass
class EvalConfig:
    tile: int = 512
    halo: int = 64
    stride: int = 384
    score: ScoreConfig = field(default_factory=ScoreConfig)


@dataclass
class ShadowConfig:
    profile: str = "BEST_V1"
    data: DataConfig = field(default_factory=DataConfig)
    local_encoder: LocalEncoderConfig = field(default_factory=LocalEncoderConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Legacy alias for scripts still reading global_encoder
    @property
    def global_encoder(self) -> DinoConfig:
        return self.dino

    def to_dict(self) -> dict:
        return asdict(self)


_NESTED: dict[str, type] = {
    "ce": CeLossConfig,
    "dice": DiceLossConfig,
    "transition": TransitionLossConfig,
    "affinity": AffinityLossConfig,
    "aux": AuxLossConfig,
    "score": ScoreConfig,
}


def _build_dataclass(cls, data: dict | None):
    if data is None:
        return cls()
    kwargs = {}
    for key, value in data.items():
        if key not in cls.__dataclass_fields__:
            continue
        nested_cls = _NESTED.get(key)
        if nested_cls is not None and isinstance(value, dict):
            value = _build_dataclass(nested_cls, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ShadowConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    section_map = {
        "data": DataConfig,
        "local_encoder": LocalEncoderConfig,
        "dino": DinoConfig,
        "model": ModelConfig,
        "loss": LossConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
    }
    kwargs = {"profile": raw.get("profile", "BEST_V1")}
    for section, cls in section_map.items():
        kwargs[section] = _build_dataclass(cls, raw.get(section, {}))
    # Backward compat: old configs use global_encoder
    if "dino" not in raw and "global_encoder" in raw:
        kwargs["dino"] = _build_dataclass(DinoConfig, raw["global_encoder"])
    return ShadowConfig(**kwargs)
