"""CSN-V4 unit and integration tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from csn_v4.constants import MODEL_NAME, NUM_SHADE_SUBTYPES
from csn_v4.data.manifest_builder import ManifestBuildConfig, build_all_manifests
from csn_v4.data.targets import build_tile_targets
from csn_v4.data.splits import assign_scene_splits, assert_no_split_leakage, discover_scenes_from_source
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import HALO, INPUT_SIZE, assert_full_coverage, build_tile_spec, iter_core_tiles
from csn_v4.heads import CategoricalLevelHead
from csn_v4.losses import CSNV4Loss
from csn_v4.metrics.region_metrics import _tile_halo_mask
from csn_v4.model import CarpetShadeNetV4


@pytest.mark.parametrize("k", range(8))
def test_d4_unique_and_inverse(k: int):
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    t = D4Transform(k)
    assert np.array_equal(arr, t.inverse.apply(t.apply(arr)))


def test_d4_eight_unique():
    arr = np.arange(24, dtype=np.uint8).reshape(4, 6)
    outs = {D4Transform(k).apply(arr).tobytes() for k in range(8)}
    assert len(outs) == 8


def test_d4_rectangular_wh_swap():
    arr = np.zeros((100, 200), dtype=np.uint8)
    t = D4Transform(1)
    out = t.apply(arr)
    assert out.shape == (200, 100)


def test_halo_four_sided():
    m = _tile_halo_mask(512, 512, HALO)
    assert m[:HALO, :].all()
    assert m[-HALO:, :].all()
    assert m[:, :HALO].all()
    assert m[:, -HALO:].all()
    assert not m[HALO:-HALO, HALO:-HALO].any()


@pytest.mark.parametrize("fw,fh", [(200, 200), (512, 512), (777, 512), (50, 2000)])
def test_full_coverage(fw: int, fh: int):
    assert_full_coverage(fw, fh)


def test_no_ordinal_head_in_v4_model():
    m = CarpetShadeNetV4.__init__.__annotations__
    head = CategoricalLevelHead(64, 64, 256)
    assert not hasattr(head, "ordinal")
    params = [n for n, _ in head.named_parameters()]
    assert all("ordinal" not in n for n in params)


def test_categorical_level_head_output_channels():
    head = CategoricalLevelHead(64, 64, 256)
    feat = torch.randn(1, 64, 32, 32)
    where = torch.randn(1, 64, 32, 32)
    trans = torch.randn(1, 64, 32, 32)
    g = torch.randn(1, 256)
    out = head(feat, where, trans, g)
    assert out.shape == (1, NUM_SHADE_SUBTYPES, 32, 32)


def test_affinity_targets_from_semantic_without_level_bmp():
    sem = np.full((64, 64), 255, dtype=np.uint8)
    sem[10:30, 10:30] = 200
    sem[30:50, 30:50] = 150
    weights = np.ones_like(sem, dtype=np.float32)
    t = build_tile_targets(sem, None, None, None, None, None, weights)
    assert (t["affinity_valid"] > 0.5).any(), "affinity_valid should be non-zero from semantic levels"


def test_affinity_loss_finite_with_invalid_targets():
    from csn_v4.config import LossConfig

    loss_fn = CSNV4Loss(LossConfig())
    b, h, w = 1, 32, 32
    outputs = {
        "refined_logits": torch.randn(b, 4, h, w),
        "where_logits": torch.randn(b, 1, h, w),
        "transition_logits": torch.randn(b, 1, h, w),
        "affinity_logits": torch.randn(b, 8, h, w) * 50,
        "level_logits": torch.randn(b, 3, h, w),
        "base_logits": torch.randn(b, 4, h, w),
    }
    targets = {
        "target_class": torch.randint(0, 4, (b, h, w)),
        "shade_mask": torch.ones(b, h, w, dtype=torch.bool),
        "transition_mask": torch.zeros(b, h, w, dtype=torch.uint8),
        "black_lock": torch.zeros(b, h, w, dtype=torch.uint8),
        "valid_mask": torch.ones(b, h, w, dtype=torch.uint8) * 255,
        "where_target": torch.ones(b, h, w),
        "level_target": torch.randint(0, 3, (b, h, w)),
        "affinity_targets": torch.full((b, 8, h, w), -1.0),
        "affinity_valid": torch.zeros(b, 8, h, w),
        "supervision_weights": torch.ones(b, h, w),
    }
    targets["affinity_valid"][:, 0, :16, :16] = 1.0
    targets["affinity_targets"][:, 0, :16, :16] = 1.0
    losses = loss_fn(outputs, targets)
    assert all(torch.isfinite(v).all() for v in losses.values())


def test_loss_nonzero_with_materialized_uint01_masks():
    """NPZ stores bool masks as uint8 0/1 — must not be compared with >127."""
    from csn_v4.config import LossConfig
    from csn_v4.masks import as_bool_mask

    assert as_bool_mask(torch.tensor([[1, 0]], dtype=torch.uint8)).tolist() == [[True, False]]
    assert as_bool_mask(torch.tensor([[255, 0]], dtype=torch.uint8)).tolist() == [[True, False]]
    assert as_bool_mask(np.array([[1, 0]], dtype=np.uint8)).tolist() == [[True, False]]

    loss_fn = CSNV4Loss(LossConfig())
    b, h, w = 1, 32, 32
    outputs = {
        "refined_logits": torch.randn(b, 4, h, w),
        "where_logits": torch.randn(b, 1, h, w),
        "transition_logits": torch.randn(b, 1, h, w),
        "affinity_logits": torch.randn(b, 8, h, w),
        "level_logits": torch.randn(b, 3, h, w),
        "base_logits": torch.randn(b, 4, h, w),
    }
    targets = {
        "target_class": torch.randint(0, 4, (b, h, w)),
        "shade_mask": torch.ones(b, h, w, dtype=torch.uint8),  # 0/1 style
        "transition_mask": torch.zeros(b, h, w, dtype=torch.uint8),
        "black_lock": torch.zeros(b, h, w, dtype=torch.uint8),
        "valid_mask": torch.ones(b, h, w, dtype=torch.uint8),  # 0/1 — broken under >127
        "where_target": torch.ones(b, h, w),
        "level_target": torch.randint(0, 3, (b, h, w)),
        "affinity_targets": torch.zeros(b, 8, h, w),
        "affinity_valid": torch.ones(b, 8, h, w),
        "supervision_weights": torch.ones(b, h, w),
    }
    losses = loss_fn(outputs, targets)
    assert float(losses["final_ce"]) > 0.0
    assert float(losses["where"]) > 0.0
    assert float(losses["level_ce"]) > 0.0
    assert float(losses["affinity"]) > 0.0
    assert float(losses["base_refined_consistency"]) > 0.0
    assert float(losses["total"]) > 0.0


def test_loss_backward():
    from csn_v4.config import LossConfig

    loss_fn = CSNV4Loss(LossConfig())
    logits = torch.randn(1, 4, 16, 16, requires_grad=True)
    outputs = {
        "refined_logits": logits,
        "where_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "transition_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "affinity_logits": torch.randn(1, 8, 16, 16, requires_grad=True),
        "level_logits": torch.randn(1, 3, 16, 16, requires_grad=True),
        "base_logits": torch.randn(1, 4, 16, 16, requires_grad=True),
    }
    targets = {
        "target_class": torch.randint(0, 4, (1, 16, 16)),
        "shade_mask": torch.ones(1, 16, 16, dtype=torch.bool),
        "transition_mask": torch.zeros(1, 16, 16, dtype=torch.uint8),
        "black_lock": torch.zeros(1, 16, 16, dtype=torch.uint8),
        "valid_mask": torch.ones(1, 16, 16, dtype=torch.uint8) * 255,
        "where_target": torch.zeros(1, 16, 16),
        "level_target": torch.full((1, 16, 16), -100),
        "affinity_targets": torch.zeros(1, 8, 16, 16),
        "affinity_valid": torch.zeros(1, 8, 16, 16),
        "supervision_weights": torch.ones(1, 16, 16),
    }
    losses = loss_fn(outputs, targets)
    losses["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_split_no_leakage_on_real_source():
    src = Path("E:/Shadowing/Dataset_V3")
    if not src.is_dir():
        pytest.skip("Dataset_V3 not available")
    scenes = discover_scenes_from_source(src)
    split = assign_scene_splits(scenes, seed=42)
    if split.val_scenes:
        assert_no_split_leakage(split.train_scenes, split.val_scenes)


def test_manifest_deterministic(tmp_path):
    src = Path("E:/Shadowing/Dataset_V3")
    if not src.is_dir():
        pytest.skip("Dataset_V3 not available")
    scenes = discover_scenes_from_source(src)[:2]
    for s in scenes:
        s["split"] = "train"
    split = assign_scene_splits(scenes, seed=42)
    split.train_scenes = scenes
    split.val_scenes = []
    cfg = ManifestBuildConfig(seed=42, offgrid_tiles_per_scene=4, spatial_offgrid_per_scene=4, subscenes_per_size=1)
    r1 = build_all_manifests(split, tmp_path / "a", cfg)
    data1 = (tmp_path / "a" / "manifests" / "train_capacity_tiles_bw.jsonl").read_bytes()
    r2 = build_all_manifests(split, tmp_path / "b", cfg)
    data2 = (tmp_path / "b" / "manifests" / "train_capacity_tiles_bw.jsonl").read_bytes()
    assert data1 == data2


def test_model_name_constant():
    assert MODEL_NAME == "CarpetShadeNetV4"
