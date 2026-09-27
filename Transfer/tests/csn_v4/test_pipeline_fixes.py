"""Regression tests for metrics, fusion ablations, inference constraints, setup."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from csn_v4.factory import load_model_from_checkpoint
from csn_v4.fusion import FourBranchFusion
from csn_v4.geometry.extract import build_supervision_weights
from csn_v4.geometry.tile_spec import build_tile_spec
from csn_v4.indexed.contract import compose_output
from csn_v4.indexed.schema import INDEX_ALLOWED, INDEX_BACKGROUND, INDEX_OUTLINE
from csn_v4.inference.tiled import (
    _apply_eligible_constraints,
    _smooth_window,
)
from csn_v4.metrics.region_metrics import (
    _fp_aware_miou,
    _presence_aware_miou,
    aggregate_region_metrics,
    compute_region_metrics,
)


def test_fp_aware_miou_penalizes_false_positive_class():
    h, w = 32, 32
    target = np.zeros((h, w), dtype=np.int64)  # only class 0 present
    pred = np.zeros((h, w), dtype=np.int64)
    pred[8:16, 8:16] = 2  # hallucinate shade class absent in GT
    mask = np.ones((h, w), dtype=bool)

    presence, _, ious, present = _presence_aware_miou(pred, target, mask)
    fp = _fp_aware_miou(pred, target, mask)
    assert present[0] == 1 and present[2] == 0
    # Presence-aware ignores class 2; FP-aware includes it with IoU=0.
    assert fp < presence
    assert ious[2] == 0.0


def test_region_metrics_exports_fp_and_per_class():
    h, w = 64, 64
    target = np.zeros((h, w), dtype=np.int64)
    target[10:40, 10:40] = 1
    pred = target.copy()
    pred[20:30, 20:30] = 2
    trans = np.zeros((h, w), dtype=np.uint8)
    valid = np.full((h, w), 255, dtype=np.uint8)
    black = np.zeros((h, w), dtype=np.uint8)
    rm = compute_region_metrics(pred, target, trans, valid, black, skip_small_component=True)
    d = rm.to_dict()
    assert "global_miou_fp" in d
    assert "per_class_iou" in d
    assert len(d["per_class_iou"]) == 4
    agg = aggregate_region_metrics([rm])
    assert "global_miou_fp" in agg
    assert "worst_orient_global_miou" in agg


def test_fusion_sources_ablation_masks_disabled_branches():
    fusion = FourBranchFusion(
        fusion_dim=16,
        convnext_dim=16,
        dino_dim=16,
        num_heads=2,
        num_blocks=1,
        sources=["convnext"],
    )
    B, C, H, W = 1, 16, 4, 4
    out, gates = fusion(
        torch.randn(B, C, H, W),
        torch.randn(B, C, H, W) * 10,
        torch.randn(B, C, H, W) * 10,
        torch.randn(B, 3, C) * 10,
        torch.zeros(B, 6),
        local_size=H,
    )
    assert out.shape == (B, 16, H, W)
    assert gates["enabled_sources"] == ("convnext",)
    # Disabled branch features must be zeroed.
    assert torch.count_nonzero(gates["branches"][:, 1]) == 0
    assert torch.count_nonzero(gates["branches"][:, 2]) == 0
    assert torch.count_nonzero(gates["branches"][:, 3]) == 0


def test_fusion_rejects_missing_convnext():
    with pytest.raises(ValueError, match="convnext"):
        FourBranchFusion(fusion_dim=8, convnext_dim=8, dino_dim=8, sources=["dino_local"])


def test_eligible_mask_and_outline_constraints():
    src = np.full((16, 16), 255, dtype=np.uint8)
    src[0, :] = 0  # outline
    pred = np.ones((16, 16), dtype=np.int64) * 2
    elig = np.zeros((16, 16), dtype=bool)
    elig[4:12, 4:12] = True
    out = _apply_eligible_constraints(pred, src, eligible_mask=elig)
    assert np.all(out[0, :] == 0)
    assert np.all(out[~elig] == 0)
    assert np.all(out[elig & (src != 0)] == 2)


def test_compose_output_locks_outside_allowed():
    x = np.full((8, 8), INDEX_BACKGROUND, dtype=np.int64)
    x[1, :] = INDEX_OUTLINE
    x[2:6, 2:6] = INDEX_ALLOWED
    pred = np.ones((8, 8), dtype=np.int64) * 2  # would be shadow_2 if applied outside
    y = compose_output(x, pred)
    assert np.array_equal(y[~ (x == INDEX_ALLOWED)], x[~ (x == INDEX_ALLOWED)])
    assert np.all(y[x == INDEX_ALLOWED] == 4)


def test_smooth_window_core_is_one():
    w = _smooth_window(512, 512, 64)
    assert w.shape == (512, 512)
    assert float(w[256, 256]) == 1.0
    assert float(w[0, 0]) < 1.0


def test_rare_shade_boosts_supervision_weights():
    spec = build_tile_spec(512, 512, 0, 0)
    valid = np.ones((512, 512), dtype=bool)
    level = np.full((512, 512), 255, dtype=np.uint8)
    level[100:120, 100:120] = 2  # dark rare
    trans = np.zeros((512, 512), dtype=np.uint8)
    trans[200:210, 200:210] = 255
    w = build_supervision_weights(
        spec, valid, transition_mask=trans, shade_level_id=level,
        rare_shade_boost=2.0, transition_boost=1.5, small_detail_boost=1.0,
    )
    ys, xs = spec.model_core_slices()
    # Compare boosted dark vs nearby non-dark core pixel
    dark_w = float(w[110, 110])
    plain_w = float(w[150, 150])
    assert dark_w > plain_w
    assert float(w[205, 205]) > plain_w


def test_load_model_from_checkpoint_signature():
    sig = inspect.signature(load_model_from_checkpoint)
    params = list(sig.parameters)
    assert params[0] == "checkpoint_path"
    assert "device" in sig.parameters
    assert sig.parameters["device"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.return_annotation != inspect.Signature.empty


def test_setup_patch_config_structured(tmp_path, monkeypatch):
    import setup_and_train as sat

    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump({
            "data": {"dataset_root": "OLD_DS"},
            "model": {"dino": {
                "repo_root": "OLD_REPO",
                "checkpoint": "OLD_CKPT",
                "cache_dir": "OLD_CACHE",
            }},
        }),
        encoding="utf-8",
    )
    active = tmp_path / "active.yaml"
    monkeypatch.setattr(sat, "CONFIG_TEMPLATE", template)
    monkeypatch.setattr(sat, "CONFIG_ACTIVE", active)

    # Create fake dino paths for validation helpers (patch_config itself only writes YAML)
    dino_root = tmp_path / "dino"
    dino_root.mkdir()
    ckpt = tmp_path / "weights.pth"
    ckpt.write_bytes(b"x")
    ds = tmp_path / "Dataset_V4"
    out = sat.patch_config(ds, dino_root, ckpt)
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert raw["data"]["dataset_root"] == ds.as_posix()
    assert raw["model"]["dino"]["repo_root"] == dino_root.as_posix()
    assert raw["model"]["dino"]["checkpoint"] == ckpt.as_posix()
    assert "OLD_" not in out.read_text(encoding="utf-8")


def test_ablation_configs_exist_and_parse_sources():
    root = Path(__file__).resolve().parents[2] / "configs" / "ablations"
    expected = {
        "csn_v4_ablation_stem_convnext.yaml": ["convnext"],
        "csn_v4_ablation_plus_local.yaml": ["convnext", "dino_local"],
        "csn_v4_ablation_plus_context.yaml": ["convnext", "dino_local", "dino_context"],
        "csn_v4_ablation_full.yaml": ["convnext", "dino_local", "dino_context", "dino_global"],
    }
    for name, sources in expected.items():
        path = root / name
        assert path.is_file(), f"missing {path}"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["model"]["fusion"]["sources"] == sources
        assert raw["train"]["seed"] == 42


def test_dataset_v2_importable():
    import dataset_v2
    from dataset_v2.pipeline import PilotConfig, discover_bmps
    assert PilotConfig is not None
    assert callable(discover_bmps)
    assert hasattr(dataset_v2, "__file__")


def test_evaluate_script_calls_factory_correctly():
    """Smoke: evaluate_csn_v4 source must unpack (model, ckpt) and pass device kwonly."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "evaluate_csn_v4.py").read_text(
        encoding="utf-8"
    )
    assert "model, _ckpt = load_model_from_checkpoint" in src or "model, ckpt = load_model_from_checkpoint" in src
    assert "device=device" in src
    assert "load_model_from_checkpoint(args.checkpoint, cfg, device)" not in src
