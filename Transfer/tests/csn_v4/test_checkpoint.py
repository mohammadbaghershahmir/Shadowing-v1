"""Checkpoint round-trip tests for CSN-V4."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from csn_v4.config import load_config
from csn_v4.factory import build_model
from csn_v4.training.checkpointing import CheckpointManager

_DINO_ROOT = Path("E:/Shadowing")
_DINO_CKPT = Path("E:/Shadowing/weight/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
_HAS_DINO = (_DINO_ROOT / "dinov3").is_dir() and _DINO_CKPT.is_file()


def _load_cfg():
    cfg = load_config("configs/csn_v4_capacity.yaml")
    cfg.model.dino.repo_root = str(_DINO_ROOT)
    cfg.model.dino.checkpoint = str(_DINO_CKPT)
    return cfg


@pytest.mark.skipif(not Path("configs/csn_v4_capacity.yaml").exists(), reason="config missing")
@pytest.mark.skipif(not _HAS_DINO, reason="DINOv3 repo/weights not available on this machine")
def test_checkpoint_atomic_roundtrip():
    cfg = _load_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.trainable_param_groups(lr_new=1e-4), lr=1e-4)
        mgr = CheckpointManager(run_dir, cfg)
        mgr.save_last(model=model, optimizer=opt, scheduler=None, scaler=None, global_step=4, optimizer_step=1)
        assert (run_dir / "last.pt").is_file()
        model2 = build_model(cfg)
        opt2 = torch.optim.AdamW(model2.trainable_param_groups(lr_new=1e-4), lr=1e-4)
        mgr.load(run_dir / "last.pt", model=model2, optimizer=opt2)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if n1.startswith("dino.backbone"):
                continue  # backbone intentionally omitted from checkpoint
            assert torch.allclose(p1, p2), n1


@pytest.mark.skipif(not Path("configs/csn_v4_capacity.yaml").exists(), reason="config missing")
@pytest.mark.skipif(not _HAS_DINO, reason="DINOv3 repo/weights not available on this machine")
def test_checkpoint_rejects_pending_grad_accum():
    cfg = _load_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.trainable_param_groups(lr_new=1e-4), lr=1e-4)
        mgr = CheckpointManager(run_dir, cfg)
        with pytest.raises(RuntimeError, match="grad_accum_pending"):
            mgr.save_last(
                model=model, optimizer=opt, scheduler=None, scaler=None,
                global_step=1, optimizer_step=0, grad_accum_pending=2,
            )


def test_checkpoint_rejects_input_mode_mismatch():
    """BW vs indexed_guided must not silently resume into each other."""
    if not _HAS_DINO:
        pytest.skip("DINOv3 repo/weights not available on this machine")
    cfg = _load_cfg()
    cfg.data.input_mode = "bw"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        model = build_model(cfg)
        opt = torch.optim.AdamW(model.trainable_param_groups(lr_new=1e-4), lr=1e-4)
        mgr = CheckpointManager(run_dir, cfg)
        mgr.save_last(model=model, optimizer=opt, scheduler=None, scaler=None, global_step=1, optimizer_step=1)
        cfg2 = _load_cfg()
        cfg2.data.input_mode = "indexed_guided"
        model2 = build_model(cfg2)
        mgr2 = CheckpointManager(run_dir, cfg2)
        with pytest.raises(ValueError, match="input_mode mismatch"):
            mgr2.load(run_dir / "last.pt", model=model2, allow_hash_override=True)
