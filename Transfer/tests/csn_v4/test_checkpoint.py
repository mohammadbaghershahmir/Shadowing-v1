"""Checkpoint round-trip tests for CSN-V4."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from csn_v4.config import load_config
from csn_v4.factory import build_model
from csn_v4.training.checkpointing import CheckpointManager


@pytest.mark.skipif(not Path("configs/csn_v4_capacity.yaml").exists(), reason="config missing")
def test_checkpoint_atomic_roundtrip():
    cfg = load_config("configs/csn_v4_capacity.yaml")
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
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


@pytest.mark.skipif(not Path("configs/csn_v4_capacity.yaml").exists(), reason="config missing")
def test_checkpoint_rejects_pending_grad_accum():
    cfg = load_config("configs/csn_v4_capacity.yaml")
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
