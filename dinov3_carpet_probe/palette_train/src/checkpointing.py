"""Checkpoint save and resume utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dinov3_carpet_probe.src.io_utils import ensure_dir


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    manifest: dict[str, Any],
    best_metric: float | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "head_state": model.head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "manifest": manifest,
        "model_key": config["model_key"],
        "checkpoint_path": model.checkpoint_path,
        "max_num_colors": config["max_num_colors"],
        "best_metric": best_metric,
        "random_state": capture_random_state(),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.head.load_state_dict(checkpoint["head_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if restore_rng and checkpoint.get("random_state") is not None:
        restore_random_state(checkpoint["random_state"])
    return checkpoint
