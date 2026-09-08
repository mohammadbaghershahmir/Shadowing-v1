"""Checkpoint save/load with full state restoration."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes(metadata_dir: Path, train_crops: str, val_crops: str) -> dict[str, str]:
    """Hash crop manifest files for reproducibility."""
    out: dict[str, str] = {}
    for name in [train_crops, val_crops]:
        p = metadata_dir / name
        if p.exists():
            out[name] = _sha256_file(p)
    return out


def git_hash() -> str:
    """Return current git commit hash or empty string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    global_step: int,
    epoch: int,
    best_score: float,
    *,
    resolved_config: dict | None = None,
    profile: str | None = None,
    manifest_hash: dict[str, str] | None = None,
    config: dict | None = None,
    rng_states: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "global_step": global_step,
        "epoch": epoch,
        "best_score": best_score,
        "resolved_config": resolved_config or config,
        "profile": profile,
        "manifest_hashes": manifest_hash,
        "git_hash": git_hash(),
        "config": config,
    }
    if rng_states:
        state["rng_states"] = rng_states

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    LOGGER.info("Saved checkpoint: %s (step=%d, score=%.4f)", path, global_step, best_score)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: torch.amp.GradScaler | None = None,
    *,
    strict: bool = True,
) -> dict:
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    LOGGER.info("Loaded checkpoint: %s (step=%d)", path, state.get("global_step", 0))
    return state


def get_rng_states() -> dict:
    import numpy as np

    states: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    import numpy as np

    torch.random.set_rng_state(states["torch"])
    np.random.set_state(states["numpy"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
