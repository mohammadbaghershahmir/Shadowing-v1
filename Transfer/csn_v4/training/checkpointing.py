"""Atomic checkpoint save/load for CSN-V4 training."""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from csn_v4.config import CSNV4Config, config_to_dict
from csn_v4.constants import CLASS_TO_GRAY, MODEL_NAME, NUM_CLASSES, SEMANTIC_GRAY_TO_CLASS
from csn_v4.data.paths import sha256_file, sha256_manifest
from csn_v4.geometry.tile_spec import CORE_SIZE, HALO, INPUT_SIZE

LOGGER = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1

# Frozen DINOv3 weights are reloaded from model.dino.checkpoint on resume;
# embedding them in every .pt roughly doubles disk use (~1GB+) and often OOMs the drive.
_SKIP_STATE_PREFIXES = ("dino.backbone.",)

_BEST_FILES = {
    "capacity": "best_capacity.pt",
    "spatial": "best_spatial_diag.pt",
    "scene_val": "best_scene_val.pt",
    "overall": "best.pt",
}


def model_state_dict_for_checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in model.state_dict().items()
        if not any(k.startswith(p) for p in _SKIP_STATE_PREFIXES)
    }


def load_model_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    """Load trainable weights; keep current frozen DINO backbone if absent from ckpt."""
    merged = model.state_dict()
    merged.update(state_dict)
    incompatible = model.load_state_dict(merged, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def get_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    if "python" in states:
        __import__("random").setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.random.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def _git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "dirty": "true" if dirty.stdout.strip() else "false",
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": "", "dirty": "unknown"}


def manifest_hashes_for_config(cfg: CSNV4Config) -> dict[str, str]:
    root = Path(cfg.data.dataset_root)
    out: dict[str, str] = {}
    for rel in (
        cfg.data.scenes,
        cfg.data.train_tiles,
        cfg.data.eval_full_images,
        "manifests/val_spatial_fixed_bw.jsonl",
        "manifests/val_scene_holdout_bw.jsonl",
    ):
        path = root / rel
        if path.is_file():
            out[rel] = sha256_manifest(path)
    return out


def dino_checkpoint_sha256(cfg: CSNV4Config) -> str:
    path = Path(cfg.model.dino.checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"DINO checkpoint missing: {path}")
    return sha256_file(path)


class CheckpointManager:
    """Save and load CSN-V4 checkpoints with hash verification."""

    def __init__(
        self,
        run_dir: str | Path,
        cfg: CSNV4Config,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.config_path = str(config_path) if config_path else None
        self.best_scores: dict[str, float] = {
            "capacity": float("-inf"),
            "spatial": float("-inf"),
            "scene_val": float("-inf"),
            "overall": float("-inf"),
        }
        self._manifest_hashes = manifest_hashes_for_config(cfg)
        self._dino_sha256 = dino_checkpoint_sha256(cfg)

    def _geometry_contract(self) -> dict[str, int]:
        return {
            "local_size": INPUT_SIZE,
            "halo": HALO,
            "core_size": CORE_SIZE,
            "context_size": self.cfg.model.context_size,
            "global_long_side": self.cfg.model.global_long_side,
        }

    def _class_mapping(self) -> dict[str, Any]:
        return {
            "num_classes": NUM_CLASSES,
            "semantic_gray_to_class": {str(k): v for k, v in SEMANTIC_GRAY_TO_CLASS.items()},
            "class_to_gray": {str(k): v for k, v in CLASS_TO_GRAY.items()},
        }

    def build_payload(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler | None,
        global_step: int,
        optimizer_step: int,
        grad_accum_pending: int = 0,
        metrics: dict[str, Any] | None = None,
        dataloader_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if grad_accum_pending > 0:
            raise RuntimeError(
                f"Refusing to save checkpoint with grad_accum_pending={grad_accum_pending}; "
                "complete or zero pending micro-batches first."
            )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "model": model_state_dict_for_checkpoint(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "micro_step": global_step,
            "grad_accum_pending": grad_accum_pending,
            "best_scores": dict(self.best_scores),
            "rng_states": get_rng_states(),
            "config": config_to_dict(self.cfg),
            "config_path": self.config_path,
            "geometry": self._geometry_contract(),
            "class_mapping": self._class_mapping(),
            "manifest_hashes": dict(self._manifest_hashes),
            "dino_checkpoint_path": str(self.cfg.model.dino.checkpoint),
            "dino_sha256": self._dino_sha256,
            "git": _git_state(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "dataloader_state": dataloader_state,
        }

    def _atomic_save(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        except OSError as exc:
            if tmp.is_file():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            free = None
            try:
                import shutil

                free = shutil.disk_usage(path.parent).free
            except OSError:
                pass
            free_msg = f" ({free / (1 << 30):.2f} GiB free on volume)" if free is not None else ""
            raise OSError(
                f"Failed to save checkpoint to {path}{free_msg}. "
                "Free disk space or point --run-dir to a larger drive."
            ) from exc
        LOGGER.info("Saved checkpoint: %s", path)

    def save_last(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler | None,
        global_step: int,
        optimizer_step: int,
        grad_accum_pending: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        payload = self.build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            optimizer_step=optimizer_step,
            grad_accum_pending=grad_accum_pending,
            metrics=metrics,
        )
        path = self.run_dir / "last.pt"
        self._atomic_save(path, payload)
        return path

    def save_best(
        self,
        kind: str,
        score: float,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler | None,
        global_step: int,
        optimizer_step: int,
        grad_accum_pending: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> Path | None:
        if kind not in _BEST_FILES:
            raise ValueError(f"Unknown best kind: {kind!r}; expected one of {sorted(_BEST_FILES)}")
        prev = self.best_scores.get(kind, float("-inf"))
        if score <= prev:
            return None
        self.best_scores[kind] = score
        payload = self.build_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            optimizer_step=optimizer_step,
            grad_accum_pending=grad_accum_pending,
            metrics=metrics,
        )
        payload["best_kind"] = kind
        payload["best_score"] = score
        path = self.run_dir / _BEST_FILES[kind]
        self._atomic_save(path, payload)
        return path

    def verify_hashes(
        self,
        state: dict[str, Any],
        *,
        allow_override: bool = False,
    ) -> None:
        if allow_override:
            return
        stored_manifest = state.get("manifest_hashes") or {}
        current = manifest_hashes_for_config(self.cfg)
        for key, expected in stored_manifest.items():
            if key not in current:
                raise ValueError(f"Manifest {key!r} no longer exists under dataset root")
            if current[key] != expected:
                raise ValueError(
                    f"Manifest hash mismatch for {key!r}: "
                    f"checkpoint={expected[:12]} current={current[key][:12]}"
                )
        stored_dino = state.get("dino_sha256")
        if stored_dino and stored_dino != self._dino_sha256:
            raise ValueError(
                f"DINO checkpoint hash mismatch: checkpoint={stored_dino[:12]} "
                f"current={self._dino_sha256[:12]}"
            )
        stored_geom = state.get("geometry") or {}
        current_geom = self._geometry_contract()
        for key, val in stored_geom.items():
            if key in current_geom and current_geom[key] != val:
                raise ValueError(f"Geometry mismatch for {key!r}: checkpoint={val} current={current_geom[key]}")

    def _verify_class_mapping(self, state: dict[str, Any]) -> None:
        stored = state.get("class_mapping") or {}
        current = self._class_mapping()
        if stored.get("num_classes") != current["num_classes"]:
            raise ValueError(
                f"num_classes mismatch: checkpoint={stored.get('num_classes')} "
                f"current={current['num_classes']}"
            )
        if stored.get("semantic_gray_to_class") != current["semantic_gray_to_class"]:
            raise ValueError("semantic_gray_to_class mismatch between checkpoint and current constants")

    def load(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: torch.amp.GradScaler | None = None,
        strict: bool = True,
        allow_hash_override: bool = False,
    ) -> dict[str, Any]:
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        schema = state.get("schema_version")
        if schema is not None and schema != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema {schema}; expected {CHECKPOINT_SCHEMA_VERSION}")

        ckpt_model = state.get("model_name")
        if ckpt_model and ckpt_model != MODEL_NAME:
            raise ValueError(f"model_name mismatch: checkpoint={ckpt_model!r} current={MODEL_NAME!r}")

        self.verify_hashes(state, allow_override=allow_hash_override)
        self._verify_class_mapping(state)

        embedded = state.get("config")
        if embedded:
            from csn_v4.config import _dict_to_config

            self.cfg = _dict_to_config(embedded)

        load_model_state_dict(model, state["model"], strict=strict)
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if state.get("rng_states"):
            set_rng_states(state["rng_states"])
        self.best_scores.update(state.get("best_scores") or {})
        LOGGER.info(
            "Loaded checkpoint %s (optimizer_step=%d)",
            path,
            int(state.get("optimizer_step", 0)),
        )
        return state
