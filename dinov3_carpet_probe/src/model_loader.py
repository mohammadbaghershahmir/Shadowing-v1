"""Load official DINOv3 backbones (Torch Hub primary, Transformers fallback)."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dinov3_carpet_probe.src.io_utils import (
    DEFAULT_WEIGHTS_DIR,
    REPO_ROOT,
    ensure_dir,
    load_models_config,
    write_json,
)


META_ACCESS_URL = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"


@dataclass
class LoadedModel:
    key: str
    model: nn.Module
    config: dict[str, Any]
    source: str
    checkpoint_path: str | None
    param_count: int
    device: str


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def unload_model(model: nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def find_local_checkpoint(model_cfg: dict[str, Any], weights_dir: Path) -> Path | None:
    hash_suffix = model_cfg["checkpoint_hash"]
    patterns = [
        f"*{hash_suffix}*.pth",
        f"*{model_cfg['key']}*.pth",
        f"*{model_cfg['hub_entry']}*{model_cfg['weights_enum'].lower()}*.pth",
    ]
    if weights_dir.exists():
        for pat in patterns:
            matches = list(weights_dir.glob(pat))
            if matches:
                return matches[0]
    # Torch hub checkpoint cache
    hub_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    if hub_ckpt.exists():
        for pat in [f"*{hash_suffix}*.pth", f"*dinov3*{hash_suffix}*"]:
            matches = list(hub_ckpt.glob(pat))
            if matches:
                return matches[0]
    return None


def find_hf_cache_snapshot(hf_id: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    dirname = "models--" + hf_id.replace("/", "--")
    snap_root = cache_root / dirname / "snapshots"
    if not snap_root.exists():
        return None
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps:
        return None
    # Prefer snapshot containing model weights
    for snap in sorted(snaps, key=lambda p: p.stat().st_mtime, reverse=True):
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")) or any(snap.glob("*.pth")):
            return snap
    return snaps[0]


def list_available_models(weights_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return models from models.yaml that have a local .pth (or HF cache) available."""
    models_yaml = load_models_config()
    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    available: list[dict[str, Any]] = []
    for key, cfg in models_yaml["models"].items():
        local = find_local_checkpoint(cfg, weights_dir)
        hf_snap = find_hf_cache_snapshot(cfg["hf_id"])
        if local is None and hf_snap is None:
            continue
        available.append(
            {
                "key": key,
                "display_name": cfg.get("display_name", key),
                "architecture": cfg.get("architecture"),
                "approx_params_m": cfg.get("approx_params_m"),
                "checkpoint_hash": cfg.get("checkpoint_hash"),
                "local_pth": str(local) if local else None,
                "hf_cache": str(hf_snap) if hf_snap else None,
                "hub_entry": cfg.get("hub_entry"),
                "weights_enum": cfg.get("weights_enum"),
            }
        )
    return available


def validate_weights(
    weights_dir: Path | None = None,
    output_json: Path | None = None,
    attempt_load: bool = False,
    repo_dir: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    models_yaml = load_models_config()
    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    ensure_dir(weights_dir)
    status: dict[str, Any] = {
        "weights_dir": str(weights_dir),
        "meta_access_url": META_ACCESS_URL,
        "models": {},
        "all_found": True,
        "access_procedure": (
            "1) Request access at "
            f"{META_ACCESS_URL} and wait for email with weight URLs. "
            "2) Download with wget/curl (not a browser) into dinov3_carpet_probe/weights/. "
            "3) Or request access on each Hugging Face model page and run `huggingface-cli login`. "
            "Do NOT substitute DINOv2 or other unofficial checkpoints."
        ),
    }
    for key, cfg in models_yaml["models"].items():
        local = find_local_checkpoint(cfg, weights_dir)
        hf_snap = find_hf_cache_snapshot(cfg["hf_id"])
        entry = {
            "key": key,
            "hf_id": cfg["hf_id"],
            "checkpoint_hash": cfg["checkpoint_hash"],
            "local_pth": str(local) if local else None,
            "hf_cache": str(hf_snap) if hf_snap else None,
            "status": "FOUND" if (local or hf_snap) else "MISSING",
        }
        if entry["status"] == "MISSING":
            status["all_found"] = False
        if attempt_load and entry["status"] == "FOUND":
            try:
                loaded = load_model(
                    key,
                    device=device,
                    repo_dir=repo_dir,
                    weights_dir=weights_dir,
                    prefer="torch_hub" if local else "transformers",
                )
                entry["loaded"] = True
                entry["param_count"] = loaded.param_count
                entry["source"] = loaded.source
                unload_model(loaded.model)
            except Exception as exc:  # noqa: BLE001
                entry["loaded"] = False
                entry["load_error"] = str(exc)
                status["all_found"] = False
        status["models"][key] = entry

    if output_json:
        write_json(output_json, status)
    return status


def _weights_enum(name: str):
    from dinov3.hub.backbones import Weights

    return Weights[name]


def load_model_torch_hub(
    model_key: str,
    *,
    repo_dir: Path | None = None,
    weights_dir: Path | None = None,
    device: str = "cuda",
) -> LoadedModel:
    models_yaml = load_models_config()
    if model_key not in models_yaml["models"]:
        raise KeyError(f"Unknown model key: {model_key}")
    cfg = models_yaml["models"][model_key]
    repo_dir = Path(repo_dir) if repo_dir else REPO_ROOT
    weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR

    local = find_local_checkpoint(cfg, weights_dir)
    weights_arg: Any
    if local is not None:
        weights_arg = str(local)
    else:
        # Try official enum (requires network access URL privileges)
        weights_arg = _weights_enum(cfg["weights_enum"])

    # Official notebook pattern: torch.hub.load(...) then model.cuda()
    model = torch.hub.load(
        str(repo_dir),
        cfg["hub_entry"],
        source="local",
        weights=weights_arg,
        trust_repo=True,
    )
    model = freeze_model(model)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        model = model.cuda()
    else:
        model = model.to(device)
    print(
        f"[gpu] loaded {model_key} on {next(model.parameters()).device} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})",
        flush=True,
    )
    return LoadedModel(
        key=model_key,
        model=model,
        config=cfg,
        source="torch_hub",
        checkpoint_path=str(local) if local else f"Weights.{cfg['weights_enum']}",
        param_count=_count_params(model),
        device=device,
    )


def load_model_transformers(
    model_key: str,
    *,
    device: str = "cuda",
) -> LoadedModel:
    from transformers import AutoModel

    models_yaml = load_models_config()
    cfg = models_yaml["models"][model_key]
    hf_id = cfg["hf_id"]
    model = AutoModel.from_pretrained(hf_id)
    model = freeze_model(model)
    model = model.to(device)
    return LoadedModel(
        key=model_key,
        model=model,
        config=cfg,
        source="transformers",
        checkpoint_path=hf_id,
        param_count=_count_params(model),
        device=device,
    )


def load_model(
    model_key: str,
    *,
    device: str = "cuda",
    repo_dir: Path | None = None,
    weights_dir: Path | None = None,
    prefer: str = "torch_hub",
) -> LoadedModel:
    """
    Primary: local Torch Hub. Fallback: Hugging Face Transformers.
    Never silently substitutes another architecture.
    """
    errors: list[str] = []
    order = [prefer, "transformers" if prefer == "torch_hub" else "torch_hub"]
    # de-dup
    seen = []
    for o in order:
        if o not in seen:
            seen.append(o)
    for source in seen:
        try:
            if source == "torch_hub":
                return load_model_torch_hub(
                    model_key, repo_dir=repo_dir, weights_dir=weights_dir, device=device
                )
            return load_model_transformers(model_key, device=device)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")
    raise RuntimeError(
        f"Failed to load official model '{model_key}'. Tried {seen}. Errors: {errors}. "
        f"Request access at {META_ACCESS_URL} and/or HF gated model pages. "
        "Do not substitute another model."
    )


def peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
