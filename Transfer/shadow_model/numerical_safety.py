"""Fail-fast numerical safety utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def assert_finite_losses(losses: dict[str, torch.Tensor]) -> None:
    for key, value in losses.items():
        if torch.is_tensor(value):
            assert_finite_tensor(value, f"loss.{key}")


class NumericalGuard:
    def __init__(self, dump_dir: Path | None = None, enabled: bool = True):
        self.dump_dir = dump_dir
        self.enabled = enabled

    def check_mask(self, mask: torch.Tensor, name: str = "M") -> None:
        if not self.enabled:
            return
        if mask.sum().item() <= 0:
            raise ValueError(f"{name}.sum() == 0: no supervised pixels")

    def check_losses(self, losses: dict[str, torch.Tensor]) -> None:
        if not self.enabled:
            return
        assert_finite_losses(losses)

    def check_grads(self, model: nn.Module | list[torch.nn.Parameter]) -> None:
        if not self.enabled:
            return
        params = model.parameters() if isinstance(model, nn.Module) else model
        for p in params:
            if p.grad is not None:
                assert_finite_tensor(p.grad, f"grad[{tuple(p.shape)}]")

    def check_params(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert_finite_tensor(p.data, f"param.{n}")


def assert_finite_tensor(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum().item())
        raise FloatingPointError(f"Non-finite values in {name}: count={bad}")


def save_nan_dump(
    path: str | Path,
    global_step: int,
    batch_meta: dict,
    losses: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    scaler: Any,
    optimizer: torch.optim.Optimizer,
) -> Path:
    dump_dir = Path(path)
    if dump_dir.suffix:
        base = dump_dir.with_suffix("")
        dump_dir = base.parent
        stem = base.name
    else:
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = f"nan_dump_step_{global_step}"

    dump_dir.mkdir(parents=True, exist_ok=True)
    json_path = dump_dir / f"{stem}.json"
    pt_path = dump_dir / f"{stem}.pt"
    payload = {
        "global_step": global_step,
        "batch_meta": batch_meta,
        "losses": {k: (float(v.item()) if torch.isfinite(v) else "non-finite") for k, v in losses.items()},
        "logits_stats": {
            k: {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean())}
            for k, t in outputs.items() if isinstance(t, torch.Tensor)
        },
        "amp_scale": float(scaler.get_scale()) if scaler is not None and hasattr(scaler, "get_scale") else None,
        "lrs": [g.get("lr") for g in optimizer.param_groups],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "losses": {k: v.detach().cpu() for k, v in losses.items() if torch.is_tensor(v)},
            "outputs": {k: v.detach().cpu() for k, v in outputs.items() if torch.is_tensor(v)},
        },
        pt_path,
    )
    return json_path
