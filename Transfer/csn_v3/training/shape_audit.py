"""Per-block forward I/O and gradient connectivity audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

REQUIRED_GRAD_PREFIXES = (
    "stem.", "backbone.", "fusion.", "decoder.", "where_head.", "transition_head.",
    "affinity_head.", "level_head.", "joint_head.", "refiner.", "mask_encoder.",
    "transition_encoder.", "dino.layer_mixer",
)


def _batch_sample(sample: dict) -> dict:
    """Add batch dimension to single-sample dataset dict."""
    out = dict(sample)
    if out["local_bw"].ndim == 3:
        out["local_bw"] = out["local_bw"].unsqueeze(0)
    if out.get("context_bw") is not None and out["context_bw"].ndim == 3:
        out["context_bw"] = out["context_bw"].unsqueeze(0)
    if out.get("full_bw") is not None and out["full_bw"].ndim == 3:
        out["full_bw"] = out["full_bw"].unsqueeze(0)
    if out.get("crop_coords_norm") is not None and out["crop_coords_norm"].ndim == 1:
        out["crop_coords_norm"] = out["crop_coords_norm"].unsqueeze(0)
    for key in ("shade_mask", "transition_mask", "target_class", "valid_mask", "black_lock",
                "where_target", "level_target"):
        if key in out and isinstance(out[key], torch.Tensor) and out[key].ndim == 2:
            out[key] = out[key].unsqueeze(0)
    if "ordinal_target" in out and out["ordinal_target"].ndim == 3:
        out["ordinal_target"] = out["ordinal_target"].unsqueeze(0)
    return out


def _to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor fields in a batch dict to device."""
    out = dict(batch)
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
    return out


class ShapeAuditor:
    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def audit_forward(self, model, sample: dict, device: torch.device | None = None) -> dict[str, Any]:
        model.eval()
        if device is not None:
            model = model.to(device)
            batch = _to_device(_batch_sample(sample), device)
        else:
            batch = _batch_sample(sample)
        with torch.no_grad():
            out = model(
                batch["local_bw"],
                context_bw=batch.get("context_bw"),
                full_bw=batch.get("full_bw"),
                crop_coords=batch.get("crop_coords_norm"),
                teacher_forcing=1.0,
                gt_shade_mask=batch.get("shade_mask"),
                gt_transition_mask=batch.get("transition_mask"),
            )
        shapes = {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
        return shapes

    def audit_backward(self, model, sample: dict, device: torch.device) -> dict[str, Any]:
        model.train()
        model = model.to(device)
        batch = _to_device(_batch_sample(sample), device)
        lb = batch["local_bw"]
        cb = batch.get("context_bw")
        fb = batch.get("full_bw")
        cc = batch.get("crop_coords_norm")

        out = model(
            lb, context_bw=cb, full_bw=fb, crop_coords=cc, teacher_forcing=1.0,
            gt_shade_mask=batch.get("shade_mask"), gt_transition_mask=batch.get("transition_mask"),
        )
        out["refined_logits"].sum().backward()

        grad_report = {}
        dino_grad = 0.0
        trainable_grad = 0.0
        zero_grad_modules = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            gnorm = param.grad.abs().sum().item()
            if name.startswith("dino.backbone"):
                dino_grad += gnorm
            elif gnorm > 0:
                trainable_grad += gnorm
            elif any(name.startswith(p) for p in REQUIRED_GRAD_PREFIXES):
                zero_grad_modules.append(name)

        grad_report["dino_backbone_grad_sum"] = dino_grad
        grad_report["trainable_grad_sum"] = trainable_grad
        grad_report["zero_grad_trainable"] = zero_grad_modules[:20]
        return grad_report

    def save(self, path: Path, forward_shapes: dict, grad_report: dict) -> None:
        report = {"forward_shapes": forward_shapes, "gradients": grad_report}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        txt = path.with_suffix(".txt")
        with txt.open("w", encoding="utf-8") as fh:
            fh.write("=== Forward Shapes ===\n")
            for k, v in forward_shapes.items():
                fh.write(f"{k}: {v}\n")
            fh.write("\n=== Gradients ===\n")
            for k, v in grad_report.items():
                fh.write(f"{k}: {v}\n")
