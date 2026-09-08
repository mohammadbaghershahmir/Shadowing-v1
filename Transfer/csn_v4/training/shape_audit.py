"""Shape audit for CSN-V4 with teacher_forcing=0."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from csn_v4.model import CarpetShadeNetV4


REQUIRED_GRAD_PREFIXES = (
    "stem.", "backbone.", "fusion.", "decoder.", "where_head.", "transition_head.",
    "affinity_head.", "level_head.", "joint_head.", "refiner.", "mask_encoder.",
    "transition_encoder.", "dino.layer_mixer",
)


def _batch_sample(sample: dict, batch_size: int = 1) -> dict:
    out = dict(sample)
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            if val.ndim >= 1 and batch_size > 1:
                out[key] = val.unsqueeze(0).expand(batch_size, *val.shape)
            elif val.ndim == 1 and key == "crop_coords_norm":
                out[key] = val.unsqueeze(0)
            elif val.ndim == 3:
                out[key] = val.unsqueeze(0)
            elif val.ndim == 2:
                out[key] = val.unsqueeze(0)
    return out


class ShapeAuditor:
    def audit_forward(self, model: CarpetShadeNetV4, sample: dict, device: torch.device, batch_size: int = 1) -> dict[str, Any]:
        batch = _batch_sample(sample, batch_size)
        lb = batch["local_bw"].to(device)
        cb = batch["context_bw"].to(device)
        fb = batch["full_bw"].to(device)
        cc = batch["crop_coords_norm"].to(device)
        model.eval()
        with torch.no_grad():
            out = model(lb, cb, fb, cc, letterbox_meta=batch.get("letterbox_meta"), teacher_forcing=0.0)
        shapes = {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
        return shapes

    def audit_backward(self, model: CarpetShadeNetV4, sample: dict, device: torch.device, batch_size: int = 1) -> dict[str, Any]:
        batch = _batch_sample(sample, batch_size)
        lb = batch["local_bw"].to(device)
        cb = batch["context_bw"].to(device)
        fb = batch["full_bw"].to(device)
        cc = batch["crop_coords_norm"].to(device)
        model.train()
        out = model(lb, cb, fb, cc, letterbox_meta=batch.get("letterbox_meta"), teacher_forcing=0.0)
        loss = out["refined_logits"].sum()
        for head in ("where_logits", "transition_logits", "level_logits", "affinity_logits"):
            if head in out:
                loss = loss + out[head].sum()
        loss.backward()
        no_grad = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None or param.grad.abs().sum() == 0:
                if any(name.startswith(p) for p in REQUIRED_GRAD_PREFIXES):
                    no_grad.append(name)
        return {"params_without_grad": no_grad, "num_no_grad": len(no_grad)}

    def save(self, path: Path, fwd: dict, grad: dict) -> None:
        report = {"forward_shapes": fwd, "gradient_report": grad, "passed": grad.get("num_no_grad", 0) == 0}
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
