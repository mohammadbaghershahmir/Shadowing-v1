"""Shape audit for CSN-V4 — never leaks trial gradients into training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from csn_v4.model import CarpetShadeNetV4


# At init, ResidualFusionBlock is zero-initialized so many side-branches
# (DINO adapters, PPM, lateral stem paths via residual) correctly receive
# ~0 gradient on a single sample. Audit therefore checks *module groups*
# that must stay on the live skip/head path, not every parameter.
CRITICAL_GROUP_PREFIXES_BW = (
    "stem.stem_in",
    "fusion.conv_proj",
    "decoder.f16_in",
    "where_head.",
    "transition_head.",
    "affinity_head.",
    "level_head.",
    "joint_head.",
    "refiner.",
)

CRITICAL_GROUP_PREFIXES_INDEXED = (
    "stem.stem_in",
    "fusion.conv_proj",
    "decoder.f16_in",
    "indexed_head.",
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


def _forward_kwargs(batch: dict, device: torch.device) -> dict:
    """Build model kwargs from a sample; support bw and indexed_guided keys."""
    kwargs: dict[str, Any] = {
        "crop_coords": batch["crop_coords_norm"].to(device),
        "letterbox_meta": batch.get("letterbox_meta"),
        "teacher_forcing": 0.0,
    }
    if "local_index" in batch:
        kwargs["local_index"] = batch["local_index"].to(device)
        kwargs["context_index"] = batch.get("context_index")
        if kwargs["context_index"] is not None:
            kwargs["context_index"] = kwargs["context_index"].to(device)
        kwargs["full_index"] = batch.get("full_index")
        if kwargs["full_index"] is not None:
            kwargs["full_index"] = kwargs["full_index"].to(device)
        if "local_bw" in batch:
            kwargs["local_bw"] = batch["local_bw"].to(device)
        else:
            kwargs["local_bw"] = (batch["local_index"].float() > 0).to(device)
            if kwargs["local_bw"].ndim == 2:
                kwargs["local_bw"] = kwargs["local_bw"].unsqueeze(0)
        if "context_bw" in batch:
            kwargs["context_bw"] = batch["context_bw"].to(device)
        if "full_bw" in batch:
            kwargs["full_bw"] = batch["full_bw"].to(device)
    else:
        kwargs["local_bw"] = batch["local_bw"].to(device)
        kwargs["context_bw"] = batch["context_bw"].to(device)
        kwargs["full_bw"] = batch["full_bw"].to(device)
    return kwargs


def _group_has_grad(model: torch.nn.Module, prefix: str) -> tuple[bool, int, int]:
    matched = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
    if not matched:
        return False, 0, 0
    with_grad = sum(
        1
        for _, p in matched
        if p.grad is not None and float(p.grad.abs().sum()) > 0.0
    )
    return with_grad > 0, with_grad, len(matched)


class ShapeAuditor:
    def audit_forward(
        self,
        model: CarpetShadeNetV4,
        sample: dict,
        device: torch.device,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        batch = _batch_sample(sample, batch_size)
        kwargs = _forward_kwargs(batch, device)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model(**kwargs)
        finally:
            model.train(was_training)
        return {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}

    def audit_backward(
        self,
        model: CarpetShadeNetV4,
        sample: dict,
        device: torch.device,
        batch_size: int = 1,
        criterion: torch.nn.Module | None = None,
    ) -> dict[str, Any]:
        """Run a trial backward for connectivity checks.

        Always restores model.train/eval mode and clears all parameter grads,
        including on failure, so training never inherits audit gradients.
        """
        batch = _batch_sample(sample, batch_size)
        kwargs = _forward_kwargs(batch, device)
        was_training = model.training
        model.zero_grad(set_to_none=True)
        model.train()
        try:
            out = model(**kwargs)
            if criterion is not None and "target_class" in sample:
                targets = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                    if k
                    in (
                        "target_class",
                        "shade_mask",
                        "transition_mask",
                        "black_lock",
                        "valid_mask",
                        "where_target",
                        "level_target",
                        "affinity_targets",
                        "affinity_valid",
                        "supervision_weights",
                        "allowed_mask",
                        "internal_target",
                    )
                }
                for k, v in list(targets.items()):
                    if isinstance(v, torch.Tensor) and v.ndim == 2:
                        targets[k] = v.unsqueeze(0)
                losses = criterion(out, targets)
                loss = losses["total"]
                loss_parts = {k: float(v.detach()) for k, v in losses.items()}
            else:
                key = "indexed_logits" if "indexed_logits" in out else "refined_logits"
                loss = out[key].float().sum()
                for head in ("where_logits", "transition_logits", "level_logits", "affinity_logits"):
                    if head in out:
                        loss = loss + out[head].float().sum()
                loss_parts = {"probe": float(loss.detach())}

            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite audit loss: {loss}")
            if float(loss.detach()) == 0.0:
                raise RuntimeError("Audit loss is exactly 0; cannot verify gradients.")

            loss.backward()

            input_mode = getattr(model.cfg.data, "input_mode", "bw")
            groups = (
                CRITICAL_GROUP_PREFIXES_INDEXED
                if input_mode == "indexed_guided"
                else CRITICAL_GROUP_PREFIXES_BW
            )

            missing_groups: list[str] = []
            group_report: dict[str, dict[str, int | bool]] = {}
            for prefix in groups:
                ok, n_grad, n_total = _group_has_grad(model, prefix)
                group_report[prefix] = {
                    "connected": ok,
                    "with_grad": n_grad,
                    "trainable": n_total,
                }
                if n_total > 0 and not ok:
                    missing_groups.append(prefix)

            # Informational: params with requires_grad but zero grad on this sample
            # (expected for zero-init residual side-branches).
            zero_grad_params = [
                name
                for name, param in model.named_parameters()
                if param.requires_grad
                and (param.grad is None or float(param.grad.abs().sum()) == 0.0)
            ]

            # Optimizer coverage: every requires_grad param should be owned by optimizer
            # if one was attached via a side channel — checked separately in train script.

            passed = len(missing_groups) == 0
            return {
                "passed": passed,
                "missing_groups": missing_groups,
                "group_report": group_report,
                "params_without_grad": zero_grad_params,
                "num_no_grad": len(zero_grad_params),
                "loss_parts": loss_parts,
                "note": (
                    "Individual zero-grad params are expected at init when "
                    "fusion/decoder ResidualFusionBlock is zero-initialized; "
                    "failure means a critical live path group has no gradient."
                ),
            }
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)

    def save(self, path: Path, fwd: dict, grad: dict) -> None:
        report = {
            "forward_shapes": fwd,
            "gradient_report": grad,
            "passed": bool(grad.get("passed", False)),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
