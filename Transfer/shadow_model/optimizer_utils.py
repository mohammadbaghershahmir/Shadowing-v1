"""Optimizer parameter coverage audit."""
from __future__ import annotations

import torch
import torch.nn as nn


def collect_optimizer_params(optimizer: torch.optim.Optimizer) -> set[int]:
    ids: set[int] = set()
    for grp in optimizer.param_groups:
        for p in grp["params"]:
            ids.add(id(p))
    return ids


def audit_optimizer_coverage(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    trainable = {
        id(p): n for n, p in model.named_parameters()
        if p.requires_grad and "dino_encoder" not in n and "dino.backbone" not in n
    }
    opt_ids = collect_optimizer_params(optimizer)
    uncovered = [n for pid, n in trainable.items() if pid not in opt_ids]
    duplicates = []
    seen: set[int] = set()
    for grp in optimizer.param_groups:
        for p in grp["params"]:
            if id(p) in seen:
                duplicates.append(id(p))
            seen.add(id(p))
    frozen_in_opt = [n for n, p in model.named_parameters() if id(p) in opt_ids and not p.requires_grad]
    if uncovered or duplicates or frozen_in_opt:
        msg = []
        if uncovered:
            msg.append(f"uncovered trainable: {uncovered[:20]}")
        if duplicates:
            msg.append(f"duplicate param ids: {len(duplicates)}")
        if frozen_in_opt:
            msg.append(f"frozen in optimizer: {frozen_in_opt[:10]}")
        raise ValueError("Optimizer coverage audit failed: " + "; ".join(msg))
