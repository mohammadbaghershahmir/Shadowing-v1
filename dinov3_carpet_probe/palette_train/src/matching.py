"""Hungarian matching for unordered palette colors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from dinov3_carpet_probe.palette_train.src.color import srgb_to_oklab


@dataclass
class MatchResult:
    pred_indices: torch.Tensor
    target_indices: torch.Tensor
    unmatched_pred_indices: torch.Tensor
    cost_matrix: torch.Tensor


def build_cost_matrix(
    pred_rgb: torch.Tensor,
    presence_logits: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    rgb_weight: float,
    perceptual_weight: float,
    presence_weight: float,
) -> torch.Tensor:
    if target_rgb.numel() == 0:
        return pred_rgb.new_zeros(pred_rgb.shape[0], 0)
    rgb_l1 = torch.cdist(pred_rgb, target_rgb, p=1) / 3.0
    pred_ok = srgb_to_oklab(pred_rgb)
    target_ok = srgb_to_oklab(target_rgb)
    perceptual = torch.cdist(pred_ok, target_ok, p=2)
    presence = torch.nn.functional.logsigmoid(presence_logits).unsqueeze(1).expand(-1, target_rgb.shape[0])
    return rgb_weight * rgb_l1 + perceptual_weight * perceptual - presence_weight * presence


def hungarian_match_single(
    pred_rgb: torch.Tensor,
    presence_logits: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    rgb_weight: float,
    perceptual_weight: float,
    presence_weight: float,
) -> MatchResult:
    cost_matrix = build_cost_matrix(
        pred_rgb,
        presence_logits,
        target_rgb,
        rgb_weight=rgb_weight,
        perceptual_weight=perceptual_weight,
        presence_weight=presence_weight,
    )
    if target_rgb.shape[0] == 0:
        pred_indices = torch.empty(0, dtype=torch.int64, device=pred_rgb.device)
        target_indices = torch.empty(0, dtype=torch.int64, device=pred_rgb.device)
        unmatched = torch.arange(pred_rgb.shape[0], dtype=torch.int64, device=pred_rgb.device)
        return MatchResult(pred_indices, target_indices, unmatched, cost_matrix)

    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    pred_indices = torch.as_tensor(row_ind, dtype=torch.int64, device=pred_rgb.device)
    target_indices = torch.as_tensor(col_ind, dtype=torch.int64, device=pred_rgb.device)
    matched_mask = torch.zeros(pred_rgb.shape[0], dtype=torch.bool, device=pred_rgb.device)
    matched_mask[pred_indices] = True
    unmatched = torch.arange(pred_rgb.shape[0], dtype=torch.int64, device=pred_rgb.device)[~matched_mask]
    return MatchResult(pred_indices, target_indices, unmatched, cost_matrix)


def hungarian_match_batch(
    pred_rgb: torch.Tensor,
    presence_logits: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    rgb_weight: float,
    perceptual_weight: float,
    presence_weight: float,
) -> list[MatchResult]:
    matches: list[MatchResult] = []
    for batch_idx in range(pred_rgb.shape[0]):
        valid_targets = target_rgb[batch_idx][target_mask[batch_idx]]
        matches.append(
            hungarian_match_single(
                pred_rgb[batch_idx],
                presence_logits[batch_idx],
                valid_targets,
                rgb_weight=rgb_weight,
                perceptual_weight=perceptual_weight,
                presence_weight=presence_weight,
            )
        )
    return matches
