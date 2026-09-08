"""Permutation-invariant palette training losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dinov3_carpet_probe.palette_train.src.color import srgb_to_oklab
from dinov3_carpet_probe.palette_train.src.matching import MatchResult, hungarian_match_batch


def _matched_oklab_distance(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    pred_ok = srgb_to_oklab(pred_rgb)
    target_ok = srgb_to_oklab(target_rgb)
    return torch.norm(pred_ok - target_ok, dim=-1)


def _presence_targets(num_queries: int, match: MatchResult, device: torch.device) -> torch.Tensor:
    target = torch.zeros(num_queries, dtype=torch.float32, device=device)
    if match.pred_indices.numel():
        target[match.pred_indices] = 1.0
    return target


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, torch.full_like(targets, alpha), torch.full_like(targets, 1.0 - alpha))
    focal = alpha_t * (1.0 - pt).pow(gamma) * bce
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    pos_loss = focal[pos_mask].mean() if pos_mask.any() else logits.new_tensor(0.0)
    neg_loss = focal[neg_mask].mean() if neg_mask.any() else logits.new_tensor(0.0)
    return focal.mean(), pos_loss, neg_loss


def compute_palette_losses(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    *,
    matcher_rgb_weight: float,
    matcher_perceptual_weight: float,
    matcher_presence_weight: float,
    loss_rgb_weight: float,
    loss_perceptual_weight: float,
    loss_presence_weight: float,
    loss_count_weight: float,
    loss_count_consistency_weight: float,
    aux_loss_weight: float = 0.5,
) -> tuple[dict[str, torch.Tensor], list[MatchResult]]:
    pred_rgb = outputs["pred_rgb"]
    presence_logits = outputs["presence_logits"]
    count_logits = outputs["count_logits"]
    target_rgb = batch["target_rgb"].to(pred_rgb.device)
    target_mask = batch["target_mask"].to(pred_rgb.device)
    target_counts = batch["target_counts"].to(pred_rgb.device)

    matches = hungarian_match_batch(
        pred_rgb,
        presence_logits,
        target_rgb,
        target_mask,
        rgb_weight=matcher_rgb_weight,
        perceptual_weight=matcher_perceptual_weight,
        presence_weight=matcher_presence_weight,
    )

    rgb_terms: list[torch.Tensor] = []
    perceptual_terms: list[torch.Tensor] = []
    presence_terms: list[torch.Tensor] = []
    presence_pos_terms: list[torch.Tensor] = []
    presence_neg_terms: list[torch.Tensor] = []
    for batch_idx, match in enumerate(matches):
        if match.pred_indices.numel():
            matched_pred = pred_rgb[batch_idx, match.pred_indices]
            matched_target = target_rgb[batch_idx, target_mask[batch_idx]][match.target_indices]
            rgb_terms.append(F.smooth_l1_loss(matched_pred, matched_target, reduction="mean"))
            perceptual_terms.append(_matched_oklab_distance(matched_pred, matched_target).mean())
        presence_target = _presence_targets(pred_rgb.shape[1], match, pred_rgb.device)
        presence_loss, presence_pos, presence_neg = _focal_bce_with_logits(
            presence_logits[batch_idx],
            presence_target,
        )
        presence_terms.append(presence_loss)
        presence_pos_terms.append(presence_pos)
        presence_neg_terms.append(presence_neg)

    zero = pred_rgb.new_tensor(0.0)
    loss_rgb = torch.stack(rgb_terms).mean() if rgb_terms else zero
    loss_perceptual = torch.stack(perceptual_terms).mean() if perceptual_terms else zero
    loss_presence = torch.stack(presence_terms).mean() if presence_terms else zero
    loss_presence_pos = torch.stack(presence_pos_terms).mean() if presence_pos_terms else zero
    loss_presence_neg = torch.stack(presence_neg_terms).mean() if presence_neg_terms else zero
    loss_count = F.cross_entropy(count_logits, target_counts)
    presence_sum = torch.sigmoid(presence_logits).sum(dim=1)
    loss_count_consistency = F.l1_loss(presence_sum, target_counts.to(presence_sum.dtype))

    loss_total = (
        loss_rgb_weight * loss_rgb
        + loss_perceptual_weight * loss_perceptual
        + loss_presence_weight * loss_presence
        + loss_count_weight * loss_count
        + loss_count_consistency_weight * loss_count_consistency
    )

    aux_losses: dict[str, torch.Tensor] = {}
    if outputs.get("aux_outputs"):
        for aux_idx, aux_output in enumerate(outputs["aux_outputs"]):
            aux_batch = {
                "pred_rgb": aux_output["pred_rgb"],
                "presence_logits": aux_output["presence_logits"],
                "count_logits": count_logits.detach(),
            }
            aux_loss, _ = compute_palette_losses(
                aux_batch,
                batch,
                matcher_rgb_weight=matcher_rgb_weight,
                matcher_perceptual_weight=matcher_perceptual_weight,
                matcher_presence_weight=matcher_presence_weight,
                loss_rgb_weight=loss_rgb_weight,
                loss_perceptual_weight=loss_perceptual_weight,
                loss_presence_weight=loss_presence_weight,
                loss_count_weight=0.0,
                loss_count_consistency_weight=0.0,
                aux_loss_weight=0.0,
            )
            aux_total = aux_loss["loss_total"].detach() * 0.0 + aux_loss["loss_rgb"] + aux_loss["loss_perceptual"] + aux_loss["loss_presence"]
            aux_losses[f"loss_aux_{aux_idx}"] = aux_total
            loss_total = loss_total + aux_loss_weight * aux_total

    losses = {
        "loss_total": loss_total,
        "loss_rgb": loss_rgb,
        "loss_perceptual": loss_perceptual,
        "loss_presence": loss_presence,
        "loss_presence_pos": loss_presence_pos,
        "loss_presence_neg": loss_presence_neg,
        "loss_count": loss_count,
        "loss_count_consistency": loss_count_consistency,
    }
    losses.update(aux_losses)
    return losses, matches
