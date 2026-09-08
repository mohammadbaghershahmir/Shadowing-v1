"""CSN-V4 loss module: categorical-only, full-crop supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v4.config import LossConfig
from csn_v4.constants import IGNORE_INDEX, NUM_CLASSES
from csn_v4.masks import as_bool_mask, build_categorical_boundary_mask


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = logits.float().squeeze(1) if logits.ndim == 4 and logits.shape[1] == 1 else logits.float()
        tgt = target.float()
        m = mask.float()
        bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
        p = torch.sigmoid(pred)
        pt = torch.where(tgt > 0.5, p, 1 - p)
        focal = (1 - pt) ** self.gamma * bce
        weight = torch.where(tgt > 0.5, self.pos_weight, 1.0)
        w = m * weight
        if pixel_weights is not None:
            w = w * pixel_weights.float()
        denom = w.sum().clamp(min=1)
        return (focal * w).sum() / denom


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        w = mask.float().unsqueeze(1)
        if pixel_weights is not None:
            w = w * pixel_weights.float().unsqueeze(1)
        labels_safe = labels.clone()
        labels_safe[labels_safe < 0] = 0
        oh = F.one_hot(labels_safe, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (w * probs * oh).sum(dim=(0, 2, 3))
        card = (w * (probs + oh)).sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        present = (w * oh).sum(dim=(0, 2, 3)) > 0
        if present.any():
            return (1.0 - dice[present].mean()).clamp(min=0.0)
        return logits.new_zeros(())


class BoundaryDiceLoss(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pixel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits.float()).squeeze(1)
        tgt = target.float()
        m = mask.float()
        if pixel_weights is not None:
            m = m * pixel_weights.float()
        if m.sum() <= 0:
            return logits.new_zeros(())
        inter = (pred * tgt * m).sum()
        card = ((pred + tgt) * m).sum()
        loss = 1.0 - (2 * inter + 1) / (card + 1)
        return loss.clamp(min=0.0, max=1.0)


def _pixel_boost_weights(boundary_boost: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 1.0:
        return torch.ones_like(boundary_boost, dtype=torch.float32)
    return 1.0 + (factor - 1.0) * boundary_boost.float()


def _combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    out = masks[0]
    for m in masks[1:]:
        out = out & m
    return out


class CSNV4Loss(nn.Module):
    def __init__(self, cfg: LossConfig, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.cfg = cfg
        self.ignore_index = ignore_index
        self.focal_bce = FocalBCELoss()
        self.final_dice = SoftDiceLoss(NUM_CLASSES)
        self.boundary_dice = BoundaryDiceLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        valid = as_bool_mask(targets["valid_mask"])
        black_lock = as_bool_mask(targets["black_lock"])
        shade_mask = as_bool_mask(targets["shade_mask"])

        sup_w = targets["supervision_weights"].float()
        if sup_w.ndim == 2:
            sup_w = sup_w.unsqueeze(0).expand(valid.shape[0], -1, -1)

        active = sup_w > 0
        sem_mask = _combine_masks(valid, ~black_lock, active)
        level_mask = _combine_masks(shade_mask, valid, active)
        trans_mask = _combine_masks(valid, ~black_lock, active)

        boundary_boost = build_categorical_boundary_mask(
            targets["target_class"],
            targets["transition_mask"],
            targets["valid_mask"],
            targets["black_lock"],
            dilate=cfg.boundary_boost.transition_dilate,
        )
        boost_w = _pixel_boost_weights(boundary_boost, cfg.boundary_boost.transition_boost)
        combined_w = sup_w * boost_w if cfg.boundary_boost.enabled else sup_w

        losses: dict[str, torch.Tensor] = {}
        logits = outputs["refined_logits"].float()
        labels = targets["target_class"]
        labels_masked = labels.clone()
        labels_masked[~sem_mask] = self.ignore_index

        if sem_mask.any():
            ce_map = F.cross_entropy(logits, labels_masked, ignore_index=self.ignore_index, reduction="none")
            w_ce = sem_mask.float() * sup_w
            if cfg.boundary_boost.enabled:
                w_ce = w_ce * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.level_ce_boost)
            losses["final_ce"] = (ce_map * w_ce).sum() / w_ce.sum().clamp(min=1)
            losses["final_dice"] = self.final_dice(
                logits, labels, sem_mask,
                combined_w if cfg.boundary_boost.enabled else sup_w,
            )
        else:
            losses["final_ce"] = logits.new_zeros(())
            losses["final_dice"] = logits.new_zeros(())

        where_w = sup_w * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.where_boost) if cfg.boundary_boost.enabled else sup_w
        losses["where"] = self.focal_bce(
            outputs["where_logits"], targets["where_target"],
            _combine_masks(valid, ~black_lock, active), where_w,
        )

        level_logits = outputs["level_logits"]
        level_tgt = targets["level_target"]
        lt = level_tgt.clone()
        lt[~level_mask] = self.ignore_index
        if level_mask.any():
            ce_lvl = F.cross_entropy(level_logits, lt, ignore_index=self.ignore_index, reduction="none")
            w_lvl = level_mask.float() * sup_w
            if cfg.boundary_boost.enabled:
                w_lvl = w_lvl * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.level_ce_boost)
            losses["level_ce"] = (ce_lvl * w_lvl).sum() / w_lvl.sum().clamp(min=1)
        else:
            losses["level_ce"] = logits.new_zeros(())

        trans_tgt = as_bool_mask(targets["transition_mask"]).float()
        trans_w = sup_w * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.transition_boost) if cfg.boundary_boost.enabled else sup_w
        losses["transition"] = self.focal_bce(
            outputs["transition_logits"], trans_tgt,
            trans_mask, trans_w,
        )
        losses["transition_boundary_dice"] = self.boundary_dice(
            outputs["transition_logits"], trans_tgt, trans_mask, trans_w,
        )

        aff_logits = outputs["affinity_logits"].float()
        aff_tgt = targets["affinity_targets"].float()
        aff_valid = targets["affinity_valid"] > 0.5
        safe_tgt = torch.where(aff_valid, aff_tgt, torch.zeros_like(aff_tgt))
        aff_boost = _pixel_boost_weights(boundary_boost, cfg.boundary_boost.affinity_boost) if cfg.boundary_boost.enabled else torch.ones_like(sup_w)
        bce_map = F.binary_cross_entropy_with_logits(aff_logits, safe_tgt, reduction="none")
        m = aff_valid & active.unsqueeze(1)
        w = m.float() * sup_w.unsqueeze(1) * aff_boost.unsqueeze(1)
        if m.any():
            losses["affinity"] = (bce_map * w).sum() / w.sum().clamp(min=1)
        else:
            losses["affinity"] = logits.new_zeros(())

        base = outputs["base_logits"]
        refined = outputs["refined_logits"]
        # Keep refined close to base on shade pixels. Use logit-space MSE with
        # detached base so residual heads stay active and the term is not
        # numerically crushed to ~0 by softmax (as happened with 0/1-mask bugs).
        m_cons = _combine_masks(shade_mask, valid, ~black_lock, active)
        if m_cons.any():
            diff = (refined.float() - base.float().detach()) ** 2
            w = (m_cons.float() * sup_w).unsqueeze(1)
            losses["base_refined_consistency"] = (diff * w).sum() / w.sum().clamp(min=1)
        else:
            losses["base_refined_consistency"] = logits.new_zeros(())

        total = (
            cfg.final_ce * losses["final_ce"]
            + cfg.final_dice * losses["final_dice"]
            + cfg.where * losses["where"]
            + cfg.level_ce * losses["level_ce"]
            + cfg.transition * losses["transition"]
            + cfg.transition_boundary_dice * losses["transition_boundary_dice"]
            + cfg.affinity * losses["affinity"]
            + cfg.base_refined_consistency * losses["base_refined_consistency"]
        )
        losses["total"] = total
        for k, v in losses.items():
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Non-finite loss component: {k}={v.item()}")
        return losses
