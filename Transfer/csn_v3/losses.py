"""CSN-V3 loss module with fixed masking and per-pixel boundary boost."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from csn_v3.config import LossConfig
from csn_v3.constants import IGNORE_INDEX, NUM_CLASSES
from csn_v3.masks import build_boundary_boost_mask, build_center_supervision_mask


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
        return (focal * w).sum() / w.sum().clamp(min=1)


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
        m = mask.float().unsqueeze(1)
        if pixel_weights is not None:
            m = m * pixel_weights.float().unsqueeze(1)
        probs = probs * m
        labels_safe = labels.clone()
        labels_safe[labels_safe < 0] = 0
        oh = F.one_hot(labels_safe, self.num_classes).permute(0, 3, 1, 2).float() * m
        inter = (probs * oh).sum(dim=(0, 2, 3))
        card = (probs + oh).sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        present = oh.sum(dim=(0, 2, 3)) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return logits.new_zeros(())


class BoundaryDiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits.float()).squeeze(1)
        tgt = target.float()
        m = mask.float()
        inter = (pred * tgt * m).sum()
        card = ((pred + tgt) * m).sum()
        return 1.0 - (2 * inter + 1) / (card + 1)


class SoftOrdinalEMDLoss(nn.Module):
    """Differentiable expected |pred_level - gt_level| using softmax."""

    def forward(self, level_logits: torch.Tensor, level_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(level_logits.float(), dim=1)
        levels = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
        pred_level = (probs * levels.view(1, -1, 1, 1)).sum(dim=1)
        diff = (pred_level - level_target.clamp(min=0).float()).abs()
        m = mask.float()
        return (diff * m).sum() / m.sum().clamp(min=1)


def _pixel_boost_weights(boundary_boost: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 1.0:
        return torch.ones_like(boundary_boost, dtype=torch.float32)
    bb = boundary_boost.float()
    return 1.0 + (factor - 1.0) * bb


class CSNV3Loss(nn.Module):
    def __init__(self, cfg: LossConfig, ignore_index: int = IGNORE_INDEX, center_halo: int = 64):
        super().__init__()
        self.cfg = cfg
        self.ignore_index = ignore_index
        self.center_halo = center_halo
        self.focal_bce = FocalBCELoss()
        self.final_dice = SoftDiceLoss(NUM_CLASSES)
        self.where_dice = SoftDiceLoss(1)
        self.boundary_dice = BoundaryDiceLoss()
        self.ordinal_emd = SoftOrdinalEMDLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        valid = targets["valid_mask"] if targets["valid_mask"].dtype == torch.bool else targets["valid_mask"] > 127
        black_lock = targets["black_lock"] if targets["black_lock"].dtype == torch.bool else targets["black_lock"] > 127
        shade_mask = targets["shade_mask"] if targets["shade_mask"].dtype == torch.bool else targets["shade_mask"] > 127

        center = build_center_supervision_mask(
            valid.shape[-2], valid.shape[-1], self.center_halo, device=valid.device,
        )
        if center.ndim == 2:
            center = center.unsqueeze(0).expand(valid.shape[0], -1, -1)
        sem_mask = valid & ~black_lock & center
        level_mask = shade_mask & valid & center
        trans_mask = valid & ~black_lock & center

        boundary_boost = build_boundary_boost_mask(
            targets["transition_mask"],
            cfg.boundary_boost.transition_dilate,
        )
        boost_w = _pixel_boost_weights(boundary_boost, cfg.boundary_boost.transition_boost)

        losses: dict[str, torch.Tensor] = {}
        logits = outputs["refined_logits"].float()
        labels = targets["target_class"]
        labels_masked = labels.clone()
        labels_masked[~sem_mask] = self.ignore_index

        if sem_mask.any():
            ce_map = F.cross_entropy(logits, labels_masked, ignore_index=self.ignore_index, reduction="none")
            w_ce = sem_mask.float()
            if cfg.boundary_boost.enabled:
                w_ce = w_ce * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.level_ce_boost)
            losses["final_ce"] = (ce_map * w_ce).sum() / w_ce.sum().clamp(min=1)
            losses["final_dice"] = self.final_dice(logits, labels, sem_mask, boost_w if cfg.boundary_boost.enabled else None)
        else:
            losses["final_ce"] = logits.new_zeros(())
            losses["final_dice"] = logits.new_zeros(())

        where_tgt = targets["where_target"]
        where_m = valid & ~black_lock & center
        where_w = _pixel_boost_weights(boundary_boost, cfg.boundary_boost.where_boost) if cfg.boundary_boost.enabled else None
        losses["where"] = self.focal_bce(outputs["where_logits"], where_tgt, where_m, where_w)

        level_logits = outputs["level_logits"]
        level_tgt = targets["level_target"]
        lt = level_tgt.clone()
        lt[~level_mask] = self.ignore_index
        if level_mask.any():
            ce_lvl = F.cross_entropy(level_logits, lt, ignore_index=self.ignore_index, reduction="none")
            w_lvl = level_mask.float()
            if cfg.boundary_boost.enabled:
                w_lvl = w_lvl * _pixel_boost_weights(boundary_boost, cfg.boundary_boost.level_ce_boost)
            losses["level_ce"] = (ce_lvl * w_lvl).sum() / w_lvl.sum().clamp(min=1)
        else:
            losses["level_ce"] = logits.new_zeros(())

        ord_logits = outputs["ordinal_logits"]
        ord_tgt = targets["ordinal_target"].float()
        if level_mask.any():
            losses["ordinal"] = F.binary_cross_entropy_with_logits(
                ord_logits[:, 0][level_mask], ord_tgt[:, 0][level_mask], reduction="mean",
            ) + F.binary_cross_entropy_with_logits(
                ord_logits[:, 1][level_mask], ord_tgt[:, 1][level_mask], reduction="mean",
            )
            losses["ordinal_emd"] = self.ordinal_emd(level_logits, level_tgt, level_mask)
        else:
            losses["ordinal"] = logits.new_zeros(())
            losses["ordinal_emd"] = logits.new_zeros(())

        trans_tgt = (targets["transition_mask"] > 127).float()
        trans_w = _pixel_boost_weights(boundary_boost, cfg.boundary_boost.transition_boost) if cfg.boundary_boost.enabled else None
        losses["transition"] = self.focal_bce(outputs["transition_logits"], trans_tgt, trans_mask, trans_w)
        losses["transition_boundary_dice"] = self.boundary_dice(
            outputs["transition_logits"], trans_tgt, trans_mask,
        )

        aff_logits = outputs["affinity_logits"]
        aff_loss = aff_logits.new_zeros(())
        aff_count = 0
        aff_targets = targets["affinity_channel_targets"]
        per_sample_targets = aff_targets if aff_targets and isinstance(aff_targets[0], list) else [aff_targets]
        aff_w_scalar = cfg.boundary_boost.affinity_boost if cfg.boundary_boost.enabled else 1.0

        for b, sample_targets in enumerate(per_sample_targets):
            if b >= aff_logits.shape[0]:
                break
            for ch, tgt_item in enumerate(sample_targets):
                if ch >= aff_logits.shape[1]:
                    break
                tgt = tgt_item[2] if isinstance(tgt_item, tuple) else tgt_item
                if not isinstance(tgt, torch.Tensor):
                    continue
                tgt = tgt.to(aff_logits.device)
                m = (tgt > -0.5) & center[b]
                if m.any():
                    bce = F.binary_cross_entropy_with_logits(
                        aff_logits[b, ch][m], tgt[m], reduction="mean",
                    )
                    aff_loss = aff_loss + bce * aff_w_scalar
                    aff_count += 1
        losses["affinity"] = aff_loss / max(aff_count, 1)

        base = outputs["base_logits"]
        refined = outputs["refined_logits"]
        m = shade_mask & valid & center
        if m.any():
            losses["overlap_consistency"] = F.mse_loss(
                F.softmax(refined, dim=1)[m.unsqueeze(1).expand(-1, 4, -1, -1)],
                F.softmax(base.detach(), dim=1)[m.unsqueeze(1).expand(-1, 4, -1, -1)],
            )
        else:
            losses["overlap_consistency"] = logits.new_zeros(())

        emd_term = losses["ordinal_emd"] if cfg.use_soft_ordinal_emd else losses["ordinal_emd"]
        total = (
            cfg.final_ce * losses["final_ce"]
            + cfg.final_dice * losses["final_dice"]
            + cfg.where * losses["where"]
            + cfg.level_ce * losses["level_ce"]
            + cfg.ordinal * (losses["ordinal"] + 0.5 * emd_term)
            + cfg.transition * losses["transition"]
            + cfg.transition_boundary_dice * losses["transition_boundary_dice"]
            + cfg.affinity * losses["affinity"]
            + cfg.overlap_consistency * losses["overlap_consistency"]
        )
        losses["total"] = total
        return losses
