"""Loss functions with FP32 computation and ramped auxiliary terms."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shadow_model.masks import (
    apply_ignore_outside_mask,
    build_main_logit_boundary,
    build_supervision_mask,
    build_transition_target,
    sample_affinity_pairs,
    sample_affinity_pairs_same_component,
)


def ramp_weight(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return 1.0
    s = int(start * total)
    e = int(end * total)
    if step <= s:
        return 0.0
    if step >= e:
        return 1.0
    return float(step - s) / max(e - s, 1)


class MaskedCELoss(nn.Module):
    def __init__(self, class_weights: list[float] | None = None, ignore_index: int = -100):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("weight", w)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if (labels != self.ignore_index).sum() == 0:
            raise ValueError("MaskedCE: all labels ignored")
        return F.cross_entropy(logits, labels, weight=self.weight, ignore_index=self.ignore_index)


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        mask_f = mask.float().unsqueeze(1)
        labels_safe = labels.clone()
        labels_safe[labels_safe < 0] = 0
        onehot = F.one_hot(labels_safe, self.num_classes).permute(0, 3, 1, 2).float() * mask_f
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dim=dims)
        card = (probs + onehot).sum(dim=dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        present = onehot.sum(dim=dims) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return logits.new_zeros(())


class TransitionLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = logits.squeeze(1).float()
        target = target.float()
        mask_f = mask.float()
        pw = torch.tensor([self.pos_weight], device=pred.device, dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weight = torch.where(target > 0.5, pw, torch.ones_like(pw))
        return (bce * weight * mask_f).sum() / mask_f.sum().clamp(min=1)


class AffinityLoss(nn.Module):
    def __init__(
        self,
        offsets: list[int] | None = None,
        max_pairs: int = 4096,
        same_component: bool = True,
    ):
        super().__init__()
        self.offsets = offsets or [1, 2, 4, 8]
        self.max_pairs = max_pairs
        self.same_component = same_component

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        total = logits.new_zeros(())
        count = 0
        sampler = sample_affinity_pairs_same_component if self.same_component else sample_affinity_pairs
        for b in range(probs.shape[0]):
            ip, iq, targets = sampler(labels[b], mask[b], self.offsets, self.max_pairs)
            if ip.shape[0] == 0:
                continue
            p_probs = probs[b, :, ip[:, 0], ip[:, 1]]
            q_probs = probs[b, :, iq[:, 0], iq[:, 1]]
            affinity = (p_probs * q_probs).sum(dim=0).clamp(1e-6, 1 - 1e-6)
            total = total + F.binary_cross_entropy(affinity, targets, reduction="mean")
            count += 1
        return total / max(count, 1)


class ShadowLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.3,
        transition_weight: float = 0.0,
        transition_ramp_start: float = 0.05,
        transition_ramp_end: float = 0.15,
        affinity_weight: float = 0.0,
        affinity_ramp_start: float = 0.10,
        affinity_ramp_end: float = 0.20,
        boundary_weight: float = 0.0,
        aux_weight: float = 0.1,
        class_weights: list[float] | None = None,
        num_classes: int = 3,
        ignore_index: int = -100,
        transition_dilate: int = 1,
        pos_weight_clip: float = 10.0,
        affinity_offsets: list[int] | None = None,
        affinity_max_pairs: int = 4096,
        affinity_same_component: bool = True,
        use_main_logit_boundary: bool = False,
        total_steps: int = 3000,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.transition_weight = transition_weight
        self.affinity_weight = affinity_weight
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight
        self.transition_dilate = transition_dilate
        self.use_main_logit_boundary = use_main_logit_boundary
        self.total_steps = total_steps
        self.tr_ramp = (transition_ramp_start, transition_ramp_end)
        self.aff_ramp = (affinity_ramp_start, affinity_ramp_end)

        self.ce = MaskedCELoss(class_weights, ignore_index)
        self.dice = SoftDiceLoss(num_classes)
        self.transition_loss = TransitionLoss(pos_weight_clip)
        self.affinity_loss = AffinityLoss(affinity_offsets, affinity_max_pairs, affinity_same_component)
        self.aux_ce = MaskedCELoss(class_weights, ignore_index)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_label: torch.Tensor,
        candidate_mask: torch.Tensor,
        image_valid_mask: torch.Tensor,
        central_valid_mask: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            M = build_supervision_mask(candidate_mask, image_valid_mask, central_valid_mask)
            if M.sum() == 0:
                raise ValueError("No supervised pixels in batch")
            label_M = apply_ignore_outside_mask(target_label, M)

            class_logits = outputs["class_logits"].float()
            losses: dict[str, torch.Tensor] = {}
            l_ce = self.ce(class_logits, label_M)
            l_dice = self.dice(class_logits, label_M, M)
            losses["ce"] = l_ce
            losses["dice"] = l_dice
            total = self.ce_weight * l_ce + self.dice_weight * l_dice

            if self.aux_weight > 0 and "aux_logits" in outputs:
                aux = F.interpolate(
                    outputs["aux_logits"].float(),
                    size=class_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                l_aux = self.aux_ce(aux, label_M)
                losses["aux"] = l_aux
                total = total + self.aux_weight * l_aux

            trans_target = build_transition_target(target_label, candidate_mask, self.transition_dilate)
            w_t = ramp_weight(global_step, self.total_steps, *self.tr_ramp)

            if self.transition_weight > 0 and "transition_logits" in outputs:
                l_trans = self.transition_loss(outputs["transition_logits"].float(), trans_target, M)
                losses["transition"] = l_trans
                total = total + w_t * self.transition_weight * l_trans

            if self.use_main_logit_boundary and self.boundary_weight > 0:
                boundary_pred = build_main_logit_boundary(F.softmax(class_logits, dim=1))
                l_bnd = F.binary_cross_entropy(
                    boundary_pred * M.float(), trans_target * M.float(), reduction="sum"
                ) / M.float().sum().clamp(min=1)
                losses["boundary"] = l_bnd
                total = total + w_t * self.boundary_weight * l_bnd

            w_a = ramp_weight(global_step, self.total_steps, *self.aff_ramp)
            if self.affinity_weight > 0:
                l_aff = self.affinity_loss(class_logits, target_label, M)
                losses["affinity"] = l_aff
                total = total + w_a * self.affinity_weight * l_aff

            losses["total"] = total
            return losses
