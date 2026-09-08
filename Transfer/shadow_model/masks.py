"""Mask construction and transition target generation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


def build_supervision_mask(
    candidate_mask: torch.Tensor,
    image_valid_mask: torch.Tensor,
    central_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """M = candidate AND image_valid AND central_valid. Returns bool [B,H,W] or [H,W]."""
    return candidate_mask & image_valid_mask & central_valid_mask


def apply_ignore_outside_mask(
    target_label: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Set target_label to ignore_index outside mask. Returns new tensor."""
    out = target_label.clone()
    out[~mask] = ignore_index
    return out


def build_transition_target(
    target_label: torch.Tensor,
    candidate_mask: torch.Tensor,
    dilate: int = 1,
) -> torch.Tensor:
    """Build internal categorical transition map.

    A pixel is positive if it has a valid class label (0/1/2) AND at least one
    4-neighbor with a DIFFERENT valid class label. Both pixels must be inside
    candidate_mask. Then dilate by `dilate` pixels within candidate for stability.
    """
    squeeze = False
    if target_label.ndim == 2:
        target_label = target_label.unsqueeze(0)
        candidate_mask = candidate_mask.unsqueeze(0)
        squeeze = True

    B, H, W = target_label.shape
    valid = candidate_mask & (target_label >= 0)
    boundary = torch.zeros(B, H, W, dtype=torch.bool, device=target_label.device)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted_label = torch.zeros_like(target_label)
        shifted_valid = torch.zeros_like(valid)

        sy = slice(max(0, -dy), H + min(0, -dy))
        sx = slice(max(0, -dx), W + min(0, -dx))
        ty = slice(max(0, dy), H + min(0, dy))
        tx = slice(max(0, dx), W + min(0, dx))

        shifted_label[:, ty, tx] = target_label[:, sy, sx]
        shifted_valid[:, ty, tx] = valid[:, sy, sx]

        diff = (target_label != shifted_label) & valid & shifted_valid
        boundary = boundary | diff

    if dilate > 0:
        kernel_size = 2 * dilate + 1
        boundary_f = boundary.float().unsqueeze(1)
        dilated = F.max_pool2d(boundary_f, kernel_size, stride=1, padding=dilate)
        boundary = (dilated.squeeze(1) > 0.5) & candidate_mask

    result = boundary.float()
    if squeeze:
        result = result.squeeze(0)
    return result


def build_main_logit_boundary_target(
    target_label: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary boundary target from class labels (same logic as transition, no dilate)."""
    return build_transition_target(target_label, candidate_mask, dilate=0)


def compute_boundary_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Differentiable neighbor disagreement: B = 1 - sum_c P_p * P_q.

    probs: [B,C,H,W] softmax probabilities.
    Returns: [B,H,W] float32 boundary score in [0,1].
    """
    B, _, H, W = probs.shape
    boundary = probs.new_zeros((B, H, W))
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        sy = slice(max(0, -dy), H + min(0, -dy))
        sx = slice(max(0, -dx), W + min(0, -dx))
        ty = slice(max(0, dy), H + min(0, dy))
        tx = slice(max(0, dx), W + min(0, dx))
        p = probs[:, :, sy, sx]
        q = probs[:, :, ty, tx]
        agree = (p * q).sum(dim=1)
        disagree = 1.0 - agree
        contrib = probs.new_zeros((B, H, W))
        contrib[:, ty, tx] = disagree
        contrib[:, sy, sx] = disagree
        boundary = torch.maximum(boundary, contrib)
    return boundary


def build_main_logit_boundary(probs: torch.Tensor) -> torch.Tensor:
    """Alias for compute_boundary_from_probs."""
    return compute_boundary_from_probs(probs)


def _label_component_map(
    target_label: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Connected components on valid candidate pixels."""
    labels_np = target_label.detach().cpu().numpy().astype(np.int32)
    valid_np = mask.detach().cpu().numpy().astype(bool) & (labels_np >= 0)
    structure = np.ones((3, 3), dtype=np.int8)
    component_map = np.zeros_like(labels_np, dtype=np.int32)
    if valid_np.any():
        labeled, _ = ndimage.label(valid_np, structure=structure)
        component_map = labeled.astype(np.int32)
    return component_map


def sample_affinity_pairs(
    target_label: torch.Tensor,
    mask: torch.Tensor,
    offsets: list[int] | None = None,
    max_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample balanced affinity pairs for categorical consistency loss."""
    if offsets is None:
        offsets = [1, 2, 4, 8]

    H, W = target_label.shape
    valid = mask & (target_label >= 0)

    all_same_y, all_same_x = [], []
    all_diff_y, all_diff_x = [], []
    all_same_qy, all_same_qx = [], []
    all_diff_qy, all_diff_qx = [], []

    for delta in offsets:
        for dy, dx in [(0, delta), (delta, 0)]:
            py = slice(0, H - abs(dy)) if dy >= 0 else slice(abs(dy), H)
            px = slice(0, W - abs(dx)) if dx >= 0 else slice(abs(dx), W)
            qy = slice(abs(dy), H) if dy >= 0 else slice(0, H - abs(dy))
            qx = slice(abs(dx), W) if dx >= 0 else slice(0, W - abs(dx))

            v_p = valid[py, px]
            v_q = valid[qy, qx]
            both_valid = v_p & v_q

            lp = target_label[py, px]
            lq = target_label[qy, qx]
            same = (lp == lq) & both_valid
            diff = (lp != lq) & both_valid

            ys_p, xs_p = torch.where(same)
            if dy >= 0:
                all_same_y.append(ys_p)
            else:
                all_same_y.append(ys_p + abs(dy))
            if dx >= 0:
                all_same_x.append(xs_p)
            else:
                all_same_x.append(xs_p + abs(dx))
            all_same_qy.append(all_same_y[-1] + dy)
            all_same_qx.append(all_same_x[-1] + dx)

            yd_p, xd_p = torch.where(diff)
            if dy >= 0:
                all_diff_y.append(yd_p)
            else:
                all_diff_y.append(yd_p + abs(dy))
            if dx >= 0:
                all_diff_x.append(xd_p)
            else:
                all_diff_x.append(xd_p + abs(dx))
            all_diff_qy.append(all_diff_y[-1] + dy)
            all_diff_qx.append(all_diff_x[-1] + dx)

    half = max_pairs // 2
    device = target_label.device

    def _gather(ys, xs, qys, qxs, n):
        if not ys:
            return torch.empty(0, 2, device=device), torch.empty(0, 2, device=device)
        y_cat = torch.cat(ys)
        x_cat = torch.cat(xs)
        qy_cat = torch.cat(qys)
        qx_cat = torch.cat(qxs)
        total = y_cat.shape[0]
        if total == 0:
            return torch.empty(0, 2, device=device), torch.empty(0, 2, device=device)
        if total > n:
            idx = torch.randperm(total, generator=generator, device=device)[:n]
            y_cat, x_cat = y_cat[idx], x_cat[idx]
            qy_cat, qx_cat = qy_cat[idx], qx_cat[idx]
        return torch.stack([y_cat, x_cat], dim=1), torch.stack([qy_cat, qx_cat], dim=1)

    p_same, q_same = _gather(all_same_y, all_same_x, all_same_qy, all_same_qx, half)
    p_diff, q_diff = _gather(all_diff_y, all_diff_x, all_diff_qy, all_diff_qx, half)

    if p_same.shape[0] == 0 and p_diff.shape[0] == 0:
        return (
            torch.empty(0, 2, device=device, dtype=torch.long),
            torch.empty(0, 2, device=device, dtype=torch.long),
            torch.empty(0, device=device),
        )

    indices_p = torch.cat([p_same, p_diff], dim=0).long()
    indices_q = torch.cat([q_same, q_diff], dim=0).long()
    targets = torch.cat([
        torch.ones(p_same.shape[0], device=device),
        torch.zeros(p_diff.shape[0], device=device),
    ])
    return indices_p, indices_q, targets


def sample_affinity_pairs_same_component(
    target_label: torch.Tensor,
    mask: torch.Tensor,
    offsets: list[int] | None = None,
    max_pairs: int = 4096,
    component_map: np.ndarray | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample affinity pairs requiring same connected component in target mask."""
    if offsets is None:
        offsets = [1, 2, 4, 8]

    if component_map is None:
        component_map = _label_component_map(target_label, mask)
    comp_t = torch.from_numpy(component_map).to(target_label.device)

    H, W = target_label.shape
    valid = mask & (target_label >= 0) & (comp_t > 0)

    all_same_y, all_same_x = [], []
    all_diff_y, all_diff_x = [], []
    all_same_qy, all_same_qx = [], []
    all_diff_qy, all_diff_qx = [], []

    for delta in offsets:
        for dy, dx in [(0, delta), (delta, 0)]:
            py = slice(0, H - abs(dy)) if dy >= 0 else slice(abs(dy), H)
            px = slice(0, W - abs(dx)) if dx >= 0 else slice(abs(dx), W)
            qy = slice(abs(dy), H) if dy >= 0 else slice(0, H - abs(dy))
            qx = slice(abs(dx), W) if dx >= 0 else slice(0, W - abs(dx))

            v_p = valid[py, px]
            v_q = valid[qy, qx]
            both_valid = v_p & v_q

            cp = comp_t[py, px]
            cq = comp_t[qy, qx]
            same_comp = (cp == cq) & both_valid
            diff_comp = (cp != cq) & both_valid

            ys_p, xs_p = torch.where(same_comp)
            if dy >= 0:
                all_same_y.append(ys_p)
            else:
                all_same_y.append(ys_p + abs(dy))
            if dx >= 0:
                all_same_x.append(xs_p)
            else:
                all_same_x.append(xs_p + abs(dx))
            all_same_qy.append(all_same_y[-1] + dy)
            all_same_qx.append(all_same_x[-1] + dx)

            yd_p, xd_p = torch.where(diff_comp)
            if dy >= 0:
                all_diff_y.append(yd_p)
            else:
                all_diff_y.append(yd_p + abs(dy))
            if dx >= 0:
                all_diff_x.append(xd_p)
            else:
                all_diff_x.append(xd_p + abs(dx))
            all_diff_qy.append(all_diff_y[-1] + dy)
            all_diff_qx.append(all_diff_x[-1] + dx)

    half = max_pairs // 2
    device = target_label.device

    def _gather(ys, xs, qys, qxs, n):
        if not ys:
            return torch.empty(0, 2, device=device), torch.empty(0, 2, device=device)
        y_cat = torch.cat(ys)
        x_cat = torch.cat(xs)
        qy_cat = torch.cat(qys)
        qx_cat = torch.cat(qxs)
        total = y_cat.shape[0]
        if total == 0:
            return torch.empty(0, 2, device=device), torch.empty(0, 2, device=device)
        if total > n:
            idx = torch.randperm(total, generator=generator, device=device)[:n]
            y_cat, x_cat = y_cat[idx], x_cat[idx]
            qy_cat, qx_cat = qy_cat[idx], qx_cat[idx]
        return torch.stack([y_cat, x_cat], dim=1), torch.stack([qy_cat, qx_cat], dim=1)

    p_same, q_same = _gather(all_same_y, all_same_x, all_same_qy, all_same_qx, half)
    p_diff, q_diff = _gather(all_diff_y, all_diff_x, all_diff_qy, all_diff_qx, half)

    if p_same.shape[0] == 0 and p_diff.shape[0] == 0:
        return (
            torch.empty(0, 2, device=device, dtype=torch.long),
            torch.empty(0, 2, device=device, dtype=torch.long),
            torch.empty(0, device=device),
        )

    indices_p = torch.cat([p_same, p_diff], dim=0).long()
    indices_q = torch.cat([q_same, q_diff], dim=0).long()
    targets = torch.cat([
        torch.ones(p_same.shape[0], device=device),
        torch.zeros(p_diff.shape[0], device=device),
    ])
    return indices_p, indices_q, targets
