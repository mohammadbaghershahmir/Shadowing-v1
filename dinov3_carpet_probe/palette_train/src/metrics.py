"""Evaluation metrics for palette prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dinov3_carpet_probe.palette_train.src.color import ciede2000_from_uint8
from dinov3_carpet_probe.palette_train.src.decode import decode_palette_prediction
from dinov3_carpet_probe.palette_train.src.matching import hungarian_match_single


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def evaluate_palette_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    matcher_rgb_weight: float,
    matcher_perceptual_weight: float,
    matcher_presence_weight: float,
) -> dict[str, float]:
    count_acc = 0
    count_abs_errors: list[float] = []
    matched_rgb_mae: list[float] = []
    matched_delta_e00: list[float] = []
    quantized_delta_e00: list[float] = []
    exact_cardinality = 0
    palette_success = 0
    false_negative = 0
    false_positive_rates: list[float] = []
    count_head_acc = 0
    presence_count_acc = 0
    count_presence_gap: list[float] = []
    set_errors: list[float] = []

    pred_rgb = outputs["pred_rgb"].detach().cpu()
    presence_logits = outputs["presence_logits"].detach().cpu()
    count_logits = outputs["count_logits"].detach().cpu()
    target_rgb = batch["target_rgb"].detach().cpu()
    target_counts = batch["target_counts"].detach().cpu()
    target_mask = batch["target_mask"].detach().cpu()

    for batch_idx in range(pred_rgb.shape[0]):
        target_valid = target_rgb[batch_idx][target_mask[batch_idx]]
        match = hungarian_match_single(
            pred_rgb[batch_idx],
            presence_logits[batch_idx],
            target_valid,
            rgb_weight=matcher_rgb_weight,
            perceptual_weight=matcher_perceptual_weight,
            presence_weight=matcher_presence_weight,
        )
        decoded = decode_palette_prediction(
            stem=batch["stems"][batch_idx],
            pred_rgb=pred_rgb[batch_idx],
            presence_logits=presence_logits[batch_idx],
            count_logits=count_logits[batch_idx],
        )
        count_head_pred = int(torch.argmax(count_logits[batch_idx]).item())
        presence_count_pred = int(torch.round(torch.sigmoid(presence_logits[batch_idx]).sum()).item())
        pred_count = decoded.num_colors
        true_count = int(target_counts[batch_idx].item())
        count_head_acc += int(count_head_pred == true_count)
        presence_count_acc += int(presence_count_pred == true_count)
        count_presence_gap.append(abs(count_head_pred - presence_count_pred))
        if pred_count == true_count:
            count_acc += 1
            exact_cardinality += 1
        count_abs_errors.append(abs(pred_count - true_count))
        false_negative += max(0, true_count - pred_count)
        false_positive_rates.append(max(0.0, (pred_count - true_count) / max(1, pred_rgb.shape[1])))

        if match.pred_indices.numel():
            matched_pred = pred_rgb[batch_idx, match.pred_indices]
            matched_target = target_valid[match.target_indices]
            matched_rgb_mae.extend(torch.abs(matched_pred - matched_target).mean(dim=-1).tolist())

            pred_q = torch.round(matched_pred.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
            target_q = torch.round(matched_target.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
            delta_q = ciede2000_from_uint8(pred_q, target_q)
            quantized_delta_e00.extend(delta_q.tolist())
            matched_delta_e00.extend(delta_q.tolist())
            c = 10.0
            set_errors.append((float(np.minimum(delta_q, c).sum()) + c * abs(pred_count - true_count)) / max(pred_count, true_count, 1))
            if pred_count == true_count and np.all(delta_q <= 1.0):
                palette_success += 1
        else:
            c = 10.0
            set_errors.append(c * abs(pred_count - true_count) / max(pred_count, true_count, 1))

    total = max(1, pred_rgb.shape[0])
    return {
        "count_accuracy": count_acc / total,
        "count_mae": _safe_mean(count_abs_errors),
        "matched_rgb_mae_mean": _safe_mean(matched_rgb_mae),
        "matched_rgb_mae_median": _safe_median(matched_rgb_mae),
        "delta_e00_mean": _safe_mean(matched_delta_e00),
        "delta_e00_median": _safe_median(matched_delta_e00),
        "delta_e00_p90": _safe_percentile(matched_delta_e00, 90),
        "delta_e00_p95": _safe_percentile(matched_delta_e00, 95),
        "delta_e00_le_1": _safe_mean([float(x <= 1.0) for x in matched_delta_e00]),
        "delta_e00_le_2": _safe_mean([float(x <= 2.0) for x in matched_delta_e00]),
        "delta_e00_le_5": _safe_mean([float(x <= 5.0) for x in matched_delta_e00]),
        "palette_exact_cardinality_rate": exact_cardinality / total,
        "palette_success_rate": palette_success / total,
        "false_positive_query_rate": _safe_mean(false_positive_rates),
        "false_negative_palette_color_rate": false_negative / max(1, int(target_counts.sum().item())),
        "count_head_accuracy": count_head_acc / total,
        "presence_count_accuracy": presence_count_acc / total,
        "count_presence_abs_gap": _safe_mean(count_presence_gap),
        "set_error": _safe_mean(set_errors),
        "quantized_delta_e00_mean": _safe_mean(quantized_delta_e00),
        "quantized_delta_e00_median": _safe_median(quantized_delta_e00),
    }
