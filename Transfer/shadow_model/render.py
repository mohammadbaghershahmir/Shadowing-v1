"""Exact-color rendering: argmax logits -> categorical RGB output."""
from __future__ import annotations
import numpy as np
import torch
from shadow_dataset.constants import CLASS_TO_RGB, TARGET_REGION, OUTLINE, UNSHADED, TARGET_ALLOWED_COLORS


def render_prediction(
    input_rgb: np.ndarray,
    class_logits: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Replace TARGET_REGION pixels with predicted class RGB.

    Args:
        input_rgb: [H, W, 3] uint8 original input
        class_logits: [3, H, W] float logits (or numpy)

    Returns:
        output_rgb: [H, W, 3] uint8 with exact allowed colors only
    """
    if isinstance(class_logits, torch.Tensor):
        class_logits = class_logits.detach().cpu().numpy()

    output = input_rgb.copy()
    pred_class = np.argmax(class_logits, axis=0)

    target_mask = (
        (input_rgb[:, :, 0] == TARGET_REGION[0])
        & (input_rgb[:, :, 1] == TARGET_REGION[1])
        & (input_rgb[:, :, 2] == TARGET_REGION[2])
    )

    for cls_id, rgb in CLASS_TO_RGB.items():
        mask = target_mask & (pred_class == cls_id)
        output[mask] = rgb

    return output


def validate_output_colors(output_rgb: np.ndarray) -> dict[str, int]:
    """Check that output contains only allowed colors.

    Returns dict with:
        invalid_color_count: number of pixels with unexpected colors
    """
    flat = output_rgb.reshape(-1, 3)
    packed = (
        (flat[:, 0].astype(np.uint32) << 16)
        | (flat[:, 1].astype(np.uint32) << 8)
        | flat[:, 2].astype(np.uint32)
    )

    allowed_packed = set()
    for color in [OUTLINE, UNSHADED, (200, 200, 200), (150, 150, 150), (100, 100, 100)]:
        allowed_packed.add((color[0] << 16) | (color[1] << 8) | color[2])

    unique, counts = np.unique(packed, return_counts=True)
    invalid = 0
    for val, cnt in zip(unique.tolist(), counts.tolist()):
        if val not in allowed_packed:
            invalid += int(cnt)

    return {"invalid_color_count": invalid}
