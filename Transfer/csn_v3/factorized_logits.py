"""Stable factorized base logits assembly."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_base_logits(where_logits: torch.Tensor, level_logits: torch.Tensor) -> torch.Tensor:
    """Factorized log-probabilities for 4 semantic classes.

    Classes: 0=white, 1=light, 2=medium, 3=dark
    """
    log_p_white = F.logsigmoid(-where_logits)
    log_p_shade = F.logsigmoid(where_logits)
    log_p_level = F.log_softmax(level_logits, dim=1)

    b, _, h, w = where_logits.shape
    out = where_logits.new_zeros((b, 4, h, w))
    out[:, 0:1] = log_p_white
    out[:, 1:2] = log_p_shade + log_p_level[:, 0:1]
    out[:, 2:3] = log_p_shade + log_p_level[:, 1:2]
    out[:, 3:4] = log_p_shade + log_p_level[:, 2:3]
    return out
