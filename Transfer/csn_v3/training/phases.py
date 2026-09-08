"""Training phase controller."""
from __future__ import annotations

from csn_v3.config import CSNV3Config


class PhaseController:
    def __init__(self, cfg: CSNV3Config):
        self.cfg = cfg
        self.phase = cfg.train.phase
        self.schedule = list(cfg.train.teacher_forcing_schedule)

    def teacher_forcing_prob(self, step: int, total: int) -> float:
        if self.cfg.train.teacher_forcing >= 0:
            base = self.cfg.train.teacher_forcing
            if base >= 0:
                return base
        if not self.schedule:
            return 0.0
        idx = min(int(step / max(total / len(self.schedule), 1)), len(self.schedule) - 1)
        return self.schedule[idx]

    def convnext_lrs(self) -> tuple[float, float]:
        if self.phase <= 3:
            return self.cfg.optimizer.lr_convnext_stage12, 0.0
        if self.phase <= 5:
            return self.cfg.optimizer.lr_convnext_stage12, self.cfg.optimizer.lr_convnext_stage34
        return self.cfg.optimizer.lr_convnext_stage12, self.cfg.optimizer.lr_convnext_stage34
