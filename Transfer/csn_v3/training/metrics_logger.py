"""CSV + matplotlib loss curve tracking for CSN-V3."""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

LOSS_COLUMNS = [
    "step", "lr", "total", "final_ce", "final_dice", "where", "level_ce",
    "ordinal", "ordinal_emd", "transition", "transition_boundary_dice",
    "affinity", "overlap_consistency",
    "global_miou", "boundary_miou", "boundary_pixel_acc", "interior_miou", "bf1_2px",
    "checkpoint_score_v3", "checkpoint_score_v4", "rotated_boundary_miou",
    "core_miou", "seam_band_miou", "small_component_recall",
]

LOSS_KEYS = {
    "lr", "total", "final_ce", "final_dice", "where", "level_ce",
    "ordinal", "ordinal_emd", "transition", "transition_boundary_dice",
    "affinity", "overlap_consistency",
}
EVAL_KEYS = {
    "global_miou", "boundary_miou", "boundary_pixel_acc", "interior_miou", "bf1_2px",
    "checkpoint_score_v3", "checkpoint_score_v4", "rotated_boundary_miou",
    "core_miou", "seam_band_miou", "small_component_recall",
}


def _write_csv_row(path: Path, fieldnames: list[str], row: dict, *, retries: int = 8) -> bool:
    for attempt in range(retries):
        try:
            with path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
            return True
        except PermissionError:
            if attempt + 1 >= retries:
                break
            time.sleep(0.25 * (attempt + 1))
    LOGGER.warning(
        "Could not append to %s (file may be open in Excel). Metrics kept in memory only.",
        path,
    )
    return False


def _blank_row(step: int) -> dict:
    return {"step": step, **{k: "" for k in LOSS_COLUMNS[2:]}}


class MetricsLogger:
    def __init__(self, run_dir: str | Path, plot_every_steps: int = 100):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.plot_every = plot_every_steps
        self.history: list[dict] = []
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=LOSS_COLUMNS, extrasaction="ignore").writeheader()

    def log_loss(self, step: int, metrics: dict[str, float]) -> None:
        row = _blank_row(step)
        for k, v in metrics.items():
            if k in LOSS_KEYS:
                row[k] = v
        self.history.append(row)
        _write_csv_row(self.csv_path, LOSS_COLUMNS, row)
        if step > 0 and step % self.plot_every == 0:
            self.plot()

    def log_eval(self, step: int, metrics: dict[str, float]) -> None:
        row = _blank_row(step)
        for k, v in metrics.items():
            if k in EVAL_KEYS or k in LOSS_KEYS:
                row[k] = v
        self.history.append(row)
        _write_csv_row(self.csv_path, LOSS_COLUMNS, row)
        self.plot()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if EVAL_KEYS.intersection(metrics):
            self.log_eval(step, metrics)
        else:
            self.log_loss(step, metrics)

    def _loss_rows(self) -> list[dict]:
        rows = []
        for r in self.history:
            if r.get("total") not in ("", None):
                rows.append(r)
        return rows

    def _eval_rows(self) -> list[dict]:
        rows = []
        for r in self.history:
            gm = r.get("global_miou")
            if gm in ("", None):
                continue
            score = float(r.get("checkpoint_score_v3") or 0)
            if float(gm) <= 0.0 and score <= 0.0:
                continue
            rows.append(r)
        return rows

    def plot(self) -> None:
        loss_rows = self._loss_rows()
        eval_rows = self._eval_rows()
        if len(loss_rows) < 2 and len(eval_rows) < 1:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        if len(loss_rows) >= 2:
            steps = [r["step"] for r in loss_rows]
            ax = axes[0, 0]
            ax.plot(steps, [float(r["total"]) for r in loss_rows], label="total")
            ax.set_title("Total Loss")
            ax.legend()

            ax = axes[0, 1]
            for k in ("transition", "affinity", "transition_boundary_dice"):
                ax.plot(steps, [float(r.get(k, 0) or 0) for r in loss_rows], label=k)
            ax.set_title("Boundary Loss Group")
            ax.legend()
        else:
            axes[0, 0].set_title("Total Loss (waiting for data)")
            axes[0, 1].set_title("Boundary Loss Group (waiting for data)")

        if eval_rows:
            esteps = [r["step"] for r in eval_rows]
            ax = axes[1, 0]
            for k in ("global_miou", "boundary_miou", "interior_miou"):
                ax.plot(esteps, [float(r[k]) for r in eval_rows], marker="o", label=k)
            ax.set_title("mIoU: Global / Boundary / Interior")
            ax.legend()

            ax = axes[1, 1]
            for k in ("boundary_pixel_acc", "bf1_2px", "checkpoint_score_v3"):
                ax.plot(esteps, [float(r[k]) for r in eval_rows], marker="o", label=k)
            ax.set_title("Boundary Acc / BF1 / Score")
            ax.legend()
        else:
            axes[1, 0].set_title("mIoU (eval every 400 micro-steps)")
            axes[1, 1].set_title("Boundary metrics (eval every 400 micro-steps)")

        plt.tight_layout()
        plot_path = self.run_dir / "loss_curves.png"
        for attempt in range(4):
            try:
                fig.savefig(plot_path, dpi=120)
                break
            except PermissionError:
                if attempt + 1 >= 4:
                    LOGGER.warning("Could not save %s (file locked).", plot_path)
                else:
                    time.sleep(0.25 * (attempt + 1))
        plt.close(fig)
