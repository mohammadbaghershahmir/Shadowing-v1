"""CSV + matplotlib loss/eval tracking for CSN-V4."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

LOSS_COLUMNS = [
    "step", "optimizer_step", "lr", "total", "final_ce", "final_dice", "where", "level_ce",
    "transition", "transition_boundary_dice", "affinity", "base_refined_consistency",
    "aff_valid_frac",
    "global_miou", "core_miou", "halo_miou", "seam_band_miou", "corner_miou",
    "boundary_miou", "bf1_2px", "small_component_recall", "small_component_f1",
    "worst_orient_global_miou", "checkpoint_score_v4",
    "score_capacity", "score_spatial", "score_scene_val",
]

LOSS_KEYS = {
    "lr", "total", "final_ce", "final_dice", "where", "level_ce",
    "transition", "transition_boundary_dice", "affinity", "base_refined_consistency",
    "aff_valid_frac",
}
EVAL_KEYS = {
    "global_miou", "core_miou", "halo_miou", "seam_band_miou", "corner_miou",
    "boundary_miou", "bf1_2px", "small_component_recall", "small_component_f1",
    "worst_orient_global_miou", "checkpoint_score_v4",
    "score_capacity", "score_spatial", "score_scene_val",
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
    LOGGER.warning("Could not append to %s (file may be open elsewhere).", path)
    return False


def _blank_row(step: int, optimizer_step: int | None = None) -> dict:
    row = {"step": step, **{k: "" for k in LOSS_COLUMNS[1:]}}
    if optimizer_step is not None:
        row["optimizer_step"] = optimizer_step
    return row


class MetricsLogger:
    def __init__(self, run_dir: str | Path, plot_every_steps: int = 100):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.eval_jsonl = self.run_dir / "eval_history.jsonl"
        self.plot_every = plot_every_steps
        self.loss_history: list[dict] = []
        self.eval_history: list[dict] = []
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=LOSS_COLUMNS, extrasaction="ignore").writeheader()

    def log_loss(self, step: int, metrics: dict[str, float], *, optimizer_step: int | None = None) -> None:
        row = _blank_row(step, optimizer_step)
        for k, v in metrics.items():
            if k in LOSS_KEYS and v is not None:
                row[k] = float(v)
        if optimizer_step is not None:
            row["optimizer_step"] = optimizer_step
        self.loss_history.append(row)
        _write_csv_row(self.csv_path, LOSS_COLUMNS, row)
        if step > 0 and step % self.plot_every == 0:
            self.plot_losses()

    def log_eval(self, step: int, metrics: dict[str, float], *, optimizer_step: int, report: dict | None = None) -> None:
        row = _blank_row(step, optimizer_step)
        for k, v in metrics.items():
            if (k in EVAL_KEYS or k in LOSS_KEYS) and v is not None and v == v:
                row[k] = float(v)
        self.eval_history.append(row)
        _write_csv_row(self.csv_path, LOSS_COLUMNS, row)
        if report is not None:
            with self.eval_jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(report, sort_keys=True) + "\n")
        self.plot_eval()
        self.plot_losses()

    def plot_losses(self) -> None:
        rows = [r for r in self.loss_history if r.get("total") not in ("", None)]
        if len(rows) < 2:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps = [int(r["step"]) for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        ax = axes[0, 0]
        ax.plot(steps, [float(r["total"]) for r in rows], color="#1f77b4", linewidth=1.2)
        ax.set_title("Total Loss")
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for k, c in (("final_ce", "#ff7f0e"), ("final_dice", "#2ca02c"), ("where", "#d62728"), ("level_ce", "#9467bd")):
            ax.plot(steps, [float(r.get(k, 0) or 0) for r in rows], label=k, color=c, linewidth=1.0)
        ax.set_title("Segmentation Losses")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for k, c in (("transition", "#8c564b"), ("transition_boundary_dice", "#e377c2"),
                     ("affinity", "#7f7f7f"), ("base_refined_consistency", "#bcbd22")):
            ax.plot(steps, [float(r.get(k, 0) or 0) for r in rows], label=k, color=c, linewidth=1.0)
        ax.set_title("Boundary / Consistency Losses")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        lrs = [float(r.get("lr", 0) or 0) for r in rows]
        ax.plot(steps, lrs, color="#17becf")
        ax.set_title("Learning Rate")
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "loss_curves.png")

    def plot_eval(self) -> None:
        rows = [r for r in self.eval_history if r.get("global_miou") not in ("", None)]
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        opt_steps = [int(r.get("optimizer_step") or r["step"]) for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        ax = axes[0, 0]
        for k, c in (("global_miou", "#1f77b4"), ("core_miou", "#ff7f0e"), ("halo_miou", "#2ca02c"), ("seam_band_miou", "#d62728")):
            ax.plot(opt_steps, [float(r.get(k, 0) or 0) for r in rows], marker="o", markersize=3, label=k, color=c)
        ax.set_title("Region mIoU (tile eval)")
        ax.set_xlabel("optimizer step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for k, c in (("boundary_miou", "#9467bd"), ("bf1_2px", "#8c564b"), ("worst_orient_global_miou", "#e377c2")):
            ax.plot(opt_steps, [float(r.get(k, 0) or 0) for r in rows], marker="o", markersize=3, label=k, color=c)
        ax.set_title("Boundary / Worst-Orientation mIoU")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for k, c in (("small_component_recall", "#7f7f7f"), ("small_component_f1", "#bcbd22")):
            vals = [float(r.get(k, 0) or 0) if r.get(k) not in ("", None) else float("nan") for r in rows]
            ax.plot(opt_steps, vals, marker="o", markersize=3, label=k, color=c)
        ax.set_title("Small Component Metrics")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for k, c in (("checkpoint_score_v4", "#1f77b4"), ("score_capacity", "#ff7f0e"),
                     ("score_spatial", "#2ca02c"), ("score_scene_val", "#d62728")):
            if any(r.get(k) not in ("", None) for r in rows):
                ax.plot(opt_steps, [float(r.get(k, 0) or 0) for r in rows], marker="o", markersize=3, label=k, color=c)
        ax.set_title("Checkpoint Scores")
        ax.set_xlabel("optimizer step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "eval_curves.png")

        # Per-orientation from latest eval report jsonl
        self._plot_orientation_bars()

    def _plot_orientation_bars(self) -> None:
        if not self.eval_jsonl.is_file():
            return
        last = None
        with self.eval_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)
        if not last or "per_orientation" not in last:
            return
        per = last["per_orientation"]
        if not per:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        keys = sorted(per.keys(), key=lambda x: int(x))
        miou = [float(per[k].get("global_miou", 0)) for k in keys]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(keys, miou, color="#1f77b4")
        ax.set_xlabel("D4 transform_id")
        ax.set_ylabel("global mIoU")
        ax.set_title(f"Per-Orientation mIoU @ opt_step={last.get('optimizer_step', '?')}")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_fig(fig, self.run_dir / "orientation_miou.png")

    @staticmethod
    def _save_fig(fig, path: Path) -> None:
        for attempt in range(4):
            try:
                fig.savefig(path, dpi=140)
                break
            except PermissionError:
                if attempt + 1 >= 4:
                    LOGGER.warning("Could not save %s (file locked).", path)
                else:
                    time.sleep(0.25 * (attempt + 1))
        import matplotlib.pyplot as plt
        plt.close(fig)
