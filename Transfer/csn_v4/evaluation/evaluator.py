"""Full-scene D4-aware evaluation for CSN-V4."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from csn_v3.masks import semantic_gray_to_class
from csn_v4.config import CSNV4Config
from csn_v4.constants import CLASS_TO_GRAY
from csn_v4.geometry.d4 import D4Transform
from csn_v4.geometry.tile_spec import assert_full_coverage, iter_core_tiles
from csn_v4.inference.tiled import tiled_predict
from csn_v4.metrics.region_metrics import RegionMetrics, aggregate_region_metrics, compute_region_metrics


REQUIRED_ORIENTATIONS = frozenset(range(8))


@dataclass
class SceneEvalResult:
    scene_id: str
    per_orientation: dict[int, RegionMetrics] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "per_orientation": {str(k): v.to_dict() for k, v in self.per_orientation.items()},
            "aggregate": self.aggregate,
        }


class CSNV4Evaluator:
    """Evaluate full scenes with scene-first D4 transforms."""

    def __init__(self, model, cfg: CSNV4Config, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

    def _class_to_gray(self, pred: np.ndarray) -> np.ndarray:
        out = np.full_like(pred, 255, dtype=np.uint8)
        for cls, gray in CLASS_TO_GRAY.items():
            out[pred == cls] = gray
        return out

    def evaluate_full_scene(
        self,
        source_bw: np.ndarray,
        target_semantic: np.ndarray,
        *,
        transform_id: int,
        scene_id: str = "eval",
        transition_mask: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
        black_lock: np.ndarray | None = None,
        skip_small_component: bool = False,
    ) -> RegionMetrics:
        """Evaluate one orientation; D4 is applied internally to all arrays."""
        tf = D4Transform(transform_id)
        src_t = tf.apply(source_bw)
        sem_t = tf.apply(target_semantic)
        h, w = src_t.shape[:2]

        specs = list(iter_core_tiles(w, h, transform_id=transform_id, scene_id=scene_id))
        assert_full_coverage(w, h, specs)

        pred = tiled_predict(
            self.model,
            src_t,
            self.device,
            context_size=self.cfg.model.context_size,
            global_long_side=self.cfg.model.global_long_side,
            transform_id=transform_id,
            scene_id=scene_id,
            input_mode=self.cfg.data.input_mode,
        )

        tgt_class = semantic_gray_to_class(torch.from_numpy(sem_t.astype(np.int64))).numpy()
        if transition_mask is not None:
            trans = tf.apply(transition_mask)
        else:
            trans = np.zeros_like(sem_t, dtype=np.uint8)
        if valid_mask is not None:
            valid = tf.apply(valid_mask)
        else:
            valid = np.ones_like(sem_t, dtype=np.uint8) * 255
        if black_lock is not None:
            black = tf.apply(black_lock)
        else:
            black = (sem_t == 0).astype(np.uint8) * 255

        return compute_region_metrics(
            pred,
            tgt_class,
            trans,
            valid,
            black,
            transform_id=transform_id,
            boundary_dilate=self.cfg.data.boundary_dilate,
            skip_small_component=skip_small_component,
        )

    def evaluate_all_orientations(
        self,
        source_bw: np.ndarray,
        target_semantic: np.ndarray,
        *,
        scene_id: str = "eval",
        transition_mask: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
        black_lock: np.ndarray | None = None,
        require_all: bool = True,
        skip_small_component: bool = False,
        orientations: list[int] | None = None,
        show_progress: bool = True,
    ) -> SceneEvalResult:
        from tqdm.auto import tqdm

        was_training = self.model.training
        self.model.eval()
        try:
            orients = list(orientations) if orientations is not None else list(range(8))
            per_orient: dict[int, RegionMetrics] = {}
            h, w = source_bw.shape[:2]
            n_tiles = len(list(iter_core_tiles(w, h, scene_id=scene_id)))
            iterator = orients
            if show_progress:
                iterator = tqdm(
                    orients,
                    desc=f"D4 {scene_id[:24]} {w}x{h}~{n_tiles}t",
                    leave=False,
                    dynamic_ncols=True,
                )
            for k in iterator:
                per_orient[k] = self.evaluate_full_scene(
                    source_bw,
                    target_semantic,
                    transform_id=k,
                    scene_id=scene_id,
                    transition_mask=transition_mask,
                    valid_mask=valid_mask,
                    black_lock=black_lock,
                    skip_small_component=skip_small_component,
                )
            if require_all and orientations is None:
                missing = REQUIRED_ORIENTATIONS - set(per_orient)
                if missing:
                    raise ValueError(f"Missing required orientations: {sorted(missing)}")
            agg_list = list(per_orient.values())
            aggregate = aggregate_region_metrics(agg_list)
            return SceneEvalResult(scene_id=scene_id, per_orientation=per_orient, aggregate=aggregate)
        finally:
            self.model.train(was_training)

    def evaluate_inverse_consistency(
        self,
        source_bw: np.ndarray,
        target_semantic: np.ndarray,
        transform_id: int,
        *,
        scene_id: str = "eval",
    ) -> dict[str, float]:
        """Compare metrics in transformed space vs inverse-mapped prediction."""
        tf = D4Transform(transform_id)
        result_t = self.evaluate_full_scene(
            source_bw, target_semantic, transform_id=transform_id, scene_id=scene_id,
        )
        pred_t = tiled_predict(
            self.model,
            tf.apply(source_bw),
            self.device,
            context_size=self.cfg.model.context_size,
            global_long_side=self.cfg.model.global_long_side,
            transform_id=transform_id,
            scene_id=scene_id,
            input_mode=self.cfg.data.input_mode,
        )
        pred_orig = tf.inverse.apply(pred_t)
        sem = target_semantic
        tgt_class = semantic_gray_to_class(torch.from_numpy(sem.astype(np.int64))).numpy()
        trans = ((tgt_class >= 1) & (tgt_class <= 3)).astype(np.uint8) * 255
        valid = np.ones_like(sem, dtype=np.uint8) * 255
        black = (sem == 0).astype(np.uint8) * 255
        result_orig = compute_region_metrics(
            pred_orig, tgt_class, trans, valid, black, transform_id=0,
            boundary_dilate=self.cfg.data.boundary_dilate,
        )
        return {
            "transformed_miou": result_t.global_miou,
            "inverse_miou": result_orig.global_miou,
            "delta_miou": abs(result_t.global_miou - result_orig.global_miou),
        }
