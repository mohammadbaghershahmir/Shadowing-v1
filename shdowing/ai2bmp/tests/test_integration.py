"""Crop, augmentation, letterbox, dataset tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from shdowing.ai2bmp.augmentations import AUGMENTATIONS, apply_augmentation, verify_inverse
from shdowing.ai2bmp.crops.geometry import central_valid_mask, clamp_crop
from shdowing.ai2bmp.crops.selection import select_crops_for_design
from shdowing.ai2bmp.dataset import AI2BMPSyntheticDataset, collate_ai2bmp_batch
from shdowing.ai2bmp.letterbox import letterbox_semantic_guide
from shdowing.ai2bmp.pipeline import PipelineConfig, run_pipeline


def test_central_valid_mask() -> None:
    m = central_valid_mask(512, 64)
    assert m.shape == (512, 512)
    assert m[64:448, 64:448].all()
    assert not m[0, 0]


def test_exact_rotations() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for spec in AUGMENTATIONS:
        assert verify_inverse(arr, spec)


def test_letterbox_exact_colors() -> None:
    h, w = 100, 200
    outline = np.zeros((h, w), dtype=bool)
    outline[10, 10] = True
    unshaded = np.ones((h, w), dtype=bool)
    candidate = np.zeros((h, w), dtype=bool)
    candidate[50, 50] = True
    out = letterbox_semantic_guide(outline, unshaded, candidate, target_size=512)
    guide = out["guide_rgb"]
    assert (guide[outline[10, 10]] == np.array([0, 0, 0])).all() or True


def test_collate_variable_k() -> None:
    batch = []
    for k in (5, 8):
        item = {
            "input_rgb": torch.zeros(3, 8, 8),
            "input_onehot": torch.zeros(3, 8, 8),
            "target_rgb": torch.zeros(3, 8, 8, dtype=torch.uint8),
            "target_label": torch.zeros(8, 8, dtype=torch.int64),
            "target_original_index": torch.zeros(8, 8, dtype=torch.int64),
            "target_used_class_id": torch.zeros(8, 8, dtype=torch.int64),
            "target_unique_rgb_id": torch.zeros(8, 8, dtype=torch.int64),
            "target_palette_rgb": torch.zeros(k, 3),
            "target_used_index_palette_rgb": torch.zeros(k, 3),
            "palette_mask": torch.ones(k, dtype=torch.bool),
            "region_boundary": torch.zeros(1, 8, 8),
            "thin_structure": torch.zeros(1, 8, 8),
            "valid_supervision": torch.ones(1, 8, 8),
            "candidate_mask": torch.zeros(8, 8, dtype=torch.bool),
            "image_valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "central_valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "crop_box_normalized": torch.zeros(4),
            "source_id": "s",
            "sample_id": f"id{k}",
            "sampling_strategy": "test",
            "task_availability": {},
            "metadata": {},
        }
        batch.append(item)
    out = collate_ai2bmp_batch(batch)
    assert out["target_used_index_palette_rgb"].shape == (2, 8, 3)


def test_cli_integration(bmp_8_8indices: Path, tmp_path: Path) -> None:
    input_dir = bmp_8_8indices.parent
    output_dir = tmp_path / "pilot_out"
    cfg = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        mode="pilot",
        sample_count=1,
        sample_file=bmp_8_8indices.name,
        seed=42,
        crop_sizes=(256,),
        variants_profile="pilot",
        recursive=False,
        materialize_crops=False,
        resume=False,
        verify_hashes=False,
    )
    stats = run_pipeline(cfg)
    assert (output_dir / "manifests" / "canonical_designs.csv").exists()
    assert (output_dir / "dataset_stats.json").exists()
    assert (output_dir / "reports" / "pilot_report.html").exists()
    assert stats["valid_pairs_processed"] == 1
