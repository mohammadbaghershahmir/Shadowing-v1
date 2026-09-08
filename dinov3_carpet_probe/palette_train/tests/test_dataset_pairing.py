from __future__ import annotations

import json

from dinov3_carpet_probe.palette_train.src.dataset import discover_pairs, validate_dataset


def test_pairing_by_stem(toy_dataset_dirs):
    image_dir, palette_dir = toy_dataset_dirs
    discovered = discover_pairs(image_dir, palette_dir)
    assert len(discovered["pairs"]) == 3
    assert discovered["missing_images"] == []
    assert discovered["missing_annotations"] == []


def test_missing_pair_detection(toy_dataset_dirs):
    image_dir, palette_dir = toy_dataset_dirs
    missing = palette_dir / "999999.json"
    missing.write_text(
        json.dumps(
            {
                "sample_id": 999999,
                "num_colors": 1,
                "rgb": [[1, 2, 3]],
                "hex": ["#010203"],
            }
        ),
        encoding="utf-8",
    )
    _, report = validate_dataset(image_dir, palette_dir)
    assert report.errors
    assert report.missing_images == ["999999.json"]
