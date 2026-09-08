from __future__ import annotations

import json

from dinov3_carpet_probe.palette_train.src.dataset import validate_dataset


def test_rgb_hex_validation(toy_dataset_dirs):
    image_dir, palette_dir = toy_dataset_dirs
    bad_path = palette_dir / "000001.json"
    payload = json.loads(bad_path.read_text(encoding="utf-8"))
    payload["hex"][1] = "#FFFFFE"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    _, report = validate_dataset(image_dir, palette_dir)
    assert any("does not match" in error for error in report.errors)


def test_duplicate_rgb_warns(toy_dataset_dirs):
    image_dir, palette_dir = toy_dataset_dirs
    dup_path = palette_dir / "000002.json"
    payload = json.loads(dup_path.read_text(encoding="utf-8"))
    payload["rgb"] = [[1, 1, 1], [1, 1, 1]]
    payload["hex"] = ["#010101", "#010101"]
    dup_path.write_text(json.dumps(payload), encoding="utf-8")
    _, report = validate_dataset(image_dir, palette_dir)
    assert any("duplicate RGB color" in warning for warning in report.warnings)
