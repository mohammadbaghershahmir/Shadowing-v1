from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def toy_dataset_dirs(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "jpg"
    palette_dir = tmp_path / "palettes"
    image_dir.mkdir()
    palette_dir.mkdir()
    for idx in range(3):
        stem = f"{idx:06d}"
        Image.new("RGB", (32, 24), color=(idx * 10, idx * 20, idx * 30)).save(image_dir / f"{stem}.jpg")
        payload = {
            "sample_id": idx,
            "num_colors": 2,
            "rgb": [[0, 0, 0], [255, 255, 255]],
            "hex": ["#000000", "#FFFFFF"],
        }
        (palette_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    return image_dir, palette_dir


@pytest.fixture
def dummy_batch() -> dict[str, torch.Tensor]:
    return {
        "target_rgb": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "target_mask": torch.tensor([[True, True], [True, False]]),
        "target_counts": torch.tensor([2, 1], dtype=torch.int64),
        "stems": ["000000", "000001"],
    }
