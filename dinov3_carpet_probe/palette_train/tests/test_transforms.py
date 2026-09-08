from __future__ import annotations

from dinov3_carpet_probe.palette_train.src.transforms import PaletteImageTransform


def test_square_letterbox_output_size(toy_dataset_dirs):
    image_dir, _ = toy_dataset_dirs
    image_path = image_dir / "000000.jpg"
    transform = PaletteImageTransform(
        long_side=512,
        pad_multiple=16,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    tensor, meta = transform(image_path)
    assert tensor.shape == (3, 512, 512)
    assert meta["processed_hw"] == (512, 512)
