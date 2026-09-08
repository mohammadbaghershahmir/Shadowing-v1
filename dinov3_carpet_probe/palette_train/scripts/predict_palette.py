"""Predict palette JSON for one JPG or a directory of JPGs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dinov3_carpet_probe.palette_train.src.checkpointing import load_checkpoint  # noqa: E402
from dinov3_carpet_probe.palette_train.src.decode import decode_palette_prediction, output_path_for_stem  # noqa: E402
from dinov3_carpet_probe.palette_train.src.model import PalettePredictor  # noqa: E402
from dinov3_carpet_probe.palette_train.src.transforms import build_transform  # noqa: E402
from dinov3_carpet_probe.src.io_utils import ensure_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resolve_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No JPG files found in {path}")
    return images


def _clean_path(value: str) -> str:
    return value.strip().strip('"').strip("'")


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(_clean_path(str(args.checkpoint)))
    input_path = Path(_clean_path(str(args.input)))
    output_dir = ensure_dir(Path(_clean_path(str(args.output_dir))))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    model = PalettePredictor(cfg).to(cfg["device"])
    load_checkpoint(checkpoint_path, model=model, optimizer=None, scheduler=None, scaler=None, restore_rng=False)
    transform = build_transform(cfg, training=False)

    model.eval()
    with torch.no_grad():
        for path in _resolve_inputs(input_path):
            image, meta = transform(path)
            outputs = model(
                image.unsqueeze(0).to(cfg["device"]),
                meta["raw_tensor"].unsqueeze(0).to(cfg["device"]),
                meta["valid_mask"].unsqueeze(0).to(cfg["device"]),
            )
            decoded = decode_palette_prediction(
                stem=path.stem,
                pred_rgb=outputs["pred_rgb"][0].detach().cpu(),
                presence_logits=outputs["presence_logits"][0].detach().cpu(),
                count_logits=outputs["count_logits"][0].detach().cpu(),
            )
            out_path = output_path_for_stem(output_dir, path.stem)
            write_json(out_path, decoded.to_dict())
            print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
