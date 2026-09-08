"""CSN-V3 PyTorch dataset — dual input (BW / magenta), D4 aug, center supervision."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from csn_v3.data.augment import apply_dihedral_csn_sample
from csn_v3.data.input import (
    build_where_target_magenta,
    load_rgb_guide,
    onehot_to_line_mask,
    rgb_uint8_to_onehot,
)
from csn_v3.data.multiscale import derive_context_crop, letterbox_full_bw, normalize_crop_coords
from csn_v3.data.targets import build_sample_targets, load_gray_bmp
from csn_v3.masks import build_center_supervision_mask


class CSNV3TrainDataset(Dataset):
    ARTIFACTS = (
        "source_bw.bmp",
        "target_semantic.bmp",
        "shade_mask.bmp",
        "shade_level_id.bmp",
        "transition_mask.bmp",
        "black_lock.bmp",
        "valid_mask.bmp",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        dataset_root: str | Path | None = None,
        input_mode: str = "bw",
        magenta_guide_dir: str | Path | None = None,
        crop_size: int = 512,
        context_size: int = 1024,
        global_long_side: int = 1024,
        center_halo: int = 64,
        allow_padded_context: bool = True,
        augment_dihedral: bool = False,
        eval_mode: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root) if dataset_root else self.manifest_path.parent.parent
        self.input_mode = input_mode
        self.magenta_guide_dir = Path(magenta_guide_dir) if magenta_guide_dir else None
        self.crop_size = crop_size
        self.context_size = context_size
        self.global_long_side = global_long_side
        self.center_halo = center_halo
        self.allow_padded_context = allow_padded_context
        self.augment_dihedral = augment_dihedral and not eval_mode
        self.eval_mode = eval_mode
        self.records: list[dict] = []
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def _load_magenta_rgb(self, rec: dict, local_dir: Path, full_dir: Path) -> np.ndarray:
        if rec.get("magenta_guide_path"):
            return load_rgb_guide(rec["magenta_guide_path"])
        oracle = local_dir / "source_oracle.bmp"
        if oracle.exists():
            from PIL import Image
            return np.asarray(Image.open(oracle).convert("RGB"), dtype=np.uint8)
        guide_path = rec.get("guide_path")
        if guide_path and Path(guide_path).exists():
            return load_rgb_guide(guide_path)
        if self.magenta_guide_dir:
            sid = rec.get("sample_id", "")
            for p in self.magenta_guide_dir.glob("*.bmp"):
                if sid in p.stem:
                    return load_rgb_guide(str(p))
        raise FileNotFoundError(f"No magenta guide for {rec.get('sample_id')}")

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        local_dir = Path(rec["local_artifact_dir"])
        full_dir = Path(rec["full_artifact_dir"])

        local_bw = load_gray_bmp(local_dir / "source_bw.bmp")
        full_bw = load_gray_bmp(full_dir / "source_bw.bmp")
        x, y = int(rec["x"]), int(rec["y"])
        w, h = int(rec.get("w", self.crop_size)), int(rec.get("h", self.crop_size))
        fh, fw = full_bw.shape[:2]

        if local_bw.shape[0] != self.crop_size:
            local_bw = local_bw[: self.crop_size, : self.crop_size]

        context_bw = derive_context_crop(full_bw, x, y, self.crop_size, self.context_size)
        full_thumb, letterbox_meta = letterbox_full_bw(full_bw, self.global_long_side)

        targets = {}
        for name in self.ARTIFACTS[1:]:
            arr = load_gray_bmp(local_dir / name)
            if arr.shape != local_bw.shape:
                arr = arr[: local_bw.shape[0], : local_bw.shape[1]]
            targets[name.replace(".bmp", "")] = arr

        tgt = build_sample_targets(
            targets["target_semantic"],
            targets["shade_mask"],
            targets["shade_level_id"],
            targets["transition_mask"],
            targets["black_lock"],
            targets["valid_mask"],
        )

        coords = normalize_crop_coords(x, y, w, h, fw, fh)
        center = build_center_supervision_mask(self.crop_size, self.crop_size, self.center_halo)

        sample: dict = {
            "local_bw": torch.from_numpy(local_bw.astype(np.float32) / 255.0).unsqueeze(0),
            "context_bw": torch.from_numpy(context_bw.astype(np.float32) / 255.0).unsqueeze(0),
            "full_bw": torch.from_numpy(full_thumb.astype(np.float32) / 255.0).unsqueeze(0),
            "crop_coords_norm": torch.from_numpy(coords),
            "letterbox_meta": letterbox_meta,
            "source_path": str(local_dir / "source_bw.bmp"),
            "full_artifact_dir": str(full_dir),
            "center_supervision": center,
            "transform_id": int(rec.get("transform_id", rec.get("dihedral_k", 0))),
            **tgt,
        }

        if self.input_mode == "magenta":
            rgb = self._load_magenta_rgb(rec, local_dir, full_dir)
            if rgb.shape[0] != self.crop_size:
                rgb = rgb[: self.crop_size, : self.crop_size]
            onehot = rgb_uint8_to_onehot(rgb)
            sample["local_onehot"] = torch.from_numpy(onehot)
            sample["local_rgb"] = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            sample["where_target"] = build_where_target_magenta(sample["local_onehot"])

        if self.augment_dihedral and sample["transform_id"] == 0:
            k = random.randint(0, 7)
            sample = apply_dihedral_csn_sample(sample, k)

        return sample
