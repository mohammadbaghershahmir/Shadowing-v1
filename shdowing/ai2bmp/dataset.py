"""PyTorch dataset for AI2BMP synthetic crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from shdowing.ai2bmp.constants import DEFAULT_TASK_AVAILABILITY
from shdowing.ai2bmp.io_utils import read_json, read_jsonl, sha256_file


class AI2BMPSyntheticDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        *,
        output_root: Path | str,
        crop_size: int = 512,
        verify_hashes: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.crop_size = int(crop_size)
        self.verify_hashes = bool(verify_hashes)
        self.rows = read_jsonl(manifest_path)
        self.rows = [r for r in self.rows if int(r.get("crop_size", crop_size)) == self.crop_size]

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, rel_path: str) -> np.ndarray:
        path = self.output_root / rel_path.replace("/", "\\")
        if self.verify_hashes:
            _ = sha256_file(path)
        return np.array(Image.open(path).convert("RGB"))

    def _load_mask(self, rel_path: str) -> np.ndarray:
        path = self.output_root / rel_path.replace("/", "\\")
        return np.array(Image.open(path).convert("L"))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x, y = int(row["x"]), int(row["y"])
        size = int(row["crop_size"])

        aligned = self._load_rgb(row["aligned_input_path"])
        target_rgb = self._load_rgb(row["target_rgb_path"])
        h, w = aligned.shape[:2]
        if aligned.shape[0] != target_rgb.shape[0] or aligned.shape[1] != target_rgb.shape[1]:
            raise ValueError("Aligned input and target RGB dimension mismatch")

        crop_in = aligned[y : y + size, x : x + size]
        crop_tgt = target_rgb[y : y + size, x : x + size]
        if crop_in.shape[0] != size or crop_in.shape[1] != size:
            raise ValueError(f"Crop shape {crop_in.shape} != requested {size}")

        index_map = np.load(self.output_root / row["original_index_map_path"])
        used_map = np.load(self.output_root / row["contiguous_used_index_map_path"])
        unique_map = np.load(self.output_root / row["contiguous_unique_rgb_map_path"])
        palette = read_json(self.output_root / row["palette_used_indices_path"])

        idx_crop = index_map[y : y + size, x : x + size]
        used_crop = used_map[y : y + size, x : x + size]
        unique_crop = unique_map[y : y + size, x : x + size]

        boundary = self._load_mask(row["boundary_mask_path"])[y : y + size, x : x + size]
        thin = self._load_mask(row["thin_structure_mask_path"])[y : y + size, x : x + size]
        valid_sup = self._load_mask(row["valid_supervision_mask_path"])[y : y + size, x : x + size]

        margin = int(row.get("valid_margin", 64))
        central = np.zeros((size, size), dtype=bool)
        if size > 2 * margin:
            central[margin : size - margin, margin : size - margin] = True

        input_f = crop_in.astype(np.float32) / 255.0
        onehot = np.zeros((3, size, size), dtype=np.float32)
        # semantic one-hot from target rgb exact colors - use palette entries
        for ci, rgb in enumerate([(0, 0, 0), (255, 255, 255), (255, 0, 255)]):
            onehot[ci] = np.all(crop_in == np.array(rgb, dtype=np.uint8), axis=-1).astype(np.float32)

        k_index = int(row.get("k_index", len(palette.get("entries", []))))
        used_palette_rgb = np.zeros((k_index, 3), dtype=np.float32)
        for i, entry in enumerate(palette.get("entries", [])[:k_index]):
            used_palette_rgb[i] = np.array(entry["rgb"], dtype=np.float32) / 255.0

        k_rgb = int(row.get("k_rgb", k_index))
        target_palette_rgb = used_palette_rgb[:k_rgb]

        return {
            "input_rgb": torch.from_numpy(input_f.transpose(2, 0, 1)),
            "input_onehot": torch.from_numpy(onehot),
            "target_rgb": torch.from_numpy(crop_tgt.astype(np.uint8).transpose(2, 0, 1)),
            "target_label": torch.from_numpy(used_crop.astype(np.int64)),
            "target_original_index": torch.from_numpy(idx_crop.astype(np.int64)),
            "target_used_class_id": torch.from_numpy(used_crop.astype(np.int64)),
            "target_unique_rgb_id": torch.from_numpy(unique_crop.astype(np.int64)),
            "target_palette_rgb": torch.from_numpy(target_palette_rgb),
            "target_used_index_palette_rgb": torch.from_numpy(used_palette_rgb),
            "palette_mask": torch.ones(k_index, dtype=torch.bool),
            "region_boundary": torch.from_numpy((boundary > 0).astype(np.float32)[None]),
            "thin_structure": torch.from_numpy((thin > 0).astype(np.float32)[None]),
            "valid_supervision": torch.from_numpy((valid_sup > 0).astype(np.float32)[None]),
            "candidate_mask": torch.zeros(size, size, dtype=torch.bool),
            "image_valid_mask": torch.ones(size, size, dtype=torch.bool),
            "central_valid_mask": torch.from_numpy(central),
            "crop_box_normalized": torch.tensor(row.get("normalized_crop_box", [0, 0, 1, 1]), dtype=torch.float32),
            "source_id": row.get("source_id", ""),
            "sample_id": row.get("sample_id", ""),
            "sampling_strategy": row.get("sampling_strategy", ""),
            "task_availability": row.get("task_availability", DEFAULT_TASK_AVAILABILITY),
            "metadata": row,
        }


def collate_ai2bmp_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    max_k = max(item["target_used_index_palette_rgb"].shape[0] for item in batch)
    max_k_rgb = max(item["target_palette_rgb"].shape[0] for item in batch)
    max_k = max(max_k, max_k_rgb)

    padded_index = []
    padded_rgb = []
    masks = []
    for item in batch:
        k = item["target_used_index_palette_rgb"].shape[0]
        pad = max_k - k
        pal = item["target_used_index_palette_rgb"]
        if pad:
            pal = torch.cat([pal, torch.zeros(pad, 3)], dim=0)
        padded_index.append(pal)
        kr = item["target_palette_rgb"].shape[0]
        pr = item["target_palette_rgb"]
        if max_k - kr:
            pr = torch.cat([pr, torch.zeros(max_k - kr, 3)], dim=0)
        padded_rgb.append(pr)
        m = item["palette_mask"]
        if pad:
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])
        masks.append(m)

    out: dict[str, Any] = {}
    tensor_keys = [
        "input_rgb",
        "input_onehot",
        "target_rgb",
        "target_label",
        "target_original_index",
        "target_used_class_id",
        "target_unique_rgb_id",
        "region_boundary",
        "thin_structure",
        "valid_supervision",
        "candidate_mask",
        "image_valid_mask",
        "central_valid_mask",
        "crop_box_normalized",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["target_used_index_palette_rgb"] = torch.stack(padded_index, dim=0)
    out["target_palette_rgb"] = torch.stack(padded_rgb, dim=0)
    out["palette_mask"] = torch.stack(masks, dim=0)
    out["source_id"] = [item["source_id"] for item in batch]
    out["sample_id"] = [item["sample_id"] for item in batch]
    out["sampling_strategy"] = [item["sampling_strategy"] for item in batch]
    out["task_availability"] = [item["task_availability"] for item in batch]
    out["metadata"] = [item["metadata"] for item in batch]
    return out
