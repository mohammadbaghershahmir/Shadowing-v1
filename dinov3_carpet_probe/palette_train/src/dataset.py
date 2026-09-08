"""Dataset discovery, validation, manifests, and palette batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from dinov3_carpet_probe.src.io_utils import ensure_dir, read_json, write_json


VALID_IMAGE_SUFFIXES = (".jpg", ".jpeg")


@dataclass(frozen=True)
class PaletteSampleRecord:
    stem: str
    sample_id: int
    image_path: str
    palette_path: str
    num_colors: int
    rgb: list[list[int]]
    hex: list[str]


@dataclass
class DatasetValidationReport:
    image_dir: str
    palette_dir: str
    image_count: int
    palette_count: int
    paired_count: int
    missing_images: list[str]
    missing_annotations: list[str]
    duplicate_palette_stems: list[str]
    warnings: list[str]
    errors: list[str]
    num_colors_stats: dict[str, Any]
    max_num_colors: int
    excluded_over_cap: list[str]
    excluded_duplicate_rgb: list[str]
    trainable_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_IMAGE_SUFFIXES


def _sorted_unique_stems(paths: Iterable[Path]) -> tuple[dict[str, Path], list[str]]:
    items: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(paths):
        stem_key = path.stem.lower()
        if stem_key in items:
            duplicates.add(path.stem)
            continue
        items[stem_key] = path.resolve()
    return items, sorted(duplicates)


def _rgb_to_hex(rgb: list[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _validate_rgb_triplet(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    out: list[int] = []
    for channel in value:
        if not isinstance(channel, int):
            return None
        if channel < 0 or channel > 255:
            return None
        out.append(channel)
    return out


def discover_pairs(image_dir: Path | str, palette_dir: Path | str) -> dict[str, Any]:
    image_dir = Path(image_dir)
    palette_dir = Path(palette_dir)
    image_paths = [p for p in image_dir.iterdir() if p.is_file() and _is_image_file(p)] if image_dir.exists() else []
    palette_paths = [p for p in palette_dir.glob("*.json")] if palette_dir.exists() else []

    image_map, image_duplicates = _sorted_unique_stems(image_paths)
    palette_map, palette_duplicates = _sorted_unique_stems(palette_paths)

    image_stems = set(image_map)
    palette_stems = set(palette_map)
    shared = sorted(image_stems & palette_stems)
    missing_images = sorted(palette_map[s].name for s in (palette_stems - image_stems))
    missing_annotations = sorted(image_map[s].name for s in (image_stems - palette_stems))

    pairs = [(image_map[s], palette_map[s]) for s in shared]
    return {
        "pairs": pairs,
        "image_count": len(image_map),
        "palette_count": len(palette_map),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "duplicate_image_stems": image_duplicates,
        "duplicate_palette_stems": palette_duplicates,
    }


def validate_palette_annotation(stem: str, payload: dict[str, Any]) -> tuple[PaletteSampleRecord | None, list[str], list[str], bool]:
    errors: list[str] = []
    warnings: list[str] = []
    has_duplicate_rgb = False

    if not stem.isdigit():
        errors.append(f"{stem}: filename stem must be numeric.")
        return None, warnings, errors, has_duplicate_rgb

    expected_sample_id = int(stem)
    sample_id = payload.get("sample_id")
    num_colors = payload.get("num_colors")
    rgb = payload.get("rgb")
    hex_values = payload.get("hex")

    if sample_id != expected_sample_id:
        errors.append(f"{stem}: sample_id={sample_id!r} does not match stem {expected_sample_id}.")
    if not isinstance(num_colors, int):
        errors.append(f"{stem}: num_colors must be an integer.")
    if not isinstance(rgb, list):
        errors.append(f"{stem}: rgb must be a list.")
    if not isinstance(hex_values, list):
        errors.append(f"{stem}: hex must be a list.")
    if errors:
        return None, warnings, errors, has_duplicate_rgb

    if num_colors != len(rgb) or num_colors != len(hex_values):
        errors.append(
            f"{stem}: num_colors={num_colors} must equal len(rgb)={len(rgb)} and len(hex)={len(hex_values)}."
        )

    rgb_out: list[list[int]] = []
    seen_rgb: set[tuple[int, int, int]] = set()
    for idx, color in enumerate(rgb):
        triplet = _validate_rgb_triplet(color)
        if triplet is None:
            errors.append(f"{stem}: rgb[{idx}] must be three integers in [0, 255].")
            continue
        rgb_out.append(triplet)
        key = tuple(triplet)
        if key in seen_rgb:
            has_duplicate_rgb = True
            warnings.append(f"{stem}: duplicate RGB color at index {idx}: {triplet}.")
            errors.append(f"{stem}: duplicate RGB color at index {idx}: {triplet}.")
        seen_rgb.add(key)

    hex_out: list[str] = []
    for idx, value in enumerate(hex_values):
        if not isinstance(value, str):
            errors.append(f"{stem}: hex[{idx}] must be a string.")
            continue
        normalized = value.upper()
        hex_out.append(normalized)
        if idx < len(rgb_out) and normalized != _rgb_to_hex(rgb_out[idx]):
            errors.append(
                f"{stem}: hex[{idx}]={normalized} does not match rgb[{idx}]={rgb_out[idx]}."
            )

    if errors:
        return None, warnings, errors, has_duplicate_rgb

    return (
        PaletteSampleRecord(
            stem=stem,
            sample_id=expected_sample_id,
            image_path="",
            palette_path="",
            num_colors=num_colors,
            rgb=rgb_out,
            hex=hex_out,
        ),
        warnings,
        errors,
        has_duplicate_rgb,
    )


def summarize_num_colors(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "percentiles": {},
            "histogram": {},
        }
    arr = np.asarray(values, dtype=np.float64)
    percentiles = {
        "p0": float(np.percentile(arr, 0)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p100": float(np.percentile(arr, 100)),
    }
    hist: dict[str, int] = {}
    for value in sorted(set(values)):
        hist[str(value)] = int(sum(v == value for v in values))
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "percentiles": percentiles,
        "histogram": hist,
    }


def validate_dataset(
    image_dir: Path | str,
    palette_dir: Path | str,
    *,
    max_num_colors_override: int | None = None,
    show_progress: bool = False,
) -> tuple[list[PaletteSampleRecord], DatasetValidationReport]:
    image_dir = Path(image_dir)
    palette_dir = Path(palette_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not palette_dir.exists():
        raise FileNotFoundError(f"Palette directory not found: {palette_dir}")

    discovered = discover_pairs(image_dir, palette_dir)
    warnings: list[str] = []
    errors: list[str] = []
    records: list[PaletteSampleRecord] = []
    num_colors: list[int] = []
    excluded_over_cap: list[str] = []
    excluded_duplicate_rgb: list[str] = []

    pair_iter = discovered["pairs"]
    if show_progress:
        pair_iter = tqdm(
            discovered["pairs"],
            desc="validate_dataset",
            unit="pair",
            dynamic_ncols=True,
        )

    for image_path, palette_path in pair_iter:
        payload = read_json(palette_path)
        record, sample_warnings, sample_errors, has_duplicate_rgb = validate_palette_annotation(image_path.stem, payload)
        warnings.extend(sample_warnings)
        errors.extend(sample_errors)
        if has_duplicate_rgb:
            excluded_duplicate_rgb.append(image_path.stem)
        if record is None:
            continue
        record = PaletteSampleRecord(
            stem=record.stem,
            sample_id=record.sample_id,
            image_path=str(image_path),
            palette_path=str(palette_path),
            num_colors=record.num_colors,
            rgb=record.rgb,
            hex=record.hex,
        )
        num_colors.append(record.num_colors)
        if max_num_colors_override is not None and record.num_colors > max_num_colors_override:
            excluded_over_cap.append(record.stem)
        else:
            records.append(record)
        if show_progress and hasattr(pair_iter, "set_postfix"):
            pair_iter.set_postfix(
                checked=len(num_colors),
                kept=len(records),
                errors=len(errors),
                warnings=len(warnings),
            )

    if discovered["missing_images"]:
        errors.append(f"Missing images for {len(discovered['missing_images'])} palette files.")
    if discovered["missing_annotations"]:
        errors.append(f"Missing annotations for {len(discovered['missing_annotations'])} image files.")
    if discovered["duplicate_image_stems"]:
        errors.append(f"Duplicate image stems found: {discovered['duplicate_image_stems']}.")
    if discovered["duplicate_palette_stems"]:
        errors.append(f"Duplicate palette stems found: {discovered['duplicate_palette_stems']}.")

    max_num_colors = max((record.num_colors for record in records), default=0)
    if max_num_colors_override is not None:
        max_num_colors = max_num_colors_override

    report = DatasetValidationReport(
        image_dir=str(image_dir),
        palette_dir=str(palette_dir),
        image_count=discovered["image_count"],
        palette_count=discovered["palette_count"],
        paired_count=len(discovered["pairs"]),
        missing_images=discovered["missing_images"],
        missing_annotations=discovered["missing_annotations"],
        duplicate_palette_stems=discovered["duplicate_palette_stems"],
        warnings=warnings,
        errors=errors,
        num_colors_stats=summarize_num_colors(num_colors),
        max_num_colors=max_num_colors,
        excluded_over_cap=sorted(excluded_over_cap),
        excluded_duplicate_rgb=sorted(set(excluded_duplicate_rgb)),
        trainable_count=len(records),
    )
    return records, report


def _manifest_identity(records: list[PaletteSampleRecord]) -> str:
    payload = "\n".join(
        f"{r.stem}|{r.image_path}|{r.palette_path}|{r.num_colors}|{','.join(map(str, sum(r.rgb, [])))}"
        for r in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_manifest(
    records: list[PaletteSampleRecord],
    *,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = np.random.default_rng(split_seed)
    stems = np.array([r.stem for r in records], dtype=object)
    perm = rng.permutation(len(stems))
    stems = stems[perm]

    n_total = len(stems)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = min(n_train, n_total)
    n_val = min(n_val, max(0, n_total - n_train))
    n_test = max(0, n_total - n_train - n_val)

    manifest = {
        "split_seed": split_seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "dataset_id": _manifest_identity(records),
        "train": [str(x) for x in stems[:n_train]],
        "val": [str(x) for x in stems[n_train : n_train + n_val]],
        "test": [str(x) for x in stems[n_train + n_val : n_train + n_val + n_test]],
    }
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def load_or_create_split_manifest(
    records: list[PaletteSampleRecord],
    *,
    manifest_dir: Path | str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, Any], Path]:
    manifest_dir = ensure_dir(manifest_dir)
    manifest_path = manifest_dir / f"split_seed{split_seed}.json"
    dataset_id = _manifest_identity(records)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("dataset_id") == dataset_id:
            return manifest, manifest_path
    manifest = build_split_manifest(
        records,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        manifest_path=manifest_path,
    )
    return manifest, manifest_path


def select_split(records: list[PaletteSampleRecord], manifest: dict[str, Any], split: str) -> list[PaletteSampleRecord]:
    valid = {"train", "val", "test"}
    if split not in valid:
        raise KeyError(f"Unknown split: {split}")
    stems = set(manifest[split])
    selected = [record for record in records if record.stem in stems]
    selected.sort(key=lambda r: r.stem)
    return selected


class PaletteTrainingDataset(torch.utils.data.Dataset):
    """Torch dataset backed by validated records."""

    def __init__(self, records: list[PaletteSampleRecord], image_transform: Any) -> None:
        self.records = list(records)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_tensor, meta = self.image_transform(record.image_path)
        rgb = torch.tensor(record.rgb, dtype=torch.float32) / 255.0
        rgb_uint8 = torch.tensor(record.rgb, dtype=torch.uint8)
        return {
            "stem": record.stem,
            "sample_id": record.sample_id,
            "image": image_tensor,
            "image_meta": meta.to_dict() if hasattr(meta, "to_dict") else meta,
            "raw_image": meta["raw_tensor"],
            "valid_mask": meta["valid_mask"],
            "target_rgb": rgb,
            "target_rgb_uint8": rgb_uint8,
            "target_hex": list(record.hex),
            "num_colors": record.num_colors,
        }


def collate_palette_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch.")
    max_colors = max(item["num_colors"] for item in batch)
    images = torch.stack([item["image"] for item in batch], dim=0)
    raw_images = torch.stack([item.get("raw_image", item["image"]) for item in batch], dim=0)
    valid_masks = torch.stack(
        [item.get("valid_mask", torch.ones(1, item["image"].shape[1], item["image"].shape[2])) for item in batch],
        dim=0,
    )
    target_rgb = torch.zeros(len(batch), max_colors, 3, dtype=torch.float32)
    target_mask = torch.zeros(len(batch), max_colors, dtype=torch.bool)
    target_rgb_uint8 = torch.zeros(len(batch), max_colors, 3, dtype=torch.uint8)
    for idx, item in enumerate(batch):
        count = item["num_colors"]
        target_rgb[idx, :count] = item["target_rgb"]
        target_mask[idx, :count] = True
        target_rgb_uint8[idx, :count] = item["target_rgb_uint8"]
    return {
        "stems": [item["stem"] for item in batch],
        "sample_ids": torch.tensor([item["sample_id"] for item in batch], dtype=torch.int64),
        "images": images,
        "raw_images": raw_images,
        "valid_masks": valid_masks,
        "image_meta": [item["image_meta"] for item in batch],
        "target_rgb": target_rgb,
        "target_rgb_uint8": target_rgb_uint8,
        "target_mask": target_mask,
        "target_counts": torch.tensor([item["num_colors"] for item in batch], dtype=torch.int64),
        "target_hex": [item["target_hex"] for item in batch],
    }
