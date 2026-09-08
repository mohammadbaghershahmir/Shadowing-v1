"""Path helpers, JSON/NPY IO, config loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROBE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROBE_ROOT.parent
DEFAULT_WEIGHTS_DIR = PROBE_ROOT / "weights"
DEFAULT_OUTPUT_ROOT = PROBE_ROOT / "outputs"
CONFIGS_DIR = PROBE_ROOT / "configs"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "models.yaml")


def load_experiment_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "experiment.yaml"
    return load_yaml(cfg_path)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_input_images(input_path: Path | str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG file, got: {path}")
        return [path.resolve()]
    if path.is_dir():
        images = sorted(path.glob("*.png")) + sorted(path.glob("*.PNG"))
        # de-dup case-insensitive on Windows
        seen: set[str] = set()
        unique: list[Path] = []
        for p in images:
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(p.resolve())
        if not unique:
            raise FileNotFoundError(f"No PNG files found in directory: {path}")
        if len(unique) > 2:
            unique = unique[:2]
        return unique
    raise FileNotFoundError(f"Input path not found: {path}")


def save_npy(path: Path | str, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)
