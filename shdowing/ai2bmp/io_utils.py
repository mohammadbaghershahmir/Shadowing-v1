"""IO helpers: atomic writes, path sanitization, relative paths."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def clean_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip().strip('"').strip("'")
    return Path(text) if text else None


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=_json_default))


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix + ".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix + ".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_csv(path: Path | str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines: list[str] = []
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write_text(path, buf.getvalue())


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    lines = [json.dumps(row, ensure_ascii=False, default=_json_default) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rel_path(path: Path | str, root: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve()).replace("\\", "/")


def save_rgb_png(path: Path | str, rgb: np.ndarray) -> None:
    """Save uint8 RGB without optimization that alters pixels."""
    path = Path(path)
    ensure_dir(path.parent)
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB, got {rgb.dtype}")
    img = Image.fromarray(rgb, mode="RGB")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".png.tmp")
    os.close(fd)
    try:
        img.save(tmp, format="PNG", compress_level=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_index_bmp(path: Path | str, mask: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    arr = mask.astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path, format="BMP")


def save_mask_bmp(path: Path | str, mask: np.ndarray, *, on: int = 255, off: int = 0) -> None:
    arr = np.where(mask.astype(bool), on, off).astype(np.uint8)
    save_index_bmp(path, arr)


def output_dir_nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file():
            return True
    return False
