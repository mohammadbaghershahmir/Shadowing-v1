"""Path resolution and hashing helpers for CSN-V4 datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def resolve_dataset_path(path: str | Path, dataset_root: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(dataset_root) / p).resolve()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing file for fingerprint: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_manifest(path: str | Path) -> str:
    p = Path(path)
    return sha256_bytes(p.read_bytes())


def rel_path_if_under(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
