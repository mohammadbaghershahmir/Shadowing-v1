"""Atomic file I/O helpers for manifests, reports, and JSON."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create output directory; fail if non-empty unless overwrite is set."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        existing = [p for p in output_dir.iterdir()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite to replace generated files inside this directory."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text atomically via a temporary file in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    text += "\n"
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for record in records:
        lines.append(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
    text = "\n".join(lines)
    if lines:
        text += "\n"
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
    return records


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def clear_generated_outputs(output_dir: Path, filenames: Sequence[str]) -> None:
    """Remove only known generated files inside output_dir (never source data)."""
    output_dir = Path(output_dir)
    for name in filenames:
        target = output_dir / name
        if target.is_file():
            target.unlink()
        elif target.is_dir() and name == "previews":
            for child in sorted(target.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if target.exists():
                target.rmdir()
