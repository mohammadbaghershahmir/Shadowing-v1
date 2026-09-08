"""Discover and pair input/target images by filename stem."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from shadow_dataset.constants import SUPPORTED_EXTENSIONS
from shadow_dataset.image_io import is_supported_image
from shadow_dataset.types import DiscoveredFile, PairPaths, path_to_str

LOGGER = logging.getLogger(__name__)


def discover_images(directory: Path | str) -> list[DiscoveredFile]:
    """List supported image files in a directory (non-recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: list[DiscoveredFile] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name.casefold()):
        if not entry.is_file():
            continue
        if not is_supported_image(entry):
            if entry.suffix.lower() in {".jpg", ".jpeg", ".webp", ".gif"}:
                LOGGER.warning("Skipping unsupported lossy/format file: %s", entry)
            continue
        files.append(DiscoveredFile(stem=entry.stem, path=entry))
    return files


def find_duplicate_stems(
    files: list[DiscoveredFile],
) -> dict[str, list[str]]:
    """Return stem -> list of paths for stems that appear more than once."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for item in files:
        by_stem[item.stem].append(item.path)

    duplicates: dict[str, list[str]] = {}
    for stem, paths in by_stem.items():
        if len(paths) > 1:
            duplicates[stem] = [path_to_str(p) for p in sorted(paths, key=str)]
    return duplicates


def pair_by_stem(
    input_files: list[DiscoveredFile],
    target_files: list[DiscoveredFile],
) -> tuple[list[PairPaths], list[dict[str, str]], list[dict[str, str]]]:
    """Pair inputs and targets by stem.

    Returns
    -------
    pairs :
        Successfully matched stem pairs (unique stems only).
    missing :
        Rows describing missing counterparts.
    duplicate_rows :
        Rows describing duplicate stems (these stems are excluded from pairs).
    """
    input_dups = find_duplicate_stems(input_files)
    target_dups = find_duplicate_stems(target_files)

    duplicate_rows: list[dict[str, str]] = []
    for stem, paths in sorted(input_dups.items()):
        duplicate_rows.append(
            {
                "stem": stem,
                "side": "input",
                "paths": ";".join(paths),
                "count": str(len(paths)),
            }
        )
    for stem, paths in sorted(target_dups.items()):
        duplicate_rows.append(
            {
                "stem": stem,
                "side": "target",
                "paths": ";".join(paths),
                "count": str(len(paths)),
            }
        )

    blocked_stems = set(input_dups) | set(target_dups)

    input_by_stem = {
        f.stem: f.path for f in input_files if f.stem not in blocked_stems
    }
    target_by_stem = {
        f.stem: f.path for f in target_files if f.stem not in blocked_stems
    }

    all_stems = sorted(set(input_by_stem) | set(target_by_stem))
    pairs: list[PairPaths] = []
    missing: list[dict[str, str]] = []

    for stem in all_stems:
        in_path = input_by_stem.get(stem)
        tg_path = target_by_stem.get(stem)
        if in_path is not None and tg_path is not None:
            pairs.append(
                PairPaths(stem=stem, input_path=in_path, target_path=tg_path)
            )
        elif in_path is not None:
            missing.append(
                {
                    "stem": stem,
                    "side": "target",
                    "present_path": path_to_str(in_path),
                    "missing_side": "target",
                }
            )
        else:
            assert tg_path is not None
            missing.append(
                {
                    "stem": stem,
                    "side": "input",
                    "present_path": path_to_str(tg_path),
                    "missing_side": "input",
                }
            )

    # Duplicate stems are also reported as missing from the usable set.
    for stem in sorted(blocked_stems):
        if stem in input_dups and stem not in target_dups and stem in {
            f.stem for f in target_files
        }:
            # Unique target exists but input duplicates block pairing.
            pass
        if stem not in {p.stem for p in pairs}:
            # Already captured in duplicate_rows; ensure missing notes if needed.
            continue

    return pairs, missing, duplicate_rows


def supported_extensions_help() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))
