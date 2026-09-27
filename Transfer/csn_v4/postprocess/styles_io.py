"""Load approved decorative style examples / designer rules (never invent geometry)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from csn_v4.constants import ALLOWED_OUTPUT_GRAY, GRAY_BLACK
from csn_v4.postprocess.constraints import gray_to_class
from csn_v4.postprocess.regions import MotifRegion


@dataclass
class MiakhiAssets:
    available: bool
    reason: str
    rules: dict = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    pattern_templates: list[np.ndarray] = field(default_factory=list)  # class maps


def default_styles_root() -> Path:
    return Path(__file__).resolve().parents[2] / "styles"


def load_miakhi_assets(styles_root: str | Path | None = None) -> MiakhiAssets:
    root = Path(styles_root) if styles_root else default_styles_root()
    miakhi = root / "miakhi"
    if not miakhi.is_dir():
        return MiakhiAssets(
            available=False,
            reason=(
                f"miakhi unavailable: missing directory {miakhi}. "
                "Add approved before/after examples or rules.json "
                "(see styles/miakhi/README.md). Cleanup is not miakhi."
            ),
        )

    rules_path = miakhi / "rules.json"
    rules: dict = {}
    if rules_path.is_file():
        rules = json.loads(rules_path.read_text(encoding="utf-8"))

    examples_dir = miakhi / "examples"
    examples: list[dict] = []
    templates: list[np.ndarray] = []
    if examples_dir.is_dir():
        for sub in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
            after = _find_gray(sub, ("after.bmp", "after.png", "y.bmp", "target.bmp"))
            before = _find_gray(sub, ("before.bmp", "before.png", "x.bmp", "source.bmp"))
            if after is None:
                continue
            after_arr = _load_gray(after)
            examples.append({
                "id": sub.name,
                "before": str(before) if before else "",
                "after": str(after),
                "shape": list(after_arr.shape),
            })
            templates.append(gray_to_class(after_arr))

    # Explicit pattern tiles (indexed gray patches approved by designer)
    patterns_dir = miakhi / "patterns"
    if patterns_dir.is_dir():
        for p in sorted(patterns_dir.glob("*.bmp")) + sorted(patterns_dir.glob("*.png")):
            arr = _load_gray(p)
            templates.append(gray_to_class(arr))
            examples.append({"id": p.stem, "after": str(p), "shape": list(arr.shape), "kind": "pattern"})

    if not templates and not rules:
        return MiakhiAssets(
            available=False,
            reason=(
                f"miakhi unavailable: {miakhi} exists but has no rules.json and no "
                "approved examples/patterns. Do not invent miakhi geometry from the name."
            ),
        )

    reason_parts = []
    if rules:
        reason_parts.append(f"rules.json keys={sorted(rules.keys())}")
    if templates:
        reason_parts.append(f"{len(templates)} template(s)")
    return MiakhiAssets(
        available=True,
        reason="; ".join(reason_parts),
        rules=rules,
        examples=examples,
        pattern_templates=templates,
    )


def _find_gray(folder: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = folder / n
        if p.is_file():
            return p
    return None


def _load_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    bad = set(int(x) for x in np.unique(arr)) - set(ALLOWED_OUTPUT_GRAY)
    if bad:
        raise ValueError(f"{path} has non-palette values {sorted(bad)}")
    return arr


def _d4_variants(tile: np.ndarray) -> list[np.ndarray]:
    """Discrete D4 orientations via nearest-neighbor (rot90 / flip)."""
    out = []
    for k in range(4):
        r = np.rot90(tile, k)
        out.append(r)
        out.append(np.flip(r, axis=1))
    return out


def propose_miakhi_for_regions(
    source_bw: np.ndarray,
    eligible: np.ndarray,
    regions: list[MotifRegion],
    assets: MiakhiAssets,
    *,
    raw_gray: np.ndarray,
) -> np.ndarray | None:
    """Build a class-proposal map aligned per motif — never one global stamp.

    Uses approved templates with discrete D4 alignment ranked by overlap with
    the motif mask / raw prediction. Regions without a reliable fit are left
    as -1 (no proposal → decide.py leaves them unchanged).
    """
    if not assets.available or not assets.pattern_templates:
        # Rules-only: optional class remap table, still no invented geometry.
        h, w = raw_gray.shape[:2]
        prop = np.full((h, w), -1, dtype=np.int64)
        remap = (assets.rules or {}).get("class_remap")
        if isinstance(remap, dict):
            raw_cls = gray_to_class(raw_gray)
            for src_s, dst_s in remap.items():
                try:
                    s_cls = gray_to_class(np.array([[int(src_s)]], dtype=np.uint8))[0, 0]
                    d_cls = gray_to_class(np.array([[int(dst_s)]], dtype=np.uint8))[0, 0]
                except Exception:
                    continue
                prop[eligible & (raw_cls == s_cls)] = int(d_cls)
            return prop
        return None

    h, w = raw_gray.shape[:2]
    proposal = np.full((h, w), -1, dtype=np.int64)
    raw_cls = gray_to_class(raw_gray)
    src = np.asarray(source_bw)
    if src.ndim == 3:
        src = src[..., 0]

    for region in regions:
        if not region.reliable:
            continue
        y0, x0, y1, x1 = region.bbox
        rh, rw = y1 - y0, x1 - x0
        if rh < 2 or rw < 2:
            continue
        best_score = -1.0
        best_patch: np.ndarray | None = None
        region_local = region.mask[y0:y1, x0:x1]
        raw_local = raw_cls[y0:y1, x0:x1]

        for tmpl in assets.pattern_templates:
            for var in _d4_variants(tmpl):
                vh, vw = var.shape[:2]
                # Tile the variant over the bbox with nearest-neighbor (no blur)
                tiled = np.zeros((rh, rw), dtype=np.int64)
                for yy in range(0, rh, max(1, vh)):
                    for xx in range(0, rw, max(1, vw)):
                        th = min(vh, rh - yy)
                        tw = min(vw, rw - xx)
                        tiled[yy : yy + th, xx : xx + tw] = var[:th, :tw]
                # Score: agreement with raw on high-confidence-looking shade pixels
                # plus coverage of eligible motif area (prefer patterns that fit mask).
                agree = (tiled == raw_local) & region_local
                score = float(agree.sum()) / max(float(region_local.sum()), 1.0)
                # Prefer templates whose shade density is close to region shade density
                shade_t = float((tiled > 0).sum()) / max(tiled.size, 1)
                shade_r = float((raw_local[region_local] > 0).sum()) / max(float(region_local.sum()), 1)
                score -= 0.25 * abs(shade_t - shade_r)
                if score > best_score:
                    best_score = score
                    best_patch = tiled

        if best_patch is None or best_score < 0.05:
            continue
        # Write proposal only inside this region mask
        patch_full = proposal[y0:y1, x0:x1]
        patch_full[region_local] = best_patch[region_local]
        proposal[y0:y1, x0:x1] = patch_full

    return proposal
