"""Write Dataset V2 artifact bundles and metadata JSON."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_v2.artifacts import SampleArtifacts, crop_array
from dataset_v2.bmp_io import save_gray_bmp, save_rgb_bmp
from dataset_v2.contact_sheet import save_contact_sheet


@dataclass
class ArtifactBundle:
    artifacts: SampleArtifacts
    metadata: dict[str, Any]


def save_artifact_bundle(out_dir: Path, bundle: ArtifactBundle) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = bundle.artifacts
    save_gray_bmp(out_dir / "target_semantic.bmp", a.target_semantic)
    save_gray_bmp(out_dir / "source_bw.bmp", a.source_bw)
    save_rgb_bmp(out_dir / "source_oracle.bmp", a.source_oracle)
    save_gray_bmp(out_dir / "shade_mask.bmp", a.shade_mask)
    save_gray_bmp(out_dir / "shade_level_id.bmp", a.shade_level_id)
    save_gray_bmp(out_dir / "shade_level_preview.bmp", a.shade_level_preview)
    save_gray_bmp(out_dir / "transition_mask.bmp", a.transition_mask)
    save_gray_bmp(out_dir / "black_lock.bmp", a.black_lock)
    save_gray_bmp(out_dir / "valid_mask.bmp", a.valid_mask)
    save_gray_bmp(out_dir / "reconstructed_target.bmp", a.reconstructed_target)
    save_gray_bmp(out_dir / "roundtrip_diff.bmp", a.roundtrip_diff)
    save_contact_sheet(out_dir / "contact_sheet.bmp", a)
    (out_dir / "metadata.json").write_text(
        json.dumps(bundle.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def crop_artifacts(artifacts: SampleArtifacts, x: int, y: int, size: int) -> SampleArtifacts:
    return SampleArtifacts(
        target_semantic=crop_array(artifacts.target_semantic, x, y, size),
        source_bw=crop_array(artifacts.source_bw, x, y, size),
        source_oracle=crop_array(artifacts.source_oracle, x, y, size),
        shade_mask=crop_array(artifacts.shade_mask, x, y, size),
        shade_level_id=crop_array(artifacts.shade_level_id, x, y, size),
        shade_level_preview=crop_array(artifacts.shade_level_preview, x, y, size),
        transition_mask=crop_array(artifacts.transition_mask, x, y, size),
        black_lock=crop_array(artifacts.black_lock, x, y, size),
        valid_mask=crop_array(artifacts.valid_mask, x, y, size),
        reconstructed_target=crop_array(artifacts.reconstructed_target, x, y, size),
        roundtrip_diff=crop_array(artifacts.roundtrip_diff, x, y, size),
    )
