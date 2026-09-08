"""Self-contained pilot HTML report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shdowing.ai2bmp.io_utils import atomic_write_text, rel_path


def _img_tag(path: Path, output_root: Path) -> str:
    rel = rel_path(path, output_root)
    return f'<img src="../{rel}" style="max-width:320px;border:1px solid #ccc;" />'


def build_pilot_html(output_root: Path, ctx: dict[str, Any]) -> Path:
    audit = ctx["audit"]
    canonical = ctx["canonical"]
    invariants = ctx.get("invariants", [])
    variants = ctx.get("variants", [])
    crops = ctx.get("crops", [])

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>AI2BMP Pilot Report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 8px;} .pass{color:green;} .fail{color:red;} h2{margin-top:2em;}</style>",
        "</head><body>",
        "<h1>AI2BMP Synthetic Behavior Dataset V1 — Pilot Report</h1>",
        "<h2>Source BMP</h2>",
        "<table>",
        f"<tr><td>Filename</td><td>{audit.filename}</td></tr>",
        f"<tr><td>Dimensions</td><td>{audit.width} x {audit.absolute_height}</td></tr>",
        f"<tr><td>bpp</td><td>{audit.bits_per_pixel}</td></tr>",
        f"<tr><td>clr_used</td><td>{audit.clr_used}</td></tr>",
        f"<tr><td>Declared palette entries</td><td>{audit.declared_palette_entry_count}</td></tr>",
        f"<tr><td>Used original indices</td><td>{audit.used_index_count}</td></tr>",
        f"<tr><td>Unique used RGB count</td><td>{audit.unique_used_rgb_count}</td></tr>",
        f"<tr><td>SHA-256</td><td><code>{audit.sha256}</code></td></tr>",
        "</table>",
        "<p><strong>Note:</strong> declared palette entries = "
        f"{audit.declared_palette_entry_count}; used original indices = {audit.used_index_count} "
        "(not {audit.declared_palette_entry_count} training colors).</p>",
        "<h2>Canonical Verification</h2>",
        _img_tag(canonical.canonical_dir / "target_rgb.png", output_root),
        _img_tag(canonical.canonical_dir / "reconstructed_rgb.png", output_root),
        _img_tag(canonical.canonical_dir / "roundtrip_diff.bmp", output_root),
        f"<p>Round-trip mismatches: {canonical.roundtrip_mismatch_count}</p>",
        "<h2>Structural Labels</h2>",
        _img_tag(canonical.design_dir / "exact_labels" / "region_boundary_mask.bmp", output_root),
        _img_tag(canonical.design_dir / "exact_labels" / "thin_structure_mask.bmp", output_root),
        "<h2>Synthetic Behavior Gallery</h2>",
    ]

    for var in variants:
        if var.get("status") != "generated":
            html_parts.append(f"<p>{var['variant_id']}: {var.get('status')}</p>")
            continue
        vdir = Path(var["variant_dir"])
        meta = var.get("metadata", {})
        html_parts.append(f"<h3>{var['variant_id']} ({var.get('family')})</h3>")
        html_parts.append(
            f"<p>native={meta.get('native_width')}x{meta.get('native_height')} "
            f"aligned={meta.get('aligned_width')}x{meta.get('aligned_height')} seed={meta.get('seed')}</p>"
        )
        html_parts.append(_img_tag(vdir / "input_native.png", output_root))
        html_parts.append(_img_tag(vdir / "input_aligned.png", output_root))

    html_parts.append("<h2>PASS/FAIL Invariants</h2><table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>")
    for chk in invariants:
        cls = "pass" if chk["status"] == "PASS" else "fail"
        html_parts.append(
            f"<tr><td>{chk['name']}</td><td class='{cls}'>{chk['status']}</td><td>{chk.get('detail','')}</td></tr>"
        )
    html_parts.append("</table>")

    html_parts.append("<h2>Crop Gallery</h2>")
    for crop in crops[:12]:
        html_parts.append(
            f"<p>{crop['sample_id']} size={crop['crop_size']} category={crop['sampling_strategy']} "
            f"@ ({crop['x']},{crop['y']})</p>"
        )

    html_parts.append("</body></html>")
    out = output_root / "reports" / "pilot_report.html"
    atomic_write_text(out, "\n".join(html_parts))
    return out
