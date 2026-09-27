"""HTML and JSON pilot review reports."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from dataset_v2.constants import DATASET_NAME, DATASET_VERSION
from dataset_v2.manifests import write_csv, PILOT_SUMMARY_FIELDS


def write_validation_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_pilot_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, PILOT_SUMMARY_FIELDS, rows)


def write_pilot_report_html(
    path: Path,
    *,
    output_dir: Path,
    validation: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_root = path.parent.parent.name

    def esc(s: Any) -> str:
        return html.escape(str(s))

    sections: list[str] = []
    sections.append(f"<h1>{esc(DATASET_NAME)} Pilot Report</h1>")
    sections.append(f"<p>dataset_version={esc(DATASET_VERSION)}</p>")
    sections.append("<h2>Validation Summary</h2>")
    sections.append(f"<pre>{esc(json.dumps(validation.get('summary', {}), indent=2))}</pre>")

    sections.append("<h2>Samples</h2>")
    for sample in samples:
        sid = sample["sample_id"]
        status = sample.get("status", "unknown")
        sections.append(f"<h3>{esc(sid)} — {esc(status)}</h3>")
        sections.append("<ul>")
        sections.append(f"<li>File: {esc(sample.get('original_filename'))}</li>")
        sections.append(f"<li>Size: {sample.get('width')}×{sample.get('height')}</li>")
        sections.append(
            f"<li>Round-trip mismatches: {sample.get('roundtrip_mismatch_count', 'n/a')}</li>"
        )
        sections.append("</ul>")
        full_rel = sample.get("full_contact_sheet")
        if full_rel:
            sections.append(
                f'<p><a href="{esc(full_rel)}">Full contact sheet</a></p>'
                f'<img src="{esc(full_rel)}" alt="contact" style="max-width:900px"/>'
            )
        if sample.get("crops"):
            sections.append("<h4>Crops</h4><ul>")
            for crop in sample["crops"]:
                sections.append(
                    f"<li>{esc(crop['crop_id'])} [{esc(crop['category'])}] "
                    f"({crop['x']},{crop['y']}) {crop['width']}×{crop['height']} "
                    f"<a href=\"{esc(crop.get('contact_sheet', ''))}\">sheet</a></li>"
                )
            sections.append("</ul>")

    rejected = validation.get("rejected_files", [])
    if rejected:
        sections.append("<h2>Rejected Inputs</h2><ul>")
        for item in rejected:
            sections.append(
                f"<li>{esc(item.get('relative_path'))}: {esc(item.get('errors'))}</li>"
            )
        sections.append("</ul>")

    body = "\n".join(sections)
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{esc(DATASET_NAME)} Pilot</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
pre {{ background: #f4f4f4; padding: 12px; }}
h3 {{ margin-top: 24px; }}
</style></head><body>
{body}
<p><em>Output root: {esc(output_dir)}</em></p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")
