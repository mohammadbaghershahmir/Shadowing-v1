"""Pilot summary CSV and validation JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shdowing.ai2bmp.io_utils import write_csv, write_json


def write_pilot_summary(output_root: Path, rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(reports / "pilot_summary.csv", rows, list(rows[0].keys()))
    write_json(reports / "validation_report.json", validation)
