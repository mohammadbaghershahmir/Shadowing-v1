#!/usr/bin/env python3
"""Audit CSN-V4 dataset manifests and scene artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.config import load_config
from csn_v4.data.manifest_validator import audit_dataset, format_audit_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CSN-V4 dataset")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_capacity.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or Path(cfg.data.dataset_root)

    report = audit_dataset(dataset_root)
    report_dir = Path(dataset_root) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "dataset_audit.json"
    md_path = report_dir / "dataset_audit.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md_path.write_text(format_audit_markdown(report), encoding="utf-8")

    print(f"Audit status: {report.get('status')} ({report.get('critical_count', 0)} errors)")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
