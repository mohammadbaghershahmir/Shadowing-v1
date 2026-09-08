#!/usr/bin/env python3
"""Overnight pipeline: materialize tiles → train with logging and auto-restart."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CONFIG = _ROOT / "configs/csn_v4_generalization.yaml"
LOG_DIR = _ROOT / "runs" / "overnight_logs"


def _log(msg: str, log_path: Path) -> None:
    line = f"{datetime.now().isoformat()} {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run(cmd: list[str], log_path: Path) -> int:
    _log(f"CMD: {' '.join(cmd)}", log_path)
    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=fh, stderr=subprocess.STDOUT)
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-materialize", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    status_path = LOG_DIR / "status.json"

    def write_status(phase: str, **extra):
        payload = {"phase": phase, "updated": datetime.now().isoformat(), **extra}
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_status("starting")

    if not args.skip_materialize:
        write_status("materialize")
        rc = _run([
            PYTHON, str(_ROOT / "scripts/materialize_csn_v4_tiles.py"),
            "--config", str(CONFIG),
            "--overwrite",
        ], log_path)
        if rc != 0:
            write_status("materialize_failed", exit_code=rc)
            return rc

    if args.materialize_only:
        write_status("materialize_done")
        return 0

    for attempt in range(1, 4):
        write_status("training", attempt=attempt)
        rc = _run([
            PYTHON, str(_ROOT / "scripts/train_csn_v4.py"),
            "--config", str(CONFIG),
            "--device", "cuda",
            "--skip-initial-eval",
        ], log_path)
        if rc == 0:
            write_status("training_complete", attempt=attempt)
            return 0
        _log(f"Training exited with code {rc}; restart attempt {attempt}/3 in 60s", log_path)
        time.sleep(60)

    write_status("training_failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
