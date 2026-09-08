#!/usr/bin/env python
"""Launch a minimal Gradio UI for palette checkpoint inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from dinov3_carpet_probe.palette_train.src.palette_ui import build_palette_ui  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_palette_ui()
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": True,
        "theme": gr.themes.Soft(),
    }
    css = getattr(demo, "app_css", None)
    if css:
        launch_kwargs["css"] = css
    demo.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
