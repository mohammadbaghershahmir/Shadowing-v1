#!/usr/bin/env python
"""Launch the professional DINOv3 carpet probe Gradio app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dinov3_carpet_probe.src.app_ui import build_app
from dinov3_carpet_probe.src.interactive_runner import InteractiveRunner


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 Carpet Probe Studio")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    p.add_argument("--weights-dir", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runner = InteractiveRunner(weights_dir=args.weights_dir, output_root=args.output_root)
    choices = runner.available_model_choices()
    print("Available models:", [c[1] for c in choices], flush=True)
    if not choices:
        print("ERROR: no local checkpoints in weights/", flush=True)
        return 2
    demo = build_app(runner=runner)
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": True,
        "theme": __import__("gradio").themes.Soft(),
    }
    css = getattr(demo, "app_css", None)
    if css:
        launch_kwargs["css"] = css
    demo.queue().launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
