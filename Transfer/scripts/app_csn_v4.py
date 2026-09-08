#!/usr/bin/env python3
"""Lightweight local UI for CSN-V4 (no Gradio/pandas) — upload BW → shaded BMP."""
from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.config import load_config
from csn_v4.factory import load_model_from_checkpoint
from csn_v4.inference.tiled import predict_full_image_gray
from csn_v3.renderer import save_output_bmp, validate_palette

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path(tempfile.gettempdir()) / "csn_v4_ui_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8"/>
<title>CSN-V4 UI</title>
<style>
 body{font-family:Tahoma,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#f6f4ef;color:#222}
 h1{font-size:1.4rem;margin-bottom:4px}
 .sub{color:#666;margin-bottom:18px}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;flex:1;min-width:280px}
 img{max-width:100%;height:auto;border:1px solid #ccc;background:#fff}
 button{background:#1f4b3a;color:#fff;border:0;padding:10px 18px;border-radius:6px;cursor:pointer;font-size:1rem}
 button:disabled{opacity:.5}
 input[type=file]{margin:8px 0}
 .status{white-space:pre-wrap;background:#111;color:#d6ffd6;padding:10px;border-radius:6px;min-height:3em;font-size:.85rem}
</style>
</head>
<body>
<h1>CSN-V4 — tiled inference</h1>
<p class="sub">آپلود تصویر BW → خروجی categorical (255/200/150/100) + دانلود BMP<br/>checkpoint: __CKPT__</p>
<form id="f">
  <input type="file" name="image" accept="image/*,.bmp" required/>
  <button id="go" type="submit">اجرا</button>
</form>
<p class="status" id="st">آماده</p>
<div class="row">
  <div class="card"><h3>ورودی</h3><img id="inp" alt="input"/></div>
  <div class="card"><h3>خروجی</h3><img id="out" alt="output"/><p><a id="dl" href="#" download="shaded.bmp">دانلود BMP</a></p></div>
</div>
<script>
const f=document.getElementById('f'), st=document.getElementById('st'), go=document.getElementById('go');
f.onsubmit=async(e)=>{
  e.preventDefault(); go.disabled=true; st.textContent='در حال inference…';
  const fd=new FormData(f);
  try{
    const r=await fetch('/infer',{method:'POST',body:fd});
    const j=await r.json();
    if(!r.ok){st.textContent=j.error||'error';return;}
    document.getElementById('inp').src=j.input_url;
    document.getElementById('out').src=j.output_url+'?t='+Date.now();
    document.getElementById('dl').href=j.bmp_url;
    st.textContent=j.status;
  }catch(err){st.textContent=String(err);}
  finally{go.disabled=false;}
};
</script>
</body></html>
"""


def _to_uint8_gray(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"))
    else:
        arr = image
        if arr.ndim == 3:
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
        if arr.dtype != np.uint8:
            if float(np.max(arr)) <= 1.0:
                arr = (arr * 255.0).round()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@lru_cache(maxsize=1)
def _load(checkpoint: str, config_path: str, device: str):
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, state = load_model_from_checkpoint(checkpoint, config_path, device=dev)
    cfg = load_config(config_path)
    model.eval()
    return model, cfg, state, dev


def run_infer(image_bytes: bytes, checkpoint: str, config_path: str, device: str) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    source = _to_uint8_gray(img)
    model, cfg, state, dev = _load(checkpoint, config_path, device)
    t0 = time.perf_counter()
    gray, n_tiles = predict_full_image_gray(
        model, source, dev,
        context_size=cfg.data.context_size,
        global_long_side=cfg.inference.global_long_side,
        input_mode=cfg.data.input_mode,
        scene_id="ui",
    )
    ok, uniq = validate_palette(gray)
    uid = uuid.uuid4().hex[:8]
    inp_path = OUTPUT_DIR / f"input_{uid}.png"
    out_path = OUTPUT_DIR / f"out_{uid}.png"
    bmp_path = OUTPUT_DIR / f"shaded_{uid}.bmp"
    Image.fromarray(source).save(inp_path)
    Image.fromarray(gray).save(out_path)
    save_output_bmp(bmp_path, gray)
    status = (
        f"OK tiles={n_tiles} palette_ok={ok} uniq={sorted(uniq)} "
        f"time={time.perf_counter()-t0:.2f}s ckpt_step={state.get('optimizer_step')} "
        f"size={source.shape[1]}x{source.shape[0]}"
    )
    return {
        "status": status,
        "input_url": f"/file/{inp_path.name}",
        "output_url": f"/file/{out_path.name}",
        "bmp_url": f"/file/{bmp_path.name}",
    }


def make_handler(checkpoint: Path, config_path: Path, device: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOGGER.info("%s - " + fmt, self.address_string(), *args)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = HTML.replace("__CKPT__", str(checkpoint)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/file/"):
                name = Path(self.path[len("/file/"):]).name
                path = OUTPUT_DIR / name
                if not path.is_file():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                ctype = "image/bmp" if path.suffix.lower() == ".bmp" else "image/png"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/infer":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            # Minimal multipart parser for a single file field named image
            content_type = self.headers.get("Content-Type", "")
            try:
                if "multipart/form-data" not in content_type:
                    raise ValueError("expected multipart/form-data")
                boundary = content_type.split("boundary=")[-1].strip().encode()
                parts = raw.split(b"--" + boundary)
                image_bytes = None
                for part in parts:
                    if b"name=\"image\"" not in part:
                        continue
                    idx = part.find(b"\r\n\r\n")
                    if idx < 0:
                        continue
                    image_bytes = part[idx + 4 :]
                    if image_bytes.endswith(b"\r\n"):
                        image_bytes = image_bytes[:-2]
                    break
                if not image_bytes:
                    raise ValueError("image field missing")
                result = run_infer(image_bytes, str(checkpoint), str(config_path), device)
                body = __import__("json").dumps(result).encode("utf-8")
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("infer failed")
                body = __import__("json").dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSN-V4 lightweight UI")
    parser.add_argument("--config", type=Path, default=Path("configs/csn_v4_generalization.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/20260802_191353_CSN_V4_bw_generalization/last.pt"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    # Warm-load model once so first browser request is faster.
    LOGGER.info("Loading checkpoint %s …", args.checkpoint)
    _load(str(args.checkpoint), str(args.config), args.device)
    handler = make_handler(args.checkpoint.resolve(), args.config.resolve(), args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    LOGGER.info("CSN-V4 UI ready at %s", url)
    print(url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
