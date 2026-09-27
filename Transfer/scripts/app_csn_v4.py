#!/usr/bin/env python3
"""Lightweight local UI for CSN-V4 — raw + postprocessed side by side."""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from csn_v4.config import load_config
from csn_v4.factory import load_model_from_checkpoint
from csn_v4.inference.tiled import predict_full_image_with_scores
from csn_v4.postprocess import PostprocessConfig, list_profiles, profile_status, run_postprocess
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
 body{font-family:Tahoma,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;background:#f6f4ef;color:#222}
 h1{font-size:1.4rem;margin-bottom:4px}
 .sub{color:#666;margin-bottom:18px}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;flex:1;min-width:260px}
 img{max-width:100%;height:auto;border:1px solid #ccc;background:#fff}
 button{background:#1f4b3a;color:#fff;border:0;padding:10px 18px;border-radius:6px;cursor:pointer;font-size:1rem}
 button:disabled{opacity:.5}
 input[type=file],select{margin:8px 0;display:block;width:100%;max-width:420px}
 .status{white-space:pre-wrap;background:#111;color:#d6ffd6;padding:10px;border-radius:6px;min-height:3em;font-size:.85rem}
 .meta{font-size:.9rem;color:#333;margin-top:8px}
 a{color:#1f4b3a}
</style>
</head>
<body>
<h1>CSN-V4 — tiled inference + style postprocess</h1>
<p class="sub">آپلود BW → خروجی خام و پس‌پردازش‌شده (هر کدام BMP جدا)<br/>checkpoint: __CKPT__</p>
<form id="f">
  <label>تصویر ورودی
    <input type="file" name="image" accept="image/*,.bmp" required/>
  </label>
  <label>ماسک editable (اختیاری — برای سبک‌های تزئینی لازم است)
    <input type="file" name="editable_mask" accept="image/*,.bmp"/>
  </label>
  <label>پروفایل سبک
    <select name="style" id="style">__STYLE_OPTS__</select>
  </label>
  <button id="go" type="submit">اجرا</button>
</form>
<p class="status" id="st">آماده</p>
<p class="meta" id="meta"></p>
<div class="row">
  <div class="card"><h3>ورودی</h3><img id="inp" alt="input"/></div>
  <div class="card">
    <h3>Raw model output</h3>
    <img id="out_raw" alt="raw"/>
    <p><a id="dl_raw" href="#" download="shaded_raw.bmp">دانلود BMP خام</a></p>
  </div>
  <div class="card">
    <h3>Postprocessed output</h3>
    <img id="out_post" alt="post"/>
    <p><a id="dl_post" href="#" download="shaded_postprocessed.bmp">دانلود BMP پس‌پردازش</a></p>
  </div>
</div>
<script>
const f=document.getElementById('f'), st=document.getElementById('st'), go=document.getElementById('go'), meta=document.getElementById('meta');
f.onsubmit=async(e)=>{
  e.preventDefault(); go.disabled=true; st.textContent='در حال inference…'; meta.textContent='';
  const fd=new FormData(f);
  try{
    const r=await fetch('/infer',{method:'POST',body:fd});
    const j=await r.json();
    if(!r.ok){st.textContent=j.error||'error';return;}
    document.getElementById('inp').src=j.input_url;
    document.getElementById('out_raw').src=j.raw_preview_url+'?t='+Date.now();
    document.getElementById('out_post').src=j.post_preview_url+'?t='+Date.now();
    document.getElementById('dl_raw').href=j.raw_bmp_url;
    document.getElementById('dl_post').href=j.post_bmp_url;
    st.textContent=j.status;
    meta.textContent=j.report_line||'';
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


def _mask_from_bytes(data: bytes | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not data:
        return None
    arr = _to_uint8_gray(Image.open(io.BytesIO(data)))
    if arr.shape[:2] != shape:
        raise ValueError(f"editable_mask shape {arr.shape} != input {shape}")
    # Non-zero (or non-white) → editable. Prefer binary masks (0/255).
    return arr > 127


@lru_cache(maxsize=1)
def _load(checkpoint: str, config_path: str, device: str):
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, state = load_model_from_checkpoint(checkpoint, config_path, device=dev)
    cfg = load_config(config_path)
    model.eval()
    return model, cfg, state, dev


def run_infer(
    image_bytes: bytes,
    checkpoint: str,
    config_path: str,
    device: str,
    *,
    style: str = "conservative_cleanup",
    editable_mask_bytes: bytes | None = None,
) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    source = _to_uint8_gray(img)
    model, cfg, state, dev = _load(checkpoint, config_path, device)
    t0 = time.perf_counter()
    gray, n_tiles, scores = predict_full_image_with_scores(
        model, source, dev,
        context_size=getattr(cfg.model, "context_size", getattr(cfg.data, "context_size", 1024)),
        global_long_side=cfg.inference.global_long_side,
        input_mode=cfg.data.input_mode,
        scene_id="ui",
    )
    raw_snapshot = gray.copy()
    ok_raw, uniq_raw = validate_palette(gray)

    editable = _mask_from_bytes(editable_mask_bytes, source.shape[:2])
    styles_root = getattr(cfg.postprocess, "styles_root", "styles")
    styles_path = Path(styles_root)
    if not styles_path.is_absolute():
        styles_path = _ROOT / styles_path

    pp_cfg = PostprocessConfig(
        profile=style,
        styles_root=styles_path,
        confidence_threshold=float(getattr(cfg.postprocess, "confidence_threshold", 0.55)),
        style_influence=float(getattr(cfg.postprocess, "style_influence", 0.5)),
        min_region_pixels=int(getattr(cfg.postprocess, "min_region_pixels", 16)),
        user_style_selected=style.lower() not in ("off", "conservative_cleanup", "cleanup", "conservative"),
    )
    result = run_postprocess(
        source,
        gray,
        scores,
        editable_mask=editable,
        config=pp_cfg,
    )
    if not np.array_equal(gray, raw_snapshot):
        raise RuntimeError("Raw prediction was mutated during postprocess")

    ok_post, uniq_post = validate_palette(result.repaired_gray)

    # Previews and BMPs must come from the *same* pixel arrays.
    uid = uuid.uuid4().hex[:8]
    inp_path = OUTPUT_DIR / f"input_{uid}.png"
    raw_preview = OUTPUT_DIR / f"raw_{uid}.png"
    post_preview = OUTPUT_DIR / f"post_{uid}.png"
    raw_bmp = OUTPUT_DIR / f"shaded_raw_{uid}.bmp"
    post_bmp = OUTPUT_DIR / f"shaded_post_{uid}.bmp"
    report_path = OUTPUT_DIR / f"report_{uid}.json"

    Image.fromarray(source).save(inp_path)
    Image.fromarray(gray).save(raw_preview)
    Image.fromarray(result.repaired_gray).save(post_preview)
    save_output_bmp(raw_bmp, gray)
    save_output_bmp(post_bmp, result.repaired_gray)
    # Verify preview PNG pixels match BMP source arrays
    assert np.array_equal(np.asarray(Image.open(raw_preview)), gray)
    assert np.array_equal(np.asarray(Image.open(post_preview)), result.repaired_gray)
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")

    pst = profile_status(style, styles_root=styles_path)
    status = (
        f"OK tiles={n_tiles} raw_palette_ok={ok_raw} post_palette_ok={ok_post} "
        f"time={time.perf_counter()-t0:.2f}s ckpt_step={state.get('optimizer_step')} "
        f"size={source.shape[1]}x{source.shape[0]}"
    )
    report_line = (
        f"style={result.report.profile} available={result.report.available} "
        f"ran={result.report.ran} changed_pixels={result.report.changed_pixels} "
        f"skipped={result.report.regions_skipped} | {result.report.reason}"
    )
    return {
        "status": status,
        "report_line": report_line,
        "input_url": f"/file/{inp_path.name}",
        "raw_preview_url": f"/file/{raw_preview.name}",
        "post_preview_url": f"/file/{post_preview.name}",
        "raw_bmp_url": f"/file/{raw_bmp.name}",
        "post_bmp_url": f"/file/{post_bmp.name}",
        "report_url": f"/file/{report_path.name}",
        # Backward-compatible aliases (raw)
        "output_url": f"/file/{raw_preview.name}",
        "bmp_url": f"/file/{raw_bmp.name}",
        "style_available": pst.available,
    }


def _parse_multipart(raw: bytes, content_type: str) -> dict[str, bytes]:
    if "multipart/form-data" not in content_type:
        raise ValueError("expected multipart/form-data")
    boundary = content_type.split("boundary=")[-1].strip().encode()
    parts = raw.split(b"--" + boundary)
    fields: dict[str, bytes] = {}
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header, _, body = part.partition(b"\r\n\r\n")
        if not body:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        # name="..."
        name = None
        for seg in header.split(b";"):
            seg = seg.strip()
            if seg.startswith(b"name="):
                name = seg.split(b"=", 1)[1].strip(b"\"").decode("utf-8", errors="replace")
        if name:
            fields[name] = body
    return fields


def make_handler(checkpoint: Path, config_path: Path, device: str):
    style_opts = "".join(
        f'<option value="{p}">{p}</option>' for p in list_profiles()
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOGGER.info("%s - " + fmt, self.address_string(), *args)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    HTML.replace("__CKPT__", str(checkpoint))
                    .replace("__STYLE_OPTS__", style_opts)
                    .encode("utf-8")
                )
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
                ctype = "image/bmp" if path.suffix.lower() == ".bmp" else (
                    "application/json" if path.suffix.lower() == ".json" else "image/png"
                )
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
            content_type = self.headers.get("Content-Type", "")
            try:
                fields = _parse_multipart(raw, content_type)
                image_bytes = fields.get("image")
                if not image_bytes:
                    raise ValueError("image field missing")
                style = fields.get("style", b"conservative_cleanup").decode("utf-8", errors="replace").strip()
                mask_bytes = fields.get("editable_mask") or None
                if mask_bytes is not None and len(mask_bytes) < 32:
                    mask_bytes = None
                result = run_infer(
                    image_bytes,
                    str(checkpoint),
                    str(config_path),
                    device,
                    style=style or "conservative_cleanup",
                    editable_mask_bytes=mask_bytes,
                )
                body = json.dumps(result).encode("utf-8")
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("infer failed")
                body = json.dumps({"error": str(exc)}).encode("utf-8")
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
