"""Tests for style-aware postprocess + dual pred_viz naming."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from csn_v4.constants import (
    ALLOWED_OUTPUT_GRAY,
    GRAY_BLACK,
    GRAY_SHADE_100,
    GRAY_SHADE_150,
    GRAY_SHADE_200,
    GRAY_WHITE,
)
from csn_v4.postprocess import PostprocessConfig, profile_status, run_postprocess
from csn_v4.postprocess.constraints import validate_result_palette
from csn_v4.training.pred_viz import compose_pred_with_fixed_input


def _make_source(h=48, w=64) -> np.ndarray:
    src = np.full((h, w), GRAY_WHITE, dtype=np.uint8)
    src[0, :] = GRAY_BLACK
    src[-1, :] = GRAY_BLACK
    src[:, 0] = GRAY_BLACK
    src[:, -1] = GRAY_BLACK
    # Internal outline box (closed motif)
    src[10:38, 10] = GRAY_BLACK
    src[10:38, 50] = GRAY_BLACK
    src[10, 10:51] = GRAY_BLACK
    src[37, 10:51] = GRAY_BLACK
    return src


def _make_pred(src: np.ndarray) -> np.ndarray:
    pred = src.copy()
    # Fill interior with shade, plus one isolated speck
    pred[12:36, 12:50] = GRAY_SHADE_200
    pred[20, 20] = GRAY_SHADE_100  # isolated-ish
    pred[src == GRAY_BLACK] = GRAY_BLACK
    return pred


def _scores_from_pred(pred: np.ndarray, *, low_conf_at=None) -> np.ndarray:
    from csn_v4.constants import NUM_CLASSES, SEMANTIC_GRAY_TO_CLASS

    h, w = pred.shape
    scores = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    for gv, cls in SEMANTIC_GRAY_TO_CLASS.items():
        scores[cls] = (pred == gv).astype(np.float32) * 0.85
        scores[cls] += (pred != gv).astype(np.float32) * 0.05
    # Normalize
    scores = scores / scores.sum(axis=0, keepdims=True).clip(min=1e-6)
    if low_conf_at is not None:
        y, x = low_conf_at
        scores[:, y, x] = 0.25
        scores[0, y, x] = 0.25  # flatten confidence
        scores[:, y, x] /= scores[:, y, x].sum()
    return scores


def test_raw_prediction_never_mutated():
    src = _make_source()
    raw = _make_pred(src)
    raw_copy = raw.copy()
    scores = _scores_from_pred(raw, low_conf_at=(20, 20))
    result = run_postprocess(src, raw, scores, config=PostprocessConfig(profile="conservative_cleanup"))
    assert np.array_equal(raw, raw_copy)
    assert np.array_equal(result.raw_gray, raw_copy)
    # repaired is a separate array
    assert result.repaired_gray is not raw
    assert result.repaired_gray is not result.raw_gray


def test_off_profile_is_identity():
    src = _make_source()
    raw = _make_pred(src)
    result = run_postprocess(src, raw, None, config=PostprocessConfig(profile="off"))
    assert result.report.profile == "off"
    assert int(result.change_map.sum()) == 0
    assert result.report.changed_pixels == 0
    assert np.array_equal(result.repaired_gray, raw)


def test_black_and_palette_preserved():
    src = _make_source()
    raw = _make_pred(src)
    scores = _scores_from_pred(raw, low_conf_at=(20, 20))
    result = run_postprocess(src, raw, scores, config=PostprocessConfig(profile="conservative_cleanup"))
    ok, uniq = validate_result_palette(result.repaired_gray, src)
    assert ok
    assert uniq.issubset(ALLOWED_OUTPUT_GRAY)
    assert np.all(result.repaired_gray[src == GRAY_BLACK] == GRAY_BLACK)


def test_edits_stay_inside_editable_mask():
    src = _make_source()
    raw = _make_pred(src)
    # Messy prediction outside a small editable window
    raw[5, 5] = GRAY_SHADE_150
    editable = np.zeros_like(src, dtype=bool)
    editable[12:36, 12:50] = True
    scores = _scores_from_pred(raw)
    # Force low confidence outside editable
    scores[:, 5, 5] = 0.2
    result = run_postprocess(
        src,
        raw,
        scores,
        editable_mask=editable,
        config=PostprocessConfig(profile="conservative_cleanup"),
    )
    # Pixel outside editable must match raw (after black lock)
    assert result.repaired_gray[5, 5] == raw[5, 5]
    # Changes only inside editable (and never on black)
    changed = result.change_map
    assert not changed[src == GRAY_BLACK].any()
    assert not changed[~editable].any()


def test_miakhi_unavailable_without_assets(tmp_path):
    src = _make_source()
    raw = _make_pred(src)
    empty = tmp_path / "styles"
    empty.mkdir()
    st = profile_status("miakhi", styles_root=empty)
    assert st.available is False
    assert "unavailable" in st.reason.lower() or "missing" in st.reason.lower()

    editable = np.ones_like(src, dtype=bool)
    editable[src == GRAY_BLACK] = False
    result = run_postprocess(
        src,
        raw,
        None,
        editable_mask=editable,
        config=PostprocessConfig(profile="miakhi", styles_root=empty),
    )
    assert result.report.available is False
    assert result.report.ran is False
    assert np.array_equal(result.repaired_gray, raw)
    # Must not misrepresent cleanup as miakhi
    assert "cleanup" not in result.report.profile


def test_miakhi_requires_editable_mask_on_bw(tmp_path):
    # Provide minimal approved pattern so profile is available
    root = tmp_path / "styles" / "miakhi" / "patterns"
    root.mkdir(parents=True)
    pat = np.full((8, 8), GRAY_WHITE, dtype=np.uint8)
    pat[2:6, 2:6] = GRAY_SHADE_150
    Image.fromarray(pat).save(root / "tile.bmp")

    src = _make_source()
    raw = _make_pred(src)
    result = run_postprocess(
        src,
        raw,
        None,
        editable_mask=None,
        config=PostprocessConfig(profile="miakhi", styles_root=tmp_path / "styles"),
    )
    assert result.report.required_editable_mask is True
    assert result.report.changed_pixels == 0
    assert np.array_equal(result.repaired_gray, raw)


def test_miakhi_with_pattern_and_mask_can_edit(tmp_path):
    root = tmp_path / "styles" / "miakhi" / "patterns"
    root.mkdir(parents=True)
    pat = np.full((8, 8), GRAY_WHITE, dtype=np.uint8)
    pat[:, ::2] = GRAY_SHADE_150
    Image.fromarray(pat).save(root / "stripes.bmp")

    src = _make_source()
    raw = np.full_like(src, GRAY_WHITE)
    raw[src == GRAY_BLACK] = GRAY_BLACK
    # Uncertain fill inside motif
    raw[12:36, 12:50] = GRAY_SHADE_200
    editable = (src != GRAY_BLACK)
    # Closed interior only
    editable[:10, :] = False
    editable[38:, :] = False
    editable[:, :10] = False
    editable[:, 51:] = False

    scores = _scores_from_pred(raw)
    scores[:, 12:36, 12:50] = 0.3  # uncertain interior
    scores[1, 12:36, 12:50] = 0.4  # class 200 slightly ahead but low
    scores = scores / scores.sum(axis=0, keepdims=True)

    result = run_postprocess(
        src,
        raw,
        scores,
        editable_mask=editable,
        config=PostprocessConfig(
            profile="miakhi",
            styles_root=tmp_path / "styles",
            confidence_threshold=0.7,
            style_influence=0.9,
            user_style_selected=True,
            min_region_pixels=8,
        ),
    )
    assert result.report.available is True
    assert result.report.ran is True
    assert np.all(result.repaired_gray[src == GRAY_BLACK] == GRAY_BLACK)
    assert not result.change_map[~editable].any()


def test_pred_viz_dual_filenames(tmp_path):
    """Simulate the naming contract used by save_holdout_pred_viz."""
    sid = "demo_scene"
    out_dir = tmp_path / "pred_viz" / "step_000010"
    out_dir.mkdir(parents=True)

    src = _make_source()
    raw = _make_pred(src)
    raw_display = compose_pred_with_fixed_input(src, raw)

    raw_full = out_dir / f"{sid}__t0_pred_full.png"
    Image.fromarray(raw_display).save(raw_full)

    result = run_postprocess(
        src, raw, _scores_from_pred(raw, low_conf_at=(20, 20)),
        config=PostprocessConfig(profile="conservative_cleanup"),
    )
    post_display = compose_pred_with_fixed_input(src, result.repaired_gray)
    post_full = out_dir / f"{sid}__t0_pred_full__postprocessed.png"
    Image.fromarray(post_display).save(post_full)

    report = out_dir / f"{sid}__t0_postprocess_report.json"
    report.write_text(json.dumps(result.report.to_dict()), encoding="utf-8")

    assert raw_full.is_file()
    assert post_full.is_file()
    assert raw_full.name == f"{sid}__t0_pred_full.png"
    assert post_full.name.endswith("__postprocessed.png")
    assert raw_full.name != post_full.name
    # Raw file contents are the raw display (not post)
    loaded_raw = np.asarray(Image.open(raw_full))
    assert np.array_equal(loaded_raw[..., 0] if loaded_raw.ndim == 3 else loaded_raw, raw_display)

    print(f"RAW_PATH={raw_full}")
    print(f"POST_PATH={post_full}")
    print(f"REPORT_PATH={report}")


def test_ui_preview_matches_bmp_array(tmp_path):
    from csn_v3.renderer import save_output_bmp

    src = _make_source()
    raw = _make_pred(src)
    result = run_postprocess(src, raw, None, config=PostprocessConfig(profile="off"))

    preview = tmp_path / "post.png"
    bmp = tmp_path / "post.bmp"
    Image.fromarray(result.repaired_gray).save(preview)
    save_output_bmp(bmp, result.repaired_gray)

    prev = np.asarray(Image.open(preview).convert("L"))
    bmp_arr = np.asarray(Image.open(bmp).convert("L"))
    assert np.array_equal(prev, result.repaired_gray)
    assert np.array_equal(bmp_arr, result.repaired_gray)
    assert np.array_equal(prev, bmp_arr)
