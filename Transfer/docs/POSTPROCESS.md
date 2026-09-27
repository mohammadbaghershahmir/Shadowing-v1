# Style-aware postprocessing (CSN-V4)

Postprocessing is a **separate stage** after model inference. It never trains,
never changes checkpoint selection, and never overwrites the raw prediction.

## Pipeline

1. Run the model once → raw categorical gray `{0,100,150,200,255}` + optional scores `[4,H,W]`.
2. Copy the raw array; postprocess the copy.
3. Save / show both.

### `pred_viz` filenames

| File | Meaning |
|------|---------|
| `{sid}__t0.png` | Existing raw contact sheet (unchanged name/format) |
| `{sid}__t0_pred_full.png` | Existing raw full-res pred (unchanged) |
| `{sid}__t0_pred_full__postprocessed.png` | Repaired full-res display |
| `{sid}__t0__postprocessed.png` | Contact sheet with repaired pred |
| `{sid}__t0_compare.png` | Raw \| Post \| change map |
| `{sid}__t0_change_map.png` | Changed pixels highlight |
| `{sid}__t0_postprocess_report.json` | Style / ran / changed / skipped |

### Profiles

| Profile | Behavior |
|---------|----------|
| `off` | Passthrough copy |
| `conservative_cleanup` | Isolated low-confidence speckles only; keeps intentional small details |
| `miakhi` | Decorative; **requires** approved assets under `styles/miakhi/` + editable mask |

See [styles/miakhi/README.md](../styles/miakhi/README.md).

## Config

```yaml
postprocess:
  profile: conservative_cleanup
  styles_root: styles
  confidence_threshold: 0.55
  style_influence: 0.5
  min_region_pixels: 16
  save_compare: true
  save_change_map: true
```

Thresholds should be chosen on **held-out validation designs**, never on final test targets.

## UI

```powershell
python scripts/app_csn_v4.py --checkpoint <best.pt> --config configs/csn_v4_generalization_local.yaml
```

Shows **Raw model output** and **Postprocessed output**, each with its own BMP
download generated from the same pixel array as the preview.

## API

```python
from csn_v4.postprocess import PostprocessConfig, run_postprocess

result = run_postprocess(
    source_bw,
    raw_pred_gray,          # not mutated
    scores,                 # [4,H,W] or None
    editable_mask=mask,     # required for miakhi
    config=PostprocessConfig(profile="conservative_cleanup"),
)
# result.raw_gray, result.repaired_gray, result.change_map, result.report
```

Ground-truth targets must not be passed as editable masks.
