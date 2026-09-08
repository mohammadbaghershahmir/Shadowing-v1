# AI2BMP Synthetic Behavior Dataset V1

`dataset_version = "AI2BMP-SYN-V1"`

Production-quality synthetic dataset builder for indexed carpet-map BMPs. Generates deterministic AI-like PNG corruption variants paired with exact canonical structural labels.

## Scope

This package implements **dataset preparation only**:

- Strict indexed BMP parsing and validation
- Canonical artifact generation (exact palette/index maps)
- Exact structural labels (regions, boundaries, thin structures)
- Provisional role/shade suggestions (`approved_for_training=false`)
- Synthetic AI-like PNG inputs (native + aligned)
- Crop manifests and PyTorch `AI2BMPSyntheticDataset`
- Pilot HTML review report

No model training, DINO, or external AI APIs.

## CLI

```powershell
cd E:\Dino\dinov3
$env:PYTHONPATH = "E:\Dino\dinov3"

python tools/build_ai2bmp_synthetic_v1.py `
  --input-dir "E:\Color-detection\Data\کلاسیک_رنگ شده_کامل" `
  --output-dir "E:\Color-detection\Data\AI2BMP_Synthetic_V1_Pilot" `
  --mode pilot `
  --sample-count 1 `
  --sample-file "1-111-6B.bmp" `
  --seed 42 `
  --crop-sizes 256 512 1024 `
  --variants-profile pilot `
  --recursive `
  --materialize-crops `
  --verify-hashes
```

Use `--resume` to write into an existing nonempty output directory.

## Output layout

```text
AI2BMP_Synthetic_V1_Pilot/
├── dataset_config.json
├── dataset_stats.json
├── designs/<sample_id>/...
├── manifests/
│   ├── canonical_designs.csv
│   ├── variants.csv
│   ├── crops.csv
│   └── all_training_samples.jsonl
└── reports/
    ├── pilot_report.html
    ├── pilot_contact_sheet.png
    └── validation_report.json
```

## BMP terminology

Do not conflate:

- `bits_per_pixel`
- `clr_used` / declared palette entries
- physically detected palette entries
- used palette indices (from index raster)
- unique used RGB colors

## Tests

```powershell
python -m pytest shdowing/ai2bmp/tests -q
```

## PyTorch loader

```python
from shdowing.ai2bmp.dataset import AI2BMPSyntheticDataset, collate_ai2bmp_batch

ds = AI2BMPSyntheticDataset(
    "manifests/all_training_samples.jsonl",
    output_root="E:/Color-detection/Data/AI2BMP_Synthetic_V1_Pilot",
    crop_size=512,
)
sample = ds[0]
```

Variable palette cardinality is handled by `collate_ai2bmp_batch`.
