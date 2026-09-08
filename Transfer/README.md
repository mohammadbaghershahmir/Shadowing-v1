# CSN-V4 Transfer Package

Self-contained setup: **only bring `converted_output`** (5-color BMP ground truth).  
V3 and V4 are built automatically on the new machine — no need to copy Dataset_V1/V2/V3.

## What you need on the new system

| Item | Description |
|------|-------------|
| **This `Transfer/` folder** | Code + configs + scripts |
| **`converted_output/`** | Raw GT BMPs (5 colors: black, white, gray 200/150/100) |
| **DINOv3 repo** | Local clone of dinov3 |
| **DINOv3 checkpoint** | `.pth` weights file |
| **Python 3.10+** | With CUDA PyTorch |

Optional: `converted_ready/` (magenta guides) — only if you use magenta training mode.

## Input format (`converted_output`)

Each file is a BMP with exactly these RGB values:

| Color | RGB | Meaning |
|-------|-----|---------|
| OUTLINE | (0,0,0) | Black lines |
| UNSHADED | (255,255,255) | White background |
| INDEX_200 | (200,200,200) | Shade class 0 |
| INDEX_150 | (150,150,150) | Shade class 1 |
| INDEX_100 | (100,100,100) | Shade class 2 |

Files can live in subfolders; all `**/*.bmp` are discovered recursively.

## One-command full pipeline

```powershell
cd Transfer
pip install -r requirements.txt

python setup_and_train.py ^
    --input-dir "F:/Shadowing/converted_output" ^
    --work-root "F:/Shadowing" ^
    --dino-root "D:/Dino/dinov3" ^
    --dino-checkpoint "D:/Dino/dinov3/dinov3_carpet_probe/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" ^
    --device cuda ^
    --steps 5000 ^
    --workers 0
```

### What this runs automatically

1. **Dataset_V3** — `tools/build_dataset_v3.py`  
   Builds scene `samples/{id}/full/` (source_bw, masks, metadata) from your BMPs.
2. **Dataset_V4** — `scripts/prepare_csn_v4_dataset.py`  
   Train/val splits, capacity tiles, scene holdout manifests.
3. **Materialize** — pre-caches NPZ tiles for fast training.
4. **Train** — `scripts/train_csn_v4.py`.

### Output layout (under `--work-root`, default = parent of `converted_output`)

```
F:/Shadowing/
├── converted_output/     ← you provide this
├── Dataset_V3/           ← built automatically (intermediate)
│   └── samples/*/full/
└── Dataset_V4/           ← built automatically
    ├── manifests/
    └── materialized/

Transfer/runs/            ← checkpoints, metrics.csv, plots
```

## Useful flags

| Flag | Purpose |
|------|---------|
| `--sample-count 24` | How many scenes to sample from BMPs |
| `--skip-v3` | Reuse existing Dataset_V3 |
| `--skip-materialize` | Skip NPZ cache (slower training) |
| `--skip-train` | Only build datasets |
| `--resume path/to/best.pt` | Continue training |

## Manual steps (if needed)

```powershell
# V3 only
python tools/build_dataset_v3.py ^
    --input-dir "F:/Shadowing/converted_output" ^
    --dataset-root "F:/Shadowing/Dataset_V3" ^
    --sample-count 24 --seed 42

# V4 manifests only
python scripts/prepare_csn_v4_dataset.py ^
    --dataset-root "F:/Shadowing/Dataset_V4" ^
    --source-dataset "F:/Shadowing/Dataset_V3" ^
    --config configs/csn_v4_generalization_local.yaml

# Train / eval / UI
python scripts/train_csn_v4.py --config configs/csn_v4_generalization_local.yaml --device cuda --workers 0
python scripts/evaluate_csn_v4.py --checkpoint runs/.../best.pt --config configs/csn_v4_generalization_local.yaml
python scripts/app_csn_v4.py --checkpoint runs/.../best.pt --config configs/csn_v4_generalization_local.yaml
```
