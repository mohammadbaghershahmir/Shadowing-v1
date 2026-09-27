# CSN-V4 / Shadowing — runbook

## Tested dependency stack (document what you actually run)

| Package | Notes |
| ------- | ----- |
| Python 3.10+ | Conda env recommended |
| `torch` + `torchvision` | CUDA build matching your driver |
| `timm>=1.0` | ConvNeXtV2 |
| `PyYAML` | Config load/patch |
| `numpy`, `Pillow`, `tqdm` | Data / materialize |
| DINOv3 source | Local clone; `--dino-root` = **parent** of the `dinov3` package |

Install Transfer deps:

```powershell
cd E:\Shadowing\Transfer
python -m pip install -r requirements.txt
python -m pip install pyyaml pillow tqdm pytest
```

`dataset_v2` ships **inside** `Transfer/dataset_v2/` (no silent stubs).

## End-to-end (BW legacy)

```powershell
cd E:\Shadowing\Transfer
python setup_and_train.py --input-dir "PATH/converted_output" --work-root "PATH" --dino-root "PATH_TO_PARENT_OF_dinov3_PKG" --dino-checkpoint "PATH/weights.pth" --device cuda --steps 5000 --workers 0
```

`setup_and_train.py` now patches YAML **structurally** and re-reads the file to verify paths.

## Indexed-guided

See [INDEXED_GUIDED.md](INDEXED_GUIDED.md). Config: `configs/csn_v4_indexed_guided.yaml`.

## Resume / eval / infer

```powershell
python scripts/train_csn_v4.py --config configs/csn_v4_generalization_local.yaml --resume PATH/last.pt --device cuda --workers 0
python scripts/evaluate_csn_v4.py --config configs/csn_v4_generalization_local.yaml --checkpoint PATH/best.pt --device cuda
python scripts/app_csn_v4.py  # if available for interactive inference
```

## Checkpoint rules (generalization)

- `best.pt` ← **independent scene holdout only**
- `best_capacity.pt` / `best_spatial_diag.pt` ← diagnostic (train scenes)
- `best_scene_val.pt` ← holdout score
- `score_on_train` cannot promote train metrics into `best.pt` in generalization mode
- `input_mode` + class schema are embedded; mismatches raise

## Geometry

`crop_size = core_size + 2 * halo` is validated at config load. Halo is stored on `TileSpec`.

## Tests

```powershell
cd E:\Shadowing\Transfer
python -m pytest tests/csn_v4 -q
```
