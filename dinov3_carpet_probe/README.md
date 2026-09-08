# DINOv3 Carpet Probe

Zero-shot diagnostic experiment: what does DINOv3 see in carpet-map RGB images?

This package does **not** extract color palettes. It freezes official DINOv3 backbones, extracts dense features, and produces PCA / similarity / clustering / correspondence analyses for a go/no-go decision before any Lab-fusion palette pipeline.

## Environment (mandatory)

Use the **Gigazaki** venv only:

```powershell
C:\Users\Allah\Gigazaki\Scripts\Activate.ps1
cd E:\Dino\dinov3
$env:PYTHONPATH = "E:\Dino\dinov3"
```

Or call Python directly:

```powershell
C:\Users\Allah\Gigazaki\Scripts\python.exe
```

## Weight access (gated)

All three models are gated. Request access at:

https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

Also request access on Hugging Face:

- `facebook/dinov3-convnext-large-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-sat493m`

Then either:

1. Place `.pth` files under `dinov3_carpet_probe/weights/`, or
2. `huggingface-cli login` after HF approval.

Do **not** substitute DINOv2 or other unofficial checkpoints.

Optional Torch-Hub dependency (install into Gigazaki only):

```powershell
C:\Users\Allah\Gigazaki\Scripts\python.exe -m pip install termcolor ftfy regex submitit
```

## Models

Backbones with a local `.pth` under `weights/` appear in the UI automatically, including:

1. ConvNeXt Large LVD-1689M  
2. ViT-B/16 LVD-1689M  
3. ViT-L/16 LVD-1689M  
4. ViT-L/16 SAT-493M  

Primary loader: local Torch Hub. Fallback: Hugging Face Transformers.

## Interactive Studio (Gradio)

```powershell
C:\Users\Allah\Gigazaki\Scripts\Activate.ps1
cd E:\Dino\dinov3
$env:PYTHONPATH = "E:\Dino\dinov3"
$env:PYTHONUNBUFFERED = "1"
python dinov3_carpet_probe/scripts/launch_app.py --port 7860
```

Open http://127.0.0.1:7860 — select a model, upload an image, tune parameters, run the full suite (PCA / similarity / clustering / correspondence), or compare models.

## Commands

```powershell
C:\Users\Allah\Gigazaki\Scripts\Activate.ps1
cd E:\Dino\dinov3
$env:PYTHONPATH = "E:\Dino\dinov3"

python -m dinov3_carpet_probe.src.environment_check

python dinov3_carpet_probe/scripts/run_probe.py --validate-weights-only

python dinov3_carpet_probe/scripts/run_probe.py `
  --input "C:\path\to\carpet.png" `
  --config dinov3_carpet_probe/configs/experiment.yaml

python dinov3_carpet_probe/scripts/launch_app.py --port 7860

python dinov3_carpet_probe/scripts/launch_similarity_ui.py `
  --run-dir dinov3_carpet_probe/outputs/<run_id> `
  --image <stem>

python dinov3_carpet_probe/scripts/build_report.py `
  --run-dir dinov3_carpet_probe/outputs/<run_id>

python -m pytest dinov3_carpet_probe/tests -q
```

## Palette Training

The isolated palette-training pipeline lives under `dinov3_carpet_probe/palette_train/`. It uses a **frozen** `vitl16_lvd` backbone and trains only a DETR-style palette head for unordered RGB set prediction.

```powershell
C:\Users\Allah\Gigazaki\Scripts\Activate.ps1
cd E:\Dino\dinov3
$env:PYTHONPATH = "E:\Dino\dinov3"

python dinov3_carpet_probe/palette_train/scripts/validate_dataset.py `
  --config dinov3_carpet_probe/palette_train/configs/palette_train.yaml

python dinov3_carpet_probe/palette_train/scripts/train_palette.py `
  --config dinov3_carpet_probe/palette_train/configs/palette_train.yaml

python dinov3_carpet_probe/palette_train/scripts/train_palette.py `
  --config dinov3_carpet_probe/palette_train/configs/palette_train.yaml `
  --resume dinov3_carpet_probe/palette_train/runs/<run_id>/checkpoint_last.pt

python dinov3_carpet_probe/palette_train/scripts/eval_palette.py `
  --config dinov3_carpet_probe/palette_train/configs/palette_train.yaml `
  --checkpoint dinov3_carpet_probe/palette_train/runs/<run_id>/checkpoint_best.pt `
  --split test

python dinov3_carpet_probe/palette_train/scripts/predict_palette.py `
  --checkpoint dinov3_carpet_probe/palette_train/runs/<run_id>/checkpoint_best.pt `
  --input E:/End-to-End/Data/jpg/000000.jpg `
  --output-dir dinov3_carpet_probe/palette_train/predictions

python dinov3_carpet_probe/palette_train/scripts/predict_palette.py `
  --checkpoint dinov3_carpet_probe/palette_train/runs/<run_id>/checkpoint_best.pt `
  --input E:/End-to-End/Data/jpg `
  --output-dir dinov3_carpet_probe/palette_train/predictions

python -m pytest dinov3_carpet_probe/palette_train/tests -v

python dinov3_carpet_probe/palette_train/scripts/run_tiny_overfit.py `
  --config dinov3_carpet_probe/palette_train/configs/palette_train.yaml `
  --num-samples 16 --epochs 200
```

## Warnings

- Upsampled feature maps are **not** pixel-level predictions.
- PCA colors are artificial — not carpet colors.
- K-means clusters are structural feature clusters — not palettes.
