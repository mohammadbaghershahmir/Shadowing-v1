# Dataset V2 prepare + global cache + overfit train (PowerShell)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Allah\Gigazaki\Scripts\python.exe"
$env:PYTHONPATH = "E:\Dino\dinov3;E:\Shadowing\code"
Set-Location E:\Shadowing\code

Write-Host "=== 1) Prepare Dataset V2 (~10k crops, margin=0, grid+corners, bake D4) ==="
& $py scripts/prepare_shadow_dataset.py `
  --input-dir "E:\Shadowing\converted_ready" `
  --target-dir "E:\Shadowing\converted_output" `
  --output-dir "E:\Shadowing\dataset_metadata_v2" `
  --crop-size 512 --valid-margin 0 `
  --include-spatial-grid --allow-padded-origins --bake-dihedral `
  --merge-train-val-crops --target-train-crops 10000 `
  --min-crops-per-image 32 --max-crops-per-image 512 `
  --grid-stride 256 --preview-count 40 --overwrite --seed 42

Write-Host "=== 2) Cache global DINO tokens for V2 (all dihedral k) ==="
& $py scripts/cache_global_features.py `
  --config configs/shadow_overfit.yaml `
  --dihedral-variants 8

Write-Host "=== 3) Overfit train (score on train mIoU, no val) ==="
& $py scripts/train_shadow.py `
  --config configs/shadow_overfit.yaml `
  --profile BEST_V1 `
  --mode overfit `
  --device cuda

Write-Host "Done. Use runs\*_BEST_V1_overfit\best.pt with app_shadow.py"
