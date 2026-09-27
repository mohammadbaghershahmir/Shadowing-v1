# Indexed-guided data preparation (schema `indexed_guided.v1`)

## Index contract (not colors)

| Index | Input `X`              | Output `Y`            |
| ----- | ---------------------- | --------------------- |
| 0     | background             | background (locked)   |
| 1     | outline                | outline (locked)      |
| 2     | allowed shading region | base fill             |
| 3     | — (never in `X`)       | first shadow          |
| 4     | — (never in `X`)       | second shadow         |

Rules:

1. Build `X` from `Y` by mapping `{3,4} → 2`; leave `{0,1,2}` unchanged.
2. Allowed mask `M = (X == 2)`.
3. Model predicts **3 internal classes** `{0,1,2}` ↔ indices `{2,3,4}` **only inside `M`**.
4. Final map = copy of `X` with `M` replaced by predicted indices. Outside `M` is bit-exact.

**Never confuse** index IDs with display gray (`DISPLAY_GRAY_V1`) or RGB (`DISPLAY_RGB_V1`).

## Inference editable region

At inference, pass the **source** index map `X∈{0,1,2}` (or an explicit `eligible_mask` in BW mode).  
**Do not** build the editable mask from ground-truth `Y`.

- Indexed path: `predict_full_image_index(model, source_index, device)` → composed `Y` with outside-`M` locked to `X`.
- BW-only path (no mask): omit `source_index` / `eligible_mask`; black outline is still locked from `source_bw==0`.

Geometric transforms apply identically to images and categorical masks (nearest-neighbor for indices).

## Legacy gray BMPs

Use versioned adapter `gray_legacy.v1` (`csn_v4.indexed.adapters.GrayLegacyAdapterV1`):

| Gray | Index |
| ---- | ----- |
| 0    | 1 (outline) |
| 255  | 0 (background) |
| 200  | 2 (allowed/base) |
| 150  | 3 (shadow_1) |
| 100  | 4 (shadow_2) |

Unknown gray values **raise**. Prefer storing native index maps going forward.

## Structural channels

From `X` only (never from `Y`):

- Channel 0: `X==0`
- Channel 1: `X==1`
- Channel 2: `X==2`

Pretrained RGB backbones use a **fixed** structural RGB palette + ImageNet norm (`indices_to_structural_rgb`). Local / context / global must share the same `X` and the same geometric transform.

## Modes

- `data.input_mode: bw` — legacy 4-class BW pipeline (unchanged checkpoints).
- `data.input_mode: indexed_guided` — this contract.

Loading a BW checkpoint under `indexed_guided` (or the reverse) **fails loudly**. Transferring backbone weights must be an explicit initialization, not a silent resume.

## Config

See `Transfer/configs/csn_v4_indexed_guided.yaml` (baseline = masked CE + Dice only).

```powershell
cd E:\Shadowing\Transfer
python setup_and_train.py `
  --input-dir "E:/Shadowing/converted_output" `
  --work-root "E:/Shadowing" `
  --dino-root "E:/Shadowing" `
  --dino-checkpoint "E:/Shadowing/weight/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" `
  --input-mode indexed_guided `
  --device cuda --steps 5000 --workers 0 --skip-train
```

Then train:

```powershell
python scripts/train_csn_v4.py `
  --config configs/csn_v4_indexed_guided.yaml `
  --device cuda --workers 0 --steps 5000 --skip-initial-eval `
  --run-dir "D:/Shadowing_runs/csn_v4_indexed"
```
