# Engineering report — Shadowing-v1 / CSN-V4 indexed_guided

Base commit reviewed: `813d0504d0c901a6e4befa4f7181764e0fdfe441` (identical to current `main` tip at start of this work).

## Problems fixed → change → evidence

| # | Problem | Change | Test / check |
|---|---------|--------|----------------|
| 1 | No `indexed_guided` contract | Package `csn_v4/indexed/` (schema, gray adapter v1, X←Y, compose lock, structural 3-ch) | `tests/csn_v4/test_indexed_guided.py` (8 contract tests) |
| 2 | Stem only 1-ch BW | `HighResStructuralStem(in_channels=1\|3)` | Import/construction via model |
| 3 | `context_dino_adapter` unused | Wired in `csn_v4/fusion.py` + `csn_v3/fusion.py` | `test_context_dino_adapter_is_used` |
| 4 | `audit_backward` polluted train grads | Save/restore mode, `zero_grad` in `finally`; train fails if audit fails | `shape_audit.py` + train script |
| 5 | `best.pt` from train spatial | Generalization: `best.pt` only from scene holdout; else `unevaluated` | `train_csn_v4.py` selection logic |
| 6 | Text path replace in setup | Structural YAML load/dump + verify | `patch_config` smoke OK |
| 7 | Halo ignored / global HALO in core mask | `TileSpec.halo`; `resolve_geometry`; YAML validated at load | `test_geometry_halo_on_spec_and_odd_small` |
| 8 | NPZ `allow_pickle=True` | Meta as `np.str_`; `allow_pickle=False` | Code review `materialize.py` |
| 9 | Checkpoint BW↔indexed silent | `class_mapping.input_mode` + schema verify | Checkpoint load raises on mismatch |
| 10 | Seed residual “for chart” | Removed `_seed_residual_deltas` from model | Model no longer seeds deltas |
| 11 | GradScaler vs scheduler drift | Scheduler advances only if scaler step accepted | Train loop |
| 12 | Test package shadowed `csn_v4` | Removed `tests/csn_v4/__init__.py`; added `tests/conftest.py` | Suite collects correctly |

## Preserved (already correct)

- Group/family/sha split leakage checks (`csn_v4/data/splits.py`)
- Atomic checkpoints; DINO backbone stripped from `.pt`
- `dataset_v2` present under `Transfer/dataset_v2/`
- BW mode remains a separate `input_mode` (not converted silently)

## Configs / docs added

- `configs/csn_v4_indexed_guided.yaml` — baseline masked CE+Dice
- `docs/INDEXED_GUIDED.md` — data contract
- `docs/RUNBOOK.md` — deps + commands
- This report

## Tests actually run

```
pytest tests/csn_v4/test_indexed_guided.py tests/csn_v4/test_coverage.py
→ 11 passed
```

Full suite may still require local Dataset_V3 for some older tests.

## Not evaluated in this environment (explicit)

| Item | Why |
|------|-----|
| Full GPU train E2E on real carpets | Heavy; needs CUDA session + data materialize time |
| Ablations (aux heads, border distance, unfreeze DINO tail) | Require fixed split + GPU hours — configs flags ready (`enable_*`) |
| Independent-design IoU tables | Need holdout scenes + trained indexed checkpoint |
| Global DINO disk cache hit-rate | Cache dir still configured; live encode remains default until cache writer wired for indexed RGB |
| Soft logit blending at seams | Core-paste kept; no seam failure measured yet → blending not enabled |

## Recommended next measured run

1. Build/adapt index maps (or gray adapter) for a small holdout set.
2. Train `csn_v4_indexed_guided.yaml` short run (e.g. 200 steps) on CUDA.
3. Report per-class IoU inside M on **scene holdout**; confirm outside-M pixel identity = 100%.
4. Only then ablate aux heads / border features with the same split id recorded in the checkpoint.
