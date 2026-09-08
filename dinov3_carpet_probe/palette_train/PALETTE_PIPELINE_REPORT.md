# Palette Prediction Pipeline Report

## 1. Executive Summary

This project implements a supervised palette-prediction pipeline on top of a frozen `vitl16_lvd` DINOv3 backbone. The input is one JPG image and the output is an unordered RGB palette with variable cardinality. The modeling choice is a DETR-style query decoder with Hungarian matching, because palette colors have no meaningful order and the number of colors varies across samples.

The core design remains:

- frozen DINOv3 visual backbone
- learned query-based palette set head
- one-to-one Hungarian matching
- permutation-invariant training and evaluation

The pipeline has been iteratively strengthened with dataset filtering, progress tracking, inference UI support, better early-stopping behavior, mask-aware preprocessing, a raw-color side-channel, and a unified set-level metric for checkpoint selection.

## 2. Task Definition

The target task is:

`JPG image -> frozen DINOv3 feature extractor -> trainable palette set head -> unordered RGB palette`

The output is serialized as:

```json
{
  "sample_id": 0,
  "num_colors": 10,
  "rgb": [[4, 4, 12], [4, 11, 105]],
  "hex": ["#04040C", "#040B69"]
}
```

Key constraints:

- DINOv3 stays frozen and untouched
- training uses only palette supervision, not segmentation or reconstruction
- palette order in JSON must not affect loss or metrics
- inference emits unique integer RGB values plus exact uppercase hex

## 3. Dataset And Validation

### 3.1 Source data

- Images: `E:/End-to-End/Data/jpg`
- Palette annotations: `E:/End-to-End/Data/palettes`
- Pairing: strictly by filename stem

### 3.2 Validation rules

Each pair is validated for:

- matching numeric `sample_id`
- `num_colors == len(rgb) == len(hex)`
- each RGB item has exactly three integers in `[0, 255]`
- each hex string matches its RGB counterpart exactly
- duplicate image stems and duplicate palette stems
- duplicate RGB values inside a palette

### 3.3 High-cardinality policy

After review, training was restricted to samples with at most `12` colors:

- samples with `num_colors > 12` are excluded from the trainable dataset
- exclusions are reported explicitly
- `max_num_colors` is therefore fixed to `12` for the trainable model

This was introduced to avoid letting a small number of extremely large palettes determine the architecture capacity and dominate presence-loss imbalance.

### 3.4 Preflight analysis

A dedicated preflight stage was added to report:

- raw cardinality distribution
- counts above `16`, `32`, `64`, and `128`
- example excluded stems
- visibility of GT palette colors in the original JPG
- visibility of GT palette colors in the resized 512x512 training view

Visibility is measured as the minimum `Delta E00` between each target color and all source pixels.

## 4. Preprocessing

The training transform now performs:

1. RGB load
2. aspect-ratio-preserving resize
3. square letterbox to fixed canvas
4. normalization with the exact `vitl16_lvd` statistics

Important changes:

- every training sample is transformed to a fixed tensor shape
- a valid-content mask is emitted alongside the image
- the raw pre-normalization tensor is preserved for chromatic side-channel extraction
- padding is no longer treated as valid evidence for palette prediction

## 5. Backbone And Feature Extraction

The backbone is `vitl16_lvd`, loaded through the official public API and frozen:

- all parameters have `requires_grad=False`
- the model remains in `eval()` mode
- no DINOv3 source file is modified

Feature extraction uses public intermediate-layer access:

- normalized patch tokens
- class token
- multi-level blocks `[5, 11, 17, 23]`

These are intentionally described as multi-level blocks, not “the last four blocks”.

## 6. Head Architecture

The palette head is query-based:

- `Q = max_num_colors + spare_queries`
- with the current trainable setting: `Q = 12 + 4 = 16`

Each query predicts:

- `pred_rgb` in `[0, 1]`
- `presence_logit`

The model also predicts:

- `count_logits`

### 6.1 Raw-color side-channel

To make precise color prediction easier, a raw-color branch was added alongside DINO features. For each patch, the system derives compact patch-level color statistics from the unnormalized input, including:

- mean RGB
- RGB standard deviation
- RGB minimum
- RGB maximum
- mean OKLab

These are concatenated with the DINO patch representation before projection into decoder memory.

### 6.2 Mask-aware count prediction

Count prediction no longer relies only on CLS. The count branch now combines:

- projected CLS token
- masked mean pooling over valid patches
- masked attention pooling over valid patches

This makes count prediction less sensitive to padding and more sensitive to sparse rare colors.

## 7. Patch Masking And Letterbox Safety

A valid-pixel mask is propagated from image space into patch space. This patch mask is used to build:

- `memory_key_padding_mask` for transformer cross-attention
- masked pooling for count prediction

The purpose is to ensure that padded letterbox regions:

- do not become fake palette evidence
- do not distort count pooling
- do not attract palette queries as if they were real image content

## 8. Matching

The palette is treated as a set. Matching is performed with Hungarian assignment between predicted queries and target colors.

The matching cost combines:

- normalized RGB L1 distance
- perceptual OKLab distance
- a stable confidence term derived from `logsigmoid(presence_logits)`

This keeps target order irrelevant and stabilizes the matching objective.

## 9. Losses

The main losses are:

- matched RGB Smooth L1
- matched perceptual OKLab loss
- presence loss
- count cross-entropy
- count consistency loss
- auxiliary decoder-layer losses

### 9.1 Per-image normalization

Matched color losses are averaged per image before being averaged across the batch. This avoids giving disproportionate influence to images with larger palettes.

### 9.2 Presence imbalance handling

Presence supervision now uses a focal-style formulation. Positive and negative presence terms are logged separately to make imbalance visible during training.

## 10. Metrics

The pipeline reports permutation-invariant metrics after Hungarian alignment:

- count accuracy
- count MAE
- matched RGB MAE mean and median
- `Delta E00` mean, median, P90, P95
- percentage of matches with `Delta E00 <= 1 / 2 / 5`
- exact-cardinality rate
- palette success rate
- false-positive query rate
- false-negative palette-color rate

### 10.1 Count agreement metrics

Because count is represented both explicitly and implicitly, the pipeline also measures:

- count-head accuracy
- presence-derived count accuracy
- absolute gap between `argmax(count_logits)` and `round(sum(sigmoid(presence_logits)))`

### 10.2 Unified set-level error

A unified `set_error` metric was added so checkpoint selection is not driven only by matched-color error:

`set_error = (sum(min(DeltaE00_matched, c)) + c * abs(K - K_hat)) / max(K, K_hat)`

with a default penalty constant `c = 10`.

This metric is now the preferred signal for:

- best checkpoint selection
- early stopping
- model comparison

## 11. Training Loop

The trainer includes:

- AdamW
- warmup + cosine LR schedule
- AMP on CUDA
- gradient clipping
- gradient accumulation
- deterministic seed handling
- validation
- checkpointing
- resume support

### 11.1 Progress tracking

Comprehensive progress reporting was added:

- dataset validation progress bar
- training epoch/batch progress bar
- validation progress bar
- per-batch loss display
- learning-rate display
- checkpoint and early-stopping summary lines

## 12. Early Stopping

Early stopping was made harder to trigger:

- increased patience
- minimum epoch gate before patience is consumed
- minimum improvement delta required to reset patience

This makes stopping more conservative and less sensitive to small metric fluctuations.

## 13. Inference

Inference proceeds as follows:

1. predict count with `count_logits`
2. rank queries by presence probability
3. take the top unique RGB predictions
4. round to 8-bit integers
5. serialize RGB + HEX

The system also includes a minimal Gradio UI where the user can:

- provide a checkpoint path
- upload an image
- view palette colors as swatches
- inspect the JSON output

## 14. Testing

Focused tests cover:

- pairing and missing-pair detection
- schema and RGB/HEX consistency
- collation behavior
- head output shapes
- matching correctness
- permutation invariance
- presence-target behavior
- count loss
- checkpoint load/save
- JSON output format
- transform shape guarantees
- patch mask effect on count behavior
- preflight visibility primitive
- unified set error

At the latest verification point, the automated test suite status was:

- `17 passed`
- `2 skipped`

## 15. Current Caveats

Some parts are implemented as infrastructure and gates but still require a fresh real-data run to produce final empirical evidence:

- full preflight report on the latest dataset state
- actual tiny-overfit success numbers on the current revised pipeline
- updated narrative report populated with new runtime numbers

In other words, the code path exists for these checks, but the new results must still be generated after the latest architectural and data-pipeline changes.

## 16. Conclusion

The project now has a substantially more robust foundation than the initial version:

- safer dataset handling
- explicit exclusion of oversized palettes
- stronger mask-aware preprocessing
- raw-color support beside DINO structure features
- better loss balancing
- more meaningful checkpoint selection
- stronger diagnostics and reporting

The next recommended operational sequence is:

1. run preflight on the real dataset
2. run tiny-overfit and verify near-memorization behavior
3. launch controlled training on the filtered `<= 12` dataset
4. analyze validation metrics with the new set-level criterion
