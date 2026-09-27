# Miakhi style assets (designer-approved)

`miakhi` is a **decorative** postprocess profile. It is **unavailable** until you
provide approved examples or explicit rules below. Conservative cleanup is **not**
miakhi — the UI/report will say `available=false` rather than pretend.

## Directory layout

```
Transfer/styles/miakhi/
  README.md          ← this file
  rules.json         ← optional explicit designer rules
  examples/
    pair_001/
      before.bmp     ← input / editable region context (palette gray)
      after.bmp      ← approved shaded result (palette {0,100,150,200,255})
    pair_002/
      before.bmp
      after.bmp
  patterns/          ← optional small indexed tiles cropped from approved work
    motif_a.bmp
```

## Palette

All BMP/PNG files must use only grayscale values:

`{0, 100, 150, 200, 255}`

- `0` = black outline (locked)
- `255` = white / unshaded
- `200` / `150` / `100` = shade classes

Do **not** supply RGB or anti-aliased edges. Use nearest-neighbor exports.

## `rules.json` (optional)

Example:

```json
{
  "name": "miakhi",
  "notes": "Designer-approved remap / constraints only — no invented geometry.",
  "class_remap": {
    "200": "150"
  },
  "min_template_overlap": 0.05
}
```

`class_remap` maps gray values inside eligible regions when no pattern templates
are available. Prefer providing `examples/` or `patterns/` so orientation,
spacing, and lengths come from real approved art.

## How the postprocessor uses these files

1. Loads templates from `examples/*/after.*` and `patterns/*`.
2. For each **reliable motif region** (from the input outline + editable mask),
   selects a D4-aligned (rot90 / mirror) template by discrete nearest-neighbor
   scoring — **not** one global stamp on the whole carpet.
3. Compares the proposal to the model’s four class scores; only uncertain /
   well-supported pixels change.
4. Restores blacks and everything outside the editable mask.

If this folder is empty of rules and examples, selecting `miakhi` reports:

> miakhi unavailable: … Do not invent miakhi geometry from the name.

## Editable mask requirement

BW inputs alone do **not** define an unambiguous editable region for decorative
styles. Provide an editable mask (or indexed `X`) when running `miakhi`.
