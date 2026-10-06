# Range cards (format `deepearth-range-card/2`)

A range card stores a species' 240 m CONUS range maps as the model that generates them. Decoding re-renders the
maps on GPU and is byte-identical to the original render.

## Why
| Storage of one species (3 maps, CONUS 240 m, 20,149 × 13,053 cells) | Size |
|---|---|
| Three uint8 GeoTIFFs, tiled, ZSTD | 13–20 MB |
| Range card | median 3.4 kB (faithful) / 6.5 kB (occurrence-trained); median 2,912× / 1,542× smaller, at least 80× |

Shared, one-time: the CONUS predictor stack (WorldClim: 21.0 GB raw float32, 7.2 GB as ZSTD GeoTIFFs; fine stack:
25.2 GB) and the ecoregion-id layer (1.5 MB). For ~5,000 species the total is dominated by the shared stack.

## Contents
A zstd-compressed tar with:
- `meta.json`
  - `format`: `deepearth-range-card/2`
  - `label`, `variables` (predictor order of the lambdas), `grid` (`EPSG:5070 240 m NLCD-aligned`), `shape` ([13053, 20149])
  - `ecoregion_ids`: the calibration area, as RESOLVE Ecoregions 2017 `ECO_ID`s resolved against the shared layer
    (lakes = 0)
  - `thresholds_ess`: per-replicate cloglog thresholds (equal training sensitivity and specificity)
  - `thresholds_p5`: per-replicate 5th-percentile training-presence thresholds
  - `sha256_uint8`: SHA-256 of each decoded map's raw bytes (`suitability`, `binary_vote`, `binary_p5`)
- `<label>_<k>.lambdas`, k = 0..4: maxent.jar's fitted models, verbatim.

## Decoding
For every cell whose ecoregion id is in `ecoregion_ids` and whose predictors are finite:
1. each replicate's cloglog suitability from its lambdas (`MaxentModel`, clamped to training ranges);
2. `suitability` = 1 + round(254 × median over replicates) (0 = outside the calibration area);
3. `binary_vote` = 1 + [majority of replicates ≥ their `thresholds_ess`] (1 = absent, 2 = present);
4. `binary_p5` = 1 + [median suitability ≥ median of `thresholds_p5`].

Exactness: per-feature contributions are summed element-wise in a fixed order, so each cell's value does not depend
on how many cells are evaluated together (`MaxentModel.linear_predictor`). Any window decodes byte-identically to the same window of the full render (`codec.decode_window`; ~10 ms per 256 × 256 window on an
RTX 3090). `codec.verify` checks a full decode against `sha256_uint8`.
