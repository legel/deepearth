# DINOv3 patch32 coverage

Current target: DINOv3-SAT493M patch embeddings shaped `(32, 32, 1024)` for
every train/test observation.

Current source: public NAIP/STAC scene URLs.

Coverage from the current manifest:

- train/test rows: `621,558`
- rows with a public NAIP/STAC candidate scene: `620,819`
- rows without a public NAIP/STAC candidate scene: `739`

The exact no-candidate rows are written on the data box at:

`/workspace/deepcal-cache-v5/gbif_naip_dinov3_patch32_v1/missing_naip_candidate_rows.csv`

Those `739` rows need a fallback remote-sensing source or an accepted masked
missing policy before the cache can truthfully satisfy real patch tensors for
all train/test rows.

Live extraction status is measured, not inferred, with:

```bash
python datatools/deepcal/env_priors/verify_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --coverage-only \
  --allow-prefix \
  --split-summary
```

Latest measured extraction coverage:

- train: `570,075 / 591,890`
- test: `26,722 / 29,668`
- total: `596,797 / 621,558`
- missing extracted rows: `24,761`

The final acceptance check is:

```bash
python datatools/deepcal/env_priors/verify_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --require-complete \
  --split-summary
```

If fallback remote-sensing chunks are needed for no-NAIP rows, verify the
combined artifact with:

```bash
python datatools/deepcal/env_priors/verify_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --fallback-dir gbif_sentinel2_dinov3_patch32_fallback_v1 \
  --require-complete \
  --split-summary
```
