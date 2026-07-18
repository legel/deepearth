# AlphaEarth geo prior — REQUIRED for the DeepCal champion

The current DeepCal champion (**arithmetic mean 0.6074**, vs 0.5962 without) depends on
**AlphaEarth** ([Google Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL),
Brown et al. 2025) — a 64-dim, 10 m, annual learned embedding of the land surface.

Unlike the other modalities (which the model reconstructs), AlphaEarth is used as a **SatCLIP-style learned
geo prior**: a small MLP projects the 64-dim embedding and *adds it to the spatial position representation that
every head reads* (alongside the RFF `smooth_geo` prior), so it informs all spatial reasoning instead of
competing for head capacity. It is enabled by `model.alphaearth_geo: true` in `autoresearch/champion.yaml`.

> **Ablation (single seed, spatial holdout, both 43 active benchmarks, full 16k steps):**
> `alphaearth_geo: true` → **arith 0.6074 / net 0.3311**, vs control `false` → **0.5962 / 0.3014** (**+0.0112 arith, +0.0297 net**).
> As an ordinary *reconstruction variable* AlphaEarth instead *hurts* the arithmetic mean (−0.004…−0.006) while
> still lifting net_score — placement as a geo prior is what converts the signal into an arithmetic-mean gain.

## Reproduce

1. **Auth Earth Engine** for a project you own: `earthengine authenticate` (register the project for Earth Engine).
   `pip install earthengine-api numpy`.
2. **Build the coords file** `ae_coords.npz` with per-observation `gbifID`, `lat`, `lon` (extract these from your
   prepared cache / observation table).
3. **Extract** (≈25–45 min for ~620k points, resumable):
   ```bash
   EE_PROJECT=<your-ee-project> AE_COORDS=ae_coords.npz \
   AE_OUT=gbif_alphaearth_tokens.npz python recipes/alphaearth/extract_alphaearth.py
   ```
4. **Place** `gbif_alphaearth_tokens.npz` in your DeepCal cache dir (`data.cache_dir` in `deepcal.yaml`).
   `autoresearch/data.py::_load_modalities` loads it automatically as the `alphaearth` modality — aligned by
   `gbifID`, z-scored on the train split, missing rows (no coverage) masked. **If the file is absent the modality
   is silently skipped and the champion will NOT reproduce** (you get the ~0.596 control).
5. **Train** with the champion config (`alphaearth_geo: true` is already set):
   ```bash
   python autoresearch/train.py autoresearch/champion.yaml --cache_dir data/deepcal --tag deepcal-champion
   ```

## Compatibility / graceful degradation

- The code is **backward compatible**: without the tokens file, `alphaearth_geo` finds no `alphaearth` values and
  is a no-op; every other config runs unchanged. `alphaearth_geo` defaults to `false`.
- The extractor writes NaN for points outside AlphaEarth coverage (e.g. offshore); the loader masks those.
- Reference extent used for the CA champion: lat 32.5–42° N, lon −124…−114° W, ~621k obs, 99.98% sampled.
