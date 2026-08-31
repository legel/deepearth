# NAIP DINOv3 patch32 inputs

Goal: produce DINOv3-SAT493M patch embeddings shaped `(32, 32, 1024)` for every train/test observation that has readable NAIP coverage.

Required cache inputs:

- `gbif_tokens/*.npz`
  - required keys: `gbifID`, `lat`, `lon`
  - optional key: `ord`
- `gbif_elev.npz`
  - optional keys: `gbifID`, `elev`
- `gbif_eventtime.npz`
  - optional keys: `gbifID`, `days` or `event_day`

Remote-sensing input options:

1. Public STAC catalog
   - default runner backend
   - builds `env_priors/naip2024_tiles.json` from Planetary Computer NAIP scene URLs
   - uses the available public acquisition year per location

2. USGS M2M token
   - place token at `/root/.usgs_m2m_token`, or set `USGS_M2M_TOKEN=/path/to/token`
   - set `NAIP_CATALOG_BACKEND=m2m`
   - the runner builds a 2024-pinned `env_priors/naip2024_tiles.json` where M2M access is available

3. Prebuilt scene catalog
   - place at `$DEEPCAL_CACHE/env_priors/naip2024_tiles.json`, or set `NAIP_TILES_JSON=/path/to/catalog.json`
   - each entry must include:
     - `entityId`
     - `displayId`
     - `bbox: [lon0, lat0, lon1, lat1]`
   - each entry may also include one of:
     - `local_path`: existing local GeoTIFF/JP2 path
     - `url`: already-authorized download URL

Best input for Lance to provide:

- A `naip2024_tiles.json` catalog with `local_path` entries pointing to mounted NAIP 2024 CNIR scenes.
- Or the same catalog with signed `url` entries if scenes are hosted remotely.

That avoids repeated catalog/download negotiation and lets the builder stream patches directly from the provided scenes.

DINOv3 model input options:

1. Hugging Face Transformers
   - default backend
   - requires an HF token with accepted access to `facebook/dinov3-vitl16-pretrain-sat493m`, or an existing local HF cache

2. Official facebookresearch/dinov3 repo
   - set `DINOV3_BACKEND=hub`
   - optionally set `DINOV3_REPO=/path/to/dinov3`
   - set `DINOV3_WEIGHTS=/path/to/dinov3_vitl16_sat493m-eadcf0ff.pth` or an accepted direct weights URL
   - uses SAT-493M normalization: mean `(0.430, 0.411, 0.296)`, std `(0.213, 0.156, 0.143)`

The official repo path does not bypass model access. It lets us use a local `.pth` or accepted weights URL if Lance/Meta provides one.
