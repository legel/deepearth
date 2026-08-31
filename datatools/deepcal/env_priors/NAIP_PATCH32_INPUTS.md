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

1. USGS M2M token
   - place token at `/root/.usgs_m2m_token`, or set `USGS_M2M_TOKEN=/path/to/token`
   - the runner builds `env_priors/naip2024_tiles.json`

2. Prebuilt scene catalog
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

That avoids repeated M2M catalog/download negotiation and lets the builder stream patches directly from the provided scenes.
