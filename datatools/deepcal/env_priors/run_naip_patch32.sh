#!/usr/bin/env bash
set -euo pipefail

: "${DEEPCAL_CACHE:=/workspace/deepcal-cache-v5}"
: "${NAIP_SAVE_PATCH32:=1}"
: "${NAIP_PATCH_DTYPE:=float16}"
: "${NAIP_PATCH_VIEW:=rgb}"
: "${NAIP_BATCH_TILES:=16}"
: "${NAIP_DLW:=8}"
: "${NAIP_EMBED_BATCH:=4}"
: "${NAIP_FETCH_TIMEOUT:=600}"
: "${NAIP_PATCH_ROWS:=16}"
: "${NAIP_SAVE_IMAGERY:=0}"
: "${USGS_M2M_TOKEN:=/root/.usgs_m2m_token}"
: "${NAIP_TILES_JSON:=$DEEPCAL_CACHE/env_priors/naip2024_tiles.json}"
: "${NAIP_CATALOG_BACKEND:=stac}"

export DEEPCAL_CACHE
export NAIP_SAVE_PATCH32
export NAIP_PATCH_DTYPE
export NAIP_PATCH_VIEW
export NAIP_BATCH_TILES
export NAIP_DLW
export NAIP_EMBED_BATCH
export NAIP_FETCH_TIMEOUT
export NAIP_PATCH_ROWS
export NAIP_SAVE_IMAGERY
export USGS_M2M_TOKEN
export NAIP_TILES_JSON
export NAIP_CATALOG_BACKEND

python datatools/deepcal/env_priors/preflight_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --dtype "$NAIP_PATCH_DTYPE" \
  --require-hf

if [[ ! -f "$NAIP_TILES_JSON" ]]; then
  if [[ "$NAIP_CATALOG_BACKEND" == "stac" ]]; then
    python datatools/deepcal/env_priors/build_naip_stac_catalog.py \
      --cache "$DEEPCAL_CACHE" \
      --out "$NAIP_TILES_JSON"
  elif [[ ! -f "$USGS_M2M_TOKEN" ]]; then
    echo "missing USGS_M2M_TOKEN=$USGS_M2M_TOKEN and no NAIP catalog exists" >&2
    exit 2
  else
    python datatools/deepcal/env_priors/build_naip_m2m2024_catalog.py
  fi
fi

python datatools/deepcal/env_priors/preflight_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --dtype "$NAIP_PATCH_DTYPE" \
  --require-catalog \
  --require-hf

python datatools/deepcal/env_priors/build_naip_m2m2024.py
python datatools/deepcal/env_priors/verify_naip_patch32.py --cache "$DEEPCAL_CACHE"
