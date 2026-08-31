#!/usr/bin/env bash
set -euo pipefail

: "${DEEPCAL_CACHE:=/workspace/deepcal-cache-v5}"
: "${NAIP_SAVE_PATCH32:=1}"
: "${NAIP_PATCH_DTYPE:=float16}"
: "${NAIP_PATCH_VIEW:=rgb}"
: "${NAIP_BATCH_TILES:=4}"
: "${NAIP_DLW:=2}"
: "${NAIP_EMBED_BATCH:=2}"
: "${NAIP_PATCH_ROWS:=16}"
: "${NAIP_SAVE_IMAGERY:=0}"
: "${USGS_M2M_TOKEN:=/root/.usgs_m2m_token}"

export DEEPCAL_CACHE
export NAIP_SAVE_PATCH32
export NAIP_PATCH_DTYPE
export NAIP_PATCH_VIEW
export NAIP_BATCH_TILES
export NAIP_DLW
export NAIP_EMBED_BATCH
export NAIP_PATCH_ROWS
export NAIP_SAVE_IMAGERY
export USGS_M2M_TOKEN

if [[ ! -f "$USGS_M2M_TOKEN" ]]; then
  echo "missing USGS_M2M_TOKEN=$USGS_M2M_TOKEN" >&2
  exit 2
fi

python datatools/deepcal/env_priors/preflight_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --dtype "$NAIP_PATCH_DTYPE" \
  --require-usgs \
  --require-hf

if [[ ! -f "$DEEPCAL_CACHE/env_priors/naip2024_tiles.json" ]]; then
  python datatools/deepcal/env_priors/build_naip_m2m2024_catalog.py
fi

python datatools/deepcal/env_priors/preflight_naip_patch32.py \
  --cache "$DEEPCAL_CACHE" \
  --dtype "$NAIP_PATCH_DTYPE" \
  --require-catalog \
  --require-usgs \
  --require-hf

python datatools/deepcal/env_priors/build_naip_m2m2024.py
python datatools/deepcal/env_priors/verify_naip_patch32.py --cache "$DEEPCAL_CACHE"
