#!/usr/bin/env bash
set -euo pipefail

: "${DEEPCAL_CACHE:=/workspace/deepcal-cache-v5}"
: "${HF_HOME:=/workspace/.hf_home}"
: "${NAIP_TILES_JSON:=$DEEPCAL_CACHE/env_priors/naip2024_tiles.json}"
: "${NAIP_SAVE_PATCH32:=1}"
: "${NAIP_PATCH_DTYPE:=float16}"
: "${NAIP_PATCH_COMPRESSED:=0}"
: "${NAIP_PATCH_VIEW:=rgb}"
: "${NAIP_BATCH_TILES:=8}"
: "${NAIP_DLW:=4}"
: "${NAIP_EMBED_BATCH:=4}"
: "${NAIP_SAVE_IMAGERY:=0}"
: "${NAIP_WATCHDOG_LOG:=/workspace/logs/naip_patch32_watchdog.log}"
: "${NAIP_RUN_LOG:=/workspace/logs/naip_patch32_full.log}"
: "${NAIP_IDLE_SECONDS:=300}"

mkdir -p "$(dirname "$NAIP_WATCHDOG_LOG")"
lock=/tmp/naip_patch32_watchdog.lock
if ! mkdir "$lock" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$lock"' EXIT

while true; do
  if ! pgrep -f "[b]uild_naip_m2m2024.py" >/dev/null; then
    {
      date -Is
      echo "starting build_naip_m2m2024.py"
    } >> "$NAIP_WATCHDOG_LOG"
    (
      cd /workspace/deepearth
      env \
        HF_HOME="$HF_HOME" \
        DEEPCAL_CACHE="$DEEPCAL_CACHE" \
        NAIP_TILES_JSON="$NAIP_TILES_JSON" \
        NAIP_SAVE_PATCH32="$NAIP_SAVE_PATCH32" \
        NAIP_PATCH_DTYPE="$NAIP_PATCH_DTYPE" \
        NAIP_PATCH_COMPRESSED="$NAIP_PATCH_COMPRESSED" \
        NAIP_PATCH_VIEW="$NAIP_PATCH_VIEW" \
        NAIP_BATCH_TILES="$NAIP_BATCH_TILES" \
        NAIP_DLW="$NAIP_DLW" \
        NAIP_EMBED_BATCH="$NAIP_EMBED_BATCH" \
        NAIP_SAVE_IMAGERY="$NAIP_SAVE_IMAGERY" \
        /venv/main/bin/python datatools/deepcal/env_priors/build_naip_m2m2024.py \
          > "$NAIP_RUN_LOG" 2>&1
    ) &
  fi
  sleep "$NAIP_IDLE_SECONDS"
done
