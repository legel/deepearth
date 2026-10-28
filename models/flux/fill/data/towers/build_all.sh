#!/bin/bash
# Build the whole tower dataset in order. Every stage resumes: finished sites and logged downloads are skipped.
#   RAW=/home/photon/4tb/fluxfill/raw/towers OUT=/home/photon/4tb/fluxfill/towers PY=python ./build_all.sh
# With WAIT_PID set, waits for that process (a Shuttle stage already running) before starting.
set -euo pipefail
cd "$(dirname "$0")"
RAW=${RAW:?} OUT=${OUT:?} PY=${PY:-python}
if [ -n "${WAIT_PID:-}" ]; then while [ -e /proc/$WAIT_PID ]; do sleep 60; done; fi
echo "[$(date -u +%FT%TZ)] fetch"
# A failed fetch (network) leaves the files already logged; the build goes on with them.
$PY fetch.py ch4 --raw "$RAW" || echo "fetch ch4 failed"
$PY fetch.py base --raw "$RAW" || echo "fetch base failed"
$PY fetch.py icos --raw "$RAW" || echo "fetch icos failed"
if [ -f "$RAW/koppen/koppen_geiger_tif.zip" ] && [ ! -d "$RAW/koppen/1991_2020" ]; then
  unzip -q -o "$RAW/koppen/koppen_geiger_tif.zip" '1991_2020/*' -d "$RAW/koppen"
fi
echo "[$(date -u +%FT%TZ)] shuttle (resume, then one retry pass for failures)"
$PY build.py shuttle --raw "$RAW" --out "$OUT" --workers "${WORKERS:-5}"
$PY build.py shuttle --raw "$RAW" --out "$OUT" --workers "${WORKERS:-5}"
for s in fluxnet_ch4 amf_base icos_l2; do
  echo "[$(date -u +%FT%TZ)] clean $s"
  $PY build.py clean --source $s --raw "$RAW" --out "$OUT"
done
echo "[$(date -u +%FT%TZ)] merge"
$PY build.py merge --raw "$RAW" --out "$OUT"
echo "[$(date -u +%FT%TZ)] tables"
$PY build.py tables --raw "$RAW" --out "$OUT"
echo "[$(date -u +%FT%TZ)] split"
$PY split.py --out "$OUT"
$PY report.py --out "$OUT"
echo "[$(date -u +%FT%TZ)] done"
