#!/bin/bash
# Full driver build. Long steps run as jobs (ops/jobs.sh); this script is the record of the order and arguments.
#   1. ERA5 and ERA5-Land point extraction from the Open-Meteo mirror (forward jobs take year files first; reverse
#      jobs take chunk files from the newest; each task skips a file already complete).
#   2. Per-site tables.
#   3. Copernicus DEM terrain.
#   4. HRRR, IMERG and MRMS at the validation towers.
#   5. Validation against the tower data (final hh/ when present, interim otherwise).
set -e
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
PY=${PY:-/home/photon/4tb/envs/fluxfill-drivers/bin/python}
PY312=${PY312:-/home/photon/4tb/envs/fluxfill-drivers312/bin/python}
ROOT=/home/photon/4tb/fluxfill/drivers
case "$1" in
  extract)
    $PY openmeteo.py extract --source "$2" --workers "${3:-28}" ${4:+--kinds chunk --reverse};;
  assemble)
    for s in era5_land era5; do $PY assemble.py --source $s --root $ROOT; done;;
  dem)
    $PY312 dem.py --workers 6;;
  validate)
    T=/home/photon/4tb/fluxfill/towers/hh; [ -d $T ] || T=/home/photon/4tb/fluxfill/towers/interim/shuttle
    $PY312 validate.py --towers $T --out $ROOT/validation/full;;
  *) echo "usage: build_all.sh extract <era5|era5_land> [workers] [reverse] | assemble | dem | validate"; exit 1;;
esac
