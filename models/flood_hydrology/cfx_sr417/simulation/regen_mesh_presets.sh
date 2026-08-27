#!/bin/bash
# Regenerate the mesh-solver viewer presets after the 2026-08-04 Manning friction fix
# (4/3 -> 7/3. Every number these files produce was computed
# under the OLD, dimensionally-wrong exponent -- this reruns them under the corrected one, exact
# same convention already established for these presets (Low=40mm/hr, Medium=100mm/hr,
# High=180mm/hr per site). Idempotent: safe to rerun.
#
# No associative arrays (macOS ships bash 3.2, which doesn't support -A) -- mm/hr looked up via
# a case statement instead.
set -euo pipefail
cd "$(dirname "$0")/.."   # cfx_sr417_corridor/

mm_for_level () {
  case "$1" in
    low) echo 40 ;;
    medium) echo 100 ;;
    high) echo 180 ;;
    *) echo "unknown level $1" >&2; exit 1 ;;
  esac
}

run_one () {
  local site=$1 level=$2 mmhr
  mmhr=$(mm_for_level "$level")
  local suffix=""; [ "$site" != "site1" ] && suffix="_${site}"
  echo "=== $site / $level (${mmhr}mm/hr) ==="
  python3 simulation/mesh_shallow_water.py --site "$site" --peak-rain-mm-hr "$mmhr"
  cp "simulation/outputs/swe_mesh_frames${suffix}.bin"   "simulation/outputs/swe_mesh_frames${suffix}_${level}.bin"
  cp "simulation/outputs/swe_mesh_summary${suffix}.json" "simulation/outputs/swe_mesh_summary${suffix}_${level}.json"
  cp "simulation/outputs/flow_tracer_paths${suffix}.bin" "simulation/outputs/flow_tracer_paths${suffix}_${level}.bin"
  cp "lidar/data/swe_surface_heightmap${suffix}.bin"     "lidar/data/swe_surface_heightmap${suffix}_${level}.bin"
  if [ "$site" = "site2" ]; then
    cp "simulation/outputs/lake_hydrograph_site2.csv" "simulation/outputs/lake_hydrograph_site2_${level}.csv"
  fi
}

for site in site1 site2; do
  for level in low medium high; do
    run_one "$site" "$level"
  done
done

echo "=== site3_1house (2500 tracers, default 100mm/hr convention) ==="
python3 site3_gee_creek/run_site3_1house_demo.py

echo "=== copying into viewer/data/ ==="
python3 viewer/preprocess/export_lidar.py

echo "DONE."
