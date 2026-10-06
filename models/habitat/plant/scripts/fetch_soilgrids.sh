#!/bin/bash
# SoilGrids 2.0 (ISRIC, 250 m, Interrupted Goode Homolosine; CC BY 4.0) soil properties at 5-15 cm, warped by
# area averaging onto (a) the CONUS 240 m render grid (EPSG:5070, NLCD-aligned) and (b) the North America 7.5"
# training grid of fine.py (170W-50W, 5N-75N). Values keep SoilGrids' integer scaling (pH x10, clay/sand g/kg,
# SOC dg/kg); nodata -32768. Ledger L10.
set -e
R=${DEEPEARTH_HABITAT_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
OUT=$R/work/soil
mkdir -p $OUT && cd $OUT
export GDAL_HTTP_MULTIRANGE=YES GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES GDAL_CACHEMAX=2048 GDAL_HTTP_MAX_RETRY=8 GDAL_HTTP_RETRY_DELAY=5
URL=/vsicurl/https://files.isric.org/soilgrids/latest/data
for v in ${@:-phh2o clay sand soc}; do
  (
  src=$URL/$v/${v}_5-15cm_mean.vrt
  [ -s ${v}_conus240.tif ] || gdalwarp -q -overwrite -t_srs EPSG:5070 -tr 240 240 -te -2493045 177285 2342715 3310005 \
      -r average -ot Int16 -srcnodata -32768 -dstnodata -32768 -multi -wo NUM_THREADS=2 \
      -co TILED=YES -co COMPRESS=ZSTD -co PREDICTOR=2 -co BIGTIFF=YES $src ${v}_conus240.tif && echo "done $v conus240"
  [ -s ${v}_na7p5s.tif ] || gdalwarp -q -overwrite -t_srs EPSG:4326 -te -170 5 -50 75 -ts 57600 33600 \
      -r average -ot Int16 -srcnodata -32768 -dstnodata -32768 -multi -wo NUM_THREADS=2 \
      -co TILED=YES -co COMPRESS=ZSTD -co PREDICTOR=2 -co BIGTIFF=YES $src ${v}_na7p5s.tif && echo "done $v na7p5s"
  ) &
done
wait
echo ALLDONE
