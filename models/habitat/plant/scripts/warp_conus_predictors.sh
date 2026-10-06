#!/bin/bash
# WorldClim 2.1 30" -> CONUS Albers (EPSG:5070) 240 m, origin aligned to the NLCD/LANDFIRE 30 m grid
# (x0 = -2493045, y1 = 3310005); 20149 x 13053 cells. Bilinear, float32, tiled ZSTD; no-data written as NaN.
# Writes work/conus240/<variable>_conus240.tif under the data root; build_conus_stack.py packs them.
set -e
R=${DEEPEARTH_HABITAT_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
W=$R/raw/worldclim
mkdir -p $R/work/conus240 && cd $R/work/conus240
TE="-2493045 177285 2342715 3310005"
for f in $W/wc2.1_30s_elev.tif $(ls $W/wc2.1_30s_bio_*.tif); do
  o=$(basename $f .tif)_conus240.tif
  [ -s $o ] && continue
  gdalwarp -q -overwrite -t_srs EPSG:5070 -tr 240 240 -te $TE -r bilinear -ot Float32 -dstnodata nan -multi -wo NUM_THREADS=4 \
    -co TILED=YES -co COMPRESS=ZSTD -co PREDICTOR=3 -co BIGTIFF=YES $f $o
  echo done $o
done
echo ALLDONE
