#!/bin/bash
# Public inputs of the native-plant habitat models, downloaded into the data root ($DEEPEARTH_HABITAT_DATA, else
# models/habitat/plant/data) in the layout the README lists. Every source is public; GBIF downloads are public once
# made. Idempotent: anything already present is skipped, so the script can be rerun after an interruption.
#
# usage: fetch_sources.sh <section> [...]
#   wcvp        World Checklist of Vascular Plants (Kew; CC BY)                               raw/wcvp
#   geo         WGSRPD level-3 areas (TDWG), RESOLVE Ecoregions 2017 (CC BY 4.0),
#               Natural Earth 10 m land and lakes, 50 m land (public domain)                   raw/geo
#   worldclim   WorldClim 2.1, 30" bio1-19 and elevation (CC BY-SA 4.0)                       raw/worldclim
#   maxent      maxent.jar 3.4.4 (MIT; github.com/mrmaxent/Maxent)                             tools/maxent.jar
#   trees       Carruthers et al. dated megatrees 0001 and 0002 (OSF 9tbha; 2.5 GB archive)   raw/trees
#   gbif [key]  the GBIF occurrence downloads of the configuration (or one download key)      raw/gbif_*/parquet
#   effort      GBIF records of all vascular plants per ~9 km cell (map API)                  raw/gbif_tracheophyta_effort_5min.csv
#   soil        SoilGrids 2.0 topsoil, warped to the training and render grids (fetch_soilgrids.sh)
#   validation  BLM AIM species indicators, FIA DataMart, VegBank                             raw/validation
#   glo90       Copernicus GLO-90 elevation tiles, 170W-50W, 5N-75N (only for fine predictors)  raw/dem_glo90
# Daru's (2024) published maps (Dryad doi:10.5061/dryad.5x69p8d9w) are downloaded by hand: extract_daru_maps.py.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=${DEEPEARTH_HABITAT_DATA:-$HERE/../data}
CONFIG=${CONFIG:-$HERE/../configs/conus.json}
get() {  # get <url> <dest>: resumable download, skipped when complete
  local url=$1 dst=$2
  [ -s "$dst" ] && return 0
  mkdir -p "$(dirname "$dst")"
  curl -sfL --retry 5 --retry-delay 20 -C - -o "$dst.part" "$url" && mv "$dst.part" "$dst" || { echo "FAILED $url"; return 1; }
}

wcvp() {
  get https://sftp.kew.org/pub/data-repositories/WCVP/wcvp.zip $R/raw/wcvp/wcvp.zip
  [ -s $R/raw/wcvp/wcvp_names.csv ] || unzip -oq $R/raw/wcvp/wcvp.zip -d $R/raw/wcvp
}

geo() {
  local d=$R/raw/geo
  get https://raw.githubusercontent.com/tdwg/wgsrpd/master/geojson/level3.geojson $d/wgsrpd_level3.geojson
  get https://storage.googleapis.com/teow2016/Ecoregions2017.zip $d/Ecoregions2017.zip
  [ -d $d/ecoregions2017 ] || unzip -oq $d/Ecoregions2017.zip -d $d/ecoregions2017
  for n in land lakes; do
    get https://naciscdn.org/naturalearth/10m/physical/ne_10m_$n.zip $d/ne_10m_$n.zip
    [ -d $d/ne_$n ] || unzip -oq $d/ne_10m_$n.zip -d $d/ne_$n
  done
  get https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip $d/ne_50m_land.zip      # CoordinateCleaner seas
  [ -d $d/ne_50m_land ] || unzip -oq $d/ne_50m_land.zip -d $d/ne_50m_land
}

worldclim() {
  local d=$R/raw/worldclim
  for f in bio elev; do
    get https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_$f.zip $d/wc2.1_30s_$f.zip
  done
  [ -s $d/wc2.1_30s_elev.tif ] || unzip -oq $d/wc2.1_30s_elev.zip -d $d
  [ -s $d/wc2.1_30s_bio_19.tif ] || unzip -oq $d/wc2.1_30s_bio.zip -d $d
}

maxent() {
  local z=$R/tools/maxent-3.4.4.zip
  get https://github.com/mrmaxent/Maxent/archive/refs/tags/v3.4.4.zip $z
  [ -s $R/tools/maxent.jar ] || unzip -p $z Maxent-3.4.4/ArchivedReleases/3.4.4/maxent.jar > $R/tools/maxent.jar
  echo "4c856e55412f70c5597b03cf9aaaf27e0782e0921f937262273b68bdcb8fee5e  $R/tools/maxent.jar" | sha256sum -c -
}

trees() {
  local d=$R/raw/trees
  get "https://osf.io/download/6940398cc46a7f528bdad312/" $d/zipped-dated-trees.zip
  for t in 0001 0002; do
    [ -s $d/${t}_.tre_dated ] || unzip -oq $d/zipped-dated-trees.zip ${t}_.tre_dated -d $d
  done
}

gbif() {  # gbif [key]: GBIF SIMPLE_PARQUET downloads into the configured directories
  local pairs
  if [ $# -gt 0 ]; then pairs="$1 raw/gbif/$1"
  else pairs=$(python3 -c "import json,sys; [print(k, v) for k, v in json.load(open(sys.argv[1]))['sources']['gbif_downloads'].items()]" $CONFIG)
  fi
  echo "$pairs" | while read -r key dir; do
    [ -d $R/$dir/parquet ] && continue
    get https://api.gbif.org/v1/occurrence/download/request/$key.zip $R/$dir/$key.zip && \
      unzip -oq $R/$dir/$key.zip -d $R/$dir/parquet && rm -f $R/$dir/$key.zip
  done
}

effort() {
  [ -s $R/raw/gbif_tracheophyta_effort_5min.csv ] || python3 $HERE/fetch_gbif_effort.py $R/raw/gbif_tracheophyta_effort_5min.csv
}

soil() {
  DEEPEARTH_HABITAT_DATA=$R bash $HERE/fetch_soilgrids.sh
}

validation() {
  local v=$R/raw/validation
  mkdir -p $v/fia
  [ -s $v/blm_aim_species.csv ] || python3 $HERE/fetch_blm_aim.py $v/blm_aim_species.csv
  for f in ENTIRE_PLOT ENTIRE_TREE REF_SPECIES; do get https://apps.fs.usda.gov/fia/datamart/CSV/$f.csv $v/fia/$f.csv; done
  [ -s $v/fia/fia_live_tree_presence.csv ] || python3 $HERE/fetch_fia.py $v/fia
  [ -s $v/vegbank/taxa.parquet ] || DEEPEARTH_HABITAT_DATA=$R python3 $HERE/fetch_vegbank.py
}

glo90() {
  local d=$R/raw/dem_glo90 lat lon t
  mkdir -p $d
  for lat in $(seq 5 74); do for lon in $(seq 51 170); do
    t=Copernicus_DSM_COG_30_$(printf "N%02d" $lat)_00_$(printf "W%03d" $lon)_00_DEM
    [ -s $d/$t.tif ] && continue
    if curl -sf --retry 3 -o $d/$t.tif.part https://copernicus-dem-90m.s3.amazonaws.com/$t/$t.tif; then
      mv $d/$t.tif.part $d/$t.tif
    else
      rm -f $d/$t.tif.part                                             # ocean tiles do not exist
    fi
  done; done
}

[ $# -gt 0 ] || { sed -n 2,19p "$0"; exit 1; }
mkdir -p $R
section=$1; shift
$section "$@"
