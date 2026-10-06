#!/bin/bash
# Dispersal rates (Daru 2024 step 4): spherical Brownian motion fitted per plant family (>= 10 located tips) and over
# the whole tree with castor::fit_sbm_const, on two dated trees of Carruthers et al. (OSF 9tbha). Tips are located at
# the area-weighted spherical centroid of their WCVP native level-3 areas (ranges/tip_geography.py); branch lengths
# are floored at 0.1 Myr (dating resolution). combine_sbm.py averages the two trees per family and adds the 'ALL'
# fallback row. Calibration areas are whole occupied ecoregions (provenance D8), so the rates change no map; every
# species summary records them.
#
# usage: fit_dispersal.sh            (data root: $DEEPEARTH_HABITAT_DATA, else models/habitat/plant/data)
# needs: the R environment of r_environment.txt (castor, ape), activated as for the pipeline (CONDA_SH, R_ENV)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
R=${DEEPEARTH_HABITAT_DATA:-$HERE/../data}
S=$R/work/sbm
mkdir -p $S
source ${CONDA_SH:-~/miniconda3/etc/profile.d/conda.sh} && conda activate ${R_ENV:-daru}
for t in 0001 0002; do
  tree=$R/raw/trees/${t}_.tre_dated
  [ -s $S/tips_$t.csv ] || (cd $HERE/.. && python3 -m ranges.tip_geography $R/raw/wcvp $R/raw/geo/wgsrpd_level3.geojson $tree $S/tips_$t.csv)
  [ -s $S/sbm_families_$t.csv ] || Rscript $HERE/../ranges/r/sbm_by_clade.R $tree $S/tips_$t.csv $S/sbm_families_$t.csv
done
python3 $HERE/combine_sbm.py $S/sbm_families_0001.csv $S/sbm_families_0002.csv $S/sbm_combined.csv
