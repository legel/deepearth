# Ensue data-channel recipes

The champion config (`autoresearch/deepcal.yaml`) uses transferable environmental channels beyond the
standard data download. Bulk token files are NOT in git — reproduce them with these recipes. A channel that
is not present in the cache is skipped automatically at train time (`build_variables` in `train.py`), so
**every config stays runnable on the standard data download**; a recipe is only needed to *enable* the extra
channel.

| channel | recipe | source | output (cache) |
|---------|--------|--------|----------------|
| phenology | `build_phenology.py` | NOAA-CDR VIIRS NDVI, 12-month seasonal cycle | `gbif_phenology_tokens.npz` |
| rsveg | `extract_rsveg.py` | MODIS 13Q1 NDVI/EVI phenology (Planetary Computer STAC) | `gbif_rsveg_tokens.npz` |

Each recipe reads observation coords from `gbif_tokens/*.npz`, samples the raster at those points, imputes
missing values with the train-median (so absence is not a location cue), and writes an `*_tokens.npz` with a
`has_<channel>` presence mask. To reproduce the champion exactly, run `build_phenology.py` before training.
