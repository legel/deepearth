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
| worldclim | `extract_worldclim.py` | WorldClim v2.1, 19 bioclim normals (1970-2000, 10-arcmin) | `gbif_worldclim_tokens.npz` |

Each recipe reads observation coords from `gbif_tokens/*.npz`, samples the raster at those points, imputes
missing values with the train-median (so absence is not a location cue), and writes an `*_tokens.npz` with a
`has_<channel>` presence mask. To reproduce the champion exactly, run `build_phenology.py` before training.


## AlphaEarth geo prior (`extract_alphaearth.py`)

The champion consumes **AlphaEarth** ([Google Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL),
Brown et al. 2025 — 64-dim, 10 m, annual) not as a reconstruction channel but as a **SatCLIP-style geo prior**
(`model.alphaearth_geo: true`): a small MLP adds it to the spatial position every head reads, so it informs all
spatial reasoning instead of competing for head capacity. Ablation (single seed, spatial holdout, 16k steps):
`true` → **0.6074 / net 0.3311** vs `false` → **0.5962 / 0.3014** (**+0.0112 arith**); as an ordinary reconstruction
variable it instead *hurts* arith (−0.004…−0.006). Extract (~25–45 min for ~620k points, resumable):

```bash
EE_PROJECT=<your-ee-project> AE_COORDS=ae_coords.npz \
AE_OUT=gbif_alphaearth_tokens.npz python autoresearch/recipes/extract_alphaearth.py
```

Place `gbif_alphaearth_tokens.npz` in the cache dir; `data.py::_load_modalities` loads it (aligned by `gbifID`,
z-scored on train, uncovered rows masked). Absent → modality skipped and the champion falls back to the ~0.596 control.

## Occurrence-densification pipeline (the champion's data lever)

The champion (0.5801) trains on 2x the occurrences via three committed steps:
1. `gbif_acquire_occurrences.py` -> downloads CA vascular-plant occurrences (GBIF, vocab species only) -> `gbif_densify_bulk.npz` (gbifID, species_local, lat, lon).
2. `build_densify.py` -> filters those to existing TRAIN cells (keeps the 0.5deg spatial holdout byte-identical) and writes them as `gbif_tokens/chunk_densify_bulk.npz` with vision zeroed.
3. At train time: `data.py` derives `has_vision` (zeroed-vision obs are masked, so occurrence-only obs add species+location without poisoning vision), and `train.py`'s `densify_weight` (0.3) down-weights them so full-modality obs keep champion-level exposure (Rule 18: fixes the batch-dilution bug).
