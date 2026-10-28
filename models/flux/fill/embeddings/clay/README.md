# Clay embeddings of the Sentinel-2 chips

Every Sentinel-2 chip of [`../../data/sentinel2/`](../../data/sentinel2/) is embedded by the Clay v1.5 encoder. The
CLS token is stored for every chip, cloudy ones included, with the chip's SCL cloud, shadow, snow and no-data
fractions so the downstream model can learn how far to trust each embedding. Patch tokens are stored for a 10-site
pilot.

## Model

- Clay Foundation Model v1.5 (weights released 2024-11-19), the latest release on 2026-10-26: HuggingFace
  `made-with-clay/Clay`, file `v1.5/clay-v1.5.ckpt`, 5,158,764,629 bytes, SHA-256
  `21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0`. No v2 exists on HuggingFace or GitHub;
  `made-with-clay/Clay-legacy` holds v0, v1 and v1.5.
- Code: github.com/Clay-foundation/model at commit `f14e698f3c237cabf8d28dec669a362d66625381` (2026-03-27), package
  `claymodel` 1.5.0, `Encoder` of `clay_mae_large`: ViT-L, dim 1024, depth 24, 16 heads, patch size 8, 311 M
  parameters, dynamic (wavelength-conditioned) patch embedding after DOFA.
- `embed.py` exports the encoder weights (`model.encoder.*`) of the Lightning checkpoint to a plain state dict and
  loads them with `strict=True` into `claymodel.model.Encoder` with mask ratio 0 and no shuffling. The DINOv2
  teacher and the decoder are not loaded.

## Inputs

From Clay's `configs/metadata.yaml` (`sentinel-2-l2a`) and its wall-to-wall tutorial (`docs/tutorials/`):

| input | value |
|---|---|
| bands | B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 (blue, green, red, rededge1-3, nir, nir08, swir16, swir22); B01 and B09 are not part of Clay's Sentinel-2 input |
| wavelengths | 0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19 um |
| normalisation | (DN - mean) / std per band, means 1105 to 2785, stds 1379 to 1888 (metadata.yaml) |
| GSD | 10 m |
| time | sin, cos of ISO week x 2 pi / 52; sin, cos of UTC hour x 2 pi / 24 |
| location | sin, cos of latitude; sin, cos of longitude (radians), at the tower |
| size | height and width multiples of the patch size 8 |

FluxFill choices (`inputs.py`):

1. Offset: DN are harmonised across processing baselines, `DN + BOA_ADD_OFFSET` = 10000 x reflectance, so a
   surface has the same input before and after baseline 04.00.
2. 20 m bands to 10 m: nearest neighbour, each 20 m pixel becomes 2 x 2 (Clay's tutorials stack Sentinel-2 at 10 m
   with `Resampling.nearest`). Both grids come from one tile, so this is exact up to a one-pixel edge where the 20 m
   window sits 10 m off the 10 m window.
3. 30 x 30 px to 32 x 32 px: edge padding (the border pixel repeated) by one pixel on each side, at the native 10 m
   GSD, giving 4 x 4 patches of 8 px. Clay's position encoding is a sine-cosine grid scaled by the GSD and its patch
   embedding is a linear map of each 8 x 8 x bands patch, so a pad keeps the true GSD and the true pixel values; the
   padded border is 6% of the pixels and lies in the edge patches only. Resampling to 32 px changes the GSD to
   9.375 m and blurs every pixel; cropping to 24 px discards 36% of the chip.
4. No-data pixels (DN 0) take the band mean (0 after normalisation).

## Sensitivity

`sensitivity.py` embeds random chips under each input mode and compares the CLS with the CLS of a true 32 x 32 px
(320 m) chip read again from the source COGs around the same tower (cosine similarity, 2026-10-26):

| mode | US-Ha1 mean (n 60) | US-Ha1 5th pct | DE-Tha mean (n 30) | DE-Tha min |
|---|---|---|---|---|
| edge pad (chosen) | 0.978 | 0.940 | 0.981 | 0.910 |
| reflect pad | 0.962 | 0.907 | 0.977 | 0.906 |
| bilinear 30 -> 32 px, GSD 9.375 m | 0.922 | 0.774 | 0.956 | 0.850 |
| central 24 x 24 crop | 0.893 | 0.782 | 0.897 | 0.591 |
| pad with band mean | 0.741 | 0.614 | 0.746 | 0.516 |

Edge and reflect padding agree with each other at 0.985 (US-Ha1) and 0.998 (DE-Tha). Results:
`/home/photon/4tb/fluxfill/clay/sensitivity_<site>.json`.

Offset harmonisation matters: for the 47 US-Ha1 chips of baseline 04.00 and later, the CLS of raw DN and of
harmonised DN agree at a mean cosine of 0.870 (minimum 0.624). Clay's metadata gives one normalisation for all
Sentinel-2 L2A chips and its documentation describes no offset handling; FluxFill harmonises so that one surface
gives one input across the 2022 baseline change and the Collection 1 reprocessing.

Sanity check, US-Ha1 (713 chips): mean CLS cosine 0.745 among clear chips (clear fraction above 0.95), 0.701 among
cloudy chips (cloud fraction above 0.95), 0.424 between the two; 0.883 among clear summer chips and 0.656 between
clear summer and clear winter chips.

## Store

`/home/photon/4tb/fluxfill/clay/sites/<site_id>.zarr`, row i matching row i of the chip store:

| array | shape | dtype |
|---|---|---|
| `cls` | (N, 1024) | float16 |
| `patches` (pilot sites) | (N, 16, 1024), 4 x 4 row-major | float16 |
| `time_us` | (N,) | int64, microseconds since 1970-01-01 UTC |
| `meta/acq_key`, `meta/item_id`, `meta/mgrs_tile` | (N,) | str |
| `meta/cloud_frac`, `shadow_frac`, `snow_frac`, `invalid_frac`, `clear_frac`, `nodata_frac`, `cld_mean` | (N,) | float32 |

Measured: the CLS store takes 1.9 KB per chip; patch tokens take 31 KB per chip (float16 compresses by about 5%),
so patch tokens for all 1.18 M projected chips would take about 37 GB. The encoder embeds about 480 chips per second
on one RTX 3090 (batch 128, 1.7 GB of GPU memory, under 150 W).

Attributes: the encoder and checkpoint SHA-256, the input mode, the tower coordinates, `n_committed` (resume).

## Run

```
python embed.py --s2 /home/photon/4tb/fluxfill/sentinel2 --out /home/photon/4tb/fluxfill/clay \
  --ckpt /home/photon/4tb/fluxfill/clay/ckpt/clay-v1.5-encoder.pt \
  --export-from /home/photon/4tb/fluxfill/clay/ckpt/clay-v1.5.ckpt \
  --pilot US-Ha1,DE-Tha,FI-Hyy,AU-How,GF-Guy,SN-Dhr,US-Var,IT-Lav,CH-Dav,US-Los --mode edge --watch 900
pytest
```

With `--watch`, the job embeds new chips as the gather appends them and ends after the gather writes `GATHER_DONE`.

## Citation and licence

- Clay Foundation (2024). Clay Foundation Model v1.5. Model weights, HuggingFace `made-with-clay/Clay`,
  doi:10.57967/hf/1537; code github.com/Clay-foundation/model; documentation clay-foundation.github.io/model. Apache
  License 2.0 (code and weights).
- Xiong, Z., Wang, Y., Zhang, F., et al. (2024). Neural plasticity-inspired multimodal foundation model for Earth
  observation (DOFA). arXiv:2403.15356.
- Embeddings derive from Copernicus Sentinel data (see the chip README).
