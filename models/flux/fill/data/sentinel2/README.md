# Sentinel-2 chips around every flux tower

Every Sentinel-2 Level-2A acquisition since the start of the mission (2015-06-23), over a 300 m x 300 m area centred
on each flux tower, at the native resolution of every band. These chips are the imagery modality of FluxFill; their
Clay embeddings are built in [`../../embeddings/clay/`](../../embeddings/clay/).

## Contents

| file | role |
|---|---|
| `sites.py` | tower coordinates from the tower table, the FLUXNET Shuttle and the FLUXNET / AmeriFlux site metadata; differences over 30 m between sources |
| `stac.py` | STAC search of both archives, one scene per acquisition, choice among reprocessings |
| `geometry.py` | chip windows on each band's native pixel grid; nearest-neighbour resampling between grids |
| `bands.py` | band groups, asset names, `BOA_ADD_OFFSET` per processing baseline, SCL classes |
| `store.py` | the per-site zarr store and its resume rule |
| `gather.py` | the threaded reader with retries and backoff; the run's disk and size guards |
| `tests/` | chip geometry, band scaling and offsets, timestamps, resumability (`pytest` in this directory) |

## Sources

1. **Earth Search, `sentinel-2-c1-l2a`** (Element 84; `https://earth-search.aws.element84.com/v1`). Sentinel-2
   Collection 1 L2A: ESA's reprocessing of the archive with processing baselines 05.00 and later, as Cloud-Optimised
   GeoTIFFs in the AWS Open Data bucket `e84-earth-search-sentinel-data` (us-west-2), no authentication.
2. **Microsoft Planetary Computer, `sentinel-2-l2a`** (`https://planetarycomputer.microsoft.com/api/stac/v1`),
   with anonymous SAS tokens. It is the fallback for acquisitions that Collection 1 on Earth Search does not hold.

Coverage measured on 2026-10-26 (items whose footprint contains the tower, per year):

| site | collection | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US-Ha1 | Earth Search C1 | 0 | 0 | 0 | 23 | 59 | 75 | 71 | 6 | 70 | 78 | 84 | 76 |
| US-Ha1 | Planetary Computer | 3 | 26 | 28 | 68 | 74 | 75 | 71 | 95 | 120 | 72 | 83 | 75 |
| DE-Tha | Earth Search C1 | 0 | 0 | 0 | 39 | 112 | 146 | 145 | 11 | 145 | 149 | 201 | 166 |
| DE-Tha | Planetary Computer | 20 | 68 | 102 | 143 | 143 | 144 | 133 | 224 | 234 | 145 | 199 | 163 |
| AU-How | Earth Search C1 | 0 | 0 | 1 | 24 | 56 | 73 | 75 | 6 | 73 | 74 | 88 | 74 |
| AU-How | Planetary Computer | 9 | 47 | 49 | 70 | 73 | 72 | 68 | 91 | 125 | 73 | 85 | 72 |

Collection 1 on Earth Search lacks 2015 to 2017, part of 2018 and 2019, and nearly all of 2022. Earth Search's older `sentinel-2-l2a` collection starts in late 2016 and holds fewer 2015 to 2017
items than the Planetary Computer, so the Planetary Computer is the single fallback. In 2022 and 2023 the Planetary
Computer holds both the original (04.00) and the reprocessed (05.10) product of many acquisitions.

### One scene per acquisition

An acquisition is one MGRS tile imaged by one datatake: the key is the tile and `s2:datatake_id` without its
`_Nxx.yy` baseline suffix. Every acquisition is kept, including every overlapping tile and every relative orbit. When
an acquisition exists in more than one processing, the chosen product is, in order of preference:

1. a Collection 1 product (processing baseline 05.00 or later);
2. Earth Search over the Planetary Computer;
3. the higher processing baseline;
4. the later generation time.

The full candidate list per site is in `<root>/scenes/<site>.jsonl` (gzip-compressed as `.jsonl.gz` after the 2026-10-27 run).

## Chips

The chip is 300 m x 300 m in the UTM CRS of the tile (EPSG from `proj:epsg`), at native resolution:

| group | bands | resolution | pixels |
|---|---|---|---|
| `r10` | B02, B03, B04, B08 | 10 m | 30 x 30 |
| `r20` | B05, B06, B07, B8A, B11, B12 | 20 m | 15 x 15 |
| `r60` | B01, B09 | 60 m | 5 x 5 |
| `scl` | Sen2Cor scene classification | 20 m | 15 x 15 |
| `cld` | Sen2Cor cloud probability (Earth Search only; 255 elsewhere) | 20 m | 15 x 15 |

Geometry: Sentinel-2 tile corners lie on a 60 m grid, so the 10, 20 and 60 m grids of a tile share an origin. Each
group's window is the window on its own grid whose centre is nearest the tower: within 5 m of the tower on each axis
for the 10 m bands, 10 m for the 20 m bands and 30 m for the 60 m bands. Each chip records the upper-left corner of
each group's window (`x_ul_r10` and so on) and the tower's offset from the 10 m chip centre (`tower_dx_m`,
`tower_dy_m`). Pixels outside the tile raster are 0 (no data).

Reads: each band is one GDAL `/vsicurl/` open of the COG (first 32 KiB for the header) and one HTTP range read of the
internal tiles that the window touches. No full tile is downloaded. A COG tile is the smallest readable unit: 1024 x
1024 px for the 10 m bands on Earth Search and 512 x 512 px for the 20 and 60 m bands and on the Planetary
Computer. One chip transfers 10.3 MB from Earth Search and 5.1 MB from the Planetary Computer (measured with
`CPL_VSIL_SHOW_NETWORK_STATS`), and stores about 20 KB.

No cloud filtering: every acquisition is gathered. Masks come from SCL and are stored, never applied by deletion.
Per-chip SCL fractions: `cloud_frac` (classes 8, 9, 10), `shadow_frac` (3), `snow_frac` (11), `invalid_frac` (0, 1),
`clear_frac` (the rest).

### Digital numbers and offsets

Bands are stored as the source DN (uint16), unchanged. Reflectance is `(DN + boa_add_offset) / 10000`, with
`boa_add_offset = -1000` for processing baseline 04.00 and later (all Collection 1 products, and products generated
from 2022-01-25) and 0 before; DN 0 is no data. The offset is stored per chip. For Earth Search items the offset
declared in the STAC `raster:bands` (offset -0.1, scale 0.0001) is checked against the baseline rule and a mismatch
stops the scene. `bands.reflectance` applies the rule.

## Store

One zarr v2 store per site, `<root>/sites/<site_id>.zarr`, Blosc zstd level 5 with bit shuffle, chunks of 256
chips. Arrays `r10`, `r20`, `r60`, `scl`, `cld`, `time_us` (int64 microseconds since 1970-01-01 UTC) and
`meta/<field>` per chip: `acq_key`, `item_id`, `source`, `platform`, `mgrs_tile`, `epsg`, `relative_orbit`,
`datatake_id`, `processing_baseline`, `boa_add_offset`, `generation_time`, `datetime_utc`, `sun_azimuth`,
`sun_elevation`, `view_azimuth`, `view_incidence` (Earth Search only; NaN for the Planetary Computer, whose items
carry no view angles), `scene_cloud_cover`, window corners, tower offsets, SCL fractions, `cld_mean`, `has_cld`.

Rows are in completion order, not time order; sort by `time_us`.

Resume: the attribute `n_committed` counts the chips fully written; on open, longer arrays are truncated back to
it, and the committed `meta/acq_key` values are the manifest of done acquisitions. A failed scene is logged in
`<root>/failures/<site>.jsonl` and read again on the next run. A rerun with `--redo-complete` searches again and adds
new acquisitions.

## Run

```
python sites.py --src <root>/sites_src --towers /home/photon/4tb/fluxfill/towers --out <root>
python gather.py --sites <root>/sites.csv --root <root> --threads 12
pytest
```

Site order: ten pilot sites, then sites whose records reach 2015 or later (newest last year first), then the rest.
The run stops if the disk falls under 30 GB free, or if, after 20 sites, the projected size of all sites exceeds 40
GB (`<root>/PROJECTION.txt`).

## Volume and throughput (2026-10-26)

- A STAC count over 40 random sites gives 1433 acquisitions per site on average (median 1337, range 710 to 5018;
  high latitudes and tile overlaps have the most). For 824 sites: about 1.18 M chips.
- Storage: 6.2 KB per chip (US-Ha1, 713 chips, 4.4 MB), so about 7.3 GB for all sites.
- Transfer: about 10.2 TB, set by the COG tile sizes above, not by the chip size.
- Measured rate on the gathering machine, over a shared WiFi link of about 9 MB/s: 1460 to 1830 chips per hour.
  The link, not the endpoint or the CPU, is the limit.

## Provenance of the full gather (2026-10-27)

- The 12 pilot and early sites were gathered on the FluxFill workstation. The other 812 sites were gathered on
  2026-10-27 from 01:59 to 10:55 PDT on a Google Cloud spot VM: n2-standard-8, us-west1-b (Oregon, next to the AWS
  us-west-2 archive), Debian 12, 100 GB balanced persistent disk, Standard network tier.
- Code: this directory, unchanged, with the same parameters, from `sites_priority.csv` (pilot, val, train, test,
  rest). Ten `gather.py` processes at 32 threads each took the priority list round-robin. Python 3.11, rasterio 1.4.4
  (GDAL 3.10.3 wheel), zarr 2.18.7, numcodecs 0.15.2, numpy 2.4.6, pandas 3.1.0.
- Check: US-Ha1 gathered on the VM matched the workstation store exactly in all 39 arrays for all 713 chips.
- Result: 824 sites, 1,349,395 chips (1,333,508 from the VM), 7.7 GB of stores and 247 MB of scene manifests.
  818 sites are complete. Six sites each lack one acquisition that fails at the source and are marked
  `complete: false` with the failure logged: AU-Wac and UK-HpF (HTTP 404), US-CAK and US-NGC (Earth Search asset on a
  60 m grid where 10 m is expected), US-EML and US-xHE (no 10 m bands).
- Transfer: 9.7 TB read by the VM at 340 to 430 MB/s, about 196k chips per hour. The VM was preempted twice
  (04:37, 05:01) and resumed from the stores.
- Cost: 8.43 instance hours at $0.233/h ($1.97), disk $0.12, external IP $0.02, about 54 GB of egress (mostly TCP
  acknowledgements of the inbound reads, 0.55 % of inbound) at $0.085/GB ($4.6). Total about $6.7. The VM and its
  disk were deleted on 2026-10-27 at 10:56 PDT.
- Scene manifests are stored gzip-compressed, `scenes/<site>.jsonl.gz` (14 times smaller). A rerun of `gather.py`
  writes a new plain `scenes/<site>.jsonl`.

## Sites

`sites.csv` holds 805 towers: the 782 sites of the FLUXNET Shuttle snapshot (ONEFlux v1.3,
`keenangroup.info/fluxnet-data-explorer/assets/shuttle_snapshot.csv`) and the 23 FLUXNET-CH4 V1.0 sites that the
Shuttle does not list, with coordinates from the FLUXNET site metadata service
(`amfcdn.lbl.gov/api/v1/site_display/FLUXNET`). Precedence: the FluxFill tower table when present, then the Shuttle,
then FLUXNET, then AmeriFlux metadata. `coordinate_diffs.csv` lists every pair of sources more than 30 m apart (14
pairs on 2026-10-26, up to 29.6 km for CN-HaM, where the sources disagree on the tower).

## Licence and citation

Contains modified Copernicus Sentinel data 2015-2026, processed by ESA. Copernicus Sentinel data are free, full and
open for any use, including commercial, under the Legal Notice on the use of Copernicus Sentinel Data and Service
Information (Commission Delegated Regulation (EU) No 1159/2013); redistribution of modified data must state that it
contains modified Copernicus Sentinel data and the year.

- Drusch, M., Del Bello, U., Carlier, S., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES
  operational services. Remote Sensing of Environment 120, 25-36. doi:10.1016/j.rse.2011.11.026
- Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Müller-Wilm, U., Gascon, F. (2017). Sen2Cor for Sentinel-2.
  Proc. SPIE 10427, Image and Signal Processing for Remote Sensing XXIII, 1042704. doi:10.1117/12.2278218
- ESA (2021). Sentinel-2 processing baseline 04.00: provision of the radiometric offset (`BOA_ADD_OFFSET`) in L2A
  products from 2022-01-25. Sentinel Online, Copernicus.
- ESA (2023-2025). Copernicus Sentinel-2 Collection-1 reprocessing, processing baselines 05.00 and 05.10.
- Element 84. Earth Search STAC API v1, collection `sentinel-2-c1-l2a`. https://earth-search.aws.element84.com/v1;
  AWS Registry of Open Data, "Sentinel-2 Level-2A COGs", https://registry.opendata.aws/sentinel-2-l2a-cogs/
- Source, M., Microsoft Open Source, McFarland, M., Emanuele, R., Morris, D., Augspurger, T. (2022).
  microsoft/PlanetaryComputer: October 2022. Zenodo. doi:10.5281/zenodo.7261897
