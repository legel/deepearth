# FluxFill tower dataset

Half-hourly eddy-covariance measurements from every flux tower with data since 2014-01-01, quality-controlled so
that only real sensor measurements remain, with the site-downscaled ERA5 drivers as a separate reanalysis modality.
Each value carries a flag that says why it is absent and a code that names its source. The dataset trains and
evaluates FluxFill, a masked generative model over 180-day windows of the site x time x variable field, and defines
the frozen 80/20 site split of FluxFill-Bench.

Data: `/home/photon/4tb/fluxfill/towers/`. Code: this directory. Built 2026-10-26 and 2026-10-27.

## Contents

| file | content |
|---|---|
| `hh/<SITE>.parquet` | one table per tower on a regular UTC grid: `time_utc`, `time_lst`, then per variable `<VAR>` (float32), `<VAR>_flag` (uint8 bit field), `<VAR>_src` (uint8 source code); `ERA5_*` reanalysis drivers; `SW_IN_POT`; ONEFlux random uncertainty `NEE_RANDUNC`, `LE_RANDUNC`, `H_RANDUNC` |
| `sites.parquet`, `sites.csv` | one row per tower: id, name, latitude, longitude, elevation, IGBP, Koppen class and its source, data hub and source network, UTC offset, step, measured years, sources, u* threshold source, coverage, inclusion, licence, citation and DOI, PIs, required acknowledgement |
| `split.csv`, `split.json` | the frozen split: train, val, test per site, with spatial block and stratum; parameters, counts and SHA-256 |
| `coverage.parquet` | measured fraction per site, variable and year, with the source used |
| `windows.parquet` | every 180-day window at a 30-day stride with the measured fraction of each main variable and a `usable` flag |
| `weights.parquet` | sampling weight per site and per window |
| `qc_counts.parquet` | values removed per site, source, variable and rule |
| `events.parquet` | site moves, instrument changes, clock offsets, failed whole-record tests, hourly records, merge notes, cross-source comparisons, co-located site IDs |
| `provenance.csv` | every source file: source, site, URL, version, download time (UTC), bytes, SHA-256, whether the raw file is kept |
| `data_dictionary.csv` | every column with unit and meaning |
| `interim/<source>/<SITE>.parquet`, `.json` | per-source cleaned tables and their sidecars (all tests, counts and metadata) |
| `report.json`, `report.md` | the statistics below |

## Sources

| source | version | sites | access | licence |
|---|---|---|---|---|
| FLUXNET Shuttle: ONEFlux FLUXNET products of AmeriFlux, ICOS (with the European Fluxes Database, ChinaFlux, JapanFlux, KoFlux, SAEON) and TERN | ONEFlux v1.3 (code 1.3.x), release r1, processed 2026 | 782 listed; 670 with data from 2014 | official `fluxnet-shuttle` library (github.com/fluxnet/shuttle, commit 0288187); AmeriFlux files from ftp.fluxdata.org, ICOS through the Carbon Portal licence acceptance, TERN from its public THREDDS server | CC-BY-4.0 |
| FLUXNET-CH4 Community Product (Delwiche et al. 2021) | V1.0 (file version 1-1) | 79 of 81 (RU-Vrk and SE-St1 have no CC-BY-4.0 years) | AmeriFlux API `data_download`, product FLUXNET-CH4, account legel | CC-BY-4.0 |
| AmeriFlux BASE-BADM | per-site BASE versions as served 2026-10-26 | 85 sites that measure FCH4 from 2014 | AmeriFlux API `data_download`, product BASE-BADM, variant FULLSET | CC-BY-4.0 |
| ICOS ETC L2 Fluxes | L2, latest version per station | 93 stations | data.icos-cp.eu licence acceptance | CC-BY-4.0 |
| Beck et al. (2023) Koppen-Geiger 1-km maps, 1991-2020 | figshare 21789074 | Koppen class where the network gives none | ndownloader.figshare.com/files/61012822 | CC-BY-4.0 |

Catalogs (snapshot of 2026-10-26, in `raw/towers/catalog/`): `fluxnet_shuttle_snapshot_20261026T163140.csv` (from
`fluxnet-shuttle listall`), the Keenan-group explorer copies `shuttle_snapshot.csv`, `icos_direct_fluxnet.csv`,
`japanflux_direct_snapshot.csv` (keenangroup.info/fluxnet-data-explorer/assets/), AmeriFlux `site_display` for
AmeriFlux and FLUXNET, `data_availability` for FLUXNET-CH4 and BASE-BADM, the amerifluxr BASE variable-availability
summary, the AmeriFlux FP variable limits, and ICOS SPARQL results for stations and L2 objects.

Every downloaded file has a row in `provenance.csv` with its URL, version, download time and SHA-256
([`sources.py` `record`](sources.py#L125)). The Shuttle archives (about 50 GB in total) are deleted after processing
to stay within the disk budget; each site's BIF, BIFVARINFO, README and data policy files are kept in
`raw/towers/shuttle_meta/<SITE>/`, and the checksum lets the archive be fetched again and verified. FLUXNET-CH4,
BASE and ICOS archives are kept in `raw/towers/`.

Not included:
- ICOS L2 carries no FCH4. None of the 93 ETC L2 Fluxes objects, and none of the interim L2 archives checked, has an
  FCH4 column (2026-10-26). European CH4 comes from FLUXNET-CH4 only. ICOS L2 supplies NEE, LE and H where the Shuttle has
  none: the 11 stations without a Shuttle product, and years outside a station's Shuttle record.
- The 29 JapanFlux2024 sites that are not in the Shuttle. They are processed outside ONEFlux, with their own QC
  conventions; 10 of them have data from 2014.

## Measured-only rule

A value enters the dataset only if the source measured it. Gap-filled values (MDS, ANN, ERA-downscaled) never enter,
as targets or as inputs.

| source | rule | code |
|---|---|---|
| ONEFlux | `NEE_VUT_REF` with `NEE_VUT_REF_QC == 0`; `LE_F_MDS`, `H_F_MDS`, `G_F_MDS`, `TA_F_MDS`, `SW_IN_F_MDS`, `LW_IN_F_MDS`, `VPD_F_MDS`, `CO2_F_MDS`, `TS_F_MDS_k`, `SWC_F_MDS_k` with their `_QC == 0` (0 = measured; 1 to 3 = MDS fill quality classes); `PA`, `P`, `WS`, `WD`, `USTAR`, `RH`, `NETRAD`, `PPFD_IN`, `PPFD_OUT`, `SW_OUT`, `LW_OUT` as published, since ONEFlux does not fill them. `*_F` (consolidated with ERA) and `*_ERA` columns are never read as tower data. | [`qc.py` `measured`](qc.py#L44), [`variables.py` `VARIABLES`](variables.py#L55) |
| FLUXNET-CH4 | the unfilled columns `FCH4`, `NEE`, `LE`, `H`, `TA`, `WTD`, ... ; never `FCH4_F`, `FCH4_F_ANNOPTLM`, `*_F`, `*_DT`, `*_NT` | [`readers.py` `CH4_COLUMNS`](readers.py#L176) |
| BASE | the unqualified or lowest positional column (`FCH4`, `FCH4_1_1_1`); never a column with an `_F` qualifier (PI gap-fill); NEE is FC + SC where SC is measured, FC alone otherwise, as ONEFlux forms it | [`readers.py` `base_column`](readers.py#L205) |
| ICOS L2 | the cleaned `NEE`, `LE`, `H`; never `*_UNCLEANED` | [`readers.py` `ICOS_COLUMNS`](readers.py#L265) |

GPP and RECO are partitioning model outputs, not measurements, and are not included.

## Quality rules

Each rule is a function in `qc.py`, runs in this order, and sets one bit of `<VAR>_flag`. A value removed by one
rule is counted under that rule only. Counts per site, source, variable and rule are in `qc_counts.parquet`; totals
are in the report below.

| bit | rule | definition | code |
|---|---|---|---|
| 1 | filled | the source holds a gap-fill at this step (ONEFlux QC 1 to 3); the fill is not stored | [`qc.py` `measured`](qc.py#L44) |
| 64 | quality | the source's steady-state and integral-turbulence flag (`*_SSITC_TEST`, Foken et al. 2004) is 2, for BASE and ICOS L2 fluxes that carry it | [`build.py` `apply_rules`](build.py#L118) |
| 2 | range | outside the AmeriFlux FP physical limits of the variable (for example NEE -100 to 100 umol m-2 s-1, FCH4 -500 to 4000 nmol m-2 s-1, TA -50 to 50 deg C), widened by 10 W m-2 for shortwave and 20 umol m-2 s-1 for PAR night offsets | [`qc.py` `out_of_range`](qc.py#L55) |
| 8 | stuck | inside a run of identical values at least 6 h long (12 h for VPD, RH, G; 24 h for PA, reported at 0.1 kPa at many sites; 2 d for soil temperature). Not tested: P, SWC, WTD. Runs of 100 % RH and 0 hPa VPD (fog) and soil temperature within 0.5 deg C of 0 (the zero curtain of freezing and thawing soil) are kept. For shortwave and PAR, steps below 5 deg solar elevation may repeat (darkness); in a run that reaches daylight only the daylight steps are removed, since a sensor reading 0 in daylight has failed while its zeros at night are right. Thresholds scale to the record's step. | [`qc.py` `stuck`](qc.py#L62) |
| 4 | spike | fluxes only: Papale et al. (2006) double-difference test in 13-day blocks, day and night separately, z = 5.5; only the centre of each displaced triple is removed. ONEFlux already despiked its fluxes, so on ONEFlux data this is a second pass; for FLUXNET-CH4, BASE and ICOS L2 it is the only despiking applied here | [`qc.py` `spikes`](qc.py#L85) |
| 16 | ustar | see the u* section | [`qc.py` `low_ustar`](qc.py#L121) |
| 32 | site | the whole variable at the site fails a record-level test: a flux whose daytime correlation with its driver has the wrong sign with abs(r) > 0.3 (NEE against SW_IN, expected negative, not tested at WAT, URB, BSV, SNO sites; H and LE against NETRAD, expected positive), or an FCH4 record whose median abs(FCH4) is below 0.05 nmol m-2 s-1 with a 99th percentile below 1 (umol reported as nmol) | [`qc.py` `sign_check`](qc.py#L227), [`qc.py` `fch4_unit_check`](qc.py#L239) |

A value is a target when it is finite. A NaN value with flag 0 means the source has no value at that step.

### u* filtering

- ONEFlux `NEE_VUT_REF` is u*-filtered by ONEFlux at the variable-u*-threshold reference (Papale et al. 2006;
  Pastorello et al. 2020); the per-year thresholds are read from the BIF `GRP_FLUX_REF` records and kept in each
  site's sidecar. In the processed archives no measured `NEE_VUT_REF` half-hour, day or night, has u* below its
  year's threshold (`oneflux_nee_below_ustar_threshold` in the sidecars): ONEFlux filters day and night.
- FLUXNET-CH4 NEE, LE and H were u*-filtered with REddyProc by the product (Delwiche et al. 2021).
- This dataset u*-filters FCH4 (all sources) and NEE from BASE and ICOS L2, at every step, day and night, as ONEFlux
  does for NEE. The threshold per site-year is, in order: the ONEFlux VUT reference threshold of the same site (the
  median over its years for years outside the ONEFlux record); for FLUXNET-CH4 sites outside ONEFlux, the 0.5th
  percentile of night-time u* under the product's own u*-filtered NEE; for BASE and ICOS L2 sites outside ONEFlux,
  the moving point test (Reichstein et al. 2005; Papale et al. 2006) on the record's own night-time NEE. A step
  without u* while a threshold exists is removed. The source used is in `sites.ustar_source`.
  [`build.py` `ustar_steps`](build.py#L76), [`build.py` `infer_ustar`](build.py#L85), [`qc.py` `ustar_mpt`](qc.py#L135)
- LE and H are not u*-filtered, following ONEFlux.

### Time

- All four sources publish TIMESTAMP_START and TIMESTAMP_END in local standard time without daylight saving. The
  dataset stores `time_lst` as published and `time_utc` = `time_lst` minus the UTC offset. The offset is the BIF
  `UTC_OFFSET` (ONEFlux), the BASE-BADM `UTC_OFFSET`, or for ICOS L2 the difference between the file's first
  timestamp and the object's acquisition start in UTC in the Carbon Portal metadata (UTC+2 in Finland and Greece,
  UTC+1 elsewhere). A FLUXNET-CH4 site outside ONEFlux takes the BASE-BADM `UTC_OFFSET` of the same
  site. A site with none uses longitude / 15, corrected by whole hours when the clock test on its own SW_IN finds the
  sun an hour or more away (Alaska is UTC-9 at 150 W). The offset source is in each sidecar.
- Hourly records (ONEFlux `_HR_` products) stay hourly: `sites.step_min` is 60 and each row is one hour. They are
  listed as `hourly` events.
- Clock test, per source and year: the shift (in steps, up to 3 h) that best aligns the mean diurnal cycle of the
  measured SW_IN with the sine of solar elevation at the stated offset ([`qc.py` `clock_offset_minutes`](qc.py#L198)).
  Offsets of 60 min or more are recorded as `clock` events; values are not shifted. A 30-min estimate is within the
  resolution of the test (tropical afternoon convection moves the radiation centroid) and is kept in the sidecar only.
- Duplicate and off-grid timestamps are dropped at the merge and counted in `merge` events.

### Deduplication: one record per tower and variable

FLUXNET site IDs are global, so a tower served by several networks is one row. For each site, variable and calendar
year, the dataset takes the first source in this order that has any measured value that year, and records the
choice in `<VAR>_src` and `coverage.parquet` ([`build.py` `merge_site`](build.py#L434)):
- FCH4: FLUXNET-CH4, ICOS L2, BASE.
- WTD: FLUXNET-CH4, BASE.
- All other variables: ONEFlux (Shuttle), FLUXNET-CH4, BASE, ICOS L2.

Where two sources measure the same steps, their agreement (n, r, slope, mean difference) is recorded as a
`source_overlap` event, and as `source_overlap_disagree` when r < 0.8 or the slope differs from 1 by more than 0.2,
which shows a reprocessing, unit or version difference ([`build.py` `source_overlaps`](build.py#L534)). Different site
IDs within 1 km of each other are recorded as `colocated` events (paired towers, or one tower renamed).

### Instruments and site moves

- Instrument changes: where the BIFVARINFO records of a variable change instrument model or height over time, each
  change is an `instrument` event with its date ([`build.py` `instrument_changes`](build.py#L268)).
- Site moves: where the BIF has several LOCATION records, each move is a `site_move` event with its distance and the
  network's comment. The site's coordinates are the latest LOCATION record.

## Coordinates and site descriptors

- Latitude and longitude are the tower location from network metadata: the ONEFlux BIF `LOCATION_LAT` and
  `LOCATION_LONG` of the latest `LOCATION_DATE_START`, at the precision the network publishes. Sites outside the
  Shuttle take the AmeriFlux or FLUXNET `site_display` BADM location, or the ICOS station metadata. The source is in
  `sites.coord_source`.
- IGBP: BIF or BADM `IGBP`.
- Koppen: the network's `CLIMATE_KOEPPEN` (normalised, for example Bsh to BSh); else the Beck et al. (2023) 1-km map
  for 1991-2020 at the tower (most common class within 3 pixels where the tower pixel is water); else the class from
  the site's ERA5 1991-2020 normals by the Beck et al. (2018) criteria ([`koppen.py` `classify`](koppen.py#L49)). The
  source is in `sites.koppen_source`.

## Inclusion

- A 180-day window (30-day stride) is usable when at least 20 % of its steps have a measured NEE, LE, H or FCH4.
- A site is included when it has at least one usable window, which needs at least 180 days of record.
- A site-year counts as measured when a flux is measured in at least 10 % of its steps; `first_year`, `last_year`
  and `n_years_measured` count those years.
- Candidate sites that are not included stay in `sites.parquet` with `included = False`.

## Balance and sampling weights

The towers are not evenly spread: most are temperate and boreal (Koppen C and D), in North America and Europe, and
forests and croplands; tropical (A), arid (B) and polar (E) sites and wetlands are fewer. The distributions are in the
report below. Two measures address this:
- The split is stratified by IGBP group x Koppen main class x FCH4, so the test set has the same mix as the training
  set and rare strata are represented in both where they have enough sites.
- `weights.parquet` gives each site a weight inversely proportional to the size of its stratum (IGBP group x Koppen
  main class), normalised to a mean of 1 over training sites and capped at 5, and each window its site's weight
  divided by the site's number of usable windows. Sampling windows by `window_weight` gives every stratum similar
  total weight and keeps long records from outweighing short ones ([`report.py` `build`](report.py#L39)).

## Split

- 80/20 by site, seed 20261026 ([`split.py` `make_split`](split.py#L77)).
- Spatial blocking: sites within 50 km of each other are linked, and each connected group (single linkage) goes
  wholly to one side, so no test site lies within 50 km of a training or validation site. At 50 km the largest block
  has 24 sites (about 3 % of the sites); 100 km would chain blocks of 38 sites and remove most of the freedom to
  stratify, and 25 km would leave towers that share weather and landscape on both sides. The split file records the
  smallest test-to-train distance.
- Stratification: blocks are taken in seeded random order; a block joins the test set when it brings the test share
  of its strata closer to 20 %. A second pass fills the remaining share with the smallest blocks. FCH4 is a stratum
  dimension, so the methane sites split in the same proportion by the same blocks.
- Validation: the same procedure (seed + 1) takes 10 % of the training sites, by block, for model selection. Test
  sites are never used for tuning.
- The split is frozen: `split.py` refuses to overwrite `split.json`, which records the SHA-256 of `split.csv`.

## Statistics

<!-- report:start -->
Generated by `report.py` from the build in `/home/photon/4tb/fluxfill/towers`.

- Candidate sites: 823. Included (at least one usable 180-day window): 678. With measured FCH4: 121. Hourly records: 7.
- Windows (180 d, 30-d stride): 45571, of which usable: 43121.
- Split (sites): {'train': 488, 'test': 136, 'val': 54}. FCH4 sites: {'train': 95, 'test': 21, 'val': 5}.
- Size: per-site tables 6.63 GB, interim tables 6.61 GB.

Values removed per rule, summed over sites and sources (each value counted under the first rule that removes it; `filled` counts source gap-fills, never stored):

| variable | measured_at_source | filled | quality | range | stuck | spike | ustar | site | final |
|---|---|---|---|---|---|---|---|---|---|
| NEE | 28893536 | 35980693 | 569769 | 982 | 7329 | 153850 | 961708 | 9552 | 27190346 |
| LE | 52935606 | 22432750 | 1271469 | 1397 | 7486 | 274233 | 0 | 0 | 51381021 |
| H | 58634808 | 18061659 | 1101041 | 918 | 9888 | 224658 | 0 | 0 | 57298303 |
| FCH4 | 4311894 | 0 | 446069 | 2460 | 147 | 55939 | 715969 | 0 | 3091310 |
| G | 51475349 | 4854796 | 0 | 9421 | 180684 | 0 | 0 | 0 | 51285244 |
| TA | 70945252 | 6742701 | 0 | 1 | 10490 | 0 | 0 | 0 | 70934761 |
| SW_IN | 71239244 | 6292720 | 0 | 3291 | 34053 | 0 | 0 | 0 | 71201900 |
| LW_IN | 56126774 | 6777466 | 0 | 41776 | 10903 | 0 | 0 | 0 | 56074095 |
| VPD | 65558922 | 8621953 | 0 | 32386 | 36014 | 0 | 0 | 0 | 65490522 |
| RH | 69487453 | 0 | 0 | 29505 | 471074 | 0 | 0 | 0 | 68986874 |
| PA | 63011301 | 0 | 0 | 13251 | 135925 | 0 | 0 | 0 | 62862125 |
| P | 55216858 | 0 | 0 | 139 | 0 | 0 | 0 | 0 | 55216719 |
| WS | 65705838 | 0 | 0 | 34 | 211846 | 0 | 0 | 0 | 65493958 |
| USTAR | 67191621 | 0 | 0 | 13873 | 3860 | 0 | 0 | 0 | 67173888 |
| NETRAD | 57454498 | 0 | 0 | 12075 | 80306 | 0 | 0 | 0 | 57362117 |
| PPFD_IN | 48995394 | 0 | 0 | 10994 | 20051 | 0 | 0 | 0 | 48964349 |
| TS_1 | 58055881 | 7253904 | 0 | 100 | 102858 | 0 | 0 | 0 | 57952923 |
| SWC_1 | 44851889 | 9971809 | 0 | 45253 | 0 | 0 | 0 | 0 | 44806636 |
| WTD | 4130722 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4130722 |

Events recorded (events.parquet):

| kind | count |
|---|---|
| instrument | 4407 |
| source_overlap | 544 |
| merge | 538 |
| colocated | 136 |
| clock | 26 |
| source_overlap_disagree | 14 |
| site_move | 8 |
| hourly | 7 |
| timestamps | 2 |
| site_test | 1 |

Sites by IGBP class:

| igbp | test | train | val | all |
|---|---|---|---|---|
| CRO | 26 | 92 | 10 | 128 |
| GRA | 24 | 84 | 9 | 117 |
| WET | 22 | 85 | 8 | 115 |
| ENF | 20 | 62 | 8 | 90 |
| DBF | 11 | 42 | 5 | 58 |
| OSH | 8 | 24 | 4 | 36 |
| EBF | 7 | 23 | 4 | 34 |
| MF | 3 | 14 | 2 | 19 |
| WSA | 2 | 11 | 1 | 14 |
| SAV | 3 | 10 | 0 | 13 |
| CSH | 1 | 8 | 1 | 10 |
| URB | 2 | 6 | 1 | 9 |
| CVM | 2 | 7 | 0 | 9 |
| BSV | 2 | 5 | 1 | 8 |
| DNF | 2 | 6 | 0 | 8 |
| WAT | 1 | 7 | 0 | 8 |
| SNO | 0 | 2 | 0 | 2 |

Sites by Koppen main class:

| koppen_main | test | train | val | all |
|---|---|---|---|---|
| C | 59 | 197 | 23 | 279 |
| D | 52 | 183 | 22 | 257 |
| B | 13 | 52 | 6 | 71 |
| A | 7 | 31 | 2 | 40 |
| E | 5 | 25 | 1 | 31 |

Sites by latitude band:

| lat_band | test | train | val | all |
|---|---|---|---|---|
| 35-50N | 62 | 265 | 30 | 357 |
| 50-66.5N | 28 | 86 | 7 | 121 |
| 23.5-35N | 14 | 47 | 12 | 73 |
| <23.5S | 18 | 22 | 3 | 43 |
| >66.5N | 4 | 29 | 0 | 33 |
| 23.5S-0 | 5 | 24 | 0 | 29 |
| 0-23.5N | 5 | 15 | 2 | 22 |

Sites by data hub:

| data_hub | test | train | val | all |
|---|---|---|---|---|
| AmeriFlux | 56 | 237 | 26 | 319 |
| ICOS | 61 | 223 | 26 | 310 |
| TERN | 19 | 20 | 2 | 41 |
| FLUXNET | 0 | 8 | 0 | 8 |

Strata with the largest and smallest sampling weights:

| stratum | sites | site_weight |
|---|---|---|
| cropland|D | 62 | 0.3087 |
| grassland|C | 61 | 0.3138 |
| forest_needle|D | 57 | 0.3358 |
| cropland|C | 57 | 0.3358 |
| wetland|D | 53 | 0.3612 |
| grassland|A | 4 | 4.7855 |
| forest_needle|B | 3 | 5.0 |
| forest_mixed|A | 1 | 5.0 |
| wetland|B | 3 | 5.0 |
| forest_mixed|B | 1 | 5.0 |
<!-- report:end -->

## Build

```bash
python -m venv --system-site-packages /home/photon/4tb/envs/fluxfill-towers
/home/photon/4tb/envs/fluxfill-towers/bin/pip install git+https://github.com/fluxnet/shuttle.git openpyxl
RAW=/home/photon/4tb/fluxfill/raw/towers OUT=/home/photon/4tb/fluxfill/towers \
  PY=/home/photon/4tb/envs/fluxfill-towers/bin/python ./build_all.sh
```

Stages: `fetch.py` (FLUXNET-CH4, BASE, ICOS L2), `build.py shuttle` (download, clean, delete each Shuttle
archive), `build.py clean` (the other sources), `build.py merge`, `build.py tables`, `split.py`, `report.py`.

Tests: `python -m pytest -q` in this directory. They cover the measured-only rule, each QC rule, the moving point
test, solar geometry and the clock test, Koppen classes, the BASE column choice, the merge priority, and the split's
50 km guarantee and balance.

## Licence and attribution

All tower data are CC-BY-4.0. Use requires, per the FLUXNET data policy (DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt,
version 20251226, in every Shuttle archive):
1. The citation of each site's data product, with its DOI or PID: `sites.citation` (Shuttle `product_citation`) and
   `sites.shuttle_pid`. FLUXNET-CH4 and BASE site DOIs are listed on each site's AmeriFlux or FLUXNET page.
2. The acknowledgement: "FLUXNET data products were produced and harmonized by eddy covariance regional networks
   and data processing centers, including AmeriFlux, ChinaFlux, European Fluxes Database, ICOS, JapanFlux, KoFlux,
   OzFlux, SAEON, and TERN. These products also include a modified version of ERA5 hourly data provided by the
   Copernicus Climate Change Service." Plus the network-specific text in `sites.acknowledgement` (AmeriFlux: "Funding
   for the AmeriFlux data service was provided by the U.S. Department of Energy Office of Science."; ChinaFlux,
   JapanFlux, OzFlux and TERN: the texts and references in that column).
3. In a Data Availability section: "The FLUXNET data products used in this work were downloaded from the FLUXNET
   Shuttle (fluxnet-shuttle library) and the AmeriFlux and ICOS data portals on 26 and 27 October 2026."

Network policies: AmeriFlux CC-BY-4.0 data policy (ameriflux.lbl.gov/data/data-policy/); ICOS licence and citation
(www.icos-cp.eu/how-to-cite); JapanFlux (ads.nipr.ac.jp/japan-flux2024/); TERN terms of use
(www.tern.org.au/terms-of-use/). PIs are listed in `sites.pis`; the policy encourages contacting them before
publication.

## References

- Pastorello, G., Trotta, C., Canfora, E. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing pipeline
  for eddy covariance data. Scientific Data 7, 225. doi:10.1038/s41597-020-0534-3
- Delwiche, K. B., Knox, S. H., Malhotra, A. et al. (2021). FLUXNET-CH4: a global, multi-ecosystem dataset and
  analysis of methane seasonality from freshwater wetlands. Earth System Science Data 13, 3607-3689.
  doi:10.5194/essd-13-3607-2021
- Knox, S. H., Jackson, R. B., Poulter, B. et al. (2019). FLUXNET-CH4 synthesis activity: objectives, observations,
  and future directions. Bulletin of the American Meteorological Society 100, 2607-2632.
  doi:10.1175/BAMS-D-18-0268.1
- Papale, D., Reichstein, M., Aubinet, M. et al. (2006). Towards a standardized processing of Net Ecosystem Exchange
  measured with eddy covariance technique: algorithms and uncertainty estimation. Biogeosciences 3, 571-583.
  doi:10.5194/bg-3-571-2006
- Reichstein, M., Falge, E., Baldocchi, D. et al. (2005). On the separation of net ecosystem exchange into
  assimilation and ecosystem respiration: review and improved algorithm. Global Change Biology 11, 1424-1439.
  doi:10.1111/j.1365-2486.2005.001002.x
- Foken, T., Gockede, M., Mauder, M., Mahrt, L., Amiro, B., Munger, W. (2004). Post-field data quality control. In
  Handbook of Micrometeorology, Kluwer, 181-208. doi:10.1007/1-4020-2265-4_9
- Vuichard, N., Papale, D. (2015). Filling the gaps in meteorological continuous data measured at FLUXNET sites with
  ERA-Interim reanalysis. Earth System Science Data 7, 157-171. doi:10.5194/essd-7-157-2015
- Wutzler, T., Lucas-Moffat, A., Migliavacca, M. et al. (2018). Basic and extensible post-processing of eddy
  covariance flux data with REddyProc. Biogeosciences 15, 5015-5030. doi:10.5194/bg-15-5015-2018
- Beck, H. E., Zimmermann, N. E., McVicar, T. R. et al. (2018). Present and future Koppen-Geiger climate
  classification maps at 1-km resolution. Scientific Data 5, 180214. doi:10.1038/sdata.2018.214
- Beck, H. E., McVicar, T. R., Vergopolan, N. et al. (2023). High-resolution (1 km) Koppen-Geiger maps for 1901-2099
  based on constrained CMIP6 projections. Scientific Data 10, 724. doi:10.1038/s41597-023-02549-6
- Kottek, M., Grieser, J., Beck, C., Rudolf, B., Rubel, F. (2006). World map of the Koppen-Geiger climate
  classification updated. Meteorologische Zeitschrift 15, 259-263. doi:10.1127/0941-2948/2006/0130
- Ueyama, M. et al. (2025). JapanFlux2024. Earth System Science Data 17, 3807. doi:10.5194/essd-17-3807-2025
- Yu, G.-R. et al. (2006). Overview of ChinaFLUX and evaluation of its eddy covariance measurement. Agricultural and
  Forest Meteorology 137, 125-137. doi:10.1016/j.agrformet.2006.02.011
- Beringer, J. et al. (2016). An introduction to the Australian and New Zealand flux tower network: OzFlux.
  Biogeosciences 13, 5895-5916. doi:10.5194/bg-13-5895-2016
- Isaac, P. et al. (2017). OzFlux data: network integration from collection to curation. Biogeosciences 14,
  2903-2928. doi:10.5194/bg-14-2903-2017
