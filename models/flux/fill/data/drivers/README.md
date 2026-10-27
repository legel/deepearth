# FluxFill meteorological drivers

Hourly surface meteorology at every tower, from sources that stay available when the tower's own sensors fail.
FluxFill reads them as a reanalysis modality next to the tower variables (docs/MODEL.md section 8). ERA5-Land is
the global baseline; sharper sources are kept only where they beat it at the towers.

Data: `/home/photon/4tb/fluxfill/drivers/<source>/`. Code: this directory. Built 2026-10-26.

## Layout

| path | content |
|---|---|
| `<source>/<SITE>.parquet` | `time_utc` (hourly valid time, UTC) and per FLUXNET variable `<VAR>` (bilinear value at the tower) and `<VAR>_NN` (nearest grid cell with a value), float32, rounded to `assemble.PRECISION` |
| `<source>/sites_used.csv` | tower coordinates used, nearest cell index and centre, cell size, cell orography, distance from tower to cell centre |
| `<source>/source.json` | units, variable mapping, time convention per variable, grid, licence, access date, every source file with its last-modified time |
| `dem/terrain.csv` | Copernicus DEM 30 m: tower elevation, slope, aspect, and mean and std of elevation within 1 km, within the ERA5-Land cell and within the ERA5 cell |
| `validation/2019_2020/validation.{csv,md}` | the validation table below (2019-2020, 36 towers); `validation/full/` is the same over all years and all towers with tower data, written after the full build |
| `_work/<source>/<variable>/<file>.npz` | extraction work files (nearest and bilinear series per site per source file). Deleted 2026-10-27 to keep the shared disk above its 40 GB floor; a coordinate refinement re-downloads the point series of the moved sites |

Everything is keyed by site id. The tower table (`towers/sites.parquet`) is rewritten later with refined
coordinates; `sites_used.csv` keeps the coordinates used, and a rerun of `openmeteo.py extract` re-reads only the
sites whose coordinates changed (`openmeteo._task`).

## Variables and time convention

| FLUXNET | unit | ERA5-Land (`era5_land`) | ERA5 (`era5`) | HRRR (`hrrr`) | IMERG, MRMS |
|---|---|---|---|---|---|
| TA | deg C | 2 m temperature | 2 m temperature | 2 m temperature | |
| VPD | hPa | es(T) - es(Td) | es(T) - es(Td) | es(T) - es(Td) | |
| SW_IN | W m-2 | (ERA5 forcing) | ssrd, mean of preceding hour | DSWRF, instantaneous | |
| LW_IN | W m-2 | not served without an account | not served without an account | DLWRF, instantaneous, from 2016-08 | |
| P | mm | (ERA5 forcing) | tp, sum of preceding hour | step-1 rate of previous run, sum of preceding hour | sum of preceding hour |
| WS | m s-1 | (ERA5 forcing) | hypot(u10, v10) | hypot(u10, v10) | |
| PA | kPa | (ERA5 forcing) | msl moved to the cell orography | surface pressure | |
| SWC_1, SWC_2 | % | layers 0-7 cm, 7-28 cm | | | |
| TS_1, TS_2 | deg C | layers 0-7 cm, 7-28 cm | | | |
| D_SNOW | cm | snow depth | | snow depth | |

State variables are instantaneous at `time_utc`. Accumulated and mean variables cover `(time_utc - 1 h, time_utc]`.
Saturation vapour pressure: Alduchov and Eskridge (1996). Pressure moves between heights hydrostatically with a
6.5 K km-1 lapse rate (`physics.py`). RH is not stored: RH = 100 (1 - VPD / es(TA)).

ERA5-Land is forced by ERA5. Its shortwave, longwave, precipitation and wind are the ERA5 fields interpolated to
0.1 deg; only temperature, humidity and pressure get a lapse-rate correction to the 0.1 deg orography
(Munoz-Sabater et al. 2021, section 2.1). So ERA5 at the tower is the ERA5-Land forcing for SW_IN, P and WS. The
baseline for FluxFill is `era5_land` (TA, VPD, SWC, TS, D_SNOW) joined with `era5` (SW_IN, P, WS, PA).

## Source survey (2026-10-26)

Resolution, coverage from 2014, access without a paid account, latency, and the cost of a point series at about
800 towers on this machine's link (measured: 700 ms round trip to AWS us-east-1 under shared load, about 2 MB/s
per stream).

| source | grid | time step | coverage | access | latency | point cost, 800 towers, 2014-2026 | decision |
|---|---|---|---|---|---|---|---|
| ERA5-Land (Munoz-Sabater et al. 2021) | 0.1 deg (9 km) | 1 h | global land, 1950- | CDS account (free) for all variables; Open-Meteo mirror (s3://openmeteo, anonymous) holds T, Td, soil, snow only | 5 days | about 4.5 GB of range requests from the mirror (61 kB per site-variable-year), latency-bound: 7 h at 100 workers | baseline: acquired |
| ERA5 (Hersbach et al. 2020) | 0.25 deg | 1 h | global, 1940- | Open-Meteo mirror (no longwave), Earthmover Icechunk (no longwave), ARCO-ERA5 GCS (all variables, one global field per chunk) | 5 days | SW, P, wind, pressure: about 2 GB from the mirror; longwave: 170 to 230 GB (ARCO or NCAR files) | acquired except LW_IN |
| HRRR (Dowell et al. 2022) | 3 km | 1 h | CONUS (318 of 823 towers), 2014-10- | anonymous: AWS noaa-hrrr-bdp-pds (GRIB2 + .idx), dynamical.org Icechunk analysis | 1 h | about 5 MB per tower-variable-90 days: 330 GB for all CONUS towers; GRIB byte ranges 1.5 MB per variable-hour: 1.5 TB | validated at 12 towers, 2019-2020 |
| NLDAS-3 beta forcing | 0.01 deg (1 km) | 1 h | North and Central America, 2001-2023 | anonymous: s3://nasa-waterinsight/NLDAS3/forcing | beta, ends 2023 | daily 12.8 GB files, chunks 6 h x 500 x 900 cells: about 130 MB per tower-day | not acquired: needs extraction inside us-west-2 |
| NLDAS-2 | 0.125 deg | 1 h | CONUS, 1979- | NASA Earthdata login (free) | 4 days | small (GES DISC subsetting) | account-gated; not sharper than ERA5-Land |
| CERRA, CERRA-Land (Schimanke et al. 2021) | 5.5 km | 3 h analyses, 1 h forecasts | Europe, 1984-, resumed to near real time in 2026 | CDS account (free) | months | CDS requests per month and variable | account-gated: the European candidate |
| MERRA-2 (Gelaro et al. 2017) | 0.5 x 0.625 deg | 1 h | global, 1980- | NASA Earthdata login | 3 weeks | small (OPeNDAP) | coarser; account-gated LW_IN alternative |
| GLDAS-2.1 | 0.25 deg | 3 h | global, 2000- | NASA Earthdata login | 1 month | small | coarser in space and time |
| IMERG V07 (Huffman et al. 2023) | 0.1 deg | 30 min | 60 N-60 S, 1998- | Final: Earthdata login; Early and Late: anonymous dynamical.org Icechunk | 4 h (Early), 14 h (Late), 3.5 months (Final) | about 25 s per tower-2 years | validated (Late) |
| MRMS QPE (Zhang et al. 2016) | 0.01 deg | 1 h | CONUS, 2014-11- | anonymous dynamical.org Icechunk | 1 h | about 15 s per tower-2 years | validated |
| MSWEP v2.8 (Beck et al. 2019) | 0.1 deg | 3 h | global, 1979- | registration form (gloh2o.org), CC BY-NC 4.0 | hours (NRT) | small | account-gated; 3-hourly |
| SARAH-3 (Pfeifroth et al. 2024) | 0.05 deg | 30 min | MSG disk: Europe, Africa, Middle East, east South America, 1983- | CM SAF registration (free) | ICDR monthly | moderate (NetCDF per day) | account-gated: the Europe-Africa radiation candidate |
| NSRDB GOES (Sengupta et al. 2018) | 2 to 4 km | 5 to 30 min | Americas, 1998- | NREL API key (free, e-mail) or AWS h5 files | about 1 year | moderate | account-gated: the Americas radiation candidate |
| Himawari SWR (JAXA P-Tree) | 5 km | 10 min, 1 h | Asia-Pacific, 2015-07- | JAXA registration | hours | moderate | account-gated |
| MODIS / VIIRS LST (MOD11A1, VNP21A1) | 1 km | 2 overpasses per day per satellite | global, 2000- / 2012- | Earthdata login, or Planetary Computer (anonymous tokens) | 1 to 8 days | small per tile | not a driver at hourly scale: skin temperature at overpass times; candidate imagery token |
| SMAP L4 (SPL4SMGP) | 9 km | 3 h | global, 2015-03- | NSIDC Earthdata login | 3 days | small | account-gated: soil moisture candidate |
| Copernicus DEM GLO-30 | 30 m | static | global | anonymous: s3://copernicus-dem-30m | | 10 s per tower | acquired |
| Open-Meteo API (historical archive) | as the model served | 1 h | | free, non-commercial; 600/min, 5,000/h, 10,000/day, calls with more than 10 variables or 2 weeks count multiple | | about 390,000 weighted calls: 39 days at the daily cap | not used; the same data read from its S3 mirror |

Provenance notes for point APIs. Open-Meteo's `era5_land` model serves ERA5-Land 0.1 deg for temperature,
humidity, soil and snow, and ERA5 0.25 deg for precipitation, radiation and wind; it serves no downward longwave
from any model. `ecmwf_ifs` (2017-) is the operational IFS at 9 km, a forecast archive, not a reanalysis.
`cerra` stops in June 2021 on Open-Meteo.

## Validation

Tower data: the measured-only half-hours of the tower dataset build (`towers/interim/shuttle`, ONEFlux v1.3 with
QC 0, 2026-10-26), 2019-2020, 36 towers with data in those years (34 with TA). Matching: `validate.py` docstring;
hourly at the driver valid time in UTC, daily over local-standard-time days. Bias is driver minus tower. Each
"matched set" uses only the site-hours where the tower and every source in the set have a value, so rows within a
set compare like with like. `oneflux_era5` is the ERA5 downscaled by ONEFlux with a regression on the tower's own
measurements (Vuichard and Papale 2015): a reference, not an independent driver. `era5_land_dem` moves ERA5-Land
TA from its bilinear cell orography to the Copernicus DEM tower elevation at 6.5 K km-1; `era5_dem` moves ERA5
PA the same way.

Time alignment check: the lag that maximises the hourly correlation between driver and tower is 0 h for ERA5
SW_IN, HRRR SW_IN and ERA5-Land TA at US-Bi1, CA-Mer, AR-CCa and BR-Xpw (lags -2 to +2 h tested).

| variable | matched set | source | sites | hours | hourly bias | hourly RMSE | hourly r | daily bias | daily RMSE | daily r |
|---|---|---|---|---|---|---|---|---|---|---|
| TA | era5_land+era5+era5_land_dem+oneflux_era5 | era5_land | 34 | 500747 | 0.60 | 2.44 | 0.984 | 0.60 | 1.82 | 0.991 |
| TA | era5_land+era5+era5_land_dem+oneflux_era5 | era5 | 34 | 500747 | 0.64 | 2.42 | 0.985 | 0.65 | 1.84 | 0.991 |
| TA | era5_land+era5+era5_land_dem+oneflux_era5 | era5_land_dem | 34 | 500747 | 0.63 | 2.48 | 0.984 | 0.63 | 1.87 | 0.991 |
| TA | era5_land+era5+era5_land_dem+oneflux_era5 | oneflux_era5 | 34 | 500747 | 0.08 | 2.28 | 0.985 | 0.09 | 1.33 | 0.995 |
| TA | era5_land+era5_land_dem+hrrr | era5_land | 12 | 195519 | 0.99 | 2.57 | 0.978 | 0.99 | 2.06 | 0.986 |
| TA | era5_land+era5_land_dem+hrrr | era5_land_dem | 12 | 195519 | 1.10 | 2.75 | 0.975 | 1.10 | 2.27 | 0.983 |
| TA | era5_land+era5_land_dem+hrrr | hrrr | 12 | 195519 | 0.72 | 2.31 | 0.981 | 0.72 | 1.90 | 0.987 |
| VPD | era5_land+era5+oneflux_era5 | era5_land | 34 | 485019 | 0.55 | 2.97 | 0.892 | 0.52 | 2.16 | 0.911 |
| VPD | era5_land+era5+oneflux_era5 | era5 | 34 | 485019 | 0.48 | 2.89 | 0.893 | 0.45 | 2.03 | 0.913 |
| VPD | era5_land+era5+oneflux_era5 | oneflux_era5 | 34 | 485019 | -0.00 | 2.76 | 0.892 | -0.01 | 1.34 | 0.952 |
| VPD | era5_land+hrrr | era5_land | 12 | 193592 | 0.69 | 3.00 | 0.899 | 0.67 | 2.32 | 0.901 |
| VPD | era5_land+hrrr | hrrr | 12 | 193592 | 0.75 | 2.55 | 0.943 | 0.73 | 1.75 | 0.949 |
| SW_IN | era5+oneflux_era5 | era5 | 34 | 518505 | 3.55 | 80.28 | 0.951 | 3.45 | 34.37 | 0.944 |
| SW_IN | era5+oneflux_era5 | oneflux_era5 | 34 | 518505 | -8.46 | 106.02 | 0.912 | -8.70 | 34.08 | 0.949 |
| SW_IN | era5+hrrr | era5 | 12 | 201262 | 0.94 | 70.87 | 0.962 | 0.92 | 28.42 | 0.963 |
| SW_IN | era5+hrrr | hrrr | 12 | 201262 | 11.52 | 91.20 | 0.947 | 11.64 | 36.67 | 0.953 |
| LW_IN | oneflux_era5 | oneflux_era5 | 29 | 1639196 | 6.94 | 44.05 | 0.796 | 6.91 | 39.68 | 0.821 |
| LW_IN | oneflux_era5+hrrr | oneflux_era5 | 11 | 182377 | 14.09 | 60.21 | 0.616 | 14.07 | 56.43 | 0.636 |
| LW_IN | oneflux_era5+hrrr | hrrr | 11 | 182377 | -18.42 | 36.00 | 0.855 | -18.40 | 28.50 | 0.913 |
| P | era5+oneflux_era5 | era5 | 28 | 387533 | 0.02 | 0.66 | 0.397 | 0.38 | 5.36 | 0.698 |
| P | era5+oneflux_era5 | oneflux_era5 | 28 | 387533 | -0.05 | 0.65 | 0.380 | -1.27 | 5.72 | 0.743 |
| P | era5+imerg | era5 | 18 | 273002 | 0.01 | 0.71 | 0.394 | 0.24 | 5.72 | 0.704 |
| P | era5+imerg | imerg | 18 | 273002 | 0.01 | 0.85 | 0.371 | 0.29 | 7.78 | 0.609 |
| P | era5+mrms | era5 | 9 | 133297 | 0.03 | 0.56 | 0.456 | 0.74 | 4.92 | 0.733 |
| P | era5+mrms | mrms | 9 | 133297 | 0.01 | 0.58 | 0.525 | 0.22 | 5.36 | 0.668 |
| P | era5+hrrr | era5 | 9 | 141994 | 0.03 | 0.55 | 0.459 | 0.71 | 4.79 | 0.738 |
| P | era5+hrrr | hrrr | 9 | 141994 | -0.01 | 0.75 | 0.329 | -0.20 | 5.22 | 0.667 |
| P | era5+hrrr+mrms+imerg | era5 | 9 | 133237 | 0.03 | 0.56 | 0.456 | 0.73 | 4.92 | 0.733 |
| P | era5+hrrr+mrms+imerg | hrrr | 9 | 133237 | -0.01 | 0.72 | 0.344 | -0.21 | 5.17 | 0.677 |
| P | era5+hrrr+mrms+imerg | mrms | 9 | 133237 | 0.01 | 0.58 | 0.525 | 0.21 | 5.37 | 0.666 |
| P | era5+hrrr+mrms+imerg | imerg | 9 | 133237 | 0.03 | 0.66 | 0.455 | 0.72 | 6.11 | 0.673 |
| WS | era5+oneflux_era5 | era5 | 34 | 463525 | 0.89 | 2.01 | 0.612 | 0.89 | 1.75 | 0.637 |
| WS | era5+oneflux_era5 | oneflux_era5 | 34 | 463525 | -0.09 | 1.11 | 0.826 | -0.09 | 0.58 | 0.932 |
| WS | era5+hrrr | era5 | 12 | 185364 | 1.01 | 1.99 | 0.582 | 1.00 | 1.75 | 0.594 |
| WS | era5+hrrr | hrrr | 12 | 185364 | 0.59 | 1.44 | 0.711 | 0.58 | 1.11 | 0.746 |
| PA | era5+era5_dem+oneflux_era5 | era5 | 33 | 487375 | -0.26 | 1.77 | 0.726 | -0.26 | 1.77 | 0.720 |
| PA | era5+era5_dem+oneflux_era5 | era5_dem | 33 | 487375 | -0.06 | 0.59 | 0.971 | -0.06 | 0.58 | 0.972 |
| PA | era5+era5_dem+oneflux_era5 | oneflux_era5 | 33 | 487375 | -0.02 | 0.28 | 0.993 | -0.02 | 0.25 | 0.995 |
| PA | era5+era5_dem+hrrr | era5 | 12 | 192949 | -0.75 | 1.33 | 0.803 | -0.75 | 1.33 | 0.801 |
| PA | era5+era5_dem+hrrr | era5_dem | 12 | 192949 | -0.28 | 0.85 | 0.899 | -0.29 | 0.84 | 0.899 |
| PA | era5+era5_dem+hrrr | hrrr | 12 | 192949 | -0.16 | 0.73 | 0.920 | -0.16 | 0.72 | 0.921 |
| SWC_1 | era5_land | era5_land | 22 | 334927 | 10.34 | 23.93 | 0.374 | 10.39 | 23.85 | 0.375 |
| TS_1 | era5_land | era5_land | 29 | 428201 | 0.58 | 3.02 | 0.960 | 0.59 | 2.56 | 0.972 |

Per site, HRRR against the ERA5-Land baseline (hourly RMSE, 12 towers, 2019-2020): HRRR is lower for TA at 10 of
12 towers, VPD 10 of 12, WS 11 of 12, PA 7 of 12 (against ERA5 moved to the tower elevation), SW_IN 0 of 12.

Reading:
- ERA5-Land (0.1 deg) does not beat ERA5 (0.25 deg) at the towers: TA hourly RMSE 2.44 vs 2.42 K, VPD 2.97 vs 2.89
  hPa (34 towers). Its gain is soil and snow, which ERA5 at 0.25 deg resolves less finely, not the near-surface
  air.
- HRRR beats ERA5-Land for TA (hourly RMSE 2.31 vs 2.57 K, daily 1.90 vs 2.06 K), VPD (2.55 vs 3.00 hPa, r 0.943
  vs 0.899) and WS (1.44 vs 1.99 m s-1, r 0.711 vs 0.582) at 12 CONUS-domain towers. It is worse for SW_IN
  (hourly RMSE 91 vs 71 W m-2, daily 37 vs 28 W m-2, bias +11.5 W m-2): the archive holds the instantaneous flux at
  the hour and HRRR has too few clouds. HRRR LW_IN is low by 18 W m-2 (r 0.855).
- The lapse-rate move of ERA5-Land TA to the tower elevation makes TA worse (2.48 vs 2.44 K at 34 towers, 2.75
  vs 2.57 K at the 12 HRRR towers). The hydrostatic move of PA to the tower elevation makes PA much better (RMSE
  1.77 to 0.59 kPa, r 0.726 to 0.971). Pressure is set by height; near-surface temperature at a tower is not.
- Precipitation: no gridded source beats ERA5 on daily totals at these towers. Daily r: ERA5 0.704 vs IMERG Late
  0.609 (18 towers); ERA5 0.733 vs MRMS 0.668 and HRRR 0.677 (9 towers). MRMS has the best hourly timing (r 0.525
  vs 0.456). Tower gauges undercatch snow and several of the nine are Canadian, outside MRMS gauge correction.
- ERA5-Land SWC_1 has r 0.374 against tower SWC_1 (22 towers): sensor depth and calibration differ by site, so the
  model should learn a per-site mapping. TS_1 has r 0.960.

Decision:
1. Baseline everywhere: ERA5-Land 0.1 deg for TA, VPD, SWC, TS, D_SNOW and ERA5 0.25 deg for SW_IN, P, WS, PA
   (its own forcing), from 2014-01-01 to five days before the build. PA is moved to the tower elevation with
   `dem/terrain.csv` at load time.
2. CONUS (318 towers): add HRRR TA, VPD, WS, PA as a second driver modality. It wins at 10 to 11 of 12 towers on
   TA, VPD and WS. Keep ERA5 SW_IN; HRRR SW_IN and LW_IN go in as extra inputs only. Full acquisition costs about
   5 MB per tower-variable-90 days from the dynamical.org archive (about 150 GB of reads for four variables at 318
   towers over 12 years), so it runs after the ERA5 extraction or inside AWS us-west-2.
3. Precipitation: ERA5 stays. MRMS (CONUS hourly timing) and IMERG (tropics) are candidates for the R3 ablation
   (docs/MODEL.md section 7, item 9), not replacements.
4. LW_IN: the ERA5 longwave needs a CDS account (the ERA5-Land time-series product returns point series in
   seconds) or a 170 to 230 GB read of ARCO-ERA5. Until then FluxFill uses the ONEFlux ERA5 LW_IN in the tower files
   for training sites and HRRR LW_IN in CONUS.
5. Europe (218 towers): no anonymous source sharper than ERA5-Land covers 2014 to now. CERRA 5.5 km (CDS account)
   is the candidate.

## Size

Measured after 20 towers (2026-10-26, 2014-01-01 to 2026-10-22, 62.5 % of hours filled at the time): 1.35 MB per
tower for `era5_land`, 1.55 MB for `era5`. Projected for 823 towers with all hours: 1.8 GB and 2.0 GB, plus about
6 GB of `_work` files; under 10 GB in total. HRRR, IMERG and MRMS at the validation towers: 6 MB. DEM: 0.2 MB.

## Terrain check of the tower coordinates

`dem/terrain.csv` against the elevation in the tower table: median absolute difference 3.6 m, 90th percentile
28.8 m (the DEM is a surface model and includes canopy). Seven towers differ by more than 100 m: BR-Sa1, CA-Ca2,
US-A39, US-Cop, US-Jo1, US-Sta, US-xKA (US-xKA: 1329 m in the table, 323 m in the DEM; Konza Prairie lies near
330 m). These are candidates for the coordinate refinement of the tower table. The tower lies a median 15 m (90th
percentile 156 m) from the mean elevation of its ERA5-Land cell.

## Account-gated sources that would add value

| account | unlocks | value for FluxFill |
|---|---|---|
| Copernicus CDS (free) | ERA5-Land time series (all variables incl. LW_IN, point series in seconds), ERA5 strd, CERRA and CERRA-Land 5.5 km | the missing global LW_IN; the European sharper driver |
| NASA Earthdata (free) | IMERG Final (gauge-calibrated), SMAP L4 9 km 3-hourly soil moisture, NLDAS-2, MERRA-2, GLDAS, MODIS and VIIRS LST | better precipitation in the tropics; a soil-moisture driver independent of ERA5-Land |
| CM SAF (free) | SARAH-3 0.05 deg 30-min surface radiation, Europe and Africa | SW_IN sharper than ERA5 for 300+ towers |
| NREL developer key (free) | NSRDB 2-4 km radiation, Americas | SW_IN sharper than ERA5 and HRRR in CONUS |
| JAXA P-Tree (free) | Himawari shortwave radiation, Asia-Pacific | SW_IN for Asian and Australian towers |

## Licences and provenance

| source | licence | URL | version | accessed |
|---|---|---|---|---|
| ERA5-Land, ERA5 | Copernicus licence (C3S); Open-Meteo mirror CC-BY-4.0 | s3://openmeteo/data/copernicus_era5_land, copernicus_era5 | as served; per-file last-modified times in `source.json` | 2026-10-26 |
| HRRR | NOAA public domain; dynamical.org CC-BY-4.0 | s3://dynamical-noaa-hrrr/noaa-hrrr-analysis/v0.2.0.icechunk | v0.2.0 | 2026-10-26 |
| IMERG V07 Late | NASA public domain; dynamical.org CC-BY-4.0 | s3://dynamical-nasa-imerg/nasa-imerg-analysis-late/v0.1.0.icechunk | v0.1.0 | 2026-10-26 |
| MRMS | NOAA public domain; dynamical.org CC-BY-4.0 | s3://dynamical-noaa-mrms/noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk | v0.3.0 | 2026-10-26 |
| Copernicus DEM GLO-30 | (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA | s3://copernicus-dem-30m | 2021 release (COG) | 2026-10-26 |

Required acknowledgement for ERA5 and ERA5-Land: "Generated using Copernicus Climate Change Service information
[2026]." Open-Meteo: "Weather data by Open-Meteo.com", CC-BY-4.0.

## Code

| file | role |
|---|---|
| `grid.py` | nearest cell, 2 x 2 corners, bilinear weights with land-sea renormalisation |
| `physics.py` | VPD, RH, surface pressure, wind speed, lapse rate |
| `openmeteo.py` | ERA5-Land and ERA5 point extraction from the Open-Meteo S3 mirror (process pool over source files; resumable; re-reads moved sites only) |
| `assemble.py` | per-site parquet tables, `sites_used.csv`, `source.json` |
| `hrrr.py` | HRRR analysis at the towers (dynamical.org Icechunk) |
| `precip.py` | IMERG Late and MRMS at the towers |
| `dem.py` | Copernicus DEM terrain at the towers |
| `validate.py` | drivers against measured tower meteorology |
| `load.py` | `baseline(site)`: the ERA5-Land baseline table with PA at the tower elevation |
| `tests/` | coordinate selection, unit conversion, time alignment (`pytest`) |

Environment: `/home/photon/4tb/envs/fluxfill-drivers` (Python 3.11, Open-Meteo extraction) and
`/home/photon/4tb/envs/fluxfill-drivers312` (Python 3.12, required by icechunk 2.x for the dynamical.org stores).

## Citations

- Munoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications.
  Earth System Science Data 13, 4349-4383. doi:10.5194/essd-13-4349-2021
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999-2049. doi:10.1002/qj.3803
- Dowell, D. C. et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating convection-allowing
  forecast model. Part I. Weather and Forecasting 37, 1371-1395. doi:10.1175/WAF-D-21-0151.1
- Schimanke, S. et al. (2021). CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to
  present. C3S Climate Data Store. doi:10.24381/cds.622a565a
- Huffman, G. J. et al. (2023). GPM IMERG Late Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07. GES DISC.
  doi:10.5067/GPM/IMERG/3B-HH-L/07
- Zhang, J. et al. (2016). Multi-Radar Multi-Sensor (MRMS) quantitative precipitation estimation: initial operating
  capabilities. Bull. Amer. Meteor. Soc. 97, 621-638. doi:10.1175/BAMS-D-14-00174.1
- European Space Agency (2021). Copernicus Global Digital Elevation Model GLO-30. doi:10.5270/ESA-c5d3d65
- Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. doi:10.5281/zenodo.7970649
- Alduchov, O. A. and Eskridge, R. E. (1996). Improved Magnus form approximation of saturation vapor pressure.
  J. Appl. Meteor. 35, 601-609.
- Beck, H. E. et al. (2019). MSWEP V2 global 3-hourly 0.1 deg precipitation. Bull. Amer. Meteor. Soc. 100, 473-500.
- Pfeifroth, U. et al. (2024). SARAH-3: satellite-based climate data records of surface solar radiation. Earth
  System Science Data 16, 5243-5265.
- Sengupta, M. et al. (2018). The National Solar Radiation Data Base (NSRDB). Renew. Sustain. Energy Rev. 89, 51-60.
- Vuichard, N. and Papale, D. (2015). Filling the gaps in meteorological continuous data measured at FLUXNET sites
  with ERA-Interim reanalysis. Earth System Science Data 7, 157-171.
- Gelaro, R. et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, version 2
  (MERRA-2). J. Climate 30, 5419-5454.
