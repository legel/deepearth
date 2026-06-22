# flood_hydrology

DeepEarth flood/lake hydrology digital twin for a 2x2km AOI around **Johns Lake**, near
17801 Champagne Dr, Winter Garden, FL (28.5216, -81.6570). Goal: lake water extent / depth /
volume estimation from satellite + DEM data, soil + imagery overlays, and rainfall-driven flood
simulation validated against Sentinel-2 and gauge records.

## Project status

**Paused** mid-investigation to prioritize other work. The core pipeline is complete and
working end-to-end: DEM/Sentinel-2/soil ingestion, 4-method water segmentation, lake mask +
bathymetry + volume, a 2D flood solver with 7 simulation scenarios, a local Three.js viewer, and
a 53-check verification dashboard (all currently passing). See **Future work** below for the
concrete open threads for whoever picks this back up: physics fidelity vs. the HEC-RAS
civil-engineering standard, expanding spatial coverage beyond the current clipped AOI, and two
known simulation gaps (spatially-uniform infiltration, no post-storm recession frames).

Branch: `feature/flood-hydrology-gsdr-monthly`.

---

## Directory map

- `sentinel2/` — Sentinel-2 L2A download, cloud masking, water segmentation
  (MNDWI, WatNet, Prithvi-EO-2.0, OmniWaterMask). Data: `sentinel2/data/`.
- `dem/` — USGS 3DEP DEM (3m, EPSG:5070), hydrology derivatives (HAND, flow
  accumulation, watershed, streams), lake mask consensus, lake bathymetry
  (FDEP/SJRWMD FWC surveys), lake depth/volume calc + visualization.
  Data: `dem/data/`.
- `soil/` — USDA SSURGO soil units + NAIP 1m aerial imagery. Data: `soil/data/`.
- `gsdr/` — Global Sub-Daily Rainfall dataset (US gauges, 1942–2014): extreme
  rainfall intensity lookups and maps. See "GSDR module reference" at the end of this file.
- `precipitation/` — NOAA Atlas 14 IDF curves, GSDR-vs-Atlas14 comparison,
  SCS Type II hyetograph generation (rainfall input to `simulation/`).
- `simulation/` — 2D local-inertia (LISFLOOD-FP style) flood inundation model,
  dx≈2.6m. Runs on Apple GPU via PyTorch MPS. Outputs depth/velocity rasters
  per scenario.
- `WatNet/` — vendored WatNet water-segmentation model.
- `viewer/` — Flask + Three.js local 3D digital-twin viewer (port 5050).
  Start: `python3 viewer/server.py`
- `ground_truth/` — independent validation of the modeled lake against
  external imagery/gauge sources. See `ground_truth/README.md`.
- `frontend_export.py` — converts `simulation/` outputs to GeoJSON for a
  separate `digitaltwin` frontend repo. Not related to the `viewer/` Three.js
  viewer; a dormant/optional export path, not part of the main pipeline.

---

## Data catalogue — sources, coverage, and role

### Raster / vector inputs

| Layer | Source | Spatial | Temporal | File(s) | Role |
|---|---|---|---|---|---|
| **DEM** | USGS 3DEP 3m | EPSG:5070, 3m, 868×868 | Static | `dem/data/winter_garden_dem.tif` | Terrain for flow routing, elevation reference |
| **Lake bathymetry (FWC)** | FDEP / SJRWMD FWC boat survey | EPSG:5070, DEM grid | Static (survey ~2010s) | `dem/data/lake_bed_dem_fwc.tif` | Lake bed elevation; authoritative depth/volume |
| **Lake bathymetry (estimated)** | Shoreline-slope extrapolation | EPSG:5070, DEM grid | — | `dem/data/lake_bed_dem_estimated.tif` | Fallback if FWC not available |
| **Lake mask** | OWM majority-vote (153 scenes) + NHD supplement | EPSG:5070, DEM grid | Consensus 2016–2026 | `dem/data/lake_mask.tif` | Authoritative lake extent (largest component = Johns Lake, 203,740 cells) |
| **Sentinel-2 bands** | ESA / Planetary Computer, L2A | EPSG:32617, 10m, ~222×223 | 205 scenes 2016–2026 | `sentinel2/data/s2_{date}_{BAND}.tif` | Multi-spectral input to water segmentation |
| **Cloud mask** | SCL + s2cloudless ensemble | S2 grid | Per scene | `sentinel2/data/cloud_mask_{date}.tif` | Quality filter |
| **OWM water mask** | OmniWaterMask v0.5.0 (CNN+NDWI) | S2 grid | 153 kept scenes | `sentinel2/data/omniwatermask_mask_{date}.tif` | Best water segmentation (F1=0.882 vs NHD) |
| **WatNet mask** | WatNet (MobileNetV2+ASPP, Metal GPU) | S2 grid | 152 scenes | `sentinel2/data/watnet_mask_{date}.tif` | F1=0.813 |
| **Prithvi mask** | Prithvi-EO-2.0 (ViT-300M, MPS) | S2 grid | 205 scenes | `sentinel2/data/prithvi_mask_{date}.tif` | F1=0.814 |
| **NAIP aerial** | USDA NAIP 1m, Planetary Computer | EPSG:5070, ~0.6m | 2023 (latest quad) | `soil/data/naip_rgb.tif` | Visual context; viewer NAIP layer |
| **SSURGO soils** | USDA SSURGO | EPSG:5070 | Static | `soil/data/ssurgo_*.shp` | Horton infiltration params; viewer soil layer |

F1 scores are against an NHD-derived reference mask (`sentinel2/method_comparison_nhd_official.csv`).
See **References** for the WatNet/Prithvi-EO-2.0/OmniWaterMask model sources.

### Precipitation sources

| Source | Coverage | Use | Status |
|---|---|---|---|
| **NOAA Atlas 14** | Contiguous US, frequency analysis (static) | Design storms: 10-yr / 100-yr return periods, 1-hr / 12-hr durations | 4 scenarios in viewer |
| **GSDR** (US_086638, Clermont-area) | 1942–1985 (station ends 1985, not 2014) | Validation: nearest-station 1-hr max = 143.5 mm > Atlas 14 100-yr 119 mm → Atlas 14 is conservative at this location | Surfaced as viewer attribution |
| **GSDR** (US_086628) | 1974–2011 | Secondary station (~30 km), 1-hr max = 93.7 mm | Checked full record — not the all-time extreme by either criterion; not used as a scenario |
| **NOAA CDO** (Clermont-area station) | 1948–present | Historical 2024-02-12 event gauge data | Works, but the gauge only captured 1.3 mm for that window — a real data-coverage gap, not a bug |
| **GSDR historical scenario (1-hr peak)** | Pre-1985 extreme | `historical_gsdr`: 1960-07-25, 143.5 mm 1-hr peak, 208 mm/24hr | Built via `precipitation/fetch_gsdr_gauge.py` |
| **GSDR historical scenario (24-hr total)** | Pre-1985 extreme | `historical_gsdr_extreme`: 1945-09-16, 245.6 mm/24hr — the all-time max GSDR storm by total depth at this location (likely a tropical system) | Built via `precipitation/fetch_gsdr_extreme.py` |

**Why Atlas 14 instead of raw GSDR for design storms:** Atlas 14 provides statistically fitted
return-period frequencies (GEV/LP3 regression over all stations) — the engineering standard for
design storms. GSDR gives raw observations only, with no return-period assignment, so it serves
as a validation check instead: the 143.5 mm observed 1-hr extreme at US_086638 confirms that
Atlas 14's 100-yr estimate (119 mm) is conservative at this location. GSDR ends in 1985 for the
nearest station, so it can't cover the 2024 target event (that needs NOAA CDO).

### Key derived data files

| File | Shape / Units | Producer | Notes |
|---|---|---|---|
| `dem/data/lake_volume.csv` | 1 row: area_ha, volume_m3, mean_depth_m, max_depth_m, water_surface_m | `dem/lake_volume.py` | **Authoritative WSE = 28.74 m NAVD88**; volume = 2.528 × 10⁶ m³; area = 144.47 ha |
| `dem/data/lake_volume_seasonal.csv` | 11 scenes | `dem/lake_volume.py` | Seasonal V/A/E timeseries; built before the OWM archive grew to 153 scenes — re-run `lake_volume.py` to refresh against the full archive |
| `sentinel2/data/cloud_summary.csv` | 205 rows | `s2_cloud_mask.py` | Per-scene cloud % (SCL+s2cloudless) |
| `sentinel2/data/scene_ranking.csv` | 205 rows | `s2_rank_scenes.py` | Tier 1/2/3 cloud ranking; 152 keep=True |
| `sentinel2/data/water_extent_timeseries.csv` | 152 rows | `s2_omniwatermask.py` | OWM date/area 2016–2026 |
| `simulation/outputs/simulation_summary.csv` | 7 rows | `flood_sim.py` | Peak results per scenario |
| `simulation/outputs/hydrograph_{scenario}.csv` | 13–25 rows | `flood_sim.py` | Per-timestep: rain, Pe, flooded_ha, lake_rise_m |
| `simulation/outputs/depth_frames_{scenario}.npz` | (n_frames, 868, 868) float32 | `flood_sim.py --save-frames` | Raw frames for viewer export |
| `viewer/data/simulation_{scenario}_frames.bin` | SIML binary, 3.3–6.4 MB | `export_simulation.py` | 256×256 downsampled, served to Three.js |
| `viewer/data/simulation_{scenario}_infiltration.bin` | SIML binary, same shape as frames.bin | `export_simulation.py` | Cumulative infiltration [mm], same SIML format |
| `viewer/data/simulation_index.json` | 7 entries | `export_simulation.py` | Scenario metadata for `/api/scenarios` |
| `precipitation/data/hyetograph_historical_gsdr_extreme.csv` | 24 hourly rows | `precipitation/fetch_gsdr_extreme.py` | 1945-09-16 event, US_086638, 245.6 mm/24hr |
| `dem/data/lake_drought.csv` | 83 rows (day 0–82) | `dem/lake_drought.py` | Drought drawdown: time_min, day, lake_level_m, area_ha, volume_m3, evap_mm_day |
| `dem/lake_drought_comparison.png` | 2-panel figure | `dem/lake_drought.py` | Wet-extreme flood extent (`historical_gsdr_extreme`'s GeoJSON) vs. drought-drawdown extent, side by side |
| `ground_truth/data/johns_lake_gauge_wse.csv` | 1,830 rows, 1959–2026 | manual export, see `ground_truth/README.md` | Orange County Water Atlas gauge WSE — used to cross-check the satellite-derived WSE above |

---

## Viewer layers — data flow

| Viewer Layer | Data Source | Computation | Notes |
|---|---|---|---|
| **DEM Wireframe** | USGS 3DEP | `export_dem.py` → `dem.bin` (256×256 Float32) | PlaneGeometry, VERT_EXAG=8 |
| **Lake Voxels** | FWC bathymetry | `export_voxels.py` → `voxels.bin` | InstancedMesh, depth = WSE − bed; largest-component filter applied |
| **Water Surface** | OWM lake mask + lake_volume.csv WSE | `export_dem.py` → `lake_mask.png` | Flat plane at y=(28.74−z_min)×8 |
| **NAIP Aerial** | USDA NAIP 1m mosaic | `export_overlays.py` → `naip_rgb.png` | Full quad coverage |
| **SSURGO Soils** | USDA SSURGO, 6 series | `export_overlays.py` → `ssurgo.png` + `ssurgo_legend.json` | tab10 colors, legend on toggle |
| **Flood Depth** | `flood_sim.py` output | `export_simulation.py` → `simulation_*_frames.bin` | Animated DataTexture, draped on `terrain.geometry`; graduated colormap floor 0.1mm |
| **Infiltration** | `flood_sim.py` `infiltration_frames` | `export_simulation.py` → `simulation_*_infiltration.bin` | Reuses Flood Depth's texture machinery, draped. **Currently spatially uniform** — see Future work |
| **NAIP Aerial (Draped)** / **SSURGO Soils (Draped)** | same PNGs as flat overlays | `overlays.js` `createDrapedOverlay()` | Same texture painted onto terrain's actual displaced surface instead of a flat plane; flat versions kept as a separate toggle |
| **S2 Ground Truth** | OWM mask 2024-02-12 | `export_simulation.py` → `s2_ground_truth_20240212.png` | Red overlay; 68,436 water pixels |

Playback speed is adjustable (0.25–4×, scales frame interval at runtime; underlying 1-second
data cadence is unchanged) and a standalone rain-preview control can trigger rain particles
independent of scenario playback.

**Coordinate system (CRITICAL — do not break):**
All layers share one local frame from `geo_meta.json`:
- X = `col × cell_x_orig − width_m/2` (west→east)
- Y = `(elev − z_min) × VERT_EXAG=8` (up)
- Z = `row × cell_y_orig − height_m/2` (north→south, row 0 = north)

`dem.bin` and `fwc_bed.bin` are row 0 = north (no flipud). PNG overlays use Three.js `flipY=true`.

---

## Simulation — physics, scenarios, and results

### Physics stack

1. **Infiltration**: Horton model — `f(t) = fc + (f0−fc)·exp(−k·t)` [m/s]; Pe = max(P−f, 0)
2. **2D overland flow**: Local inertia SWE (Bates et al. 2010, LISFLOOD-FP), Manning friction
3. **Lake storage**: VAE curve from FWC bathymetry; weir outflow `Q = 1.7·L·head^1.5`
4. **Stability**: CFL adaptive time step, α=0.5
5. **Solver backend**: PyTorch MPS (Apple GPU via `~/miniforge3/envs/prithvi/bin/python`); falls back to NumPy CPU if torch unavailable

**Note on local-inertia limitation**: peak velocities ~3.5 m/s → Fr > 1 in channels → the
convective-acceleration term this scheme drops becomes significant. See **Future work** item 1.

### Soil preset (always use for demos and S2 validation)

`--soil-preset central-fl-antecedent`: f0=76, fc=25 mm/hr, k=2.0 hr⁻¹
(wet-season antecedent moisture, ~70% initial saturation, Florida HSG-A sandy soils).
Raw SSURGO f0 values (e.g., 1656 mm/hr for Florahome sand) absorb all design storms — Pe=0, no surface flooding — physically correct for dry conditions but wrong for wet-season demos.

### Scenario results

| Scenario | Rain | Peak flood | Max depth | Max vel | Lake rise | Wall time |
|---|---|---|---|---|---|---|
| Flash 1hr 100yr | 119 mm | 199.8 ha | 3.43 m | 3.46 m/s | +0.076 m | 62 s |
| Flash 1hr 10yr | 75 mm | 102.6 ha | 2.56 m | 3.00 m/s | +0.036 m | 56 s |
| Sustained 12hr 100yr | 240 mm | 118.1 ha | 2.82 m | 2.01 m/s | +0.048 m | 606 s |
| Sustained 12hr 10yr | 151 mm | 13.0 ha | 2.20 m | 1.46 m/s | +0.011 m | 452 s |
| Historical GSDR 1960-07-25 | 208 mm / 24 hr | 194.8 ha | 6.55 m | 2.93 m/s | +0.153 m | 1540 s |
| Historical GSDR extreme 1945-09-16 | 245.6 mm / 24 hr | 108.8 ha | 2.55 m | 2.27 m/s | +0.041 m | 214 s |

All 7 scenarios (including `historical_20240212`) are exported to the viewer
(`simulation_index.json`). `historical_20240212` runs but the NOAA gauge only captured 1.3 mm
total for that window, so it produces no flooding (0.0 ha) — a real data-coverage limitation,
not a bug.

**Physics insight:** despite having *more* total rainfall than the 1960 event (245.6 mm vs.
208 mm), the 1945 storm floods *less* (108.8 ha vs. 194.8 ha) — it spreads the rain over 24 hr
at a lower peak intensity (65 mm/hr vs. 143.5 mm/hr), so Horton infiltration absorbs more of it
before the flood threshold is hit. **Peak rainfall intensity, not storm total, drives
flash-flood severity at this AOI.**

Start server: `python3 viewer/server.py` → http://localhost:5050/

### Run commands

```bash
# GPU (recommended — ~20× faster than CPU):
~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \
    --scenario all --save-frames --soil-preset central-fl-antecedent

# After sim finishes, export to viewer:
~/miniforge3/envs/prithvi/bin/python viewer/preprocess/export_simulation.py

# Start / restart server:
python3 viewer/server.py
```

**Gotcha:** running with a single `--scenario <name>` overwrites
`simulation/outputs/simulation_summary.csv` with *only* that scenario's row (it doesn't
merge/append), and overwrites that scenario's `.npz` regardless of `--save-frames` on others.
Follow any single-scenario test run with `--scenario all --save-frames` to restore the full
summary before anything downstream (e.g. the viewer export) relies on it.

---

## Consistency check — lake mask / WSE / volume

**Authoritative values (all modules must agree):**

| Quantity | Authoritative value | Source | flood_sim.py |
|---|---|---|---|
| Lake extent | 203,740 cells | largest-component of `lake_mask.tif` | matches |
| Water surface elevation | 28.74 m NAVD88 | `lake_volume.csv` (FWC + OWM) | matches |
| Lake area | 144.47 ha | `lake_volume.csv` | viewer matches; sim approx (140.9 ha from DEM grid) |
| Lake volume | 2.528 × 10⁶ m³ | `lake_volume.csv` | sim VAE curve is lower (1.82 × 10⁶) — see FWC volume gap below |
| Max depth | 3.97 m | `lake_volume.csv` / FWC bed | matches |

**FWC volume gap (known, not a new issue):** `lake_bed_dem_fwc.tif` has NaN cells where the boat
survey didn't reach. These are filled with the DEM surface elevation (depth→0), underestimating
volume by ~27%. Fix requires interpolating the FWC survey to fill gaps.

**144.47 ha vs. the per-scene OWM average (~122 ha) — not a disagreement:** `lake_volume.csv`'s
144.47 ha comes from the **majority-vote consensus mask** (`lake_mask.tif`, pixels wet in ≥50%
of 153 OWM scenes) + largest-component filter — a different quantity than the **per-scene OWM
polygon area** in `sentinel2/data/water_extent_timeseries.csv` (mean 122.14 ha across all 153
scenes; wet-season 120.71 ha, dry-season 122.39 ha — seasonality itself is not a meaningful
bias, −1.4%). A majority-vote mask includes marginal/fringe pixels wet in most-but-not-all
scenes, pulling consensus area above the typical single-scene area. Two different
methodologies, not two conflicting measurements — see `ground_truth/README.md` for the full
check.

**AOI does not cover the full lake:** the 2x2km AOI (`lake_mask.tif`, 868×868 cells, 521.87 ha
total grid area) clips Johns Lake on 3 of 4 edges (north 502 wet cells, west 450, east 127,
south only 5) — confirmed by reading the mask directly. Johns Lake's documented full surface
area is ~2,580 acres ≈ 1,044 ha (Orange County Water Atlas, waterbody ID 7935), more than
double the current AOI. The 144.47 ha figure is one basin/portion of the lake near the target
address, not the whole lake. See **Future work** item 2.

---

## Known issues (resolved)

| Issue | Fix applied |
|---|---|
| `cloud_summary.csv` counted SCL class 0 (no-data) as cloud | `CLOUD_CLASSES = {1,3,8,9,10}`; divide by valid pixels only |
| Voxels included irrigation ponds/canals | `_largest_component()` in `export_voxels.py` |
| Lake depth hover broken over voxel faces | `instanceDepths` array + `getDepthAtInstance()` in `voxelLayer.js` |
| SSURGO only 2 colors | Canonical series grouping + tab10 colormap in `export_overlays.py` |
| No SSURGO legend | `layerControls.js` fetches `ssurgo_legend.json` on toggle |
| NAIP black edges | `RADIUS_KM 1.1→1.8`, mosaic from full quad tile |
| SSURGO nodata gap (black patch) | `scipy.ndimage.distance_transform_edt` fill in `export_overlays.py` |
| Flood simulation on CPU (NumPy) — too slow for 12hr scenarios | Ported `local_inertia_step` to PyTorch MPS (`_local_inertia_step_torch`); ~20× speedup |
| `floodLayer.js` `setFrame()` forced `mesh.visible=hasWater` — hid layer at frame 0 | Removed visibility toggle from `setFrame()`; `layerControls.js` now sole owner of visibility |
| Bilinear downsampling spread depth noise to all pixels | Graduated colormap floor (0.1mm, was 1cm) in `floodLayer.js` |
| Lake mask not filtered in `flood_sim.py` — used all 5 components (216k cells) | `_largest_component()` now applied in `flood_sim.py main()` |
| WSE₀ in simulation used raw mask DEM mean (29.23 m) | Now reads from `lake_volume.csv` (28.74 m); `_load_wse_from_lake_volume()` helper |
| NAIP/SSURGO/Flood Depth/Infiltration rendered as flat floating planes, not draped on terrain | `terrain.js` exposes the displaced `geometry`; `overlays.js` `createDrapedOverlay()` and `floodLayer.js`'s `terrainGeometry` option reuse it |
| Flooded depressions appeared to "pop into existence" | Visibility floor lowered to 0.1mm with two new low-alpha colormap buckets so depth fades in gradually |
| Rain particles imperceptible at scene scale | `PARTICLE_SIZE` bumped 1.4 → 6.0 |
| Layer list grouped flat overlays separately from draped variants | Reordered so each layer sits next to its draped counterpart |
| Controls + simulation panels ate ~470px of viewport side by side | Wrapped both in `#left-stack` (flex column, shared width, internal scroll) |

---

## Ground-truth validation

Independent validation of this repo's satellite-derived lake metrics against external imagery
and gauge sources — full methodology and numbers in `ground_truth/README.md`. Summary:

- **PlanetScope** (3m commercial imagery) was reviewed as a finer-resolution alternative to
  Sentinel-2's 10m: realistic shoreline-position precision ~2–5m for this lake's
  turbid/vegetated-fringe water, vs. Sentinel-2's 10m pixel — relevant if foot-scale shoreline
  tracking is ever required.
- **Drone RTK imagery** is available (survey-grade GNSS-tagged capture) but its footprint is a
  single ~100–150m property, not lake-wide — useful for cm-precision terrain/DEM validation at
  one site, not a substitute for lake-extent ground truth.
- **Gauge cross-check passes**: the Orange County Water Atlas "JOHNS" station (waterbody 7935)
  has 1,830 water-surface-elevation readings spanning 1959–2026. Restricted to 2016+ (this
  repo's Sentinel-2 archive window): gauge mean 29.07 m, stdev 0.35 m, n=94 — this repo's
  satellite-derived WSE (28.74 m) sits 0.33 m below the gauge mean, within 1 stdev of normal
  lake-level variation.
- **AOI under-coverage** (see Consistency check above) was found and confirmed this way: real
  lake area ~1,044 ha vs. 144.47 ha currently modeled.
- **Clay Foundation Model** was scoped as a 5th segmentation method and deprioritized — it
  ships embeddings only (no water-segmentation head), so adding it would mean training a
  decoder from scratch, and it would still run on the same 10m Sentinel-2 input, so it can't
  address the resolution question above.

---

## Future work

1. **Adopt full Saint-Venant convective-acceleration physics.** The current solver
   (local-inertia, Bates et al. 2010 / LISFLOOD-FP) keeps the local-acceleration term but drops
   convective acceleration (`v·∂v/∂x` — how flow velocity changes as water moves through
   space). HEC-RAS, the USACE civil-engineering standard, explicitly names flash floods as one
   of 8 cases where full Saint-Venant (Shallow Water) should be used instead of the simpler
   Diffusion Wave approximation — and this repo's two flash scenarios are exactly where Fr > 1
   is reached, i.e. where the dropped term is most likely to matter numerically.
   - Concrete next step: a HEC-RAS 7.0.1 reference run on the flash scenarios at this AOI, for
     a real fidelity-gap number against `flood_sim.py`.
   - Longer-term idea: NVIDIA's HydroGraphNet (a graph-neural-network flood surrogate) uses a
     physics-informed mass-conservation loss, trained against HEC-RAS output — a reusable loss
     pattern if a learned surrogate for `_local_inertia_step_torch` is ever built. Not a
     drop-in module; it needs HEC-RAS-quality ground truth at this AOI to train.
   - Links: see References.

2. **Expand spatial coverage beyond the current clipped AOI.** Johns Lake's true area
   (~1,044 ha) is more than double what's currently modeled (144.47 ha, clipped on 3 of 4
   edges). A bounding box of 7.37×3.87 km (buffered around the real NHD shoreline polygon) has
   already been validated as a test-only DEM pull with **zero edge-clipping**, so the box itself
   is ready to use. Open question before re-running the full pipeline at that extent: continue
   with Sentinel-2 (10m) or move to a finer-resolution source (PlanetScope, ~3m, ~2–5m
   realistic shoreline precision per the Ground-truth validation findings above) — resolution
   matters more at full-lake scale than it did for the current basin-scale AOI. Stretch goal:
   generalize the pipeline beyond one named lake to other Florida lakes/watersheds.

3. **Spatially-varying infiltration + post-storm recession in the viewer.** Two known solver
   gaps, not viewer bugs:
   - The Infiltration layer is currently a single scalar Pe broadcast across the whole grid
     (no per-cell SSURGO Horton parameters wired in), so the rendered map has zero spatial
     structure under the default full-AOI rain mask. Fix: wire per-mukey SSURGO Horton `f0`/
     `fc`/`k` (already computed in `soil/data/soil_parameters.json`) into `flood_sim.py`'s
     per-cell infiltration calculation.
   - `flood_sim.py`'s post-storm drainage loop only updates the static peak-depth raster — it
     never appends a frame, so saved animations stop at the last rain timestep instead of
     showing recession. Fix: snapshot frames during the drainage loop too, then re-run
     `--scenario all --save-frames` and re-export.

---

## References

**Data sources**
- Sentinel-2 L2A via Microsoft Planetary Computer: `https://planetarycomputer.microsoft.com/api/stac/v1`
- NOAA Atlas 14 (HDSC): `https://hdsc.nws.noaa.gov`
- NOAA Climate Data Online: `https://www.ncdc.noaa.gov/cdo-web`
- Orange County Water Atlas, Johns Lake (waterbody 7935): `https://orange.wateratlas.usf.edu/waterbodies/lakes/7935/johns-lake`
- USGS 3DEP, USDA SSURGO/NAIP — standard USDA/USGS public data portals
- FDEP / SJRWMD FWC lake bathymetry surveys
- GSDR (Global Sub-Daily Rainfall): Zenodo record [8369987](https://zenodo.org/records/8369987) — see "GSDR module reference" below

**Segmentation models**
- WatNet: `https://github.com/xinluo2018/WatNet`
- Prithvi-EO-2.0: `https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` (via `terratorch`)
- OmniWaterMask: PyPI package `omniwatermask`

**Simulation & benchmarking**
- Bates, P. D., Horritt, M. S., & Fewtrell, T. J. (2010). A simple inertial formulation of the
  shallow water equations for efficient two-dimensional flood inundation modelling. *Journal of
  Hydrology*, 387(1–2), 33–45. (LISFLOOD-FP local-inertia scheme used by `flood_sim.py`)
- HEC-RAS (USACE): `https://www.hec.usace.army.mil/software/hec-ras/download.aspx`; 2D governing
  equations: `https://www.hec.usace.army.mil/confluence/rasdocs/r2dum`
- NVIDIA HydroGraphNet: `https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/flood_modeling/hydrographnet`

---

## Pipeline / data flow

1. **Download**: `dem/dem_download.py` → DEM; `sentinel2/s2_download.py` → S2 bands;
   `soil/fetch_naip.py` + `soil/ssurgo_download.py` → soil/NAIP;
   `dem/fetch_*bathymetry.py` → FWC lake-bed surveys.
2. **Cloud masking** (`sentinel2/s2_cloud_mask.py`): SCL + s2cloudless → `cloud_mask_{date}.tif`.
3. **Water segmentation**: MNDWI / WatNet / Prithvi-EO-2.0 / OmniWaterMask → binary masks.
4. **Lake mask consensus** (`dem/lake_utils.get_lake_mask_and_fwc`): OWM majority-vote (153 scenes) + NHD → `lake_mask.tif` → largest-component filter → Johns Lake only.
5. **Lake depth/volume** (`dem/lake_volume.py`): FWC bathymetry + lake mask → `lake_volume.csv` (WSE=28.74 m, V=2.528×10⁶ m³).
6. **Simulation** (`simulation/flood_sim.py`): Atlas 14 / GSDR hyetograph → Horton infiltration → 2D SWE (PyTorch MPS) → depth/velocity rasters per scenario.
7. **Viewer export**: `viewer/preprocess/export_*.py` → binary data files in `viewer/data/`.
8. **Verify** (`verify_all.py`): 53-check dashboard.

---

## DL model environments

| Model | Python | Key packages | Run via |
|---|---|---|---|
| **MNDWI** | system python3 (3.9) | rasterio, numpy | `python3 sentinel2/s2_water_segment.py` |
| **WatNet** | system python3 | TF 2.16.2 + tensorflow-metal | `python3 sentinel2/s2_water_segment.py` |
| **Prithvi-EO-2.0** | `~/miniforge3/envs/prithvi/bin/python` | PyTorch + terratorch + MPS | `~/miniforge3/envs/prithvi/bin/python sentinel2/s2_water_segment_v2.py` |
| **OmniWaterMask** | `~/miniforge3/envs/prithvi/bin/python` | omnicloudmask, lightgbm | `~/miniforge3/envs/prithvi/bin/python sentinel2/s2_omniwatermask.py --from-ranking` |
| **Flood solver (GPU)** | `~/miniforge3/envs/prithvi/bin/python` | PyTorch 2.12 + MPS | `~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py` |

**Critical NumPy constraint (WatNet/TF 2.16.2):** pin `numpy<2.0.0` and `ml_dtypes~=0.3.1` in system env.

**S2 bands downloaded:** `B01, B02, B03, B04, B05, B08, B8A, B09, B11, B12, SCL`

---

## Conventions

- Dates as `YYYYMMDD` strings everywhere.
- **Two CRSs**: S2 products EPSG:32617 (UTM 17N, ~10m); DEM/lake EPSG:5070 (Albers, 3m).
  Reproject with `rasterio.warp.reproject(..., resampling=Resampling.nearest)`.
- Every script: `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`, can run from any cwd.
- `matplotlib.use("Agg")` for headless plotting.
- s2cloudless and OWM run via subprocess calling `~/miniforge3/envs/prithvi/bin/python`.

---

## Simulation binary format (SIML) — `viewer/data/simulation_*_frames.bin` / `*_infiltration.bin`

```
[0:4]        b'SIML'           magic
[4:8]        uint32            n_frames
[8:12]       uint32            rows (256)
[12:16]      uint32            cols (256)
[16:16+n×4]  float32[n]        times_min
[16+n×4:]    float32[n×256×256] frames.bin: depth [m]; infiltration.bin: cumulative infiltration [mm]
```

Same header for both files (`viewer/static/js/floodLayer.js`'s `urlSuffix` option picks which
one to load). Render-side colormap floor is 0.1mm (not 1cm) — graduated low-end buckets fade
thin film in gradually instead of popping at a hard threshold.

---

## Verification

- `python3 verify_all.py` — 53 checks across DEM/Sentinel-2/soil/simulation.
  Run after any change to `dem/data/` or `sentinel2/data/`.
- New scripts: print pass/fail checks against known reference values.
- `ground_truth/README.md` — independent cross-checks against external imagery/gauge sources
  (separate from `verify_all.py`'s internal consistency checks).

---

## GSDR module reference

Detailed reference for the `gsdr/` directory specifically — scripts, data schema, and setup for
the Global Sub-Daily Rainfall dataset. The rest of this README covers the full project; this
section is the GSDR module's own usage guide.

### Contents

```
models/flood_hydrology/
├── gsdr/
│   ├── gsdr_build_index.py               One-time station index builder
│   ├── gsdr_intensity_matrix.py          Query: sub-daily max precip by radius/duration
│   ├── gsdr_visualizations.py            Dataset exploration figures
│   ├── gsdr_extreme_maps.py              US extreme rainfall maps (5 durations, all-time)
│   ├── gsdr_monthly_maps.py              US monthly maximum rainfall maps (24 maps)
│   ├── gsdr_us_index.csv                 Pre-built station metadata index
│   ├── gsdr_station_maxima.csv           Pre-computed per-station all-time maxima cache
│   ├── gsdr_station_monthly_maxima.csv   Pre-computed per-station monthly maxima cache
│   └── outputs/
│       ├── fig01_us_stations.html        Interactive US station map
│       ├── fig02_record_length.png       Record length distribution
│       ├── fig03_temporal_coverage.png   Active stations per year
│       ├── fig04_missing_data.png        Missing data rate distribution
│       ├── fig05_intensity_scatter.png   1-hr vs 24-hr max scatter
│       ├── fig06_peak_event_year.png     Year of all-time 1-hr max per station
│       ├── extreme_grid_1hr.html/.png    Extreme rainfall map — 1-hour duration
│       ├── extreme_grid_3hr.html/.png    Extreme rainfall map — 3-hour duration
│       ├── extreme_grid_6hr.html/.png    Extreme rainfall map — 6-hour duration
│       ├── extreme_grid_12hr.html/.png   Extreme rainfall map — 12-hour duration
│       ├── extreme_grid_24hr.html/.png   Extreme rainfall map — 24-hour duration
│       ├── max_1_hour_january_rainfall_us.png    Monthly map — 1-hr, January
│       ├── max_1_hour_february_rainfall_us.png   Monthly map — 1-hr, February
│       ├── ...                                   (10 more 1-hr monthly maps)
│       ├── max_12_hour_january_rainfall_us.png   Monthly map — 12-hr, January
│       ├── max_12_hour_february_rainfall_us.png  Monthly map — 12-hr, February
│       └── ...                                   (10 more 12-hr monthly maps)
└── README.md
```

### Dataset — GSDR (Global Sub-Daily Rainfall, US open-access subset)

| Item | Detail |
|---|---|
| Full name | Global Sub-Daily Rainfall dataset (INTENSE project) |
| Source | NOAA + partner agencies; Lewis et al., *Journal of Climate*, 2019 |
| US gauges | 6,605 quality-controlled hourly stations |
| Time span | ~1948–2014 |
| Unit | **mm** — continuous hourly precipitation accumulation |
| Format | INTENSE format: 21-line header + one mm value per line; missing = −999 |
| Download | [Zenodo record 8369987](https://zenodo.org/records/8369987) — download `QC_d data - US.zip` |

#### Column structure of `gsdr_us_index.csv`

| Column | Description |
|---|---|
| `ID` | Station identifier (e.g. `US_086638`) |
| `LAT` | Latitude (decimal degrees, WGS-84) |
| `LON` | Longitude (decimal degrees, WGS-84) |
| `START` | Record start (YYYYMMDDHH) |
| `END` | Record end (YYYYMMDDHH) |
| `N_RECORDS` | Total number of hourly records |
| `PCT_MISSING` | Percentage of missing values |

#### Column structure of `gsdr_station_maxima.csv`

Pre-computed all-time maximum rolling accumulation per station for each duration. Used by `gsdr_extreme_maps.py` to avoid reprocessing all gauge files.

| Column | Description |
|---|---|
| `ID` | Station identifier |
| `max_1hr_mm` | All-time max 1-hour accumulation (mm) |
| `max_3hr_mm` | All-time max 3-hour accumulation (mm) |
| `max_6hr_mm` | All-time max 6-hour accumulation (mm) |
| `max_12hr_mm` | All-time max 12-hour accumulation (mm) |
| `max_24hr_mm` | All-time max 24-hour accumulation (mm) |

#### Column structure of `gsdr_station_monthly_maxima.csv`

Pre-computed per-month all-time maximum rolling accumulation per station for 1-hr and 12-hr windows. Used by `gsdr_monthly_maps.py`.

| Column | Description |
|---|---|
| `ID` | Station identifier |
| `max_1hr_jan_mm` | All-time max 1-hour accumulation in any January (mm) |
| `max_1hr_feb_mm` | All-time max 1-hour accumulation in any February (mm) |
| `...` | Columns follow the same pattern through December |
| `max_12hr_jan_mm` | All-time max 12-hour accumulation in any January (mm) |
| `...` | Columns follow the same pattern through December |

### Setup

#### Python dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly kaleido
```

#### GSDR raw data (required for `gsdr_intensity_matrix.py` and `gsdr_visualizations.py`)

1. Download `QC_d data - US.zip` from [Zenodo record 8369987](https://zenodo.org/records/8369987)
2. Extract to a local folder (≈ 4.4 GB uncompressed)
3. Set the environment variable:

```bash
export GSDR_QC_DIR="/path/to/QC_d data - US"
```

If `GSDR_QC_DIR` is not set, scripts default to `~/Desktop/GSDR/QC_d data - US`.

> `gsdr_extreme_maps.py` uses the pre-built cache files included in this package and does **not** require raw gauge files.
>
> `gsdr_monthly_maps.py` uses `gsdr_station_monthly_maxima.csv` if present and does **not** require raw gauge files. If the cache is absent it will read all gauge files (~15 min).

### Scripts

All scripts use `__file__`-based absolute paths and can be run from any directory.

#### `gsdr_build_index.py`

Scans the 21-line header of every gauge file and builds `gsdr_us_index.csv`. A pre-built index is included — only rerun if you have a different GSDR download.

```bash
python3 gsdr/gsdr_build_index.py
```

Runtime: ~45 seconds. Requires raw GSDR data.

#### `gsdr_intensity_matrix.py`

Given a geospatial coordinate, finds all GSDR stations within each search radius and returns the **all-time observed maximum precipitation accumulation** for each duration window. Values are raw historical observations — no statistical distribution fitting.

```bash
python3 gsdr/gsdr_intensity_matrix.py --lat 28.5652 --lon -81.5868
python3 gsdr/gsdr_intensity_matrix.py --lat 37.38575 --lon -122.00037
```

| Argument | Default | Description |
|---|---|---|
| `--lat` | required | Latitude (decimal degrees, WGS-84) |
| `--lon` | required | Longitude (decimal degrees, WGS-84) |
| `--radii` | `10 50 100` | Search radii in km |
| `--durations` | `1 3 6 12 24` | Accumulation windows in hours |
| `--no-csv` | — | Print results only, skip saving CSV |

**Output:** `gsdr/outputs/gsdr_intensity_{lat}_{lon}.csv`

Example output (Winter Garden, FL):

| Duration | ≤10 km | ≤50 km | ≤100 km |
|---|---|---|---|
| 1-hr | N/A | 143.5 mm (5.65 in) | 143.5 mm (5.65 in) |
| 6-hr | N/A | 232.4 mm (9.15 in) | 248.4 mm (9.78 in) |
| 24-hr | N/A | 363.0 mm (14.29 in) | 363.0 mm (14.29 in) |

N/A indicates no station within that radius. Values increase monotonically across radii since each larger radius is a superset of the smaller.

Requires raw GSDR data.

#### `gsdr_visualizations.py`

Generates 6 dataset exploration figures saved to `gsdr/outputs/`. Pre-generated outputs are included. Figs 1–4 run in seconds; Figs 5–6 load all gauge files and take 10–20 minutes.

```bash
python3 gsdr/gsdr_visualizations.py
```

| Figure | Description |
|---|---|
| `fig01_us_stations.html` | Interactive US map — 6,605 hourly gauges coloured by record length |
| `fig02_record_length.png` | Distribution of years of data per station |
| `fig03_temporal_coverage.png` | Active stations per year across full record |
| `fig04_missing_data.png` | Missing data rate per station; shows the 50% filter threshold |
| `fig05_intensity_scatter.png` | 1-hr vs 24-hr all-time max — distinguishes flash-flood vs slow-rain regimes |
| `fig06_peak_event_year.png` | Year each station recorded its all-time 1-hr maximum |

Figs 5–6 require raw GSDR data.

#### `gsdr_extreme_maps.py`

Generates 5 interactive HTML maps and 5 static PNG exports showing the spatial distribution of extreme sub-daily rainfall across the US, one per duration. Uses the pre-built cache — does **not** require raw GSDR data.

```bash
python3 gsdr/gsdr_extreme_maps.py
```

| Design element | Detail |
|---|---|
| Circle position | Centre of 1°×1° grid cell |
| Circle size | Number of GSDR stations in that cell; size legend inset on each map |
| Circle colour | All-time maximum accumulation in inches |
| Colour palette | ColorBrewer YlGnBu 9-class (colorblind-verified, journal standard) |
| Colour scale | Shared log₁ₚ normalisation across all 5 maps (0–15 in) |
| Hover tooltip | Actual inches and mm, station count, best station ID |
| PNG resolution | 1400×800 px, 2× scale |

#### `gsdr_monthly_maps.py`

Generates 24 static PNG maps showing the spatial distribution of monthly-maximum sub-daily rainfall across the US. Two durations (1-hr, 12-hr) × 12 months = 24 maps. Uses the pre-built monthly cache — does **not** require raw GSDR data once the cache exists.

```bash
python3 gsdr/gsdr_monthly_maps.py
```

| Design element | Detail |
|---|---|
| Circle position | Centre of 1°×1° grid cell |
| Circle size | Number of GSDR stations in that cell; size legend inset on each map |
| Circle colour | All-time maximum accumulation for that month in inches |
| Colour palette | ColorBrewer YlGnBu 9-class (colorblind-verified, journal standard) |
| Colour scale | Shared log₁ₚ normalisation across the 12 monthly maps for each duration |
| PNG resolution | 1400×800 px, 2× scale |

Output filenames follow the pattern `max_{duration}_hour_{month}_rainfall_us.png`, e.g.:

```
outputs/max_1_hour_january_rainfall_us.png
outputs/max_1_hour_july_rainfall_us.png
outputs/max_12_hour_january_rainfall_us.png
outputs/max_12_hour_july_rainfall_us.png
```
