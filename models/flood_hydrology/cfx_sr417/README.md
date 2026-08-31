# CFX SR417 Corridor — flood digital twin

> **2026-08-29 — magnitude numbers in this document are superseded.**
> Four defects were found in the solver's integration loop (broken clock delivering 7-11 % of
> the storm, phantom infiltration, an unstable CFL constant, a broken final-frame guard). All
> are fixed and the solver now conserves mass to -0.001 %, but every magnitude figure below —
> runoff coefficients, peak discharges, depths, flooded areas, probability surfaces — predates
> the fixes. Timing results are robust and largely stand. See [`../NEXT_STEPS.md`](../NEXT_STEPS.md).

A reproducible pipeline that turns **a coordinate** into a working flood digital twin: fetch
every public dataset for that location, condition the terrain, run a calibrated shallow-water
solver over a standard storm ensemble, and serve the result as an interactive 3D viewer.

Built for the Central Florida Expressway Authority's SR417 corridor, and validated at a second,
gauge-matched site where simulated discharge can be checked against a real USGS record.

One of two sites under [`../`](../README.md) (Johns Lake is the other,
[`../johns_lake/`](../johns_lake/README.md)). Shared code lives in [`../floodtwin`](../floodtwin).

---

## Sites

Every site is a registry entry, not a hardcoded constant. `site_registry.py` resolves a
`--site` flag to coordinates **and** an output directory, and refuses a `--site` combined with a
contradicting `--lat/--lon`.

| Site | Centre | Box | Purpose |
|---|---|---|---|
| `main_aoi` | 28.36687, −81.43299 | 2 × 2 km | The CFX test-landscape AOI near Lake Nona, south Orlando |
| `site3` | 28.690514, −81.287539 | 6 × 6 km | **Gauge-matched validation** — Gee Creek nr Longwood, USGS 02234400 |
| `site1` | 28.363317, −81.431574 | ~160 m | 5-house cluster: does water shed off roofs correctly? |
| `site2` | 28.366330, −81.434606 | ~390 m | Retention pond + 3 houses: does runoff *accumulate* correctly? |
| `site3_1house` | 28.701821, −81.261779 | ~120 m | Single isolated house, fine-scale demo |
| `site3_crop{,_coarse}` | 28.703899, −81.290643 | ~500 m | Surrogate-training crops at site3's pour point |

`site1`/`site2` sit inside `main_aoi` and reuse its data; the `site3_*` crops sit inside `site3`.
Only `main_aoi` and `site3` own a fetched data tree.

## Pipeline

```
coordinate
   │
   ├── fetch/      DEM (3DEP) · soils (SSURGO) · impervious (NLCD) · imagery (NAIP)
   │               hydrography (3DHP) · flood zones (FEMA NFHL) · roads+buildings (OSM)
   │               precipitation (ASOS/GHCND) · design storms (NOAA Atlas 14) · LiDAR (LAZ)
   │
   ├── condition/  stream burn → depression breach → D8 flow → accumulation → HAND → streams
   │               + terrain derivatives (slope, aspect, hillshade, curvature, TPI, TRI)
   │
   ├── simulate/   raster local-inertial solver (Bates et al. 2010 / LISFLOOD-FP)
   │               mesh shallow-water solver on a fused ground + LiDAR-roof triangulation
   │
   ├── analyse/    design-storm ensemble → per-cell annual exceedance probability
   │               FEMA × HAND risk classes · simulated-vs-mapped flood extent
   │
   └── viewer/     Flask + Three.js, http://localhost:5051
```

### Quick start

> **Use Python 3.9 for the pipeline — and `python3` already *is* 3.9.6 here.** There is no
> `python3.9` binary on PATH; checking for one and concluding 3.9 is unavailable is a mistake
> worth not repeating. `richdem`, which performs the depression breaching the solver depends on,
> does not build on 3.11 — its vendored pybind11 predates the Python 3.11 change that made
> `PyFrameObject` opaque. A separate 3.11 venv at `.venv` serves `research/` only, which needs
> torch. `requirements.txt` documents both constraints.

```bash
python3 -m pip install --user -r requirements.txt   # python3 IS 3.9 here

# 1. Fetch everything for a site
python3 dem/dem_download.py            --site main_aoi
python3 soil/ssurgo_download.py        --site main_aoi
python3 soil/fetch_nlcd.py             --site main_aoi
python3 imagery/fetch_naip.py          --site main_aoi
python3 hydrography/fetch_3dhp.py      --site main_aoi
python3 floodplain/fetch_fema_nfhl.py  --site main_aoi
python3 infrastructure/fetch_roads_buildings.py --site main_aoi

# 2. Condition the terrain
python3 dem/dem_hydro.py
python3 dem/dem_terrain.py

# 3. Simulate
python3 simulation/flood_sim_ian.py --cell-size 5 --dt 20 --save-frames   # historical event
python3 analysis/flood_probability.py --site main_aoi                     # storm ensemble

# 4. View
python3 viewer/server.py     # → http://localhost:5051
```

Omitting `--site` keeps every script's original single-AOI behaviour, so no existing invocation
changes meaning.

## Flood scenarios

The standard, cross-site scenario set is the **NOAA Atlas 14 design-storm ensemble**:
return periods **T ∈ {1, 2, 5, 10, 25, 50, 100, 200, 500} yr** at a 24-hour duration, converted
to SCS Type II hyetographs. Because Atlas 14 is queried per coordinate, this ensemble runs
unchanged at any new site — which is what makes cross-site comparison meaningful.

`analysis/flood_probability.py` runs the ensemble and inverts the per-cell peak-depth curve to
an **annual exceedance probability**, then to any horizon via `P(≥1 in N yr) = 1 − (1 − AEP)^N`.

| | `main_aoi` | `site3` |
|---|---|---|
| Grid | 208,390 cells @ 5 m | 1,870,036 cells @ 5 m |
| Peak depth, 1 yr → 500 yr | 0.373 → 0.617 m | 0.533 → 1.162 m |
| Peak flooded, 1 yr → 500 yr | 6.7 → 26.3 ha | 50.1 → 199.9 ha |
| Any resolvable pluvial risk | **11.03 ha** | **56.33 ha** |
| ≥1 %/yr | **5.97 ha** | **33.54 ha** |
| ≥10 %/yr | **1.73 ha** | **13.68 ha** |
| Depth vs log T, R² | 0.9715 | 0.9703 |

**Both surfaces were rebuilt on 2026-08-27** after three defects that suppressed runoff were
fixed (see `../NEXT_STEPS.md`). The previous figures were 6.85 / 3.98 / 1.30 (main_aoi) and
30.08 / 18.53 / 8.47 (site3) — the correction roughly doubles mapped risk area at both sites.
site3's peak depths *fell* while its areas *rose* (100 yr: 1.249 m / 84.7 ha → 0.967 m /
143.4 ha), which is the expected signature of removing depression storage the DEM downsampling
had invented: water spreads and moves instead of pooling in pits that were never there.

Both surfaces are monotone in T, as required. A 1 %-AEP cell returns P = 0.2606 over 30 years,
matching FEMA's published "26 % chance over a 30-year mortgage" figure.

**Historical events** are an optional per-site validation layer, run only where ground truth
exists — currently Hurricane Ian (2022-09-28/30) at both sites.

## Validation

**Gee Creek (`site3`, USGS 02234400)** is the only place in this project where simulated
discharge can be compared against a real gauge without a disqualifying scale mismatch.

| | Simulated | Observed |
|---|---|---|
| Peak outflow, domain boundary | 411.6 cfs | — |
| Peak discharge, at the gauge cell | 101.6 cfs (still rising at t = 72 h) | 1,190 cfs at t = 37.52 h |
| Runoff coefficient | 8.90 % | **28.9 – 31.4 %** |
| Rising limb (50 % of peak) | 0.72 h difference | gauge samples at 0.25 h |

Run `analysis/validate_gauge_site3.py` rather than quoting these — it recomputes them from the
raw NWIS record and reports the baseflow/window sensitivity. **A 19.6 % observed runoff
coefficient appears throughout the older text below and cannot be reproduced from that record
under any standard choice**; like-for-like over the simulated window it is 28.9–31.4 %, so the
model was always further from truth than recorded. Treat every figure in the "Known limitations"
section that predates 2026-08-27 as superseded.

**What is validated is the rising limb, not the peak.** Quoting a "1.24 h peak-timing error"
overstates the result: both hydrographs have broad, flat tops, and that difference is *smaller
than the width of either peak's own plateau*, so the argmax is not resolved well enough to
support it.

| | simulated | observed |
|---|---|---|
| plateau within 99% of peak | 1.61 h wide | 3.00 h wide |
| plateau within 95% of peak | 4.06 h wide | 7.00 h wide |

Three timing metrics on the same pair of hydrographs give three different answers, and the
spread is the finding:

| metric | difference | reading |
|---|---|---|
| **rising limb** (50% of peak) | **0.09 h** | **real agreement — the model gets flood onset right, to ~5 minutes** |
| peak argmax | 1.72 h | inside the resolution limit above; consistent with agreement, not evidence of it |
| centroid (centre of mass) | **7.44 h** | the model's response is centred 7 h early — it recedes far too fast |

These are measured on the current solver, with the finite-storage infiltration cap active. The
cap improved both meaningful metrics — rising limb 0.47 -> 0.09 h and centroid 9.13 -> 7.44 h —
and moved the argmax 1.24 -> 1.72 h, which is not informative either way since it sits inside
the plateau width. The rising-limb agreement is robust to the baseflow assumption.

The 7.44 h centroid gap is the honest failure, and it is expected: the observed hydrograph
recedes over more than a day on channel storage and groundwater that a surface-only solver with
no baseflow does not represent at all. The model reproduces how a flood *starts* here; it does
not reproduce how one *ends*.

Timing is the metric this comparison legitimately rests on. **Magnitude is structurally
invalid** and is reported only for completeness: the solver sums outflow across all four domain
edges rather than one channel, has no baseflow, and D8 delineation captures only 11.65 km² of
the gauge's documented 33.15 km² watershed — a known limitation of D8 routing on Florida's flat,
depression-dominated terrain, not a bug.

The main AOI has **no** valid gauge comparison: the nearest Shingle Creek gauge (02263800)
drains 231 km², about 44× this AOI's 5.24 km². It is cross-checked spatially instead
(`analysis/fema_sim_extent_crossref.py`): of 28.21 ha simulated flood extent, **10.05 %** falls
inside a mapped FEMA SFHA, and the simulation reproduces only 1.37 % of the SFHA's own 207.49 ha.
The low overlap is expected — FEMA's SFHA maps *channel* flooding driven by the upstream
watershed, while this solver models only direct rainfall-runoff ponding inside the box. FEMA
geometries are clipped to the AOI before any area maths; the raw fetched features are unclipped
and would make every percentage meaningless.

## Layout

| Path | Contents |
|---|---|
| `dem/ soil/ imagery/ hydrography/ floodplain/ infrastructure/ precipitation/ lidar/ boundary/` | Per-domain fetch + processing, each with its own `data/` |
| `simulation/` | Solvers and run drivers |
| `analysis/` | Flood probability, FEMA × HAND, extent cross-reference |
| `viewer/` | Flask server, Three.js front end, preprocess exporters |
| `site3_gee_creek/` | site3's own data tree and thin site-specific wrappers |
| `research/` | Learned surrogates — **exploratory, not part of the pipeline** ([README](research/README.md)) |

## Known limitations

These are real and documented rather than hidden:

- **Infiltration is unbounded — this is the dominant source of error, and it is diagnosed.**
  `horton_rate()` returns `fc + (f0-fc)*exp(-k*t)`: the rate decays to `fc` and stays there
  indefinitely. `cum_infil` is tracked but never limits anything, so the soil can absorb water
  forever. At site3's `fc_eff` of 23.3 mm/hr over a 72-hour event that is 1,678 mm of capacity
  against 392 mm of rain, so essentially all rainfall infiltrates.

  Measured against the Gee Creek gauge for Hurricane Ian:

  | | runoff coefficient |
  |---|---|
  | observed (2.54 of 12.98 million m³) | **19.6 %** |
  | simulated (0.19 of 18.31 million m³) | **1.0 %** |

  The 19.3x gap fully accounts for the 13x peak-discharge shortfall then measured (91.4 vs
  1,190 cfs); the
  35% D8 watershed capture would only explain ~2.9x. **The error is parameterisation, not
  geometry.**

  Physically, the model implements *infiltration-excess* (Hortonian) runoff only. Central
  Florida's flat terrain and shallow water table generate runoff mainly by *saturation-excess*:
  the profile fills, then everything runs off. The model's soil never fills. This also explains
  the validation pattern exactly — the rising limb matches (0.47 h), because early runoff comes
  from impervious and already-saturated ground, which the model does capture; magnitude and
  recession fail, because the bulk saturation-excess response is absent entirely.

  Two standard methods bracket the truth, and neither is right as parameterised here:

  | method | runoff coefficient, P = 392 mm |
  |---|---|
  | current Horton, uncapped | 1.0 % |
  | **observed** | **19.6 %** |
  | SCS-CN with the on-disk mean CN of 54.5 | 55.4 % |

  **Implemented, and it is not sufficient on its own.** `soil/fetch_soil_storage.py` now pulls
  SSURGO's `wtdepannmin` and the solver caps cumulative infiltration at
  water-table-depth x drainable porosity (`--no-soil-storage` restores the old behaviour). For
  site3 that is a mean cap of 206 mm, with 26 % of cells depressional and therefore zero-storage.

  Measured effect on the Ian run:

  | | before | after | observed |
  |---|---|---|---|
  | runoff coefficient | 1.0 % | **1.7 %** | **19.6 %** |
  | peak outflow | 91.4 cfs | 129.8 cfs | 1,190 cfs |
  | peak flooded area | 127.6 ha | **246.1 ha** | — |

  The mechanism works — flooded area nearly doubled, so the cap is genuinely forcing water to
  stay on the surface — but that water does not reach a boundary. Runoff coefficient moved only
  1.0 → 1.7 %, leaving an 11.5x gap. **Infiltration was a real error but not the dominant one.**

  **What this isolates is conveyance, not storage** — measured, not assumed. Three checks:

  - The water is not trapped in depressions. The conditioned DEM contains **240 true pits**
    (0.00% of cells); depression breaching worked. What it does contain is **11.02% flats** —
    cells whose lowest neighbour is at exactly equal elevation.
  - Twenty-four hours after rain stops, flooded area is still **rising** (+0.13 ha/hr), not
    falling. Water is spreading laterally under its own surface gradient, seeking an outlet.
  - Extrapolated, draining the standing 244 ha would take **~80 days**. Within any realistic
    event window the water is immobile, though not strictly trapped.

  So the limiting factor is the rate at which water can be conveyed across near-zero gradient to
  an outlet, not how much the ground can hold. That is the D8 under-capture (11.65 of
  33.15 km²) seen from the water side: if the channel network in the DEM is not continuous,
  water has nowhere efficient to go regardless of how much runoff is generated. Restoring
  connectivity — not adding storage — is the open problem.

  A cap of ~315 mm would reproduce the observed 19.6 % arithmetically, but that is calibration
  against a single event with the routing error still present, so it would be fitting one bug
  with another. The physically-derived 206 mm is kept deliberately.

- **Pluvial only.** No inflow boundary condition, so no channel overtopping at any site.
- **D8 under-capture.** 11.65 / 33.15 km² at Gee Creek. Central Florida's isolated wetlands and
  cypress domes only connect to the channel network during high-water events.
- **No convective acceleration.** All solvers here use the Bates et al. (2010) local-inertial
  family, which drops that term by design. Most likely to matter at `site3`'s fast-responding
  creek watershed.
- **Sub-grid pit trapping.** At native (~0.9 m) resolution the solver traps water in 1–2 cell
  pits. Trust the wet-cell percentile depths (median ~7–8 cm, p90 ~14 cm), not the absolute peak.
- **Stationary.** Atlas 14 carries no climate trend, so these are present-day probabilities.
- **Stream delineation is unstable on this terrain.** Conditioning the same DEM with two
  defensible depression-breaching algorithms (richdem vs WhiteboxTools) and routing both with
  pysheds gives stream networks agreeing at only **IoU 0.29**, despite elevations matching
  within 1 mm on 98.3% of cells — because with ~14 m of relief over 2 km, D8 routing is decided
  by sub-millimetre differences at pits and flats. The delineated network therefore carries
  more uncertainty than a single run suggests, which is worth weighing against the documented
  D8 under-capture at Gee Creek. Everything shipped here uses richdem throughout, so results
  are internally consistent; see `requirements.txt` for the measurements.
- **Mesh solver does not scale.** Excellent at the ~160–400 m demo sites; ~2.5 h of wall time
  for an 8-minute synthetic event at site3's full extent. Use the raster solver for real events.

## Licence

MIT, via the repository root [`LICENSE`](../../LICENSE).
