# CFX SR417 Corridor — flood digital twin

A reproducible pipeline that turns **a coordinate** into a working flood digital twin: fetch
every public dataset for that location, condition the terrain, run a calibrated shallow-water
solver over a standard storm ensemble, and serve the result as an interactive 3D viewer.

Built for the Central Florida Expressway Authority's SR417 corridor, and validated at a second,
gauge-matched site where simulated discharge can be checked against a real USGS record.

Sibling project: [`../flood_hydrology`](../flood_hydrology) (Johns Lake). Shared code lives in
[`../floodtwin`](../floodtwin).

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

```bash
pip install -r requirements.txt

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
| Peak depth, 1 yr → 500 yr | 0.373 → 0.617 m | 0.939 → 1.456 m |
| Peak flooded, 1 yr → 500 yr | 6.7 → 26.3 ha | 31.3 → 124.7 ha |
| Any resolvable pluvial risk | 6.85 ha | 30.08 ha |
| ≥1 %/yr | 3.98 ha | 18.53 ha |
| ≥10 %/yr | 1.30 ha | 8.47 ha |
| Depth vs log T, R² | 0.9715 | 0.9703 |

Both surfaces are monotone in T, as required. A 1 %-AEP cell returns P = 0.2606 over 30 years,
matching FEMA's published "26 % chance over a 30-year mortgage" figure.

**Historical events** are an optional per-site validation layer, run only where ground truth
exists — currently Hurricane Ian (2022-09-28/30) at both sites.

## Validation

**Gee Creek (`site3`, USGS 02234400)** is the only place in this project where simulated
discharge can be compared against a real gauge without a disqualifying scale mismatch.

| | Simulated | Observed |
|---|---|---|
| Peak outflow | 91.4 cfs | 1,190 cfs |
| Peak time | t = 36.28 h | t = 37.52 h |
| **Timing error** | **1.24 h** | — |

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

- **Pluvial only.** No inflow boundary condition, so no channel overtopping at any site.
- **D8 under-capture.** 11.65 / 33.15 km² at Gee Creek. Central Florida's isolated wetlands and
  cypress domes only connect to the channel network during high-water events.
- **No convective acceleration.** All solvers here use the Bates et al. (2010) local-inertial
  family, which drops that term by design. Most likely to matter at `site3`'s fast-responding
  creek watershed.
- **Sub-grid pit trapping.** At native (~0.9 m) resolution the solver traps water in 1–2 cell
  pits. Trust the wet-cell percentile depths (median ~7–8 cm, p90 ~14 cm), not the absolute peak.
- **Stationary.** Atlas 14 carries no climate trend, so these are present-day probabilities.
- **Mesh solver does not scale.** Excellent at the ~160–400 m demo sites; ~2.5 h of wall time
  for an 8-minute synthetic event at site3's full extent. Use the raster solver for real events.

## Licence

MIT, via the repository root [`LICENSE`](../../LICENSE).
