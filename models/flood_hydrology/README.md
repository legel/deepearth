# flood_hydrology — physics-based flood digital twins

A reproducible pipeline that turns **a coordinate** into a working flood digital twin: fetch
every public dataset for that location, condition the terrain, run a calibrated shallow-water
solver over a standard storm ensemble, and serve the result as an interactive 3D viewer.

```
models/flood_hydrology/
├── floodtwin/     shared library — physics constants, flood-probability method
├── johns_lake/    Johns Lake, Winter Garden FL — lake-focused twin
└── cfx_sr417/     CFX SR417 corridor + Gee Creek — gauge-validated twin
```

The two sites are peers. Everything method-level that must be identical between them lives in
`floodtwin/`; everything location-specific lives in the site directory.

---

## Sites

| | `johns_lake` | `cfx_sr417` |
|---|---|---|
| Centre | 28.5216, −81.6570 | 28.36687, −81.43299 (+ site3 at 28.6905, −81.2875) |
| Character | 4.9 % impervious, sandy, lake-dominated | 28.7 % impervious, roadway corridor |
| Ground truth | lake stage + Sentinel-2 extent (static) | **USGS gauge 02234400 discharge (dynamic)** |
| Extra capability | Sentinel-2 archive, 4-method water segmentation, GSDR sub-daily rainfall, lake volume/forecast | LiDAR meshes, mesh SWE solver, FEMA cross-reference, learned surrogates (`research/`) |

## Standard scenario set

Every site runs the **NOAA Atlas 14 design-storm ensemble** — return periods
T ∈ {1, 2, 5, 10, 25, 50, 100, 200, 500} yr at 24 hr, converted to SCS Type II hyetographs.
Atlas 14 is queried per coordinate, so this runs unchanged anywhere, which is what makes
cross-site comparison mean anything.

`floodtwin/probability.py` inverts the per-cell peak-depth curve to an annual exceedance
probability, then to any horizon via `P(≥1 in N yr) = 1 − (1 − AEP)^N`.

| | cfx main AOI | cfx site3 | johns_lake |
|---|---|---|---|
| any resolvable risk | 6.85 ha | 30.08 ha | 13.95 ha |
| ≥1 %/yr | 3.98 ha | 18.53 ha | 5.77 ha |
| ≥10 %/yr | 1.30 ha | 8.47 ha | 0.00 ha |

Johns Lake floods nothing below the 50-yr storm. That is a correct result, not a broken one:
flooding there tracks **peak intensity**, not storm total, and a 24-hr design storm spreads the
rain below the soil's infiltration rate. A 24hr/1yr storm delivers 94.2 mm and floods nothing;
a 1hr/100yr storm delivers a near-identical 98.0 mm at 25× the peak intensity and floods
152.5 ha. **A single fixed duration is not discriminating at every site.**

## Validation

**One dynamic validation exists**, at `cfx_sr417`'s site3 against USGS 02234400 for Hurricane Ian:

| metric | difference | reading |
|---|---|---|
| **rising limb** (50 % of peak) | **0.09 h** | flood onset reproduced to ~5 minutes |
| peak argmax | 1.72 h | *unresolvable* — the 99 % plateau is 1.60 h wide in the model, 3.00 h at the gauge |
| centroid | 7.44 h | the model recedes far too fast |

Peak magnitude is 129.8 cfs against 1,190 observed. That gap is **not** mainly the 35 % D8
watershed capture — it is conveyance (see below).

`johns_lake` has strong **static** validation (water mask F1 0.88 / IoU 0.79 vs NHD; lake level
within 1 σ of the gauge mean) but **no** viable event validation: its gauge samples ~27-daily,
and its one satellite-flagged candidate had no rain at five independent stations. Its
`historical_20240212` scenario is not a validation case.

## Known limitations

- **Conveyance-limited on flat terrain.** Runoff is generated correctly but cannot reach an
  outlet: 24 h after rain stops, flooded area is still *rising*; the standing water would need
  ~80 days to clear. This is the D8 under-capture seen from the water side, and it is the
  dominant remaining error.
- **Pluvial only.** No inflow boundary condition, so no channel overtopping.
- **No baseflow or channel storage** — hence the 7.44 h recession error.
- **Stationary.** Atlas 14 carries no climate trend; these are present-day probabilities.
- **Stream delineation is unstable here.** Two defensible depression-breaching algorithms give
  stream networks agreeing at only IoU 0.29, because with ~14 m of relief over 2 km, D8 routing
  turns on sub-millimetre differences.

## Environment

**Python 3.9 for the pipeline.** `richdem`, which performs the depression breaching the solver
depends on, does not build on 3.11 — its vendored pybind11 predates Python 3.11 making
`PyFrameObject` opaque. `pysheds` 0.5 also requires numpy < 2 (`np.in1d`, removed in 2.0). A
separate 3.11 venv serves `cfx_sr417/research/` only, which needs torch. See each site's
`requirements.txt`.

```bash
python3.9 -m pip install --user -r cfx_sr417/requirements.txt
python3 cfx_sr417/viewer/server.py     # → http://localhost:5051
python3 johns_lake/viewer/server.py    # → http://localhost:5050
```

**Next steps** are in [`NEXT_STEPS.md`](NEXT_STEPS.md) — two parallel tracks, with the
measurements each is scored against.

Site-level detail is in [`cfx_sr417/README.md`](cfx_sr417/README.md) and
[`johns_lake/README.md`](johns_lake/README.md). A session log covering the most recent cleanup,
unification and validation work is in [`../PROGRESS_2026-08-26.md`](../PROGRESS_2026-08-26.md).

## Licence

MIT, via the repository root [`LICENSE`](../../LICENSE).
