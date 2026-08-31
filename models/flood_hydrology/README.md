# flood_hydrology — physics-based flood digital twins

> **2026-08-29 — magnitude numbers in this document are superseded.**
> Four defects were found in the solver's integration loop (broken clock delivering 7-11 % of
> the storm, phantom infiltration, an unstable CFL constant, a broken final-frame guard). All
> are fixed and the solver now conserves mass to -0.001 %, but every magnitude figure below —
> runoff coefficients, peak discharges, depths, flooded areas, probability surfaces — predates
> the fixes. Timing results are robust and largely stand. See [`NEXT_STEPS.md`](NEXT_STEPS.md).

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
| any resolvable risk | **11.03 ha** | **56.33 ha** | 13.95 ha † |
| ≥1 %/yr | **5.97 ha** | **33.54 ha** | 5.77 ha † |
| ≥10 %/yr | **1.73 ha** | **13.68 ha** | 0.00 ha † |

Both cfx columns were re-run on 2026-08-27 after three defects that had been suppressing runoff
were fixed: the solver destroyed its own DEM conditioning when downsampling, site3's stream burn
had never actually executed, and the soil-storage cap had only ever been applied to site3.
Before those fixes the figures were 6.85 / 3.98 / 1.30 (main AOI) and 30.08 / 18.53 / 8.47
(site3). The correction roughly **doubles** mapped risk area at both sites.

The signature is the one removing fabricated depression storage should produce — **peak depths
fall while flooded areas rise**, because water spreads and moves instead of pooling in pits the
downsampling had invented. At site3 the 100-yr storm went from 1.249 m / 84.7 ha to 0.967 m /
143.4 ha. Monotonicity in T holds throughout (50.1 ha at 1 yr to 199.9 ha at 500 yr).

† `johns_lake` is **not** affected by the resampling defect — verified, its solver reads the DEM
at native 2.64 m with no downsampling step — but it has not been re-run, so it has not been
checked against the other two fixes. See [`NEXT_STEPS.md`](NEXT_STEPS.md).

Johns Lake floods nothing below the 50-yr storm. That is a correct result, not a broken one:
flooding there tracks **peak intensity**, not storm total, and a 24-hr design storm spreads the
rain below the soil's infiltration rate. A 24hr/1yr storm delivers 94.2 mm and floods nothing;
a 1hr/100yr storm delivers a near-identical 98.0 mm at 25× the peak intensity and floods
152.5 ha. **A single fixed duration is not discriminating at every site.**

## Validation

**One dynamic validation exists**, at `cfx_sr417`'s site3 against USGS 02234400 for Hurricane Ian:

| metric | difference | reading |
|---|---|---|
| rising limb (50 % of peak) | 0.72 h | resolved — 2.9x the gauge's own sampling interval |
| peak argmax | — | *unresolvable*; the 99 % plateau is wider than the difference |

**The gauge samples at 15 min (0.25 h), so no timing difference below that is claimable.** An
earlier "0.09 h" rising limb sat under that limit and was never a measurement of accuracy — the
same resolution ceiling that already disqualified the peak-argmax metric.

Peak magnitude is 411.6 cfs at the domain boundary, and 101.6 cfs at the gauge cell itself,
against 1,190 observed. Runoff coefficient is 8.90 % against an observed **28.9–31.4 %**.

A 19.6 % observed runoff coefficient appears in earlier write-ups. **It cannot be reproduced
from the NWIS record** under any standard baseflow-separation or integration-window choice;
like-for-like over the simulated window the figure is 28.9–31.4 %. Run
`analysis/validate_gauge_site3.py`, which reports the full sensitivity, rather than quoting a
single number.

`johns_lake` has strong **static** validation (water mask F1 0.88 / IoU 0.79 vs NHD; lake level
within 1 σ of the gauge mean) but **no** viable event validation: its gauge samples ~27-daily,
and its one satellite-flagged candidate had no rain at five independent stations. Its
`historical_20240212` scenario is not a validation case.

## Known limitations

- **Magnitude is short by ~3x, and the gauge-cell hydrograph does not peak.** Runoff reaches
  8.90 % against an observed 28.9–31.4 %, and discharge at the gauge cell rises monotonically
  for the full 72 h rather than peaking. The water no longer accumulates indefinitely — that
  was fixed on 2026-08-27, and flooded area now drains where it previously kept rising — but
  the storm response is too small and too slow. **Conveyance has been ruled out by
  measurement**: along-channel bed slope on the solver grid is 1.94e-3, and at the existing
  n=0.040 a channel depth of 0.14–0.62 m already yields 0.30–0.80 m/s, the documented range for
  this stream class. Manning's *n* is not the bottleneck.
- **Pluvial only.** No inflow boundary condition, so no channel overtopping.
- **No baseflow or channel storage** — hence the 7.44 h recession error.
- **Stationary.** Atlas 14 carries no climate trend; these are present-day probabilities.
- **Stream delineation is unstable here.** Two defensible depression-breaching algorithms give
  stream networks agreeing at only IoU 0.29, because with ~14 m of relief over 2 km, D8 routing
  turns on sub-millimetre differences.

## Environment

**Python 3.9 for the pipeline — and `python3` already *is* 3.9.6 on this machine.** There is no
`python3.9` binary on PATH; checking for one and concluding 3.9 is unavailable is a mistake
worth not repeating. `richdem`, which performs the depression breaching the solver
depends on, does not build on 3.11 — its vendored pybind11 predates Python 3.11 making
`PyFrameObject` opaque. `pysheds` 0.5 also requires numpy < 2 (`np.in1d`, removed in 2.0). A
separate 3.11 venv serves `cfx_sr417/research/` only, which needs torch. See each site's
`requirements.txt`.

```bash
python3 -m pip install --user -r cfx_sr417/requirements.txt   # python3 IS 3.9 here
python3 cfx_sr417/viewer/server.py     # → http://localhost:5051
python3 johns_lake/viewer/server.py    # → http://localhost:5050
```

**Next steps** are in [`NEXT_STEPS.md`](NEXT_STEPS.md) — two parallel tracks, with the
measurements each is scored against.

Site-level detail is in [`cfx_sr417/README.md`](cfx_sr417/README.md) and
[`johns_lake/README.md`](johns_lake/README.md). Session logs: [`../PROGRESS_2026-08-29.md`](../PROGRESS_2026-08-29.md) (solver
repair — four integration-loop defects, timing solved, magnitude numbers withdrawn) and
[`../PROGRESS_2026-08-26.md`](../PROGRESS_2026-08-26.md) (cleanup, unification, validation).

## Licence

MIT, via the repository root [`LICENSE`](../../LICENSE).
