# Track C — Extreme storm vs. longest drought: lake extent comparison

**Status: DONE 2026-06-18.** All 4 tasks below implemented. Found the true
GSDR extreme is a different event than the one already in `historical_gsdr`
(1945-09-16, 245.6mm/24hr by total depth, vs. 1960's 143.5mm/1hr peak
intensity) — added as scenario `historical_gsdr_extreme`. Drought side built
as `dem/lake_drought.py` (VAE-curve drawdown, not a 2D solver run) using the
longest GSDR dry spell (82 days) and USGS central-FL evaporation rates. See
"Track C" under "Completed work tracks" in `CLAUDE.md` for the full summary
and findings (peak intensity vs. total rainfall on flood severity; steep
near-surface basin means drought barely shrinks lake area while draining
~14% of volume). All 7 scenarios re-run, exported, `verify_all.py` 53/53.

Repo: `flood_hydrology`. Touches `precipitation/`, `simulation/flood_sim.py`
(`SCENARIOS` dict only), and probably a new small `dem/lake_drought.py`.
Benefits from Track A's infiltration-toggle landing first (so the extreme-storm
run can be sanity-checked the same way), but isn't strictly blocked on it —
the `SCENARIOS` mechanism already supports new entries with zero code changes.

## Why

From the meeting notes:

> "Simulate the biggest in history — look at the data, the most extreme
> precipitation event — and the longest drought? Like how the lake look like
> in comparison, show the 2 extents."

This is a sanity-check / storytelling scenario pair: show the lake at its
most extreme high-water and low-water states, both literally and as a way to
stress-test whether the pipeline (and its ground-truth assumptions, see Track
D) holds up at the extremes.

## Current state

- `simulation/flood_sim.py`'s `SCENARIOS` dict (line 162) currently has:
  `flash_1hr_100yr`, `flash_1hr_10yr`, `sustained_12hr_100yr`,
  `sustained_12hr_10yr`, `historical_20240212` (blocked, needs
  `NOAA_CDO_TOKEN`), `historical_gsdr` (done — 1960-07-25 event, 143.5mm/1hr
  peak, 208mm/24hr, from station `US_086638`). Each entry just points to a
  hyetograph CSV filename + label — **no code changes needed to add a new
  storm**, only a new CSV in `precipitation/data/`.
- `precipitation/fetch_gsdr_gauge.py` already exists and was used to build
  the `historical_gsdr` hyetograph from `US_086638`. CLAUDE.md notes a second
  available station, `US_086628` (1974–2011, 1-hr max 93.7mm), "available, not
  yet used" — worth checking if a different historical event (possibly more
  extreme, possibly longer-duration) is buried in either station's full record
  rather than just the single peak event already extracted.
- There is **no drought-scenario mechanism at all** — `flood_sim.py` is an
  inundation solver (2D shallow water + infiltration), not built to model
  multi-week/month drawdown. The lake-level/volume side of that is in
  `dem/lake_volume.py`'s VAE (volume-area-elevation) curve, which is the
  right tool for a drought scenario: declining lake level → look up
  corresponding area/volume from the VAE curve, no need to run the 2D solver
  for zero/near-zero rainfall over weeks.

## Tasks

1. **Find the true historical extreme.** Re-examine the full GSDR record for
   both `US_086638` and `US_086628` (not just the single event already
   extracted for `historical_gsdr`) — confirm whether 1960-07-25 (143.5mm/1hr)
   is actually the all-time max in the available record, or whether a longer
   record search turns up something more extreme. Use
   `precipitation/fetch_gsdr_gauge.py` as the starting point/reference for how
   to parse the raw GSDR text files (`~/Desktop/GSDR/QC_d data - US/US_*.txt`).
2. **Build the drought scenario.** This does NOT need a new `flood_sim.py`
   run. Instead:
   - Write a small new script, e.g. `dem/lake_drought.py`, that reuses
     `dem/lake_volume.py`'s VAE curve logic (read that file first to find the
     actual function names — described in CLAUDE.md as "Lake storage: VAE
     curve from FWC bathymetry").
   - Model a drawdown by stepping the lake level down from the current
     authoritative WSE (28.74 m, from `dem/data/lake_volume.csv`) by some
     plausible drought rate (look at NOAA/USGS regional climate data or the
     GSDR record for the longest observed dry spell — this determines the
     rate/duration, don't just guess a number), and look up area/volume at
     each step from the VAE curve.
   - Output a small CSV (analogous to a hydrograph) with `time, lake_level_m,
     area_ha, volume_m3` for the drought drawdown — same shape of output as a
     `simulation/outputs/hydrograph_*.csv` row set so it's easy to compare
     side-by-side with the wet scenarios.
3. **Add the extreme-storm scenario** to `SCENARIOS` the normal way (new
   hyetograph CSV + dict entry), run via:
   ```bash
   ~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \
       --scenario <new_extreme_name> --save-frames \
       --soil-preset central-fl-antecedent
   ```
4. **Side-by-side comparison.** Produce a figure or pair of viewer scenario
   entries showing the extreme-wet flooded extent next to the drought-drawn-down
   lake extent — reuse whatever plotting pattern `dem/lake_volume.py` or
   `dem/dem_visualize.py` already use for consistency (check those files for
   the existing matplotlib style before writing new plotting code).

## Gotcha (found verifying Track A, 2026-06-18)

`flood_sim.py --scenario <single_name>` overwrites
`simulation/outputs/simulation_summary.csv` with **only that scenario's
row** — it does not merge. Before testing your new extreme-storm scenario in
isolation, know that doing so will wipe the other 5 scenarios' rows from the
summary CSV (and that scenario's `depth_frames_*.npz`). After any
single-scenario test run, restore full state with:
```bash
~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \
    --scenario all --save-frames --soil-preset central-fl-antecedent
```
Don't run this in parallel with another `flood_sim.py` invocation from a
different terminal — both write the same `simulation_summary.csv` and can
race/corrupt it.

## Files touched

`precipitation/data/` (new hyetograph CSV), `precipitation/fetch_gsdr_gauge.py`
(if extended to search the full record rather than one known event),
`simulation/flood_sim.py` (`SCENARIOS` dict, one new entry),
new file `dem/lake_drought.py`.

## Verification

- New extreme scenario appears in `simulation/outputs/simulation_summary.csv`
  after running, and gets picked up by
  `viewer/preprocess/export_simulation.py` → `viewer/data/simulation_index.json`
  gets a new entry.
- Drought drawdown CSV shows monotonically declining area/volume consistent
  with the VAE curve (no non-physical jumps).
- `python3 verify_all.py` still passes (this track only adds, doesn't modify
  existing scenario behavior).
