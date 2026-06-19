# Track A — Simulation physics sanity check

Repo: `flood_hydrology`. Touches only `simulation/flood_sim.py`. No dependency
on any other track — start here in a fresh terminal.

## Why

Team lead's notes are explicit: don't trust the full design-storm scenarios
until a basic physical sanity check passes. The ask, paraphrased from messy
meeting notes:

> "Can we localize the rain — on top of the hill? Make sure it actually
> flows. Turn off infiltration first to check — once we feel good, [re-enable
> it]. Can we visualize infiltration as well?"

Right now the simulation only ever rains uniformly over the whole domain with
infiltration always on, so there's no way to visually confirm water moves
downhill the way it should before trusting the bigger scenarios.

## Current state (confirmed by reading `simulation/flood_sim.py`)

- Rainfall is a **scalar applied uniformly** to the whole grid every step:
  `rain_rate_ms = incr_mm[step_i] / 1000 / dt_hyet_s` (line 632), then
  `Pe_ms = horton.effective_rainfall(rain_rate_ms, t_s)` (line 635) — both are
  plain floats, not arrays. There is no spatial rainfall mask anywhere.
- `class HortonInfiltration` (line 216) always computes infiltration; there's
  no flag to disable it. `SOIL_PRESETS` (line 152) is a dict of two presets
  (`"central-fl-antecedent"` and SSURGO-derived defaults) — neither is a
  zero-infiltration preset.
- Infiltration is recorded only as a **scalar rate** in the hydrograph CSV:
  `"infilt_mm_hr": round(horton.rate(t_s) * 1000 * 3600, 2)` (line 691). It's
  never accumulated into a spatial map, and never saved as a frame array like
  depth is.
- Time stepping is CFL-adaptive (`adaptive_dt`, line 306); `--dt` (line 936)
  lets you force a fixed step, default is adaptive.
- `--save-frames` (line 938) and `--soil-preset` (line 941, choices = keys of
  `SOIL_PRESETS`) already exist as CLI flags — `main()`'s arg parsing starts
  around line 920s.

## Tasks

1. **Spatial rainfall mask.** Add a `--rain-mask {full,hilltop}` CLI arg
   (default `full` = current behavior, unchanged). For `hilltop`: build a
   boolean array the shape of the DEM grid, true within some radius (start
   with ~50–100 m, i.e. a handful of cells at dx≈2.6m) of the grid's
   highest-elevation cell(s). Change `rain_rate_ms` (and anywhere it's
   currently a scalar broadcast into `h +=`) into a `(ny, nx)` array masked by
   this boolean array, so non-hilltop cells get zero rain. Horton infiltration
   computation should still run per-cell using whatever the local Pe is.

2. **Infiltration off-switch.** Add `--no-infiltration` (or add a `"none"`
   entry to `SOIL_PRESETS` with `f0_mm_hr=0, fc_mm_hr=0`) so `Pe == P` exactly
   — i.e. `HortonInfiltration.effective_rainfall` should short-circuit to
   return the raw rain rate unchanged when this is set. Don't remove the
   `HortonInfiltration` class or fork the step loop — just make the rate zero.

3. **Infiltration visualization array.** Add a `cumulative_infiltration_mm`
   array (shape `(ny, nx)`, or scalar broadcast if rain is still uniform —
   shape it like the depth array regardless), accumulated each step as
   `f(t) * dt` integrated over the run. Save it into the same `.npz` that
   `--save-frames` writes (`simulation/outputs/depth_frames_{scenario}.npz`),
   as a new array key (e.g. `infiltration_frames`), same `(n_frames, ny, nx)`
   shape as the depth frames, so Track B can reuse the existing depth-texture
   rendering path in the viewer for it.

4. **Run the sanity check:**
   ```bash
   ~/miniforge3/envs/prithvi/bin/python simulation/flood_sim.py \
       --scenario flash_1hr_10yr --rain-mask hilltop --no-infiltration \
       --save-frames --soil-preset central-fl-antecedent
   ```
   Then write a quick standalone script (or a `python3 -c` one-liner) that
   loads `depth_frames_flash_1hr_10yr.npz`, plots a few frames with
   `matplotlib.pyplot.imshow`, and confirms: water appears first near the
   hilltop cell(s), then visibly spreads/flows toward lower elevation over
   subsequent frames, pooling in the lake basin or local depressions —
   not appearing uniformly across the grid.

5. **Re-enable infiltration**, same hilltop rain mask, and confirm flooded
   area is smaller / delayed relative to the no-infiltration run — i.e.
   infiltration is doing something physically sensible, not just present in
   name. This is the point where "we feel good again" per the notes.

## Gotcha discovered while verifying this track (2026-06-18)

Running `flood_sim.py --scenario <single_name>` **overwrites
`simulation/outputs/simulation_summary.csv` with only that one scenario's
row** — it doesn't merge/append. Any single-scenario run (including this
track's sanity-check runs) destroys the other scenarios' rows in that CSV,
and also overwrites that scenario's `depth_frames_{name}.npz` regardless of
whether `--save-frames` was passed for the *other* scenarios' files. If you
run a one-off single-scenario test, **follow it with
`--scenario all --save-frames --soil-preset central-fl-antecedent`** to
restore the full summary before anyone else relies on it (e.g. the viewer
export pipeline). This caused real data loss during this track's own
verification and had to be repaired.

## Files touched

`simulation/flood_sim.py` only, plus one throwaway/small verification script
(can live in `simulation/` or be a one-off, your call — doesn't need to be
checked in if it's just for eyeballing).

## Verification

- New hilltop+no-infiltration run visually shows downhill flow (see task 4).
- `python3 verify_all.py` still shows 53/53 passing after the change (the
  existing `flash_1hr_10yr` / etc. scenarios with `--rain-mask full` —
  the default — must produce numerically identical results to before this
  change, since `full` should be a no-op vs. current behavior).
- Re-run the two already-passing scenarios without new flags and confirm
  `simulation/outputs/simulation_summary.csv` peak values are unchanged.

## Follow-up found via Track B viewer review (2026-06-18) — not yet actioned

While verifying the new Infiltration layer in the viewer (`plans/track_B_viewer_ux.md`
task 4), found two things worth a decision before more viewer work goes into
either layer. Not fixed as part of this — both need a `flood_sim.py` change
plus a full rerun, so deliberately left for whoever's driving Track A next
(possibly the same session already mid-rerun on this file as of 2026-06-18
21:xx — check for conflicts before editing `flood_sim.py`).

1. **Infiltration has no spatial variation under the default config.**
   Loaded the actual exported `simulation_{scenario}_infiltration.bin` and
   checked per-cell stats: at every single timestep, `min == p50 == max`
   across the entire 256×256 grid (e.g. `sustained_12hr_100yr` t=720min: all
   cells = 192.314mm, exactly). This is expected given the current
   architecture — `HortonInfiltration` computes one scalar Pe per step
   (confirmed at `flood_sim.py` line ~655, `Pe_scalar = horton.effective_rainfall(...)`)
   broadcast to the whole `cumulative_infiltration_m` array, and soil
   parameters aren't spatially varying either (single dominant preset, not
   per-cell SSURGO). So a colored 2D "map" of this field is mathematically
   guaranteed to be a flat, uniform tint — real data, zero spatial
   information. Two options if this should be a real spatial map:
   - Wire actual per-cell SSURGO Horton `f0`/`fc`/`k` into the infiltration
     calc (bigger change — `soil/data/soil_parameters.json` already has
     per-mukey values, just isn't consumed spatially yet).
   - Or accept it's uniform under `--rain-mask full` and only expect spatial
     structure when paired with `--rain-mask hilltop` (where Pe does vary
     because rain itself is masked spatially) — in which case the viewer
     should probably label the layer as a single number (HUD stat) rather
     than a colored overlay for the `full`-mask scenarios, and only show the
     spatial overlay for hilltop-style runs.

2. **No scenario captures post-storm recession.** `flood_sim.py`'s
   post-storm drainage loop (the `for _ in range(60):` loop right after the
   main hyetograph loop, "Post-storm drainage …") only updates `h_peak` — it
   never calls `frame_depths.append()` / `frame_infiltration.append()`, so
   `--save-frames` never captures anything past the last rain timestep. The
   two flash scenarios (`flash_1hr_100yr`, `flash_1hr_10yr`) still have
   nonzero `Pe` at their final saved frame, so their viewer animations look
   like they just stop mid-flood rather than receding. (The sustained 12hr
   scenarios already show partial recession within existing frames, since
   their SCS Type II rain curve tapers to ~0 well before the nominal storm
   end — e.g. `sustained_12hr_100yr` flooded area drops 74ha→18ha between
   t=540 and t=720min.) Fix would be to snapshot frames during the drainage
   loop too (same `frame_interval_min` cadence, extending `times_min` past
   the hyetograph's own range), then rerun `--scenario all --save-frames` and
   re-export.
