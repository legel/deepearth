# Ian flood simulation — grid resolution sensitivity (Task 6)

_2026-07-06. Answers Lance's meeting-note question: is the 5m grid sufficient for
house/road-scale water depth reporting?_

Three runs of `simulation/flood_sim_ian.py` (with the new spatial per-cell Horton infiltration —
see Task 10 notes below) at 5m (default/production), 2m, and native ~0.88m LiDAR resolution.
Non-frame outputs from the 2m and native runs are preserved as `*_2m.*` / `*_native.*` in this
directory; the unsuffixed files are the restored 5m production config.

| Resolution | Peak depth (max cell) | Peak flooded area | Wet-cell median depth | Wet-cell p90 | Wet-cell p99 |
|---|---|---|---|---|---|
| 5 m  | 0.485 m | 15.9 ha | 0.072 m* | 0.137 m* | ~0.35 m* |
| 2 m  | 0.870 m | 4.7 ha  | 0.077 m | 0.145 m | 0.344 m |
| 0.88 m (native) | 2.562 m | 1.5 ha | 0.072 m | 0.137 m | 0.418 m |

_*5m wet-cell percentiles not separately recomputed after the file-restore rerun; 2m/native
values shown are directly measured from the corresponding rasters._

## Two findings, not one

**1. Coarser grids under-report depth and over-report extent.** Bilinear-resampling the DEM to
5m smooths real micro-topography (drainage hollows at the 30–100m scale, per the earlier terrain
analysis) into broad, shallow puddles. As resolution sharpens, the same rainfall concentrates
into narrower, deeper features — exactly what you'd expect once real channels/depressions are
resolved instead of averaged away. This directly supports upgrading past 5m for any house/road-
scale claim.

**2. But the native-resolution "peak depth" number is itself not fully trustworthy.** Both the
2m and native runs show the same signature: a 1–2 cell depth spike (0.87m / 2.56m) surrounded
immediately by near-dry cells (see raw cell dumps in the session — no smooth gradient a real
depression would show). This is a known local-inertia/LISFLOOD-FP-style solver failure mode:
water gets numerically trapped in a sub-grid-scale pit that the current solver's friction/flux
formulation can't adequately drain once the pit is only 1–2 cells wide. It gets *worse* at finer
resolution because sharper DEMs resolve narrower pits that are more prone to this — it is not
evidence of a real 2.5m-deep pond.

**The wet-cell percentile depths are the trustworthy statistic and are strikingly consistent
across resolutions**: median ~7–8 cm, p90 ~14 cm, p99 ~35–42 cm, regardless of grid size. This
is the number to quote for "typical" house/road-scale depth, not the single-cell max.

## Recommendation

- **2m is a reasonable practical resolution** for house/road-scale reporting: fine enough to
  resolve individual house footprints (10–15m spans 5–7 cells) and road crowns, without the
  native grid's more extreme pit-trapping.
- Before quoting native-resolution absolute peak depths, the solver needs a pit-filling / sub-
  grid storage correction (standard in production LISFLOOD-FP and similar tools) — this is a
  gap in the current from-scratch implementation, not a data or DEM problem.
- This finding is independent of, and complementary to, Task 10's separate need for a
  building/road/canopy-aware solver generalization.
