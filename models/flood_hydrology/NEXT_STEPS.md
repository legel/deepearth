# Next steps — two parallel tracks

Written 2026-08-26 as a handoff. Background and measurements are in
[`../PROGRESS_2026-08-26.md`](../PROGRESS_2026-08-26.md); site detail is in each site's README.

## Where things stand

The pipeline is clean, unified and pushed. One physics bug was found by gauge validation, fixed,
and measured. **The dominant remaining error is now isolated and it is not parameterisation.**

| | value |
|---|---|
| Flood-onset timing vs USGS 02234400 (Hurricane Ian) | **0.09 h** — validated |
| Runoff coefficient, simulated vs observed | **1.7 % vs 19.6 %** — 11.5× gap |
| Peak discharge | 129.8 cfs vs 1,190 cfs |
| Hydrograph centroid | 7.44 h early — no baseflow or channel storage |

**Both tracks below are scored against the same two numbers: rising limb (0.09 h) and runoff
coefficient (1.7 % vs 19.6 %).** A change either moves them or it doesn't. That discipline is
what caught the unbounded infiltration; keep it.

---

## Track B — channel connectivity (the dominant error)

**The finding this comes from.** Water is generated correctly and then cannot leave the domain.
Twenty-four hours after rain stops, flooded area is still *rising* (+0.13 ha/hr); extrapolated,
the standing 244 ha would take ~80 days to clear. It is **not** trapped in depressions — the
conditioned DEM has 240 true pits (0.00 %), so breaching worked. An earlier claim of ~750,000
pits was a detector artifact that counted *flats* (11.02 %) as pits; don't repeat it.

The constraint is conveyance across near-zero gradient. This is the same phenomenon as the D8
under-capture (11.65 of 33.15 km² at Gee Creek), seen from the water side.

**First test, before building anything:** is site3's burned creek channel *continuous* through
the domain? `dem_hydro.py` burns 3DHP flowlines 1.5 m into the DEM. If that burn left gaps —
because 3DHP flowlines are discontinuous, or the burn didn't reach the domain edge — then a
single fact explains both the under-capture and the 80-day drainage, and the fix is local.

```
cfx_sr417/dem/data/hydro/dem_burned.tif      # post-burn, pre-breach
cfx_sr417/dem/data/hydro/dem_conditioned.tif # what the solver reads
cfx_sr417/site3_gee_creek/hydrography/data/  # the 3DHP flowlines burned in
```

Concretely: trace the burned channel from the Gee Creek pour point
(28.7041629, −81.2906221) downslope and check it reaches a domain boundary without
interruption. If it does, connectivity isn't the answer and the next candidate is conveyance
capacity — Manning's *n* and channel cross-section at 5 m resolution.

**Known, related, undocumented:** `dem_hydro.py` computes a fully conditioned DEM
(breach + `fill_pits` + `fill_depressions` + `resolve_flats`) for D8 and HAND but **never saves
it**. The solver reads the breach-only `dem_conditioned.tif`. With no pits to fill the practical
impact is small, but HAND/streams and the solver genuinely run on different DEMs.

---

## Track A — segmentation-derived parameters (Lance's Step 1+2)

Lance is picking this up. It fills two documented holes:

1. **Tree canopy is absent from the physics mesh entirely.** The mesh is bare-earth DEM plus
   LiDAR building roofs — no vegetation in the flow surface at all.
2. **Ground Manning's *n* is a single uniform 0.040** everywhere non-roof (roofs 0.015). That is
   the whole spatial variation.

| step | output | consumed by |
|---|---|---|
| SAM3 on NAIP 0.6 m — segment and name everything | per-class polygons | new `cfx_sr417/segmentation/` |
| VLM per segment → `{material, Smax, Ks, Manning's n}` | per-class parameter JSON | rasterised to the solver grid |
| rasterise to the 5 m solver grid | 3 parameter rasters | `flood_sim_ian.py` as spatial arrays |

**The solver already takes spatially-varying inputs at this interface** —
`load_spatial_horton()` builds per-cell `f0`/`fc`/`k` from SSURGO, and `load_soil_storage_capacity()`
does the same for `Smax`. Feeding segmentation-derived values in is a substitution, not new
solver machinery. Only Manning's *n* needs promoting from scalar to array.

**Canopy: use LiDAR, not imagery.** The 2018 point cloud carries vegetation returns (classes
3/4/5), so canopy height is recoverable directly rather than inferred from nadir NAIP — real
geometry, from data already on disk. See `cfx_sr417/lidar/cache_bbox_points.py` for the
classification-filtered loader.

**Compute:** ASU allocation for SAM3; hosted inference only if VLM iteration speed matters.

---

## Also open, lower priority

- **Duration axis.** The standard set is T ∈ {1…500} yr at 24 hr. Johns Lake floods nothing
  below the 50-yr storm at that duration because flooding there tracks *peak intensity*, not
  storm total, and it is only 4.9 % impervious against the CFX AOI's 28.7 %. Running
  T × {1 hr, 12 hr, 24 hr} would make the comparison discriminate at every site.
- **Cold-start test.** Nothing has been built at a genuinely fresh coordinate. Site3 re-runs can
  silently pass by reusing cached files; only a new location proves "coordinate → twin".
- **Johns Lake AOI.** 144 ha modelled against a real lake of ~1,044 ha, clipped on three of four
  edges. A validated zero-clip bbox (7.37 × 3.87 km) exists in
  `johns_lake/ground_truth/aoi_expansion_test_data/` and has never been run.
- **`historical_20240212` is not a validation case.** Five independent stations report 1.1–3.9 mm
  over that window; the lake rose 17 ha with no rain. Don't treat its 0.0 ha result as a model
  failure.

## Don't redo these — already measured and settled

- **whitebox is not a richdem substitute.** Stream-network IoU 0.29 against richdem despite
  elevations agreeing within 1 mm on 98.3 % of cells.
- **Python 3.9 for the pipeline.** richdem does not build on 3.11 (vendored pybind11 predates
  `PyFrameObject` becoming opaque); pysheds 0.5 needs numpy < 2 (`np.in1d`).
- **The "1.24 h peak-timing error" cannot be claimed.** Peak plateaus are 1.60 h (model) and
  3.00 h (gauge) wide, so the argmax is not resolved that finely. The rising limb is the
  defensible metric.
