# Track G — Expand AOI to cover the full Johns Lake

Repo: `flood_hydrology`. **Scoping only in this pass — do not execute without
confirming with the user/team first.** This is the single most expensive
track in the backlog: it cascades through nearly every directory in the
pipeline. Read this whole file before starting any step.

## Why

Track D (`ground_truth/track_D_findings.md`, §4) found, and independently
verified (read `dem/data/lake_mask.tif` directly with rasterio):

- Current AOI is a 2x2km box, `dem/data/lake_mask.tif` is 868×868 cells at
  ~2.64×2.62 m resolution (EPSG:5070) = **521.87 ha total grid area**.
- Water-mask pixels touch the AOI boundary on **3 of 4 edges**: north (502
  cells), west (450 cells), east (127 cells); only south is clear (5 cells).
  This means the lake is being cut off by the download extent, not just
  smaller than expected.
- Johns Lake's actual documented surface area is **~2,580 acres ≈ 1,044 ha**
  (Orange County Water Atlas, waterbody ID 7935:
  `orange.wateratlas.usf.edu/waterbodies/lakes/7935/johns-lake`) — roughly
  **2× the current AOI's total grid area**, and the current AOI isn't even
  fully water, so the gap is bigger than 2× in real terms.
- The repo's "authoritative" 144.47 ha lake area
  (`dem/data/lake_volume.csv`) is therefore the area of one basin/portion of
  the lake near the target address (17801 Champagne Dr), not the whole lake.

This matters for: Track C's drought/extreme-storm lake-extent comparison
(currently only shows part of the lake), Track D's PlanetScope/drone
ground-truth discussion (any of those need to know the real extent to be
useful), and basically every "authoritative value" in CLAUDE.md's
Consistency Check table.

## What this actually requires — read before committing

Expanding the AOI is not a config tweak. It cascades through:

1. **DEM** (`dem/dem_download.py`) — cheap, fast, re-download at larger extent.
2. **Sentinel-2** (`sentinel2/s2_download.py`) — re-download all ~205 scenes'
   bands at the new extent. Meaningfully larger download volume, but still
   just a download.
3. **Cloud masking + all 4 water segmentation methods** (`s2_cloud_mask.py`,
   `s2_water_segment.py`, `s2_water_segment_v2.py` for Prithvi,
   `s2_omniwatermask.py`) — **this is the expensive step.** Re-running
   MNDWI/WatNet/Prithvi-EO-2.0/OmniWaterMask across ~205 scenes at a larger
   grid means re-running every GPU inference pass that currently exists in
   the repo. This alone could be hours of wall time (Prithvi and OWM are the
   heavier models per CLAUDE.md's DL environments table).
4. **Soil/NAIP** (`soil/fetch_naip.py`, SSURGO download) — re-fetch at larger
   extent, cheap.
5. **FWC lake bathymetry** (`dem/fetch_*bathymetry.py`) — check first whether
   the FWC boat survey already covers the full lake (likely yes, since boat
   surveys aren't bounded by our AOI choice) — may only need re-clipping to
   the new DEM grid, not re-fetching.
6. **Lake mask consensus** (`dem/lake_utils.get_lake_mask_and_fwc`) — rebuild
   over the new extent and full 153-scene OWM archive (this is also Priority-1
   item 3 already sitting in CLAUDE.md's OPEN list — worth doing together).
7. **`dem/lake_volume.py`** — recompute WSE/area/volume on the new mask +
   bathymetry. Every "authoritative value" in CLAUDE.md's Consistency Check
   table changes.
8. **Simulation domain** (`simulation/flood_sim.py`) — check whether grid
   dimensions are hardcoded anywhere (868×868 appears in multiple places per
   CLAUDE.md) vs. read dynamically from the DEM file shape. A larger domain
   means more cells, longer runtime, and **must be checked against PyTorch
   MPS memory limits** before assuming it'll just work — the CFL-adaptive
   timestep also means runtime scales worse than linearly with cell count for
   the sustained 12-hr scenarios (CLAUDE.md already notes those take
   600s+ at current size).
9. **Viewer exports** (`viewer/preprocess/export_*.py`) — every binary
   (`dem.bin`, `voxels.bin`, overlays, simulation frames) and `geo_meta.json`
   needs re-export against the new grid/extent.
10. **`verify_all.py`** — reference values/shape checks (e.g. 868×868, lake
    area ranges) are tied to the current AOI and need updating, or the
    dashboard will show false failures after this change.

## Recommended approach — validate cheaply before committing to the expensive part

Don't jump straight to step 3 (the 205-scene × 4-method re-run). Sequence it
so you can back out cheaply if the new extent still isn't right:

1. **Determine the real bounding box needed.** Get Johns Lake's actual
   shoreline geometry (NHD waterbody polygon, or the Orange County Water
   Atlas waterbody-7935 boundary) rather than guessing a bigger square — fit
   a box around the real shoreline plus a buffer (e.g. +200–300m for
   watershed context, matching the rationale for the current AOI's buffer).
2. **Re-download DEM only** at the new extent (cheap, fast) and build a
   **preliminary lake mask from NHD alone** (skip the OWM majority-vote step
   for now) just to confirm the new box visually contains the whole lake with
   no edge-clipping, the same edge-check method already used to find this
   problem (`mask[0,:].sum()`, etc. on the four edges — should now show ~0 on
   all four edges, or just open water/wetland fringe, not abrupt clipping).
3. **Only after step 2 confirms the extent is right**, proceed to re-download
   Sentinel-2 and re-run the full 4-method segmentation pipeline (step 3
   above) — this is the point of no easy return, so don't start it on an
   unverified extent.
4. Continue through steps 4–10 in order; re-run `verify_all.py` last to
   define new reference values for the dashboard.

## Decision point before starting

This is a multi-hour-to-multi-day effort depending on GPU availability and
how much of steps 3/8 can run unattended overnight. Recommend treating "start
Track G" as its own explicit go/no-go decision (with Lance/team lead) rather
than picking it up opportunistically between other tracks — the existing
144.47 ha figure and all scenario results downstream of it are usable today;
this fixes a real scope gap but isn't blocking anything else in the backlog
from proceeding in the meantime.

**Status (2026-06-18): cheap validation (steps 1-2) done and passed; the
expensive part (step 3 onward) is blocked pending team-lead discussion.**

Validation run (`ground_truth/track_G_validate_extent.py`, re-confirmed
2026-06-18 21:5x): fetched Johns Lake's real NHDPlus HR shoreline polygon
(**989.8 ha**, 10 parts — close to the 1,044 ha Water Atlas figure, small gap
likely NHD-vs-county-survey methodology, not a problem), fit a buffered
bounding box (**7.37 km E-W × 3.87 km N-S** — notably non-square; the lake is
elongated, unlike the current 2×2km square AOI), downloaded a test-only 3m
DEM at that extent (2023×2853 cells, `ground_truth/track_G_test_data/`, not
touching production data), and rasterized the NHD polygon onto it.
**Result: zero wet-pixel clipping on all 4 edges** (north/south/west/east all
0, vs. the current AOI's 502/450/127/5), with NHD mask area 1,012.7 ha on a
3,901.6 ha grid. **The new extent fully contains the lake — go for the
expensive part is now a real option, not a guess.**

Beyond the go/no-go on *when* to spend the multi-hour-to-multi-day effort for
steps 3 onward, there's a second open question on *what to re-run with*:
continue re-downloading/re-running the existing Sentinel-2 + 4-method
segmentation pipeline at this now-validated larger extent, or explore a
different/additional dataset for the expanded ROI (e.g. higher native
resolution than Sentinel-2's 10m, given the expanded area will include more
open water and shoreline where resolution matters more — see
`RESEARCH_FINDINGS.md` for the PlanetScope/drone-RTK analysis that's directly
relevant to this choice). Raise both questions together with Lance/team lead
before starting step 3 — the dataset choice changes which of steps 2–3 above
is even the right thing to re-run, but the bounding box itself (7.37×3.87km,
+250m buffer around the real NHD shoreline) is now validated and ready to use
regardless of which dataset decision follows.

## Files touched

Essentially the whole pipeline: `dem/`, `sentinel2/`, `soil/`,
`simulation/flood_sim.py`, `viewer/preprocess/export_*.py`, `verify_all.py`.
Recommend keeping the current AOI's outputs around (e.g. git tag or branch
before starting) since CLAUDE.md's existing "authoritative" numbers are
referenced throughout and a full AOI swap invalidates most of them at once —
useful to be able to diff old vs. new rather than just losing the old numbers.

## Verification

- New `lake_mask.tif` shows ~0 wet cells on all 4 AOI edges (vs. current
  502/450/127/5).
- New `lake_volume.csv` area is consistent with the ~1,044 ha documented full
  lake extent (allowing for the mask only capturing actual open water, not
  the full 2,580-acre figure which may include some wetland fringe).
- `python3 verify_all.py` passes with updated reference values.
- Re-run at least one simulation scenario end-to-end on the new domain and
  confirm wall time is still tractable (compare against the current table in
  CLAUDE.md's "Scenario results" section).
