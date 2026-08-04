# Task handoff: USGS 3DHP Flow Network Derivatives — real independent cross-check for the watershed-area-mismatch problem

**Written 2026-07-27 by a parallel Claude Code session, handed off so the primary session can
stay focused on the GNN surrogate training work (see CLAUDE.md's 2026-07-27 "how do we
combine/scale the 3D mesh solver" entries for that other thread — don't touch
`simulation/train_mesh_gnn_site3.py`, `simulation/run_gnn_training_sweep.py`, or anything under
`site3_gee_creek/gnn_training/` while that's in flight; check for a running
`train_mesh_gnn_site3.py` / `run_gnn_training_sweep.py` process before starting, and don't kill
it).**

## Why this matters (read this first)

This project's single biggest unresolved validation problem, referenced repeatedly throughout
CLAUDE.md, is the **watershed-area mismatch**: this project's own D8-delineated catchments never
match the documented drainage area of the real USGS gauges being compared against.
- Original AOI vs. Shingle Creek gauge 02263800: AOI is **44x smaller** than the gauge's real
  231 km² drainage area.
- site3 (Gee Creek, gauge 02234400): D8 delineation recovers only **35%** (11.65 km² of
  33.15 km² documented) — the best fix found so far, but still not a perfect match.

Every real gauge comparison this project has done (the Ian shape/timing comparisons, today's
real site3-vs-Gee-Creek discharge comparison) has had to explicitly caveat this area mismatch.
**USGS 3DHP's Flow Network Derivative (FND) data might offer a real, independent way to
characterize the TRUE upstream network extent** at each gauge, without needing another D8
delineation attempt — worth checking directly rather than assumed either way.

## What's already been confirmed (don't re-derive this — verified today, 2026-07-27)

1. **3DHP's `Catchment` layer (drainage-area polygons) is EMPTY for this entire region** — not
   a fetch failure, a real national-rollout gap. Verified via direct REST query against
   `https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/80/query`
   for every site's real bounding box (original AOI, site1, site2, site3, site3_crop) — 0
   features everywhere. The service's own metadata confirms catchments are being "populated in
   the future" nationally and this part of Florida isn't covered yet. See CLAUDE.md's
   2026-07-27 update to Future-work item 9 for the full writeup — **don't re-check this, it's
   confirmed empty.**

2. **The 880MB CSV was downloaded and checked (2026-07-27, same session) — real, negative
   result: don't re-download it.** Per
   https://www.usgs.gov/3d-hydrography-program/3dhp-flow-network-derivatives, national bulk
   files are at `https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Hydrography/3DHP/temp_tables/`:
   - `3dhp_all_csv_FY26_FlowNetwork.zip` — **downloaded and inspected already** (see point 3
     below for why it turned out not useful). Extracts to `flownetwork.csv`, 1.68GB, 30,086,800
     rows, only 5 columns: `fromid3dhp, toid3dhp, downmain, upmain, globalid` — pure
     connectivity (a directed edge list of which reach flows into which), NOT the attribute
     data (arbolatesum/streamorder/etc.) the task actually needs. Confirmed 334 rows matching
     site3's own known reach IDs, so it IS the right geography, just the wrong content —
     everything useful here is already derivable from `uphydrosequence`/`dnhydrosequence` on
     the live Flowline REST layer (point 3), which also has the actual attributes this file
     lacks. Files are sitting in `hydrography/data/fnd/` (2.5GB total, zip + extracted CSV) —
     ask the user whether to delete before doing anything else with disk space.
   - `3dhp_all_gdb_FY26_FlowNetworkDerivatives.zip` — geodatabase, **~2.48GB**, NOT downloaded
     yet. Different name than the CSV ("FlowNetworkDerivatives" vs "FlowNetwork") — might
     contain more/different tables (a GDB can hold several), but given the CSV's negative
     result, **get explicit user permission before downloading this one too** — don't assume
     it's needed just because the CSV wasn't sufficient.
   - `Prob_flow_July2026.zip` — flow probability stats, **~1.2MB**, not downloaded, still
     untested — small enough to be low-risk if it seems relevant, but still ask first per the
     standing file-download rule.

3. **The FND-equivalent fields are ALREADY populated directly on the live Flowline REST layer**
   — no bulk download needed. Verified by querying
   `https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50/query`
   for site3's bbox with `outFields=id3dhp,streamorder,arbolatesum,hydrosequence,pathlength,levelpath,mainstemid`
   — got back **real, non-null values** for all 48 flowline segments in site3's box, e.g.:
   ```
   {'id3dhp': '7TEYH', 'streamorder': 3, 'arbolatesum': 50.211176470000005,
    'hydrosequence': 9719304, 'pathlength': 298.25089528, 'levelpath': 9719304,
    'mainstemid': 'https://geoconnex.us/ref/mainstems/2502587'}
   ```
   Contrast this with the OLDER already-downloaded `hydrography/data/3dhp_flowlines.geojson`
   (from `hydrography/fetch_3dhp.py`, which hits the OLDER `hydro.nationalmap.gov` MapServer
   endpoint) — same field names exist there but every value is `None`. **The newer
   `3dhp.nationalmap.gov` FeatureServer is a materially better data source than what
   `fetch_3dhp.py` currently uses** — this alone might be worth fixing regardless of the FND
   task specifically.

## The actual task

1. **Query the real flowline network attributes for the gauge-relevant reaches specifically**,
   not just the whole AOI box:
   - Original AOI / Shingle Creek: find the flowline(s) nearest gauge 02263800 (Shingle Creek
     at Airport, lat/lon in `infrastructure/data/` or look it up fresh via NWIS site service)
     and gauge 02263692 (Oak Ridge Rd, upstream). Get their `arbolatesum`, `streamorder`,
     `pathlength`, `mainstemid`.
   - site3 / Gee Creek: same for gauge 02234400 — its coordinates are already in
     `lidar/test_sites.py`'s `site3` entry (`gauge_lat=28.7041629, gauge_lon=-81.2906221`).
     You'll likely need to query a small box right around the gauge point itself (not the whole
     site3 box) to find the SPECIFIC reach at/near the gauge, since `arbolatesum` is a
     per-reach cumulative value that should be highest right at the pour point / gauge location
     (it accumulates length walking downstream).

2. **Real caveat to think through, don't gloss over**: `arbolatesum` is a LENGTH (km of total
   upstream flowline network), not an AREA (km²) — it is NOT a direct substitute for drainage
   area. Think about whether/how it's still useful:
   - As a purely qualitative cross-check: does the reach nearest 02263800 (231 km² documented
     drainage area) show a much larger `arbolatesum` than the reach nearest 02234400 (33.15 km²
     documented)? If the RATIO of arbolatesum values roughly tracks the ratio of documented
     drainage areas, that's a real, independent (non-D8) signal that the gauge-area figures
     this project has been using are being interpreted consistently — worth stating plainly if
     so, and equally worth stating plainly if it DOESN'T track.
   - Do NOT try to force a conversion formula from arbolatesum to area unless you find a
     real, cited hydrologic relationship for doing so (e.g. a real regional regression) — don't
     invent one.

3. **Write up the real finding either way** (positive or negative) in CLAUDE.md, following this
   project's own established convention: a dated, detailed entry under "Project status" at the
   top, stating what was checked, the real numbers found, and an honest interpretation — not
   just "done," and not overselling a weak signal as a strong validation.

4. **Optional, lower priority**: consider whether `hydrography/fetch_3dhp.py` should be
   repointed from the older `hydro.nationalmap.gov` MapServer to the newer
   `3dhp.nationalmap.gov` FeatureServer, given the newer one clearly has richer/populated data
   for this region. If you do this, follow the project's established pattern of monkey-patching/
   parameterizing rather than silently changing production behavior without flagging it, and
   re-verify the existing viewer's "Hydrography (3DHP)" layer still works after any change.

## Relevant existing code/conventions to reuse, not reinvent

- `hydrography/fetch_3dhp.py` — existing fetch script, uses the OLDER endpoint; has the
  `bbox_from_center()` helper (copied verbatim across this project) you'll want too.
- `lidar/test_sites.py` — site registry with real lat/lon/radius for every site; `site3`'s own
  entry already has the gauge's lat/lon and a long comment explaining the area-mismatch
  history — read that comment in full before starting, it has directly relevant prior context.
- CLAUDE.md's own "Hydrology chain & Hurricane Ian case study" section has the real documented
  gauge discharge numbers for 02263800/02263692 already looked up — reuse those, don't
  re-fetch NWIS site metadata unless you need something not already there.
- This project's data catalogue table convention (see CLAUDE.md's "Data catalogue" section) —
  if you produce a new output file, add a row there too, matching the existing table format.

## What NOT to do

- Don't touch `simulation/train_mesh_gnn_site3.py`, `simulation/run_gnn_training_sweep.py`,
  `simulation/run_sim_gpu` in `mesh_shallow_water.py`, or anything under
  `site3_gee_creek/gnn_training/` — that's the primary session's active work.
- Don't download the 880MB/2.48GB bulk FND files without asking the user first and stating the
  real size — and given section 3 above, you probably don't need them at all.
- Don't invent an arbolatesum-to-drainage-area conversion formula without a real citable source.
