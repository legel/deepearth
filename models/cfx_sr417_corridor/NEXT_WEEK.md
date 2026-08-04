# CFX SR417 Corridor — Next Week Tasks
_Logged 2026-06-29 from post-meeting notes (Lance / team review)_
_Status updated 2026-07-07 ahead of the follow-up team-lead meeting — see CLAUDE.md's
2026-07-06/07 status entry for full detail on each item below._

---

## Task 1 — Road and building vector layers — ✅ DONE
`infrastructure/fetch_roads_buildings.py` (OSM/Overpass): 646 road segments (101.0 km), 786
buildings (44.1 ha). Added to viewer as a "Roads & Buildings" toggle layer.

## Task 2 — Soil/landcover dataset for non-built surfaces — ✅ DONE
NLCD 2021 impervious surface ported (`soil/fetch_nlcd.py`; mean 28.7% impervious for this AOI).
gNATSGO (Lance's 2026-06-29 email) cross-checked: its `Valu1` table is not SDA-queryable
(bulk file-geodatabase only); substituted the SDA `muaggatt` table instead, which **validates
our existing SSURGO-derived HSG/CN methodology exactly** for all 8 real soil map units. See
`soil/data/gnatsgo_comparison.md`. Also fixed a real bug this check surfaced: the "Water" map
unit was defaulting to HSG-B/nonzero infiltration instead of the project's own zero-runoff
convention.

## Task 3 — NAIP imagery — ✅ DONE
`imagery/fetch_naip.py` (Microsoft Planetary Computer STAC). NAIP 2021 (0.6m, 100% AOI
coverage) added to viewer as a BASE LAYERS toggle.

## Task 4 — DINOv2 land classification model evaluation — 🔶 RESEARCHED, not built
The ArcGIS pretrained model Lance pointed at is a generic feature-embedding backbone — no
road/building/tree classes, ArcGIS-ecosystem-locked, no path to use it standalone. Recommended
fallback: fine-tune open DINOv2 (Meta/HuggingFace) on LandCover.ai or Inria Aerial Image
Labeling. No fine-tuning pipeline built yet. Full detail: `RESEARCH_FINDINGS_2026-07-06.md`.

## Task 5 — Segment Anything Model (SAM) evaluation — 🔶 RESEARCHED, not built
Apache 2.0/free. **SAM 2** (not SAM 1) is the better target — same license, adds video, strict
superset. Complementary to DINOv2 as originally framed (classify region → delineate individual
object). No inference pipeline built yet.

## Task 6 — Fine-scale water depth analysis at house and road scale — ✅ DONE (nuanced finding)
Ran the Ian sim at 5m/2m/native ~0.88m. Coarser grids genuinely under-report depth (5m smooths
real micro-topography), but the native-resolution "peak depth" (2.56m) is a 1–2 cell solver
pit-trapping artifact, not a real feature — confirmed by inspecting raw cell neighborhoods. The
trustworthy cross-resolution statistic is the wet-cell percentile depth (median ~7-8cm, p90
~14cm, p99 ~35-42cm). **Follow-up need surfaced**: the solver needs a pit-filling/sub-grid
storage correction before native-resolution absolute peaks can be trusted. Full detail:
`simulation/outputs/RESOLUTION_ANALYSIS.md`.

## Task 7 — Most recent USGS 3DEP LiDAR acquisition — ✅ DONE
Confirmed via USGS TNM API: our existing "1m" DEM traces to acquisition project
`FL_Peninsular_FDEM_2018_D19_DRRA` (2018, statewide FDEM/3DEP QL1 post-Irma/Michael program).
Nothing newer covers this AOI yet.

## Task 8 — Mesh generation from LiDAR point cloud — ✅ DONE (for the 2 bridge crossings)
`lidar/build_lidar_pointcloud.py` downloaded the raw point cloud (6 LAZ tiles, ~1.57 GB, USGS
TNM API direct download — PDAL wasn't installable in this environment without brew/conda, used
`laspy`+`scipy.spatial.Delaunay` instead) and built real 2.5D Delaunay TIN meshes of the two
SR417 bridge crossings. This also directly answered a follow-up question raised after reviewing
the DEM: **the bare-earth DEM was found to incorrectly drop SR417 ~7.5-8.4m to grade level at
both crossings** (bridge-deck LiDAR returns get classified non-ground and stripped from a
bare-earth DTM) — confirmed via the raw point cloud's classification codes (53,855 bridge-deck
points present) and a DSM-vs-DEM profile comparison. Full write-up:
`lidar/data/BRIDGE_VALIDATION.md`. Rest of the AOI's point cloud not remeshed (already matches
the existing DEM outside the two bridges).

## Task 9 — Rename Terrain → Topography; add LiDAR layer to viewer — ✅ DONE
Layer-panel section renamed to "TOPOGRAPHY" (`layerControls.js`). Raw-LiDAR viewer layer added:
"LiDAR Bridge Correction" (`lidarBridges.js`, the two bridge-crossing meshes from Task 8).

## Task 10 — Generalized water-on-any-surface simulation — 🔶 PARTIAL
The solver now uses spatially-varying per-cell SSURGO Horton infiltration instead of a single
domain-wide mean (a real step toward per-surface differentiation, using data — `mukey_map.tif`
— that already existed but was unused). Full generalization to roads/roofs/canopy as distinct
hydraulic surfaces is still blocked on data that doesn't exist yet: building *heights* (Task 1's
footprints are 2D only) and a canopy-height model (needs Task 8's point cloud).

---

_Source: meeting notes 2026-06-29. 8 of 10 tasks fully or substantively done (as of 2026-07-07,
after the raw LiDAR point-cloud work); the remaining 2 (DINOv2/SAM builds, full Task 10
generalization) have concrete next steps scoped but need another work session — see CLAUDE.md
Future-work items 16 and 18._
