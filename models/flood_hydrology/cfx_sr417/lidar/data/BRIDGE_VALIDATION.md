# SR417 bridge-crossing DEM validation (2026-07-07)

_Prompted by a direct question: "is the highway DEM elevated but cut off at 2 intersections?"
Answer: yes — confirmed, quantified, and now corrected with actual raw-LiDAR-derived meshes._

## The question

SR417 (Central Florida GreeneWay) is a limited-access toll expressway — it should cross any
surface street via a grade-separated overpass, not an at-grade intersection. Within this
project's 2×2 km AOI, SR417 crosses exactly two real surface streets: **Town Loop Boulevard**
(~28.3666°N, -81.4329°W) and **John Young Parkway/CR423** (~28.3727°N, -81.4258°W) — found by
intersecting the OSM-derived `infrastructure/data/roads.geojson` SR417 geometry against every
other road in the AOI.

## What the existing bare-earth DEM shows

Sampling `dem/data/hydro/dem_conditioned.tif` along the SR417 centerline through both crossings
(±150m at 10m steps) shows the highway's elevation:
- Riding a steady, gently-graded plateau (~33-35m) for 100+ meters on either side of each
  crossing (its embankment), then
- **Dropping ~7.5-8.4m to grade level in a ~40m-wide notch exactly at the crossing**, then
  jumping straight back up to the plateau.

A real overpass does not do this. This is the textbook signature of a bare-earth DTM: USGS's
standard 3DEP DEM product classifies bridge-deck LiDAR returns as non-ground and excludes them
from the bare-earth surface, so at a bridge the DTM shows whatever ground is visible
**underneath** the structure (here, the cross-street's own grade) instead of the highway.

## Confirmation via raw point cloud

Downloaded the raw LAS/LAZ point cloud for the AOI (6 tiles, ~1.57 GB, USGS TNM API,
`FL_Peninsular_FDEM_2018_D19_DRRA` project — the same 2018 acquisition already confirmed as this
project's DEM source). **70,903,812 points** fall inside the AOI. Classification histogram:

| Class | Points | % |
|---|---|---|
| unclassified | 44,077,001 | 62.16% |
| ground | 22,605,396 | 31.88% |
| building | 3,774,751 | 5.32% |
| low point noise | 219,531 | 0.31% |
| water | 129,241 | 0.18% |
| **bridge deck (ASPRS class 17)** | **53,855** | **0.08%** |
| high noise | 44 | 0.00% |
| class 20 (reserved) | 43,993 | 0.06% |

Bridge-deck points are genuinely present in the source data — the DEM's bare-earth
classification correctly excludes them per its own definition, it just means "bare-earth" isn't
the right surface to represent the highway itself.

Built a first-return/all-points DSM (max elevation per 1m cell — captures whatever is physically
highest: bridge deck, canopy, roofline) and re-sampled the same SR417 centerline profile:

**Town Loop Boulevard** — max **+7.57 m** DSM-over-DEM difference, right at the crossing (DSM
holds steady ~34.0-34.1m through the whole profile; DEM drops to 26.5m for ~40m).

**John Young Parkway/CR423** — max **+8.40 m** DSM-over-DEM difference (DSM holds steady
~35.3-35.5m; DEM drops to ~27m for ~40m).

Both confirmed: the point-cloud DSM shows a continuous, smoothly-graded elevated roadway through
both crossings, with no dip — exactly what a real limited-access overpass should look like.

## What was built

- `lidar/build_lidar_pointcloud.py` — downloads the raw point cloud (TNM API direct LAZ tile
  download, not PDAL/EPT streaming — PDAL isn't installable in this environment without
  brew/conda; used `laspy`+`lazrs` instead, both pure-pip), filters to the AOI, computes the
  classification histogram, rasterizes the DSM, profiles both crossings, and builds a focused
  2.5D Delaunay TIN mesh (via `scipy.spatial.Delaunay`, per the meshing-method research in
  `RESEARCH_FINDINGS_2026-07-06.md`) of each crossing's immediate ~120m neighborhood.
- `lidar/data/bridge_mesh_town_loop_blvd.obj` (125,195 points, 232,217 triangles) and
  `bridge_mesh_john_young_pkwy.obj` (239,740 points, 360,138 triangles) — real, raw-LiDAR-derived
  bridge surfaces, exported directly in the viewer's own scene-space coordinate convention
  (matching `terrain.js`'s `VERT_EXAG`/`z_min`/origin transform) so they drop into the Three.js
  scene with zero extra positioning math.
- `lidar/data/lidar_dsm_1m.tif`, `classification_histogram.json`,
  `bridge_crossing_validation.json` (full offset-by-offset profile data for both crossings).
- Wired into the viewer as a new **"LiDAR Bridge Correction"** BASE LAYERS toggle
  (`viewer/static/js/lidarBridges.js`, `viewer/preprocess/export_lidar.py`) — off by default,
  bright red-orange material to stand out against the terrain.

## CRS note (worth remembering)

The LAZ files carry no machine-readable CRS VLR (`laspy`'s `parse_crs()` returns `None`).
Coordinate magnitudes (~5.1e5 easting, ~1.47e6 northing) initially looked like they might match
EPSG:6437, but that transform landed far outside the tile bounds. **EPSG:2881 (NAD83 / Florida
East, US survey feet)** is the correct CRS — confirmed by transforming the AOI center through it
and checking the result falls inside a tile's own header `mins`/`maxs`. If future point-cloud
work touches this same acquisition, use EPSG:2881, not EPSG:6437.

## Limitations / not done

- Only the two crossings were meshed in detail (~120m windows) — the rest of the AOI's
  bare-earth ground-only point-cloud surface already matches the existing DEM closely (spot
  checks agreed within centimeters), so a full-AOI point-cloud mesh would be redundant with the
  existing 1m DEM outside these two bridge locations.
- The bridge meshes are additive/visual only — they have not been blended back into
  `dem_conditioned.tif` or the hydrology-conditioned DEM used by the flood simulation. If the
  Ian simulation or any future erosion/drainage model needs the highway's true elevated geometry
  (e.g. to correctly route stormwater around/under the embankment), that DEM-patching step is
  still open.
- PDAL was not installed (no brew/conda in this environment); the pure-pip `laspy`+`scipy` path
  worked fine for this AOI's scale but would need PDAL for genuinely large-scale point-cloud
  work (e.g. if this pipeline is later extended to the full ~100-mile CFX corridor).
