# Research Findings — 2026-07-06

Scoping pass on four harder items from Lance's 2026-06-29 meeting (Tasks 4, 5, 7, 8), ahead
of the 2026-07-07 follow-up. Research/scoping only — no code written. Confirmed by a prior
pass that this repo and `models/flood_hydrology` currently have **zero** code for DINOv2, SAM,
`.las`/`.laz` handling, or PDAL.

---

## 1. DINOv2 (Task 4) — ArcGIS pretrained model

Source: https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-dinov2.htm

**What it does:** This is **not** a road/building/tree classifier. Esri's ArcGIS DINOv2 is a
general-purpose **feature-embedding backbone** — it "generates high-quality feature
representations from RGB imagery" for reuse in downstream tasks (similarity search,
clustering, semantic analysis, fine-tuning). The doc page explicitly contrasts it with
task-specific models: *"Unlike task-specific models that directly produce classifications or
detections, DINOv2 focuses on learning rich feature representations that can be reused across
a variety of geospatial applications."* There are no baked-in classes at all — not roads, not
buildings, not trees.

**Input spec:** 3-band RGB imagery. No documented resolution requirement, so NAIP's native
0.6–1.0 m would be usable as input, but the model produces embeddings, not a segmentation map,
so resolution compatibility is moot without a downstream head trained on those embeddings.

**License/cost:** The page doesn't state pricing directly, but distribution is via ArcGIS
Living Atlas as a `.dlpk` package "for use directly in ArcGIS Pro" — i.e., it's built for the
Esri ecosystem (ArcGIS Pro / ArcGIS Online credits for hosted inferencing), not a
license-free standalone download. No ONNX/PyTorch export or standalone-Python-API path is
documented on this page.

**Ecosystem lock-in:** Effectively locked into ArcGIS Pro/Online as distributed. Esri does
publish separate, actually-task-specific pretrained models for related needs (Tree Point
Classification via PointCNN, Building Point Classification, a Road/Sidewalk/Crosswalk
classifier) — but those are different, separately-licensed ArcGIS models, not "DINOv2 with
classes." We don't currently hold an ArcGIS Pro/Online license, so any of these paths cost
money we haven't budgeted.

**Fallback — open DINOv2 (Meta AI / HuggingFace `facebook/dinov2-*`):** Since ArcGIS's model
gives us no road/building/tree classes and has a licensing cost, the realistic path is: use the
raw open-source DINOv2 backbone (Apache 2.0, free) purely as a frozen feature extractor, and
train a lightweight segmentation head on top of it against an open labeled dataset. Two
realistic open datasets found for road/building/tree segmentation on aerial imagery comparable
to NAIP:

- **Inria Aerial Image Labeling Dataset** — 0.3 m RGB aerial tiles over 10 US/European cities
  (2.25 km²/tile), building footprint masks. Free, research-license.
- **LandCover.ai** — 0.25–0.5 m aerial imagery over Poland, pixel-labeled into buildings,
  woodlands (trees), water, and roads — the closest match to our exact 3-of-4 target classes
  in one dataset.
- (Also surfaced: DeepGlobe — building/road/water extraction on 2448×2448 satellite tiles;
  SpaceNet — building/road challenges; both viable secondary options.)

**Bottom line:** ArcGIS's DINOv2 is a paid, license-gated embedding model with no road/building/
tree classes out of the box — it does not meet the stated need as-is. Recommend skipping it and
instead fine-tuning the free `facebook/dinov2-*` backbone with a segmentation head trained on
LandCover.ai (best class match) and/or Inria (buildings) against our NAIP imagery once pulled.

---

## 2. SAM / SAM 2 (Task 5) — Segment Anything

Source: https://github.com/facebookresearch/segment-anything

**License:** Apache 2.0, confirmed directly from the repo: *"The model is licensed under the
Apache 2.0 license."* Free and unrestricted for our use.

**How it works:** SAM is **promptable segmentation**, not semantic classification — it takes a
point, box, or mask prompt and returns a high-quality object mask for that one prompted region
(or can auto-generate masks for everything in an image). This is architecturally complementary
to DINOv2 as described in the meeting: DINOv2 (or a head trained on it) flags "this region is
building," then SAM, prompted at that region, delineates the precise pixel-accurate boundary of
that individual building/tree/road segment. Trained on 11M images / 1.1B masks — very strong
zero-shot generalization, no fine-tuning needed for boundary delineation itself.

**Model sizes:** Three checkpoints, all downloadable free:
| Checkpoint | Params | File size |
|---|---|---|
| ViT-B | 91M | 375 MB |
| ViT-L | 308M | 1.25 GB |
| ViT-H (default) | 636M | 2.56 GB |

ViT-H gives the best accuracy but only marginal gains over ViT-L; ViT-B is much cheaper to run.
For a 2×2 km AOI at 0.6–1 m NAIP resolution (a few thousand pixels per side), ViT-L is a
reasonable default — single consumer/workstation GPU (8–16 GB VRAM) should suffice for
inference; exact official memory figures aren't published, but community reports place ViT-H
inference comfortably within a single modern GPU (not multi-GPU territory) for our tile sizes.

**Successor — SAM 2:** Confirmed to exist (Meta, 2024) — the original repo now points to it:
*"Please check out our new release on Segment Anything Model 2 (SAM 2)."* SAM 2 extends
promptable segmentation to video (real-time tracking across frames) in addition to images, and
is a strict superset of SAM 1's image capability with improved accuracy/speed. For our
still-imagery use case (NAIP tiles, PlanetScope scenes), SAM 2 is a drop-in upgrade with no
downside — recommend targeting SAM 2 directly rather than SAM 1.

**Bottom line:** SAM 2 (Apache 2.0, free) is the right tool for precise object-boundary
delineation once DINOv2 (or our fine-tuned head) flags candidate regions; use ViT-L checkpoint
as the accuracy/compute sweet spot.

---

## 3. USGS 3DEP LiDAR acquisition recency (Task 7)

Queried `https://tnmaccess.nationalmap.gov/api/v1/products` for the AOI bbox
(`-81.443,28.357,-81.423,28.377`) against both the LPC point-cloud dataset and the 1m DEM
dataset.

**Result — both point cloud and DEM for this AOI come from the same single project:**

- **Project name:** `FL_Peninsular_FDEM_2018_D19_DRRA`
- **Collection year:** **2018**
- **USGS product publication:** LPC tiles published 2021-04-20; the corresponding 1m DEM tile
  (`USGS 1 Meter 17x45y314 FL_Peninsular_FDEM_2018_D19_DRRA`) published 2023-02-10
- **Format:** LAZ (compressed LAS) point cloud tiles, 9 tiles cover our bbox
- **Context (via web search):** This is part of a joint USGS/Florida Division of Emergency
  Management (FDEM) "Florida Statewide Lidar" program — QL1 lidar acquired across ~35
  peninsular-Florida counties following Hurricanes Irma (2017) and Michael (2018), with final
  USGS-delivered products completed mid-2022. Our AOI's 1m DEM (`sr417_corridor_dem_1m.tif`,
  0.88 m in EPSG:5070) is this project rasterized server-side by py3dep — same source, just
  missing the acquisition metadata in our local file until now.
- **Older, superseded coverage found for comparison:** an earlier `FL_Osceola_2015` DEM tile
  also covers part of this bbox (published 2020-03-30, 2015 collection) — confirms 2018 is
  the newer of the two, not the older.
- **Nothing newer than 2018 exists yet** for this bbox — a broader search for 2022/2023
  Florida lidar recapture (post-Hurricane Ian) found ongoing statewide FDEM/3DEP lidar programs
  in general, but no re-flown acquisition specifically covering this AOI was found in TNM's
  product index as of this query.

**Bottom line:** Our existing "1m DEM" is derived from lidar flown in **2018** under project
`FL_Peninsular_FDEM_2018_D19_DRRA` (part of the statewide FDEM/3DEP post-Irma/Michael program);
this is confirmed to be the most current 3DEP coverage available for this AOI — no action
needed beyond recording the provenance in the data catalogue.

---

## 4. Raw LiDAR point cloud fetch + meshing method (Task 8)

### 4a. How to fetch the raw point cloud

Two viable USGS access paths, both pointing at the same underlying `FL_Peninsular_
FDEM_2018_D19_DRRA` data identified in Section 3:

1. **TNM API direct LAZ tile download** — the same `tnmaccess.nationalmap.gov/api/v1/products`
   query used above returns direct download URLs for the 9 LAZ tiles covering the bbox; this is
   the simplest path since we already have the working query.
2. **USGS 3DEP Entwine Point Tile (EPT) service** (`https://usgs.entwine.io`, maintained by
   Hobu Inc. + USGS, hosted as an AWS Public Dataset — GitHub: `hobuinc/usgs-lidar`). This
   indexes 10+ trillion lidar points across 950+ resources as streamable EPT/LAZ octrees. Rather
   than downloading whole tiles, `PDAL`'s `readers.ept` can stream **only the points inside our
   AOI polygon/bbox** directly from the S3-hosted EPT resource, which is the more efficient
   approach for a 2×2 km AOI carved out of a much larger statewide dataset. (Note: the page
   itself returned HTTP 403 to a plain fetch during this research — it appears to require a
   browser/viewer context or JS-rendered index rather than serving flat directory listings; the
   GitHub repo `hobuinc/usgs-lidar` is a reliable fallback path to find the correct EPT resource
   ID for Florida.)

Recommend PDAL + EPT streaming (path 2) over bulk LAZ tile download (path 1) once actual
point-cloud work starts, since it avoids pulling full statewide tiles for a small AOI.

### 4b. Meshing method recommendation

Lance's framing ("nearby points connected") describes triangulated surface meshing generically;
the four standard candidates and how they fit our terrain (2×2 km, mostly flat flatwoods with
localized road-embankment slopes up to 68°):

| Method | Fit for this AOI |
|---|---|
| **Delaunay triangulation (2.5D TIN)** | Best fit. Standard, deterministic, fast at this AOI size; because it's a 2.5D height-field triangulation (one z per x,y), it naturally preserves sharp embankment edges/slope breaks without over-smoothing — as long as the point density is high enough to resolve the embankment geometry (2018 QL1 lidar is dense enough for this). |
| **Greedy projection triangulation** | Reasonable fallback for genuinely non-2.5D geometry (overhangs, verticals) — not really needed here since embankments, even at 68°, are still single-valued height surfaces, not overhangs. Adds complexity for no benefit at this AOI. |
| **Ball-pivoting** | Designed for noisy/incomplete/multi-view point clouds (e.g., photogrammetry with gaps) — unnecessary overhead for clean, single-source airborne lidar with regular ground coverage. |
| **Poisson surface reconstruction** | Explicitly the wrong tool here — it's a smoothing, watertight-solid reconstruction method built for noisy object scans; it would blur exactly the sharp embankment edges/slope breaks we most want to preserve. Avoid. |

**Recommendation:** 2.5D Delaunay triangulation (i.e., a standard TIN), same category of
algorithm already implicit in how the DEM raster itself was derived, just applied directly to
the point cloud instead of a resampled grid if/when finer embankment detail than the current 1m
grid is needed. Only fall back to greedy projection if a specific area turns out to have true
overhangs (unlikely for a road-embankment corridor).

**Open-source tooling:** **PDAL** is the natural choice — it can both stream the EPT point
cloud (Section 4a) and run Delaunay-based meshing/DEM generation (`filters.delaunay`,
`writers.gdal`) in one pipeline, and it's already the natural counterpart to
`richdem`/`pysheds`/`py3dep`, which this project already depends on. **Open3D** is a good
secondary option specifically for the meshing step (`open3d.geometry.TriangleMesh` from a point
cloud, ball-pivoting/Poisson available if ever needed) with a friendlier Python API for
visualization/QA. **CGAL** is the most rigorous/industrial-strength option for the Delaunay
triangulation itself but is a heavier C++-first dependency with a less ergonomic Python binding
— reasonable if performance at larger scale becomes a concern, not necessary at 2×2 km.
None of PDAL/Open3D/CGAL are currently installed or used anywhere in this project.

**Bottom line:** Fetch via PDAL + USGS EPT streaming (or direct TNM LAZ tiles as a simpler
first step), then mesh with 2.5D Delaunay triangulation (TIN) via PDAL — it's the standard,
edge-preserving choice for single-valued airborne lidar and specifically avoids the
over-smoothing risk that Poisson reconstruction would introduce on the embankment slopes.

---

## Summary for the meeting

| Task | Verdict |
|---|---|
| 4. DINOv2 (ArcGIS) | Doesn't cover our classes, is Esri-licensed/paid. Use free `facebook/dinov2-*` + LandCover.ai/Inria fine-tuning instead. |
| 5. SAM | Apache 2.0, free. Use **SAM 2** (not SAM 1) with ViT-L checkpoint, as the boundary-delineation complement to a DINOv2-based classifier. |
| 7. LiDAR recency | Our existing 1m DEM = 2018 lidar, project `FL_Peninsular_FDEM_2018_D19_DRRA` (FDEM/3DEP statewide post-Irma/Michael program). Confirmed current — nothing newer exists yet for this AOI. |
| 8. Raw point cloud + meshing | Fetch via PDAL streaming from USGS EPT (`usgs.entwine.io` / `hobuinc/usgs-lidar`) or direct TNM LAZ tiles; mesh with 2.5D Delaunay triangulation (TIN) via PDAL — avoids Poisson's over-smoothing of embankment edges. |
