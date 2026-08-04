# Small-area 3D droplet-based water-flow prototype — scoping plan

_2026-07-07, from team meeting feedback. This is a plan, not an implementation — the current
solver is a fundamentally different architecture (grid-based 2.5D), and this is a genuine
redesign, not a parameter tweak._

**Status update, same day**: v1 (built) used one Delaunay mesh from raw ground+building LiDAR
points combined — Lance (team lead) reviewed it and flagged that water was settling on rooftops
(a LiDAR-noise artifact), giving explicit correction: DEM = modeled ground (authoritative, less
noisy), LiDAR = trees/buildings sitting above it, and water should essentially never rest on a
raised LiDAR surface. Rebuilt as v2 (also built, same day) — see `lidar/droplet_flow_test.py`'s
docstring for the fused two-surface architecture. Confirmed Lance's "less noise on the ground"
prediction directly: mean ground-surface downhill magnitude dropped from 0.167 (raw LiDAR
points) to 0.042 (DEM) once the ground source changed. **Still open, not yet started**: Lance
separately noted LiDAR (once fused with vision-based surface classification — the DINOv2/SAM
work) could give a much more precise **Manning's roughness** estimate per surface than the
current single global `MANNING_N=0.040` in `flood_sim_ian.py` — a real, concrete future
direction once the segmentation model exists, not an immediate to-do.

## Why the current solver isn't what was asked for

`simulation/flood_sim_ian.py` is a **2.5D** solver: one scalar depth `h` per grid cell, water
moves between cells via horizontal fluxes (`qx`, `qy`) computed from the water-surface slope.
It cannot represent flow over a true 3D surface — water running down a roof, off an eave, along
a wall, or around a building at arbitrary angles — because it only ever has one height value per
(x,y) column. The meeting's ask (full 3D, gravity-driven flow shaped by real topography,
"take every droplet and compute where it goes next") is a **Lagrangian particle-tracing
approach on an actual 3D mesh**, not a grid Eulerian solver — a different method entirely.

## Proposed first step (the "dumb version," per the meeting notes)

1. **Pick a small test area with both a slope and a house** — the 5-house cluster already
   identified this session (lon=-81.4316, lat=28.3633, ~160m box, 19 buildings within 60m,
   `lidar_pointcloud_5houses.bin`) is a ready-made candidate; confirm it has meaningful slope
   (check `dem/data/terrain/slope_deg.tif` in that footprint) or pick a neighboring cluster that
   does.
2. **Build a real mesh from the LiDAR points in that area** — 2.5D Delaunay triangulation
   (`scipy.spatial.Delaunay`, same method already used for the 2 bridge-crossing meshes in
   `lidar/build_lidar_pointcloud.py`), but this time keeping the **first-return/all-points**
   surface (roofs, walls-as-steep-triangles, ground) rather than a single bare-earth value —
   this is what makes it "full 3D": the mesh has real roof/wall/ground triangles, not one
   height per column.
3. **Per-triangle gravity-driven flow direction**: for each mesh triangle, compute the downhill
   direction as the projection of gravity (0,0,-1) onto the triangle's plane, normalized — this
   is a standard, cheap computation (subtract the component of gravity along the surface
   normal). This is what "topography shapes the direction from gravity" means concretely.
4. **Droplet tracer**: seed N droplets (start with a **small number of droplets and a small
   total water volume** per the meeting notes — a demo-scale test, not a real storm), each
   droplet:
   - finds which triangle it's currently on (point-in-triangle test, or nearest-triangle by
     projected distance),
   - moves along that triangle's downhill direction at a fixed step size,
   - when it crosses a triangle edge, switches to the neighboring triangle and recomputes
     direction (this is the "two points connected, form a surface, test water flow" step from
     the meeting notes — walking a triangulated mesh via its edge-adjacency graph),
   - stops when it reaches a local sink (no downhill neighbor) — record that as a puddle/outlet.
   - This is **explicitly the "dumb version"**: no momentum, no depth field, no interaction
     between droplets, no infiltration — literally just "where does this droplet go next."
     Refining it (droplet volume/spreading, merging into a depth field, interception by
     roofs vs. ground) is follow-on work once the basic mesh-walk is proven to work at all.

## What this needs that doesn't exist yet

- A proper edge-adjacency structure over the Delaunay triangulation (which triangles share
  which edges) — `scipy.spatial.Delaunay` exposes `neighbors` per simplex, which gives this
  directly; no new library needed.
- A decision on droplet "volume" units for later visual/quantitative reporting — the meeting
  raised this (cubic centimeters?) but this first version doesn't need it: the "dumb version"
  only needs droplet **positions over time**, not volume/depth, to prove the concept.
- A renderer for droplet paths in the viewer (a `THREE.Line` per droplet trail, or animated
  `THREE.Points` — straightforward once the Python-side path data exists, following the same
  "compute in Python, export a small binary/JSON, load in JS" pattern used everywhere else in
  this project).

## Separately, but related: impervious-surface infiltration — done this session

The meeting's "classify hard materials so no infiltration" note is **already implemented** as
of this session, independent of the full-3D question — `simulation/flood_sim_ian.py`'s
`apply_impervious_mask()` now forces zero infiltration under every OSM-mapped road/building
polygon (24.1% of the domain), using vector data already on disk. This does **not** need the
segmentation model — that's for classifying imagery into surface types more generally
(driveways, parking lots, canopy vs. bare soil — anything not already an OSM-tagged
road/building), which is a separate, larger, not-yet-started piece of work.

## Recommendation

Build the small-area droplet prototype as a standalone script (`lidar/droplet_flow_test.py` or
similar) against the existing 5-house LiDAR extract, entirely separate from `flood_sim_ian.py` —
don't try to retrofit the grid solver into 3D. If the droplet-on-mesh approach proves out
visually, the next real design question is how (or whether) to reconcile it with the existing
grid-based Ian simulation, which is a bigger architectural decision to make once there's a
working prototype to look at, not before.
