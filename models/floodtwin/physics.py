"""Physical constants and shared parameter tables for the flood digital twins.

Every value here was previously duplicated across two or more modules. Centralising them
removes the risk that two copies of "the same" constant silently drift apart — which had
already happened once, between the solver's impervious-road mask and the viewer's road
overlay (see ROAD_BUFFER_M below).

This module must stay free of import-time side effects: no file reads, no network calls, no
dependency on any site's data being present on disk. Anything that needs site data belongs in
a site config, not here.

Deliberately NOT centralised
----------------------------
Some constants look shared but are legitimately per-solver or per-site, and unifying them
would change physics or reported results. These stay where they are used:

CFL_ALPHA
    0.3 in the CFX corridor solvers (conservative, tuned for steep road-embankment terrain)
    versus 0.5 in the Johns Lake solver. A real tuning difference, not an accident.

"flooded" / "wet" depth thresholds
    The raster solver reports flood EXTENT at 0.05 m; the fine-scale mesh solver reports WET
    cells at 0.01 m, because it resolves thin films on roofs and pavement that the raster
    solver cannot. Same idea, different question, different defensible value. See
    FLOODED_DEPTH_THR_M / WET_DEPTH_THR_M below for the documented reference values.

Manning's n
    Varies by land cover and by which surface is being modelled (ground versus roof), and in
    one solver is a per-class dict. It is a model parameter, not a constant.
"""

# ── Fundamental constants ────────────────────────────────────────────────────────────────

G = 9.81
"""Acceleration due to gravity [m/s^2]."""

FT_TO_M = 0.3048006096012192
"""US survey foot -> metre.

Note this is the US SURVEY foot, not the international foot (0.3048 exactly). USGS 3DEP LiDAR
for Florida is distributed in State Plane coordinates using the survey foot, so the two differ
by ~2 ppm — about 3 mm over a 1.5 km AOI. Small, but it is the correct factor for this data.
"""

MANNING_EXP = 7.0 / 3.0
"""Exponent on flow depth in the Bates et al. (2010) semi-implicit friction denominator:

    q^(n+1) = [q - g*hf*dt*d(eta)/dx] / [1 + g*dt*n^2*|q| / hf^MANNING_EXP]

MUST be 7/3 when q is UNIT DISCHARGE (m^2/s), which it is in every solver here — the depth
update divides the flux difference by dx, so q carries m^2/s rather than a velocity.

Two independent confirmations:

1. Dimensional analysis. [g][dt][n^2][q] / [hf^p] = m^(7/3 - p) * s^0, which is dimensionless
   only at p = 7/3.
2. Steady state must reduce to Manning's equation. Setting q^(n+1) = q^n gives
   q = hf^((p+1)/2) * sqrt(-S) / n; Manning is q = hf^(5/3) * sqrt(-S) / n, so (p+1)/2 = 5/3,
   giving p = 7/3.

Verified numerically by iterating the update to a fixed point on a uniform slope: p = 7/3
reproduces Manning to -0.00% at every depth tested (0.02-1.0 m), whereas p = 4/3 over-predicts
discharge by +216% at h = 0.10 m and +607% at h = 0.02 m. The two agree only at h = 1 m, where
hf^0 = 1, which is why an incorrect 4/3 can survive visual review.
"""

MIN_DEPTH = 1e-4
"""Wet/dry threshold [m]. Below this a cell is treated as dry and carries no flux."""


# ── Reference depth thresholds ───────────────────────────────────────────────────────────
# These are REFERENCE values, documented here so the two thresholds are visibly distinct and
# their difference is intentional. Solvers may still define their own; see the module
# docstring for why they are not forced to share one value.

FLOODED_DEPTH_THR_M = 0.05
"""Depth [m] at or above which a raster cell is reported as FLOODED in extent products."""

WET_DEPTH_THR_M = 0.01
"""Depth [m] at or above which a fine-scale mesh triangle is reported as WET.

Lower than FLOODED_DEPTH_THR_M because the mesh solver resolves thin films on roofs and
pavement that a 5 m raster cannot represent at all.
"""


# ── Impervious-surface parameters ────────────────────────────────────────────────────────

ROAD_BUFFER_M = {
    "motorway": 16, "motorway_link": 12, "trunk": 14, "trunk_link": 10,
    "primary": 10, "primary_link": 8, "secondary": 8, "secondary_link": 6,
    "tertiary": 6, "tertiary_link": 5, "residential": 5, "unclassified": 5,
    "service": 3, "track": 3, "path": 2, "footway": 2, "pedestrian": 3,
    "proposed": 3, "construction": 3,
}
"""Half-width [m] to buffer an OSM road centreline by, per `highway` tag.

OSM road geometry is a centreline with no width, so a buffer is required to turn it into a
real impervious surface. Widths approximate a typical carriageway for each class.

This table is shared by the SOLVER (which uses it to zero infiltration under paved surfaces)
and by the VIEWER export (which uses it to draw the roads overlay). They must agree: if they
do not, the map shows a different impervious footprint than the physics actually used.
"""

ROAD_BUFFER_DEFAULT_M = 5.0
"""Buffer half-width [m] for an OSM `highway` value not present in ROAD_BUFFER_M.

Matches "residential", the most common unclassified case. Previously the solver used 5.0 while
the viewer overlay used 4, so any unrecognised road class was masked and drawn at different
widths.
"""

IMPERVIOUS_FC_MM_HR = 0.0
"""Horton final infiltration capacity [mm/hr] for hard surfaces (roads, roofs): zero."""
