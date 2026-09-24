"""Physical constants shared by the solver, the surface parameterisation and the viewer export.

Import-time side-effect free: no file reads, no network, no dependency on site data. Anything
needing site data belongs in `sites.SiteConfig`.
"""

from typing import Dict, Final

G: Final = 9.81
"""Acceleration due to gravity [m/s^2]."""

FT_TO_M: Final = 0.3048006096012192
"""US *survey* foot -> metre. USGS 3DEP LiDAR for Florida ships in State Plane survey feet;
the international foot (0.3048) differs by ~2 ppm, or ~3 mm over a 1.5 km domain."""

MANNING_EXP: Final = 7.0 / 3.0
"""Depth exponent in the Bates et al. (2010) semi-implicit friction denominator:

    q[t+1] = (q - g*hf*dt*d(eta)/dx) / (1 + g*dt*n^2*|q| / hf**MANNING_EXP)

Must be 7/3 because `q` is unit discharge [m^2/s], not velocity. Two derivations agree:
dimensional analysis leaves m**(7/3 - p) * s**0, dimensionless only at p = 7/3; and steady
state (q[t+1] == q[t]) gives q = hf**((p+1)/2) * sqrt(-S)/n, which matches Manning's
q = hf**(5/3) * sqrt(-S)/n only at p = 7/3.

A 4/3 shipped here for months. It over-predicts discharge by +216% at h = 0.10 m and +607% at
h = 0.02 m, and agrees only at h = 1 m where hf**0 = 1 -- which is why it survived review while
being worst in this project's own operating regime (median wet depth 7-8 cm). See
`tests/test_physics.py`, which fails if the exponent regresses.
"""

MIN_DEPTH: Final = 1e-4
"""Wet/dry threshold [m]. Below this a cell is dry and carries no flux."""

FROUDE_CAP: Final = 0.9
"""Upper bound on the Froude number at a face: |q| <= FROUDE_CAP * hf * sqrt(g * hf)."""

FLOODED_DEPTH_THR_M: Final = 0.05
"""Depth [m] at or above which a raster cell counts as flooded in extent products."""

IMPERVIOUS_FC_MM_HR: Final = 0.0
"""Horton final infiltration capacity [mm/hr] for roads and roofs."""

ROAD_BUFFER_M: Final[Dict[str, float]] = {
    "motorway": 16, "motorway_link": 12, "trunk": 14, "trunk_link": 10,
    "primary": 10, "primary_link": 8, "secondary": 8, "secondary_link": 6,
    "tertiary": 6, "tertiary_link": 5, "residential": 5, "unclassified": 5,
    "service": 3, "track": 3, "path": 2, "footway": 2, "pedestrian": 3,
    "proposed": 3, "construction": 3,
}
"""Half-width [m] to buffer an OSM road centreline by, per `highway` tag.

OSM roads are width-less centrelines, so a buffer is what turns them into an impervious
surface. The solver (zeroing infiltration under pavement) and the viewer overlay (drawing the
roads layer) must read the same table, or the map shows a footprint the physics never used.
"""

ROAD_BUFFER_DEFAULT_M: Final = 5.0
"""Fallback half-width [m] for an unrecognised `highway` value, matching "residential"."""


def road_buffer_m(highway: str) -> float:
    """Half-width [m] for an OSM `highway` tag, falling back to the residential default."""
    return ROAD_BUFFER_M.get(highway, ROAD_BUFFER_DEFAULT_M)
