"""floodtwin — shared library for the DeepEarth flood digital-twin models.

This package holds the code that is genuinely common to every flood digital twin in
`models/`, so that a twin for a new location is built by supplying a coordinate and a site
config rather than by copying a sibling project and editing constants.

Sub-modules
-----------
physics
    Physical constants and shared parameter tables used by every solver and by the viewer
    export path. Import-time side-effect free.

Consuming projects add `models/` to `sys.path` and import from here, e.g.

    import os, sys
    sys.path.insert(0, os.path.join(<repo>, "models"))
    from floodtwin.physics import G, MANNING_EXP, ROAD_BUFFER_M
"""

__all__ = ["physics"]
