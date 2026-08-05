#!/usr/bin/env python3
"""ONE resolver every fetch script and solver uses to answer "where am I fetching, and where
does the output go?" — closing the reproducibility gap found in INTERNSHIP_AUDIT_2026-08-03.md §4.

The problem this fixes
----------------------
`lidar/test_sites.py` has been the site registry since 2026-07-20, but it was only ever read by
the LiDAR/mesh-solver code path (`droplet_flow_test.py`, `mesh_shallow_water.py`). Every
DEM/soil/imagery/roads/FEMA/hydrography fetch script instead carried its OWN module-level
`DEFAULT_LAT`/`DEFAULT_LON`/`DEFAULT_RADIUS` constants — the main AOI's coordinates, duplicated
verbatim across 8 separate files, with no connection to the registry at all.

That duplication is the direct cause of the audit's single clearest reproducibility gap: site3's
DEM/SSURGO/NLCD/precipitation data exists on disk but for months had no saved script that
produced it, because those fetches were run by hand-typing different `--lat/--lon/--radius_km`
values into scripts that had no idea a registry existed. Hand-typed coordinates can silently
drift from what the registry records; a resolved registry entry cannot.

What this module adds
---------------------
`--site <name>` on every fetch script. When passed, lat/lon/radius come from the registry (the
single source of truth) and the output directory is resolved to that site's own data tree.
When omitted, every script behaves EXACTLY as before — same defaults, same output paths — so
this is purely additive and no existing invocation or docstring becomes wrong.

The registry itself is NOT duplicated here. `SITES` is imported from `lidar/test_sites.py`,
which stays authoritative for the fine-scale sites and keeps all its existing selection-rationale
comments. This module only ADDS the two AOI-scale entries that were previously implicit in those
8 copies of the same three constants:

  main_aoi  — the CFX SR417 corridor 2x2km box (28.36687, -81.43299, r=1.0km)
  site3     — already in test_sites.py; its `data_root` is registered here so fetch scripts
              write into site3_gee_creek/ instead of overwriting the main AOI's files (the exact
              failure mode `fetch_naip_site3.py`'s own docstring warns about: both AOIs would
              otherwise share the filename `naip_2021_RGB.tif`)

Usage in a fetch script (3 lines, no other changes):

    import site_registry                                   # noqa: E402
    site_registry.add_site_arg(parser)                     # after building the parser
    args = site_registry.resolve(parser.parse_args())      # instead of parser.parse_args()

`resolve()` fills args.lat/lon/radius_km from the registry when --site is given, attaches
args.site_data_root, and refuses to silently accept a --site together with a conflicting
explicit --lat/--lon (which would reintroduce exactly the drift this module exists to prevent).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "lidar") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "lidar"))

from test_sites import SITES as _FINE_SITES  # noqa: E402  authoritative fine-scale registry

# The CFX main AOI. These are the same three numbers that were previously duplicated as
# DEFAULT_LAT/DEFAULT_LON/DEFAULT_RADIUS in dem_download.py, ssurgo_download.py, fetch_nlcd.py,
# fetch_naip.py, fetch_roads_buildings.py, fetch_fema_nfhl.py, fetch_3dhp.py and
# generate_cfx_corridor_kml.py. Those constants are deliberately LEFT IN PLACE so every existing
# no-flag invocation keeps working unchanged; this entry is what `--site main_aoi` resolves to,
# and is now the one place to change them.
_AOI_SITES = {
    "main_aoi": dict(
        label="CFX SR417 corridor test-landscape AOI (Lake Nona / south Orlando)",
        lat=28.36687, lon=-81.43299, radius_km=1.0,
        data_root=_HERE,
    ),
}

# Every fine-scale site in test_sites.py that has its OWN fetched data tree (rather than reusing
# a parent AOI's). site1/site2/site3_crop/site3_1house all sit inside a parent box and reuse its
# DEM/soil/imagery — fetching for them independently would be wrong, so they map to their parent.
_DATA_ROOTS = {
    "main_aoi":          _HERE,
    "site1":             _HERE,                                        # inside main AOI
    "site2":             _HERE,                                        # inside main AOI
    "site3":             os.path.join(_HERE, "site3_gee_creek"),       # own tree
    "site3_crop":        os.path.join(_HERE, "site3_gee_creek"),       # inside site3
    "site3_crop_coarse": os.path.join(_HERE, "site3_gee_creek"),       # inside site3
    "site3_1house":      os.path.join(_HERE, "site3_gee_creek"),       # inside site3
}

# Sites that own their data tree — i.e. the ones it is meaningful to run a FETCH for. Running a
# fetch for site1 would just re-download the main AOI at a 80m radius and clobber it.
FETCHABLE = ("main_aoi", "site3")

SITES = {**_FINE_SITES, **_AOI_SITES}


def get_site(name):
    """Registry entry for `name`, with `data_root` guaranteed present."""
    if name not in SITES:
        raise ValueError(f"Unknown site {name!r} — choices are {sorted(SITES)}")
    entry = dict(SITES[name])
    entry.setdefault("data_root", _DATA_ROOTS.get(name, _HERE))
    entry["name"] = name
    return entry


def data_dir(name, category):
    """Absolute path to `<site data_root>/<category>/data`, created if missing.

    category is the existing per-category folder name already used throughout both projects
    ('dem', 'soil', 'imagery', 'infrastructure', 'floodplain', 'hydrography', 'precipitation').
    For main_aoi this returns exactly the paths the scripts already use, so nothing moves.
    """
    d = os.path.join(get_site(name)["data_root"], category, "data")
    os.makedirs(d, exist_ok=True)
    return d


def add_site_arg(parser, fetchable_only=True):
    """Add `--site` to an existing argparse parser. Call after the --lat/--lon args are added."""
    choices = list(FETCHABLE) if fetchable_only else sorted(SITES)
    parser.add_argument(
        "--site", default=None, choices=choices,
        help="Resolve lat/lon/radius_km AND the output directory from the shared site registry "
             "(site_registry.py) instead of hand-typed coordinates. This is the reproducible "
             "path — prefer it over --lat/--lon. Omit to keep the script's own legacy defaults.")
    return parser


def resolve(args, category=None):
    """Fill args.lat/lon/radius_km + args.site_data_root from --site, if given.

    Returns the same namespace (mutated) so it can wrap parse_args() directly. If --site was NOT
    passed, args is returned untouched and args.site_data_root is None — every existing caller
    keeps its current behaviour exactly.

    Raises if --site is combined with an explicit --lat/--lon that disagrees with the registry:
    silently letting a hand-typed coordinate win over the registry is precisely the drift this
    module exists to prevent, so it fails loudly instead.
    """
    args.site_data_root = None
    name = getattr(args, "site", None)
    if not name:
        return args

    entry = get_site(name)

    # Detect an explicit --lat/--lon that contradicts the registry. argparse gives us no direct
    # "was this flag passed?" signal, so compare against sys.argv, which is unambiguous.
    for flag, key in (("--lat", "lat"), ("--lon", "lon"), ("--radius_km", "radius_km")):
        if flag in sys.argv and hasattr(args, key):
            given, registered = float(getattr(args, key)), float(entry[key])
            if abs(given - registered) > 1e-9:
                raise SystemExit(
                    f"\n  Refusing to run: --site {name} says {key}={registered}, but {flag} "
                    f"{given} was also passed.\n"
                    f"  Pick one. If the registry value is wrong, fix it in lidar/test_sites.py "
                    f"(or site_registry.py for AOI-scale sites)\n"
                    f"  so the correction is recorded once and every script picks it up — do not "
                    f"override it per-invocation.\n")

    for key in ("lat", "lon", "radius_km"):
        if hasattr(args, key):
            setattr(args, key, entry[key])

    args.site_data_root = entry["data_root"]
    if category:
        args.site_data_dir = data_dir(name, category)

    print(f"  [site_registry] --site {name}: {entry['label']}")
    print(f"  [site_registry] lat={entry['lat']}  lon={entry['lon']}  "
          f"radius_km={entry['radius_km']}  data_root={os.path.relpath(entry['data_root'], _HERE) or '.'}")
    return args


if __name__ == "__main__":
    print(f"{'site':<20}{'lat':>13}{'lon':>13}{'radius_km':>11}  {'fetchable':<10} data_root")
    for n in sorted(SITES):
        e = get_site(n)
        rel = os.path.relpath(e["data_root"], _HERE)
        print(f"{n:<20}{e['lat']:>13.6f}{e['lon']:>13.6f}{e['radius_km']:>11.3f}  "
              f"{'yes' if n in FETCHABLE else 'no (reuses parent)':<10} {rel}")
