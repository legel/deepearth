"""
Named small-area test sites for the fine-resolution water-flow work
(droplet_flow_test.py, mesh_shallow_water.py).

Two sites, deliberately different, so the same physics can be checked against two different
kinds of ground truth: site1 is pure slope+roofs (does water shed off buildings correctly?),
site2 adds a real water body (does water that runs off actually accumulate and raise a
measurable "lake level" the way a real stormwater retention pond would?).

site2 was chosen by directly querying hydrography/data/3dhp_waterbodies.geojson +
infrastructure/data/buildings.geojson for small (<1 ha) ponds near SR417 with a SMALL number
of nearby houses (not a whole subdivision) — see chat history 2026-07-20 for the search. The
box radius (0.196 km) is the smallest that fully contains the pond's own ~182x175m footprint
plus exactly 3 nearby houses (149-224m from the pond) without sweeping in the denser
subdivision that starts appearing past ~250m from this particular pond.
"""
import os

_PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SITES = {
    "site1": dict(
        label="5-house residential cluster (slope + roofs, no water body)",
        lat=28.363316587590315, lon=-81.4315738078251, radius_km=0.08,
    ),
    "site2": dict(
        label="Retention pond + 3 houses near SR417",
        lat=28.366330, lon=-81.434606, radius_km=0.196,
        # 3DHP waterbody id3dhp='LX9ZK', featuretype=3 ("Lake"), ~0.6-0.7ha, centroid matches
        # the site center above. Confirmed a genuine ~1.5m depression in dem_conditioned.tif
        # (flat ~23.7-23.8m interior = existing bare-earth-DTM water-surface fill, 25.25m mean
        # in the surrounding 10-40m ring) — a real, usable "bowl" for runoff to accumulate in.
        pond_id3dhp="LX9ZK",
    ),
    "site3": dict(
        label="Gauge-matched validation site — Gee Creek near Longwood (USGS 02234400)",
        lat=28.690514, lon=-81.287539, radius_km=2.99,
        # Added 2026-07-27, direct response to Lance's repeated "reconstruct a real flood" /
        # ground-truth ask. Unlike site1/site2, this is NOT a candidate erosion/planting test
        # landscape — it's a real USGS-gauged watershed picked SPECIFICALLY so this project's
        # own simulated outflow can be compared against a REAL observed discharge record, for
        # the first time without the ~44x watershed-area mismatch that made the original
        # 02263800 Shingle Creek gauge comparison invalid (see CLAUDE.md's 2026-07-23/24 entry).
        #
        # Selection process (see CLAUDE.md's 2026-07-26/27 entries for the full search): started
        # from Lance's own June 21 2026 kickoff email, which named (28.64627,-81.23725) on SR417
        # as an example of "sloping terrain with engineered drainage" — searched USGS NWIS for
        # small-drainage-area stream gauges near THAT coordinate specifically (not just near the
        # original AOI), filtered for ones with real Hurricane Ian-window discharge data (most
        # small FL gauges are either discontinued or lake/canal-chain structures whose real
        # contributing area can't be D8-traced — both checked and ruled out for closer
        # candidates). Gee Creek (02234400) is 8.3km from Lance's own flagged coordinate,
        # documented drainage area 12.80 sq mi (33.15 km^2).
        #
        # CORRECTED 2026-07-27: the "~9x Ian flood response (baseflow ~4 cfs -> peak 35.4 cfs
        # on 2022-09-30)" figure originally recorded here during site selection was WRONG —
        # re-fetched directly from USGS NWIS's actual instantaneous-values (IV) service
        # (site3_gee_creek/infrastructure/data/gee_creek_ian_discharge.csv, 15-min resolution,
        # Sep 26-Oct 2 2022) rather than trusting the earlier figure, and found the real
        # numbers are substantially different: baseline ~45.2 cfs (2022-09-27 19:30 UTC) ->
        # peak 1,190 cfs (2022-09-29 13:31 UTC) -- a ~26.3x flood response, not ~9x, and the
        # peak fell on Sep 29, not Sep 30. Root cause of the original error not diagnosed (most
        # likely a wrong query window or a daily-mean vs instantaneous-value mixup during the
        # original search) -- flagged here so this wrong figure doesn't propagate further.
        #
        # Real, honest caveat: this project's own pysheds D8 delineation (breach + flowdir +
        # accumulation, pour point snapped to the gauge's own coordinates) recovers only
        # 11.65 km^2 (35% of the documented 33.15 km^2) — confirmed NOT a box-size artifact
        # (the delineated catchment doesn't touch the DEM download box's edge). Most likely
        # explanation: central Florida's naturally depression-dominated flat terrain (isolated
        # wetlands/cypress domes that only connect to the channel network during high-water
        # events like Ian) — a well-documented general challenge for D8-based delineation in
        # this state, not a bug in this specific search. This site therefore validates against
        # ~35% of the gauge's official contributing area, not 100% — still a dramatically
        # better basis for comparison than the original ~44x-off Shingle Creek attempt, but not
        # a perfect 1:1 match either; keep that caveat attached to any simulated-vs-observed
        # comparison built from this site.
        gauge_site_no="02234400",
        gauge_lat=28.7041629, gauge_lon=-81.2906221,
        documented_drainage_area_km2=33.15,
        delineated_drainage_area_km2=11.65,
        pour_point_5070=(1434214.3, 736979.1),
        # Data-source path overrides — site3 lives in its own site3_gee_creek/ tree (37km from
        # the original AOI, a genuinely separate location), never the shared production data/
        # directories droplet_flow_test.py/mesh_shallow_water.py default to. Added 2026-07-27
        # alongside the matching optional-parameter support in both scripts (see their function
        # docstrings) — site1/site2 have no such keys, so .get(...) returns None for them and
        # both scripts fall back to their original module-level constants unchanged.
        dem_cond_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "dem", "data", "hydro",
                                    "dem_conditioned.tif"),
        buildings_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                     "buildings.geojson"),
        roads_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                 "roads.geojson"),
        mukey_map_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "mukey_map.tif"),
        mukey_legend_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                        "mukey_map_legend.csv"),
        soil_json_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "soil_parameters.json"),
        nlcd_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                "nlcd_impervious.tif"),
        # bbox_cache_dir: added 2026-07-27 after load_points_in_bbox() repeatedly got killed
        # reading site3's 25 large (up to 655MB) LAZ tiles in one uninterrupted call (see
        # lidar/cache_bbox_points.py's own docstring for the full story). When set, both
        # droplet_flow_test.py and mesh_shallow_water.py use the pre-cached, per-tile-filtered
        # points from this directory (built once via cache_bbox_points.py) instead of re-reading
        # the raw LAZ files directly. site1/site2 have no such key and are unaffected.
        bbox_cache_dir=os.path.join(_PROJ_DIR, "site3_gee_creek", "lidar", "data", "bbox_cache"),
        # ground_decimate: added 2026-07-27 — site3's 6x6km box has 46.78M valid cells at native
        # ~0.88m DEM resolution; a full Delaunay triangulation over all of them is neither
        # tractable in this environment nor actually useful (this project's own full-AOI Ian
        # solver already runs at 5m, not native resolution, for exactly this kind of larger
        # area). decimate=8 -> ~7m effective ground resolution, ~731K points -- a tractable
        # triangulation size, comparable to or finer than the Ian solver's own convention.
        ground_decimate=8,
        # roof_max_points: added 2026-07-27 — site3's 10,739 meshed buildings averaged ~7,477
        # roof triangles EACH (80.3M total) at full LiDAR point density, far more than a roof
        # shape needs and big enough to make OBJ export impractical (a verbose ASCII format).
        # 200 points/building -> roughly 400 triangles/building max, ~4.3M total: a real
        # reduction (~18x) while still capturing a moderately complex roof outline.
        roof_max_points=200,
    ),
    "site3_crop": dict(
        label="Site3 GNN-training crop — near Gee Creek pour point (USGS 02234400)",
        # Added 2026-07-27, direct response to the "how do we combine/scale the 3D mesh solver
        # with a full watershed" question. The 3D mesh solver (mesh_shallow_water.py) is the
        # real per-point physics Lance asked about, but is confirmed intractable at full site3
        # scale (2.5hr wall time for an 8-minute synthetic event, 0.02% boundary outflow — see
        # CLAUDE.md's 2026-07-27 "real Hurricane Ian event" entries). The fix is a learned GNN
        # surrogate trained on the mesh solver's own output — but that surrogate needs REAL
        # training data from site3's own domain, not reused from site1/site2 (different
        # location near the ORIGINAL AOI, different soil/land-cover — training there and
        # deploying at site3 would risk real, silent domain-shift error). This site is that
        # training-data source: a small crop centered directly ON site3's own pour point
        # (28.7038988, -81.2906429 -- distinct from the gauge's own coordinate,
        # gauge_lat/gauge_lon above, which is the USGS station location the D8 delineation was
        # snapped to; the pour point is the actual grid cell used for delineation) --
        # deliberately at the exact location most relevant to the eventual full-scale gauge
        # comparison, since water dynamics right at the outlet are what the final comparison
        # most needs the surrogate to get right.
        #
        # Radius chosen empirically (2026-07-27): checked real building density around the pour
        # point directly (not guessed) -- 0 buildings within 100m, 6 within 250m, 10 within
        # 300m -- comparable density to site1 (5 houses) / site2 (3 houses). Chose 250m (0.25km
        # radius, ~500m box) as a reasonable middle ground: enough to capture a handful of real
        # roof+ground interactions without being so large the mesh solver becomes slow again
        # (this needs to run MANY times for many synthetic training scenarios, unlike the other
        # sites' one-off runs). Confirmed the pour point sits ~2km from the nearest site3 DEM
        # edge in every direction, so this crop is nowhere near the download-box boundary.
        lat=28.703898850286457, lon=-81.29064288498651, radius_km=0.25,
        gauge_site_no="02234400",
        gauge_lat=28.7041629, gauge_lon=-81.2906221,
        documented_drainage_area_km2=33.15,
        delineated_drainage_area_km2=11.65,
        pour_point_5070=(1434214.3, 736979.1),
        # Reuses every one of site3's own already-fetched data sources (same soil, same DEM,
        # same roads/buildings, same LiDAR) -- this crop is a sub-region of site3's existing
        # 6x6km download, not a new AOI fetch. bbox_cache_dir points at the SAME cache
        # cache_bbox_points.py already built for full site3 -- load_cached_points() filters to
        # whatever lon/lat box is actually requested, so reusing it here just means this much
        # smaller query naturally returns a subset of the same cached tiles, no re-caching
        # needed.
        dem_cond_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "dem", "data", "hydro",
                                    "dem_conditioned.tif"),
        buildings_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                     "buildings.geojson"),
        roads_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                 "roads.geojson"),
        mukey_map_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "mukey_map.tif"),
        mukey_legend_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                        "mukey_map_legend.csv"),
        soil_json_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "soil_parameters.json"),
        nlcd_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                "nlcd_impervious.tif"),
        bbox_cache_dir=os.path.join(_PROJ_DIR, "site3_gee_creek", "lidar", "data", "bbox_cache"),
        # No ground_decimate/roof_max_points overrides -- at this crop's much smaller ~500m-box
        # scale (vs. full site3's 6x6km), native resolution should be tractable the same way it
        # already is for site1/site2, which also use no decimation. This resolution is right
        # for PHYSICS validation (confirmed: 8-scenario GPU sweep completed cleanly, mass
        # balance closing to ~0.000% on every run) -- but real, measured 2026-07-27: a single
        # MeshGraphKAN forward+backward pass on this mesh's 710,302 nodes took 134.7s, making
        # direct GNN TRAINING on it intractable (1200 samples x 20 epochs would take ~37 days).
        # See site3_crop_coarse below for the GNN-training-specific coarser version of this
        # exact same location.
    ),
    "site3_crop_coarse": dict(
        label="Site3 GNN-training crop (COARSE, for GNN training only) — same pour-point "
              "location as site3_crop, decimated to HydroGraphNet's own reference node-count "
              "scale (~4,787 nodes)",
        # Added 2026-07-27 after directly measuring that site3_crop's full-resolution mesh
        # (710,302 nodes) makes MeshGraphKAN training intractable (one forward+backward pass:
        # 134.7s: 1200 samples x 20 epochs would take ~37 days) -- the same reason the original
        # train_mesh_gnn.py proof-of-concept used a coarse ~30m grid (~5,000-6,000 nodes)
        # instead of the production 5m grid. This is the SAME real-world location as
        # site3_crop (identical lat/lon/radius -- same pour point, same soil/terrain domain,
        # so no new domain-shift risk is introduced), just decimated for a tractable GNN
        # training graph size. ground_decimate=10 -> ~654,085/100 ~ 6,541 ground triangles;
        # roof_max_points=20 -> a much smaller roof mesh per building. Total ~6,500-7,000
        # nodes, matching HydroGraphNet's own reference case (4,787 nodes) closely.
        lat=28.703898850286457, lon=-81.29064288498651, radius_km=0.25,
        gauge_site_no="02234400",
        gauge_lat=28.7041629, gauge_lon=-81.2906221,
        documented_drainage_area_km2=33.15,
        delineated_drainage_area_km2=11.65,
        pour_point_5070=(1434214.3, 736979.1),
        dem_cond_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "dem", "data", "hydro",
                                    "dem_conditioned.tif"),
        buildings_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                     "buildings.geojson"),
        roads_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                 "roads.geojson"),
        mukey_map_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "mukey_map.tif"),
        mukey_legend_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                        "mukey_map_legend.csv"),
        soil_json_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "soil_parameters.json"),
        nlcd_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                "nlcd_impervious.tif"),
        bbox_cache_dir=os.path.join(_PROJ_DIR, "site3_gee_creek", "lidar", "data", "bbox_cache"),
        ground_decimate=10,
        roof_max_points=20,
    ),
    "site3_1house": dict(
        label="Site3 single-house demo crop — falling rain + physics-driven flow, same "
              "convention as site1/site2's demo layers",
        # Added per direct request for a small, single-building demo area WITHIN site3 (site1/
        # site2 already have this kind of layer at the original AOI; site3 never had one, since
        # its own registered crops (site3_crop/site3_crop_coarse) were built for GNN TRAINING
        # DATA volume, not a single clean demo house). Found by directly querying site3's own
        # buildings.geojson for a truly isolated building (no neighbor within 80m, real house-
        # sized footprint 80-400 sq m, not a shed) -- same "check real density, don't guess"
        # method used to pick site3_crop's own location. This candidate: ~399 sq m footprint,
        # nearest other building >80m away.
        lat=28.701821422947734, lon=-81.26177949874166, radius_km=0.06,
        # Reuses site3's own already-fetched data (same DEM/soil/roads/buildings/LiDAR cache) --
        # a sub-region of the existing 6x6km site3 download, not a new AOI fetch.
        dem_cond_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "dem", "data", "hydro",
                                    "dem_conditioned.tif"),
        buildings_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                     "buildings.geojson"),
        roads_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "infrastructure", "data",
                                 "roads.geojson"),
        mukey_map_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "mukey_map.tif"),
        mukey_legend_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                        "mukey_map_legend.csv"),
        soil_json_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                     "soil_parameters.json"),
        nlcd_path=os.path.join(_PROJ_DIR, "site3_gee_creek", "soil", "data",
                                "nlcd_impervious.tif"),
        bbox_cache_dir=os.path.join(_PROJ_DIR, "site3_gee_creek", "lidar", "data", "bbox_cache"),
        # No ground_decimate/roof_max_points -- at this ~120m-box, single-building scale, native
        # resolution is comfortably tractable the same way it already is for site1/site2.
    ),
}


def get_site(name):
    if name not in SITES:
        raise ValueError(f"Unknown site {name!r} — choices are {list(SITES)}")
    return SITES[name]
