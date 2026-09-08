"""Site and storm registry.

This is the ONLY source of coordinates, gauge metadata and file locations in the package.
Every path a stage reads or writes is derived from `SiteConfig.root`, so adding a site is a
dict entry rather than a directory tree plus a set of wrapper scripts.

The predecessor registry was opt-in: modules kept their own `DEFAULT_LAT`/`DEFAULT_LON` so that
existing no-flag invocations would not change meaning, which left 42 coordinate literals across
18 files alongside a registry that was supposed to have replaced them. Here there is no default
site and no module-level coordinate anywhere; a stage cannot run without being told which site
it is running for.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parent

KM_PER_DEG_LAT = 111.0
"""Kilometres per degree of latitude, as used by every dataset fetch in this pipeline.

A spherical-Earth value would be 111.195. The 0.18 % difference is deliberate and load-bearing:
this constant fixes the bounding box that every raster was downloaded on, so changing it shifts
the DEM grid and silently invalidates the conditioned terrain, the delineation and the gauge
validation built on them. Fix the geodesy only alongside a full re-fetch and re-validation.
"""


@dataclass(frozen=True)
class Gauge:
    """A USGS streamgauge used as observed truth.

    Attributes:
        site_no: NWIS site number.
        lat: Station latitude, WGS84.
        lon: Station longitude, WGS84.
        documented_area_km2: Drainage area USGS publishes for the station.
        delineated_area_km2: Area this pipeline's own D8 delineation recovers. The gap is
            real and must travel with every comparison: central Florida's depression-dominated
            flat terrain only connects isolated wetlands to the channel network during
            high-water events, so D8 under-captures. It is not a box-size artifact -- the
            delineated catchment does not touch the DEM download boundary.

            An 11.65 km2 figure appears in older write-ups. It predates the stream-burn and
            accumulation-threshold fixes and is superseded. Quote the smaller number -- the model
            is scored against ~11 % of the gauge's real contributing area, not ~35 %. The value
            here is what `cli.py terrain` produces from a freshly fetched 3DEP DEM, recorded in
            `docs/terrain_site3.json`; a 3.71 figure in older notes is the same measurement
            rounded from a slightly different accumulation threshold.
        baseflow_cfs: Pre-storm baseflow, subtracted when separating storm runoff.
    """

    site_no: str
    lat: float
    lon: float
    documented_area_km2: float
    delineated_area_km2: float
    baseflow_cfs: float

    @property
    def capture_fraction(self) -> float:
        """Delineated area as a fraction of the documented area."""
        return self.delineated_area_km2 / self.documented_area_km2


@dataclass(frozen=True)
class Storm:
    """A real rainfall event with an observed discharge record to score against.

    Attributes:
        name: Registry key.
        label: Human-readable name.
        start: Inclusive UTC start of the hyetograph window, "YYYY-MM-DD HH:MM".
        end: Inclusive UTC end of the hyetograph window.
        gauge_start: Start of the NWIS record to fetch, usually wider than the sim window.
        gauge_end: End of the NWIS record to fetch.
    """

    name: str
    label: str
    start: str
    end: str
    gauge_start: str
    gauge_end: str


@dataclass(frozen=True)
class SiteConfig:
    """One modelling domain: where it is, what forces it, and where its files live.

    Attributes:
        name: Registry key; also the data and output subdirectory name.
        label: Human-readable description.
        lat: Domain centre latitude, WGS84.
        lon: Domain centre longitude, WGS84.
        radius_km: Half-width of the square WGS84 box the fetch stage REQUESTS. The delivered
            grid is larger: 3DEP serves in EPSG:5070, where a lat/lon box acquires ~8.9 deg of
            Albers convergence at this longitude, and its axis-aligned bounds are ~14 % wider
            per side, plus a service buffer. site3's 2.99 km half-width becomes a 272x274 grid
            at 25 m -- 46.6 km2 modelled against the 35.8 km2 this number implies. 5070 is
            equal-area, so these are true metres and every area downstream is computed from
            the delivered grid, never from this value.
        asos_station: IEM ASOS identifier supplying the observed hyetograph.
        cell_size_m: Production solver resolution.
        cfl_alpha: CFL safety factor. 0.15, reduced from 0.30 after a measured -517.8 % mass
            residual with 8.99 m depths oscillating under zero rain at 5 m.
        gauge: Observed-discharge station, when the site has a valid one.
    """

    name: str
    label: str
    lat: float
    lon: float
    radius_km: float
    asos_station: str
    cell_size_m: float = 5.0
    cfl_alpha: float = 0.15
    gauge: Optional[Gauge] = None

    # ── Geometry ─────────────────────────────────────────────────────────────────────────

    def bbox(self) -> Tuple[float, float, float, float]:
        """Domain bounding box in WGS84 as (west, south, east, north)."""
        import math

        dlat = self.radius_km / KM_PER_DEG_LAT
        dlon = self.radius_km / (KM_PER_DEG_LAT * math.cos(math.radians(self.lat)))
        return (self.lon - dlon, self.lat - dlat, self.lon + dlon, self.lat + dlat)

    # ── Paths ────────────────────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Fetched-data root for this site. Everything under it is reproducible."""
        return _ROOT / "data" / self.name

    @property
    def out(self) -> Path:
        """Solver-output root for this site."""
        return _ROOT / "outputs" / self.name

    def path(self, *parts: str) -> Path:
        """A path under this site's data root, creating the parent directory."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def out_path(self, *parts: str) -> Path:
        """A path under this site's output root, creating the parent directory."""
        p = self.out.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Terrain
    @property
    def dem(self) -> Path:
        return self.path("dem", "dem.tif")

    @property
    def dem_burned(self) -> Path:
        return self.path("dem", "dem_burned.tif")

    @property
    def dem_conditioned(self) -> Path:
        return self.path("dem", "dem_conditioned.tif")

    @property
    def flow_accum(self) -> Path:
        return self.path("dem", "flow_accum.tif")

    @property
    def hand(self) -> Path:
        return self.path("dem", "hand.tif")

    @property
    def streams(self) -> Path:
        return self.path("dem", "stream_network.geojson")

    @property
    def watershed(self) -> Path:
        return self.path("dem", "watershed.geojson")

    # Soil and land cover
    @property
    def mukey_map(self) -> Path:
        return self.path("soil", "mukey_map.tif")

    @property
    def mukey_legend(self) -> Path:
        return self.path("soil", "mukey_map_legend.csv")

    @property
    def soil_params(self) -> Path:
        return self.path("soil", "soil_parameters.json")

    @property
    def soil_storage(self) -> Path:
        return self.path("soil", "soil_storage.csv")

    @property
    def nlcd_impervious(self) -> Path:
        return self.path("soil", "nlcd_impervious.tif")

    # Imagery, hydrography, infrastructure
    @property
    def naip_rgb(self) -> Path:
        return self.path("imagery", "naip_rgb.tif")

    @property
    def naip_nir(self) -> Path:
        return self.path("imagery", "naip_nir.tif")

    @property
    def flowlines(self) -> Path:
        return self.path("hydrography", "flowlines.geojson")

    @property
    def waterbodies(self) -> Path:
        return self.path("hydrography", "waterbodies.geojson")

    @property
    def fema_zones(self) -> Path:
        return self.path("floodplain", "fema_zones.geojson")

    @property
    def roads(self) -> Path:
        return self.path("infrastructure", "roads.geojson")

    @property
    def buildings(self) -> Path:
        return self.path("infrastructure", "buildings.geojson")

    # Forcing and observations
    @property
    def atlas14(self) -> Path:
        return self.path("precipitation", "atlas14.json")

    def asos(self, storm: str) -> Path:
        """Observed hourly rainfall for one storm."""
        return self.path("precipitation", f"asos_{self.asos_station}_{storm}.csv")

    def discharge(self, storm: str) -> Path:
        """Observed 15-minute gauge discharge for one storm."""
        return self.path("gauge", f"discharge_{storm}.csv")

    # Surface parameterisation
    @property
    def landcover(self) -> Path:
        return self.path("surface", "landcover.tif")

    @property
    def manning_n(self) -> Path:
        return self.path("surface", "manning_n.tif")

    @property
    def impervious_frac(self) -> Path:
        return self.path("surface", "impervious_frac.tif")


STORMS: Dict[str, Storm] = {
    "ian": Storm(
        name="ian",
        label="Hurricane Ian, September 2022",
        start="2022-09-28 00:00",
        end="2022-09-30 23:00",
        gauge_start="2022-09-26",
        gauge_end="2022-10-03",
    ),
    "milton": Storm(
        name="milton",
        label="Hurricane Milton, October 2024",
        start="2024-10-06 00:00",
        end="2024-10-10 23:00",
        gauge_start="2024-10-04",
        gauge_end="2024-10-15",
    ),
}

SITES: Dict[str, SiteConfig] = {
    "main_aoi": SiteConfig(
        name="main_aoi",
        label="CFX SR417 corridor test landscape, Lake Nona, Orlando FL",
        lat=28.36687,
        lon=-81.43299,
        radius_km=1.0,
        asos_station="MCO",
        # No gauge. The nearest, Shingle Creek 02263800, drains 231 km2 against this domain's
        # 5.24 km2 -- a 44x mismatch that makes a discharge comparison meaningless. Results
        # here are plausibility checks, never validation.
        gauge=None,
    ),
    "site3": SiteConfig(
        name="site3",
        label="Gee Creek near Longwood FL, gauge-matched validation site",
        lat=28.690514,
        lon=-81.287539,
        radius_km=2.99,
        # KSFB (Orlando Sanford Intl) at 10.8 km, not MCO at 29.1 km. Chosen on proximity and
        # then confirmed on reliability against an independent GHCND daily station: r = 0.634
        # over 1,956 overlapping days, and it stayed online through Ian. That check exists
        # because the main AOI's nearest station, ISM, reported 0.0 mm through the whole storm.
        asos_station="SFB",
        gauge=Gauge(
            site_no="02234400",
            lat=28.7041629,
            lon=-81.2906221,
            documented_area_km2=33.15,
            delineated_area_km2=3.72,
            baseflow_cfs=45.2,
        ),
    ),
}


def get_site(name: str) -> SiteConfig:
    """Look up a site, failing with the valid options rather than a KeyError."""
    assert name in SITES, f"unknown site {name!r}; valid sites: {sorted(SITES)}"
    return SITES[name]


def get_storm(name: str) -> Storm:
    """Look up a storm, failing with the valid options rather than a KeyError."""
    assert name in STORMS, f"unknown storm {name!r}; valid storms: {sorted(STORMS)}"
    return STORMS[name]
