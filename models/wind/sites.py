"""Site and day registry.

The only source of coordinates, station metadata and file locations in the package. Every path
a stage reads or writes derives from `SiteConfig.root`; there is no default site.
"""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parent

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Day:
    """One calendar day whose hourly record drives a time series.

    Attributes:
        name: Registry key.
        label: Human-readable name.
        date: "YYYY-MM-DD", UTC.
    """

    name: str
    label: str
    date: str

    @property
    def year(self) -> int:
        return int(self.date[:4])


@dataclass(frozen=True)
class SiteConfig:
    """One modelling parcel: where it is, what forces it, and where its files live.

    Attributes:
        name: Registry key; also the data and output subdirectory name.
        label: Human-readable description.
        lat: Parcel centre latitude, WGS84.
        lon: Parcel centre longitude, WGS84.
        radius_m: Parcel disc radius [m]; results are shown out to here.
        anchor_utm: (E, N, h) of the scene origin in EPSG:32610.
        asos_station: IEM ASOS identifier supplying speed, direction and gust.
        station_lat: Station latitude, WGS84.
        station_lon: Station longitude, WGS84.
        station_z0_m: Roughness length of the station's fetch [m].
        z0_m: Roughness length of the parcel's upwind fetch [m].
        d_m: Displacement height of the parcel's upwind fetch [m].
        z_blend_m: Height at which the station and parcel profiles share one speed [m].
        cell_sizes_m: Production horizontal resolutions.
        stretch: Vertical cell-size ratio between adjacent levels.
        tallest_m: Tallest obstacle on the parcel [m]; the grid top clears twice this.
        buffer_m: Ring beyond the disc that the grid also covers [m], so that a structure cut
            by the disc edge is whole; set from the bundle, 0 without one.
    """

    name: str
    label: str
    lat: float
    lon: float
    radius_m: float
    anchor_utm: Tuple[float, float, float]
    asos_station: str
    station_lat: float
    station_lon: float
    station_z0_m: float = 0.03
    z0_m: float = 0.8
    d_m: float = 5.0
    z_blend_m: float = 60.0
    cell_sizes_m: Tuple[float, ...] = (0.2, 0.1)
    stretch: float = 1.06
    tallest_m: float = 94.0
    buffer_m: float = 0.0

    def station_km(self) -> float:
        """Great-circle distance from the parcel centre to the station [km]."""
        p1, p2 = math.radians(self.lat), math.radians(self.station_lat)
        dphi, dlam = p2 - p1, math.radians(self.station_lon - self.lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

    @property
    def fetch_radius_m(self) -> float:
        """Radius the grid covers [m]: the disc plus its buffer."""
        return self.radius_m + self.buffer_m

    @property
    def display_radius_m(self) -> float:
        """Radius results are shown out to [m]."""
        return self.radius_m

    def cells_across(self, dx: float, multiple: int = 128) -> int:
        """Horizontal cell count covering the fetch disc, rounded up to a multigrid-friendly multiple."""
        n = math.ceil(2 * self.fetch_radius_m / dx)
        return multiple * math.ceil(n / multiple)

    def levels(self, dx: float, multiple: int = 8) -> int:
        """Vertical level count whose stretched top, starting at `dx`, clears twice `tallest_m`."""
        n = math.log(1 + 2 * self.tallest_m * (self.stretch - 1) / dx) / math.log(self.stretch)
        return multiple * math.ceil(n / multiple)

    # ── Paths ────────────────────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Fetched-data root for this site."""
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

    def asos(self, year: int) -> Path:
        """Hourly speed, direction and gust for one year."""
        return self.path("wind", f"asos_{self.asos_station}_{year}.csv")

    @property
    def bundle(self) -> Path:
        """Default bundle directory: `semantics/` and the surface rasters live under it."""
        return self.root / "bundle"


DAYS: Dict[str, Day] = {
    "summer": Day(name="summer", label="Summer sea breeze, 15 July 2025", date="2025-07-15"),
    "winter": Day(name="winter", label="Winter storm season, 15 January 2025",
                  date="2025-01-15"),
}

SITES: Dict[str, SiteConfig] = {
    "campanile": SiteConfig(
        name="campanile",
        label="Sather Tower parcel, UC Berkeley",
        lat=37.87217,
        lon=-122.25778,
        radius_m=112.32,
        anchor_utm=(565278.3393294326, 4191891.744672124, 119.81544136816949),
        asos_station="OAK",
        station_lat=37.7178,
        station_lon=-122.2331,
    ),
}


def get_site(name: str) -> SiteConfig:
    """Look up a site, failing with the valid options rather than a KeyError."""
    assert name in SITES, f"unknown site {name!r}; valid sites: {sorted(SITES)}"
    return SITES[name]


def get_day(name: str) -> Day:
    """Look up a day, failing with the valid options rather than a KeyError."""
    assert name in DAYS, f"unknown day {name!r}; valid days: {sorted(DAYS)}"
    return DAYS[name]


def aoi_document(site: SiteConfig, bundle: Path) -> Dict[str, object]:
    """The bundle's `surface/aoi_<site>.json`, checked against this registry entry.

    The anchor and the display radius are copies of the bundle's numbers; a disagreement means
    the registry and the bundle describe different discs and nothing downstream would overlay.
    """
    doc = json.loads((bundle / "surface" / f"aoi_{site.name}.json").read_text())
    anchor, radius = tuple(doc["scene_frame"]["anchor_utm"]), doc["aoi"]["radius_m"]
    assert anchor == site.anchor_utm and radius == site.radius_m, (
        f"bundle disc {anchor}, r {radius} is not the registry's {site.anchor_utm}, r {site.radius_m}")
    return doc


def bundled(site: SiteConfig, bundle: Path, buffer_m: Optional[float] = None) -> SiteConfig:
    """The site carrying the buffer its bundle was fetched with.

    Args:
        site: Registry entry.
        bundle: Bundle directory.
        buffer_m: Override [m]; 0 reproduces a run over the display disc alone.
    Returns:
        The site with `buffer_m` set.
    """
    doc = aoi_document(site, bundle)
    return replace(site, buffer_m=doc["aoi"]["buffer_m"] if buffer_m is None else buffer_m)


def box_of(site: SiteConfig, bundle: Path) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    """The bundle's boxes when the site follows its ordered polygon: {"fetch": ..., "display": ...}, each
    (x0, y0, x1, y1) scene metres; None for a disc site, which then runs exactly as before."""
    b = aoi_document(site, bundle)["aoi"].get("box_scene_m")
    return {k: tuple(float(v) for v in b[k]) for k in ("fetch", "display")} if b else None


def box_cells(box: Tuple[float, float, float, float], dx: float, multiple: int = 128) -> Tuple[int, int]:
    """(nx, ny) covering a fetch box, each rounded up to a multigrid-friendly multiple as `cells_across`."""
    return tuple(multiple * math.ceil(math.ceil((box[k + 2] - box[k]) / dx) / multiple) for k in (0, 1))
