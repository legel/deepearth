"""WGS 84 longitude/latitude -> NAD83 / Conus Albers (EPSG:5070) in PyTorch, exactly as PROJ transforms them.

Why exactness matters here. The 240 m CONUS rasters (predictor stack, ecoregion ids, the environmental field) were
warped from WGS 84 by gdalwarp. To read a record's *own* 240 m cell of those rasters, its coordinates must go through
the same transformation gdalwarp applied. That transformation is not a single formula: WGS 84 and NAD83 differ by up
to ~2 m, and EPSG defines several datum transformations between them. PROJ (the engine of gdalwarp and pyproj)
chooses one per point:

1. Candidate operations: pyproj's ``TransformerGroup(EPSG:4326 -> EPSG:5070)``, each with its accuracy, its area of
   use and (for most) a horizontal shift grid: NOAA's HARN grids for the US states, NRC Canada's grids, and the null
   "NAD83 to WGS 84 (1)" (accuracy 4 m) everywhere else.
2. Per point, as PROJ's ``proj_trans``: among the operations whose area of use contains the point, the most accurate
   first, ties in PROJ's own order of the operation list; an operation whose grid does not cover the point is
   skipped.
3. Inverse horizontal grid shift (WGS 84 -> NAD83), as PROJ's ``pj_hgrid_apply`` inverse: fixed-point iteration
   t <- t - (t + shift(t) - input), at most 10 iterations, tolerance 1e-12 rad; the shift is bilinear between the
   grid's nodes (latitude and longitude offsets in arc-seconds, longitude positive east).
4. Albers equal-area conic on the GRS80 ellipsoid (Snyder 1987, eqs. 3-12 and 14-3 to 14-7).

A grid file can hold nested subgrids (finer grids inside coarser ones); which one serves a point follows PROJ's grid
hierarchy (``ShiftGrid``). On 2 million random CONUS points the result differs from pyproj by at most 7e-8 m
(tests/test_geodesy.py holds it to 1e-6 m).

``grid_positions`` turns the projected coordinates into fractional (row, column) positions on a north-up raster:
the position of a record on the 240 m grid, used to read the environmental field around it and its ecoregion.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ARCSEC = math.pi / (180.0 * 3600.0)
CONUS_BOX = (20.0, 52.0, -130.0, -60.0)       # (south, north, west, east): where the conic projection is used


def albers_5070(lat: torch.Tensor, lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NAD83 geographic coordinates (degrees) -> EPSG:5070 x, y (m): standard parallels 29.5 and 45.5 N, origin
    23 N, 96 W, GRS80 ellipsoid. Evaluated in float64."""
    a, f = 6378137.0, 1 / 298.257222101
    e2 = f * (2 - f)
    e = math.sqrt(e2)

    def q(phi):
        s = torch.sin(phi)
        return (1 - e2) * (s / (1 - e2 * s * s) - 1 / (2 * e) * torch.log((1 - e * s) / (1 + e * s)))

    def m(phi):
        return math.cos(phi) / math.sqrt(1 - e2 * math.sin(phi) ** 2)

    p1, p2, p0, l0 = (math.radians(v) for v in (29.5, 45.5, 23.0, -96.0))
    t = lambda v: torch.tensor(v, dtype=torch.float64)                       # noqa: E731
    q1, q2, q0 = q(t(p1)), q(t(p2)), q(t(p0))
    n = (m(p1) ** 2 - m(p2) ** 2) / (q2 - q1)
    C = m(p1) ** 2 + n * q1
    rho0 = a * torch.sqrt(C - n * q0) / n
    phi, lam = torch.deg2rad(lat.double()), torch.deg2rad(lon.double())
    rho = a * torch.sqrt(C - n * q(phi)) / n                   # 0-dim CPU constants combine with any device
    th = n * (lam - l0)
    return rho * torch.sin(th), rho0 - rho * torch.cos(th)


class _Subgrid:
    """One grid of a PROJ GeoTIFF horizontal-offset file (latitude and longitude offsets at regular nodes)."""

    def __init__(self, ds, device, parent=None):
        """``parent``: (latitude band, longitude band, positive direction, unit) of the file's first grid, which child
        subgrids inherit (PROJ GeoTIFF grids: band 1 latitude_offset, band 2 longitude_offset unless described)."""
        descr = [d or "" for d in ds.descriptions]
        if "latitude_offset" in descr:
            lat_b, lon_b = descr.index("latitude_offset") + 1, descr.index("longitude_offset") + 1
            pos = (ds.tags(lon_b).get("positive_value") or "east").lower()
            unit = (ds.tags(lat_b).get("UNITTYPE") or ds.tags().get("UNITTYPE") or "arc-second").lower()
        else:
            lat_b, lon_b, pos, unit = parent if parent else (1, 2, "east", "arc-second")
        if unit not in ("arc-second", "arcsecond"):
            raise ValueError(f"{ds.name}: offsets in {unit!r}, expected arc-seconds")
        self.conv = (lat_b, lon_b, pos, unit)
        self.dlat = torch.from_numpy(ds.read(lat_b).astype(np.float64)).to(device)
        self.dlon = torch.from_numpy(ds.read(lon_b).astype(np.float64) * (1.0 if pos == "east" else -1.0)).to(device)
        T = ds.transform
        self.lon0, self.dx = T.c + 0.5 * T.a, T.a                  # first node and spacing (degrees)
        self.lat0, self.dy = T.f + 0.5 * T.e, T.e                  # T.e < 0: rows run south
        self.H, self.W = self.dlat.shape
        self.children: list[_Subgrid] = []
        self.type = ""

    def extent(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the nodes, degrees."""
        lats = (self.lat0, self.lat0 + (self.H - 1) * self.dy)
        return self.lon0, min(lats), self.lon0 + (self.W - 1) * self.dx, max(lats)

    def contains(self, other: "_Subgrid") -> bool:
        w, s, e, n = self.extent()
        ow, os_, oe, on = other.extent()
        return ow >= w and oe <= e and os_ >= s and on <= n

    def insert(self, g: "_Subgrid") -> None:
        """PROJ ``HorizontalShiftGrid::insertGrid``: into the first child containing it, else as a new child."""
        for c in self.children:
            if c.contains(g):
                c.insert(g)
                return
        self.children.append(g)

    def covers(self, lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        """Point within the node extent, with PROJ's tolerance of (resX + resY) * 1e-5 (``isPointInExtent``)."""
        w, s, e, n = self.extent()
        eps = (abs(self.dx) + abs(self.dy)) * 1e-5
        return (lon + eps >= w) & (lon - eps <= e) & (lat + eps >= s) & (lat - eps <= n)

    def shift(self, lon: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear (dlon, dlat) in radians at points given in degrees."""
        c = ((lon - self.lon0) / self.dx).clamp(0, self.W - 1)
        r = ((lat - self.lat0) / self.dy).clamp(0, self.H - 1)
        c0 = c.floor().clamp(max=self.W - 2).long()
        r0 = r.floor().clamp(max=self.H - 2).long()
        fc, fr = c - c0, r - r0

        def bil(g):
            return ((g[r0, c0] * (1 - fc) + g[r0, c0 + 1] * fc) * (1 - fr)
                    + (g[r0 + 1, c0] * (1 - fc) + g[r0 + 1, c0 + 1] * fc) * fr)
        return bil(self.dlon) * ARCSEC, bil(self.dlat) * ARCSEC


class ShiftGrid:
    """A PROJ GeoTIFF horizontal-offset grid file, with PROJ's grid hierarchy (grids.cpp, ``insertIntoHierarchy``).

    Subgrids (GeoTIFF IFDs) are read in file order. A subgrid naming a parent already read, whose extent contains it,
    becomes that parent's child; a named grid without a parent is top-level; otherwise (no parent named, or a parent
    whose extent does not contain it) PROJ's bounding-box method applies: the first top-level grid of the same TYPE
    whose extent contains it adopts it (recursively into the first child containing it), else it becomes a new
    top-level grid. For a point PROJ then takes the first top-level grid containing it and descends into the first
    child containing it, recursively. (For nested NTv2 grids such as Quebec's this choice changes positions by up to
    1 m.)"""

    def __init__(self, path: str | Path, device="cpu"):
        import rasterio
        with rasterio.open(path) as r:
            names = sorted(r.subdatasets or [], key=lambda n: int(n.split(":")[1]))       # IFD order
            tags0 = r.tags()
        by_name: dict[str, _Subgrid] = {}
        self.top: list[_Subgrid] = []
        parent_conv = None
        for name in (names or [str(path)]):
            with rasterio.open(name) as d:
                g = _Subgrid(d, device, parent_conv)
                t = d.tags() if names else tags0
            parent_conv = parent_conv or g.conv
            g.type = t.get("TYPE", tags0.get("TYPE", ""))
            gname, pname = t.get("grid_name", ""), t.get("parent_grid_name", "")
            if gname:
                by_name[gname] = g
            if pname and pname in by_name and by_name[pname].contains(g):
                by_name[pname].children.append(g)
                continue
            if not pname and gname:
                self.top.append(g)
                continue
            for cand in self.top:                                   # bounding-box method
                if cand.type == g.type and cand.contains(g):
                    cand.insert(g)
                    break
            else:
                self.top.append(g)

    def _assign(self, lon: torch.Tensor, lat: torch.Tensor) -> list[tuple[_Subgrid, torch.Tensor]]:
        """(subgrid, mask) pairs: PROJ's grid for each point (points in no grid: none)."""
        out = []
        left = torch.ones_like(lon, dtype=torch.bool)

        def descend(g, m):
            rest = m.clone()
            for c in g.children:
                mc = rest & c.covers(lon, lat)
                if bool(mc.any()):
                    descend(c, mc)
                    rest &= ~mc
            if bool(rest.any()):
                out.append((g, rest))
        for g in self.top:
            m = left & g.covers(lon, lat)
            if bool(m.any()):
                descend(g, m)
                left &= ~m
        return out

    def covers(self, lam: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        lon, lat = torch.rad2deg(lam), torch.rad2deg(phi)
        return torch.stack([g.covers(lon, lat) for g in self.top]).any(0)

    def shift(self, lam: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear (dlam, dphi) in radians at points given in radians, each from PROJ's grid for it."""
        lon, lat = torch.rad2deg(lam), torch.rad2deg(phi)
        dl, dp = torch.zeros_like(lam), torch.zeros_like(phi)
        for g, m in self._assign(lon, lat):
            a, b = g.shift(lon[m], lat[m])
            dl[m], dp[m] = a, b
        return dl, dp


def _transformer_group():
    from pyproj import CRS
    from pyproj.transformer import TransformerGroup
    return TransformerGroup(CRS(4326), CRS(5070), always_xy=True)


def fetch_grids(grid_dir: str | Path) -> None:
    """Download the PROJ grid files of every WGS 84 -> EPSG:5070 operation (NOAA HARN, NRC Canada; from
    cdn.proj.org) into ``grid_dir``."""
    Path(grid_dir).mkdir(parents=True, exist_ok=True)
    _transformer_group().download_grids(directory=str(grid_dir), open_license=True)


def _find_grid(name: str, grid_dir: Path) -> Path:
    import pyproj.datadir
    for d in (grid_dir, pyproj.datadir.get_data_dir(), pyproj.datadir.get_user_data_dir()):
        p = Path(d) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"PROJ grid {name} is neither in {grid_dir} nor in pyproj's data directories; "
                            f"download the grids with geodesy.fetch_grids({str(grid_dir)!r})")


class WGS84ToConusAlbers:
    """Callable (lat, lon in degrees, tensors on any device) -> (x, y) metres in EPSG:5070, as PROJ/gdalwarp.

    ``grid_dir``: directory of the PROJ grid files the operations name (``fetch_grids``); pyproj's own data
    directories are searched for a file not found there."""

    def __init__(self, grid_dir: str | Path, device="cpu"):
        grid_dir = Path(grid_dir)
        self.ops = []
        for t in _transformer_group().transformers:
            b = t.area_of_use.bounds                                # (west, south, east, north) degrees
            words = t.definition.split()
            grids = [p.split("=", 1)[1] for p in words if p.startswith("grids=")]
            if len(grids) > 1 or (grids and "inv" not in words):    # WGS 84 -> NAD83 is the inverse shift
                raise ValueError(f"unexpected operation: {t.definition}")
            g = ShiftGrid(_find_grid(grids[0], grid_dir), device) if grids else None
            self.ops.append(dict(acc=t.accuracy if t.accuracy >= 0 else float("inf"), bounds=b, grid=g,
                                 name=t.description))
        self.ops.sort(key=lambda o: o["acc"])                       # stable: ties keep PROJ's list order

    @staticmethod
    def _inverse_shift(g: ShiftGrid, lam, phi):
        dl, dp = g.shift(lam, phi)
        tl, tp = lam - dl, phi - dp
        for _ in range(10):
            dl, dp = g.shift(tl, tp)
            el, ep = tl + dl - lam, tp + dp - phi
            tl, tp = tl - el, tp - ep
            if bool((el.abs().max() < 1e-12) & (ep.abs().max() < 1e-12)):
                break
        return tl, tp

    def __call__(self, lat: torch.Tensor, lon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lat, lon = lat.double(), lon.double()
        lam, phi = torch.deg2rad(lon), torch.deg2rad(lat)
        out_l, out_p = lam.clone(), phi.clone()
        done = torch.zeros_like(lam, dtype=torch.bool)
        for o in self.ops:
            w, s, e, n = o["bounds"]
            sel = ~done & (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
            if o["grid"] is not None:
                sel = sel & o["grid"].covers(lam, phi)
            if not bool(sel.any()):
                continue
            if o["grid"] is not None:
                out_l[sel], out_p[sel] = self._inverse_shift(o["grid"], lam[sel], phi[sel])
            done |= sel
        return albers_5070(torch.rad2deg(out_p), torch.rad2deg(out_l))


def grid_positions(lat, lon, transform: Sequence[float], to5070: WGS84ToConusAlbers,
                   box: Sequence[float] = CONUS_BOX, chunk: int = 8_000_000, device="cpu") -> np.ndarray:
    """Fractional (row, column) of points on a north-up EPSG:5070 raster with affine ``transform`` (a, b, c, d, e, f)
    (x = c + col * a, y = f + row * e; cell centres at +0.5), float32 [n, 2]. Points outside ``box`` (south, north,
    west, east degrees; records elsewhere in the world, where the conic projection means nothing) get -1e6, i.e. off
    every grid."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    s_, n_, w_, e_ = box
    lat, lon = np.asarray(lat), np.asarray(lon)
    out = np.empty((len(lat), 2), np.float32)
    for i in range(0, len(lat), chunk):
        la = torch.as_tensor(np.asarray(lat[i:i + chunk]), device=device).double()
        lo = torch.as_tensor(np.asarray(lon[i:i + chunk]), device=device).double()
        x, y = to5070(la, lo)
        rc = torch.stack([(y - f) / e, (x - c) / a], 1).float()
        inside = (la >= s_) & (la <= n_) & (lo >= w_) & (lo <= e_)
        out[i:i + chunk] = torch.where(inside[:, None], rc, torch.full_like(rc, -1e6)).cpu().numpy()
    return out
