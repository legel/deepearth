"""Reading the map store: every species' 240 m map from one shared, transform-coded field and a short code per species.

A store is a directory:

* ``store.json``: the codec (``klt``), quantizer step ``delta``, tile size, channel count ``d`` and the grid shape of
  each region, e.g. ``{"conus": [13053, 20149]}``;
* ``field_<region>.zst`` and ``index_<region>.npy``: the integer field q(x), one vector of channels per 240 m cell,
  in tiles of 128 x 128 cells. ``index`` holds per tile (byte offset, bytes, k, k16): the tile keeps its first k
  channels (the rest are zero there), the first k16 of them as int16 stored as two byte planes (all low bytes, then
  all high bytes), the others as int8, and the payload is zstd-compressed. Stores written before the int8 split have
  a 3-column index and hold every channel as int16 (k16 = k);
* ``valid_<region>.npy``: bit-packed mask of the cells with climate data;
* ``species.npz``, one row per species: ``species`` (label, e.g. ``Quercus_lobata``), ``codes`` [S, d] and
  ``offsets`` [S] (the score is f_s(x) = q(x) . codes[s] + offsets[s]), ``quantiles`` [S, 254] (quantiles of f_s
  over the species' own background points), ``p5`` [S] (5th percentile of f_s over its training presences),
  ``calibration`` (space-separated RESOLVE ecoregion ids), ``inferred`` (True for a species without records,
  mapped from its relatives) and, for a model with a learned calibration penalty, ``penalty`` [S].

What a map value means, for species s at cell x:

* score f_s(x): the joint model's log relative intensity of records (higher = more suitable);
* served score g_s(x): f_s(x), lowered by the species' learned penalty pi_s where x lies outside its calibration
  ecoregions (stores with ``penalty``; without it the calibration area is a hard rule: nothing is served outside);
* suitability, 0..255: 1 + the number of the species' 254 background quantiles that g_s(x) exceeds, i.e. the share
  of its calibration-area background that x outscores, in 254 steps; 0 where there is no climate and, under the hard
  rule, outside the calibration ecoregions;
* range, yes/no: g_s(x) >= P5 where a suitability is served (the threshold keeps 95% of its training presences).

The calibration mask needs the region's 240 m ecoregion-id raster (``ecoregion_id_conus240.tif`` in the grid
directory given as ``grids``); scores and the stored field do not. Only numpy and zstandard are needed to read
scores (rasterio and pyproj for the mask and for coordinates). Everything runs on a CPU.

    >>> st = Store("cards_conus_klt", grids={"conus": "work/conus240"})
    >>> s = st.index("Quercus lobata")
    >>> st.suitability("conus", 6000, 6512, 1500, 2012, s)        # uint8 [512, 512]
    >>> st.in_range("conus", 6000, 6512, 1500, 2012, s)           # bool [512, 512]
    >>> st.at("conus", lon=[-122.27], lat=[37.87], species=s)     # scores, suitability, range at points
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import zstandard

TILE = 128
ECOREGION_RASTER = "ecoregion_id_conus240.tif"     # the 240 m ecoregion-id layer of a region grid (lakes = 0)


class Store:
    """Reader of a map store (CPU). ``scores`` is the decode fast path (tiles decompressed in parallel and scored in
    their stored channel-major layout); ``window`` and ``cells`` return the integer field itself; ``suitability``,
    ``in_range`` and ``at`` the served maps. ``grids``: region -> 240 m grid directory, needed for the calibration
    mask and for coordinates."""

    def __init__(self, path: str | Path, grids: dict[str, str | Path] | None = None, cache_tiles: int = 64,
                 threads: int = 8):
        self.path = Path(path)
        self.meta = json.loads((self.path / "store.json").read_text())
        if self.meta.get("codec") != "klt":
            raise ValueError(f"{self.path}: codec {self.meta.get('codec')!r} is not supported")
        self.tile = int(self.meta.get("tile", TILE))
        self.T = dict(np.load(self.path / "species.npz"))
        self.grids = {k: Path(v) for k, v in (grids or {}).items()}
        self._f, self._eco, self._grid, self._cache, self.cache_tiles = {}, {}, {}, OrderedDict(), cache_tiles
        self._row = {str(n): i for i, n in enumerate(self.T.get("species", []))}
        self._tl = threading.local()
        self._pool = ThreadPoolExecutor(threads)

    # ---------------------------------------------------------------------------------------------- metadata
    def regions(self) -> list[str]:
        return list(self.meta["shape"])

    def shape(self, region: str) -> tuple[int, int]:
        return tuple(self.meta["shape"][region])

    @property
    def species(self) -> np.ndarray:
        return self.T["species"]

    def index(self, name: str) -> int:
        """Row of a species, by label (``Quercus_lobata``) or name (``Quercus lobata``)."""
        key = str(name).strip().replace(" ", "_")
        if key not in self._row:
            raise KeyError(f"{name!r} is not in the store")
        return self._row[key]

    def calibration(self, s: int) -> list[int]:
        """RESOLVE ecoregion ids of species row ``s``'s calibration area."""
        return [int(x) for x in str(self.T["calibration"][s]).split()]

    # -------------------------------------------------------------------------------------------------- tiles
    def _field(self, region):
        if region not in self._f:
            index = np.load(self.path / f"index_{region}.npy")
            if index.shape[-1] not in (3, 4):
                raise ValueError(f"{self.path}: index_{region}.npy has {index.shape[-1]} columns, expected "
                                 f"(offset, bytes, k[, k16])")
            self._f[region] = (index, np.memmap(self.path / f"field_{region}.zst", dtype=np.uint8, mode="r"))
        return self._f[region]

    def _zd(self):
        zd = getattr(self._tl, "zd", None)
        if zd is None:                                                         # decompressors are per thread
            zd = self._tl.zd = zstandard.ZstdDecompressor()
        return zd

    def planes(self, region: str, ty: int, tx: int) -> tuple[np.ndarray, np.ndarray]:
        """One tile as stored, channel-major: (int16 [k16, h, w], int8 [k - k16, h, w]); k = 0 without land."""
        index, blob = self._field(region)
        row = [int(v) for v in index[ty, tx]]
        o, n, k = row[:3]
        k16 = row[3] if len(row) > 3 else k
        H, W = self.shape(region)
        h, w = min(self.tile, H - ty * self.tile), min(self.tile, W - tx * self.tile)
        if n == 0:
            return np.zeros((0, h, w), np.int16), np.zeros((0, h, w), np.int8)
        raw = np.frombuffer(self._zd().decompress(bytes(blob[o:o + n])), np.uint8)
        m = 2 * k16 * h * w
        q16 = np.empty(k16 * h * w, np.int16)
        b = q16.view(np.uint8).reshape(-1, 2)
        lo_hi = raw[:m].reshape(2, -1)
        b[:, 0], b[:, 1] = lo_hi[0], lo_hi[1]                                  # little-endian low, high bytes
        return q16.reshape(k16, h, w), raw[m:].view(np.int8).reshape(k - k16, h, w)

    def tile_values(self, region: str, ty: int, tx: int) -> np.ndarray:
        """int16 [h, w, k] of one tile (cached; k = 0 for a tile without land)."""
        key = (region, ty, tx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        q16, q8 = self.planes(region, ty, tx)
        q = np.moveaxis(np.concatenate([q16, q8.astype(np.int16)]), 0, -1)
        self._cache[key] = q
        if len(self._cache) > self.cache_tiles:
            self._cache.popitem(last=False)
        return q

    def _tiles(self, r0, r1, c0, c1):
        t = self.tile
        return [(ty, tx) for ty in range(r0 // t, (r1 - 1) // t + 1) for tx in range(c0 // t, (c1 - 1) // t + 1)]

    # ---------------------------------------------------------------------------------------- the field
    def valid(self, region: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """bool [h, w]: cells with climate data."""
        v = np.load(self.path / f"valid_{region}.npy", mmap_mode="r")
        return np.unpackbits(v[r0:r1, c0 // 8:(c1 + 7) // 8], axis=1)[:, c0 % 8:c0 % 8 + (c1 - c0)].astype(bool)

    def window(self, region: str, r0: int, r1: int, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray]:
        """(int16 q [h, w, k], valid [h, w]) over a window; channels beyond k are zero everywhere in it."""
        t = self.tile
        tiles = {key: self.tile_values(region, *key) for key in self._tiles(r0, r1, c0, c1)}
        k = max(v.shape[-1] for v in tiles.values())
        q = np.zeros((r1 - r0, c1 - c0, k), np.int16)
        for (ty, tx), v in tiles.items():
            a0, a1 = max(r0, ty * t), min(r1, ty * t + v.shape[0])
            b0, b1 = max(c0, tx * t), min(c1, tx * t + v.shape[1])
            q[a0 - r0:a1 - r0, b0 - c0:b1 - c0, :v.shape[-1]] = v[a0 - ty * t:a1 - ty * t, b0 - tx * t:b1 - tx * t]
        return q, self.valid(region, r0, r1, c0, c1)

    def cells(self, region: str, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        """(float32 q [n, channels], valid [n]) at scattered cells; zeros and False off the grid."""
        rr, cc = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        H, W = self.shape(region)
        t = self.tile
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        G = np.zeros((len(rr), self.T["codes"].shape[1]), np.float32)
        v = np.zeros(len(rr), bool)
        vm = np.load(self.path / f"valid_{region}.npy", mmap_mode="r")
        idx = np.flatnonzero(ok)
        v[idx] = (np.asarray(vm[rr[idx], cc[idx] // 8]) >> (7 - cc[idx] % 8)) & 1
        key = (rr[idx] // t) * 100000 + cc[idx] // t
        order = np.argsort(key, kind="stable")
        for grp in np.split(order, np.flatnonzero(np.diff(key[order])) + 1):
            if not len(grp):
                continue
            i = idx[grp]
            q = self.tile_values(region, int(rr[i[0]] // t), int(cc[i[0]] // t))
            G[i, :q.shape[-1]] = q[rr[i] % t, cc[i] % t]
        return G, v

    # --------------------------------------------------------------------------------------------- scores
    def scores(self, region: str, r0: int, r1: int, c0: int, c1: int, species) -> np.ndarray:
        """float32 [len(species), h, w]: f_s over a window (also at cells without climate, where it is meaningless;
        see ``valid``). A species' scores can differ in the last float32 bit between calls with different species
        lists (the matrix product's kernel depends on its shape); the served maps score one species at a time."""
        species = np.atleast_1d(species)
        codes = self.T["codes"][species].astype(np.float32)
        offs = self.T["offsets"][species].astype(np.float32)
        out = np.empty((len(species), r1 - r0, c1 - c0), np.float32)
        out[:] = offs[:, None, None]
        t = self.tile

        def one(key):
            ty, tx = key
            q16, q8 = self.planes(region, ty, tx)
            k16, h, w = q16.shape
            a0, a1 = max(r0, ty * t), min(r1, ty * t + h)
            b0, b1 = max(c0, tx * t), min(c1, tx * t + w)
            win = (slice(None), slice(a0 - ty * t, a1 - ty * t), slice(b0 - tx * t, b1 - tx * t))
            f = None
            for part, lo in ((q16, 0), (q8, k16)):
                if len(part):
                    x = part[win].reshape(len(part), -1).astype(np.float32)
                    g = codes[:, lo:lo + len(part)] @ x
                    f = g if f is None else f + g
            if f is not None:
                out[:, a0 - r0:a1 - r0, b0 - c0:b1 - c0] += f.reshape(len(species), a1 - a0, b1 - b0)
        list(self._pool.map(one, self._tiles(r0, r1, c0, c1)))
        return out

    def cell_scores(self, region: str, rows, cols, species) -> tuple[np.ndarray, np.ndarray]:
        """(float32 [n, len(species)] scores, valid [n]) at scattered cells."""
        species = np.atleast_1d(species)
        G, v = self.cells(region, rows, cols)
        return G @ self.T["codes"][species].T.astype(np.float32) + self.T["offsets"][species].astype(np.float32), v

    # ---------------------------------------------------------------------------------------- served maps
    def ecoregions(self, region: str) -> np.ndarray:
        """The region's 240 m ecoregion-id layer (uint16, 0 = none or lake)."""
        if region not in self._eco:
            self._eco[region] = self._grid_raster(region)[0]
        return self._eco[region]

    def _grid_raster(self, region: str):
        if region not in self.grids:
            raise KeyError(f"no grid directory for region {region!r} (Store(..., grids={{{region!r}: ...}}))")
        import rasterio
        with rasterio.open(self.grids[region] / ECOREGION_RASTER) as r:
            return r.read(1), r.transform, r.crs.to_wkt()

    @property
    def has_penalty(self) -> bool:
        """True for a store of a model with a learned calibration penalty (maps continue outside the area)."""
        return "penalty" in self.T

    def inside(self, region: str, r0: int, r1: int, c0: int, c1: int, s: int) -> np.ndarray:
        """bool [h, w]: cells of species row ``s``'s calibration ecoregions that have climate data."""
        return np.isin(self.ecoregions(region)[r0:r1, c0:c1], self.calibration(s)) & self.valid(region, r0, r1, c0, c1)

    def served(self, f: np.ndarray, s: int, in_area: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(served scores g, served cells) of species row ``s`` from its scores ``f``, the calibration-area membership
        and the climate mask of the same cells: with a learned penalty g = f - penalty outside the area, served
        wherever there is climate; under the hard rule g = f, served inside the area only."""
        if self.has_penalty:
            return f - float(self.T["penalty"][s]) * ~in_area, valid
        return f, in_area & valid

    def suitability_of(self, g: np.ndarray, s: int, served: np.ndarray) -> np.ndarray:
        """uint8 suitability of served scores ``g`` of species row ``s``: 1 + the number of its background quantiles
        that g exceeds (1..255) where ``served``, else 0."""
        q = (1 + np.searchsorted(self.T["quantiles"][s], g)).astype(np.uint8)
        return np.where(served, q, 0).astype(np.uint8)

    def _served_window(self, region, r0, r1, c0, c1, s):
        f = self.scores(region, r0, r1, c0, c1, [s])[0]
        in_area = np.isin(self.ecoregions(region)[r0:r1, c0:c1], self.calibration(s))
        return self.served(f, s, in_area, self.valid(region, r0, r1, c0, c1))

    def decode(self, region: str, r0: int, r1: int, c0: int, c1: int, s: int) -> np.ndarray:
        """uint8 [h, w] suitability map of species row ``s``: 1..255 where served (with climate; under the hard rule
        only inside its calibration ecoregions), 0 elsewhere."""
        g, served = self._served_window(region, r0, r1, c0, c1, s)
        return self.suitability_of(g, s, served)

    suitability = decode

    def in_range(self, region: str, r0: int, r1: int, c0: int, c1: int, s: int) -> np.ndarray:
        """bool [h, w]: species row ``s``'s binary range, g_s >= P5 where served."""
        g, served = self._served_window(region, r0, r1, c0, c1, s)
        return (g >= self.T["p5"][s]) & served

    # -------------------------------------------------------------------------------------------- points
    def rowcol(self, region: str, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Grid rows and columns of the 240 m cells containing lon/lat (WGS84 degrees); may fall off the grid."""
        if region not in self._grid:
            from pyproj import Transformer                                      # before rasterio (its bundled PROJ)
            _, t, crs = self._grid_raster(region)
            self._grid[region] = (t, Transformer.from_crs(4326, crs, always_xy=True))
        t, tr = self._grid[region]
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        x, y = tr.transform(lon, lat)
        if not np.isfinite(np.asarray(x)[np.isfinite(lon) & np.isfinite(lat)]).all():
            raise RuntimeError("non-finite coordinate transform (import pyproj before rasterio)")
        return (np.floor((t.f - np.asarray(y)) / -t.e).astype(np.int64),
                np.floor((np.asarray(x) - t.c) / t.a).astype(np.int64))

    def at(self, region: str, lon=None, lat=None, species: int | Sequence[int] = 0, rows=None, cols=None) -> dict:
        """Scores, suitability and range of one or more species rows at points, given as lon/lat or as grid
        rows/cols: {"score": float32 [n, m] (f_s, before any penalty), "suitability": uint8 [n, m], "range": bool
        [n, m], "valid": bool [n]} for n points and m species. Off the grid and without climate: suitability 0,
        range False."""
        if rows is None:
            rows, cols = self.rowcol(region, lon, lat)
        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        species = np.atleast_1d(species)
        f, valid = self.cell_scores(region, rows, cols, species)
        H, W = self.shape(region)
        on = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        eco = np.zeros(len(rows), np.int64)
        eco[on] = self.ecoregions(region)[rows[on], cols[on]]
        suit = np.zeros(f.shape, np.uint8)
        rng = np.zeros(f.shape, bool)
        for j, s in enumerate(species):
            g, served = self.served(f[:, j], int(s), np.isin(eco, self.calibration(int(s))), valid & on)
            suit[:, j] = self.suitability_of(g, int(s), served)
            rng[:, j] = (g >= self.T["p5"][int(s)]) & served
        return {"score": f, "suitability": suit, "range": rng, "valid": valid}
