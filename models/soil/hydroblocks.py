"""Surface soil moisture baseline for a site: SMAP-HydroBlocks seasonal means and percentiles at 30 m.

Source: SMAP-HydroBlocks (Vergopolan et al. 2021, Sci Data 8, 264; Zenodo 10.5281/zenodo.5206725, CC-BY-4.0),
top 5 cm, 2015-2019. Built nationally by scripts/hydroblocks/national.py into four int16 COGs at
$HYDROBLOCKS_BASE (a folder or bucket of your own; README.json there says how each layer was computed).

    from soil.hydroblocks import site_window
    w = site_window(-122.07, 37.40, -122.03, 37.43)
    w['wet_mean']          # float32 [rows, cols], m3/m3, NaN where the source has no value
    w['geotransform']      # GDAL order, lon/lat degrees of the window's top-left cell corner

WHY A TIFF READER AGAIN. Same reason as soil/polaris.py: no GDAL here. A site touches one or two 512 px tiles,
so a window is the header (16 KB, cached), one ranged read of each tile index row and one read per run of
adjacent tiles. Nothing national is downloaded.

GRID. Native to the source: geographic lon/lat on GRS80 (NAD83 inputs, EPSG:4269 in practice), cells of one
arc-second with edges on integer arc-seconds. A cell is about 31 m north-south and 31 m * cos(lat) east-west.

Standard library plus numpy.
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = os.environ.get('HYDROBLOCKS_BASE', 'hydroblocks/v1')
LAYERS = ('wet_mean', 'dry_mean', 'p10', 'p90')
FILES = {k: 'smaphb_%s_30m.tif' % k for k in LAYERS}
SCALE = 1e-4            # stored int16 * SCALE = m3/m3
NODATA = -32768
SOURCE = ('SMAP-HydroBlocks v1.1 (Vergopolan et al. 2021, doi:10.1038/s41597-021-01050-2; '
          'data doi:10.5281/zenodo.5206725, CC-BY-4.0), 0-5 cm, 2015-2019')
CRS = 'EPSG:4269'       # GRS80 geographic as the source writes it; see README.json crs_note
_HEAD = 16 * 1024       # the COG's IFD and inline tags; the tile index arrays are read by range
_GAP = 64               # bytes between two tiles still fetched as one read (a COG's leader and trailer are 8)

_T_W, _T_H, _T_BITS, _T_COMP, _T_PRED = 256, 257, 258, 259, 317
_T_TW, _T_TH, _T_OFFS, _T_CNTS, _T_FMT = 322, 323, 324, 325, 339
_T_SCALE, _T_TIE, _T_NODATA = 33550, 33922, 42113
_FMT = {1: 'B', 2: 's', 3: 'H', 4: 'I', 6: 'b', 8: 'h', 9: 'i', 11: 'f', 12: 'd', 16: 'Q', 17: 'q'}
_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 7: 1, 8: 2, 9: 4, 11: 4, 12: 8, 16: 8, 17: 8}

_cache: dict = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ transport ----
def fetch(uri: str, start: int, length: int) -> bytes:
    """`length` bytes of `uri` from `start`: a local path, or gs:// by the JSON API with `gsutil cat -r` behind it."""
    if not uri.startswith('gs://'):
        with open(uri, 'rb') as f:
            f.seek(start)
            return f.read(length)
    try:
        from sims import gcs_api
        return gcs_api.get_range(uri, start, length)
    except (ImportError, OSError):
        pass                                # off GCE (no token) or a short read: the CLI is slower, not wrong
    out = subprocess.run(['gsutil', 'cat', '-r', '%d-%d' % (start, start + length - 1), uri],
                         capture_output=True, timeout=120)
    if out.returncode != 0:
        raise OSError('gsutil cat -r %s: %s' % (uri, out.stderr.decode(errors='replace')[-300:]))
    return out.stdout


# ------------------------------------------------------------------ tiff ----
class Cog:
    """One tiled int16 GeoTIFF (classic or BigTIFF, deflate or none, predictor 1 or 2) read by range."""

    def __init__(self, uri: str, get=fetch):
        self.uri, self._get = uri, get
        head = get(uri, 0, _HEAD)
        if head[:2] != b'II':
            raise RuntimeError('%s: expected a little-endian TIFF' % uri)
        magic = struct.unpack_from('<H', head, 2)[0]
        self.big = magic == 43
        if magic not in (42, 43):
            raise RuntimeError('%s: not a TIFF (magic %d)' % (uri, magic))
        ifd = struct.unpack_from('<Q', head, 8)[0] if self.big else struct.unpack_from('<I', head, 4)[0]
        nfmt, esz, cfmt, vsz = ('<Q', 20, '<HHQ', 8) if self.big else ('<H', 12, '<HHI', 4)
        nlen = struct.calcsize(nfmt)
        if ifd + nlen > len(head):
            head = get(uri, 0, ifd + nlen + 4096)
        n = struct.unpack_from(nfmt, head, ifd)[0]
        if ifd + nlen + n * esz > len(head):
            head = get(uri, 0, ifd + nlen + n * esz + 16)
        self._head = head
        tags, self._arrays = {}, {}
        for i in range(n):
            off = ifd + nlen + i * esz
            tag, typ, count = struct.unpack_from(cfmt, head, off)
            size = _SIZE.get(typ, 1) * count
            at = off + struct.calcsize(cfmt)
            if tag in (_T_OFFS, _T_CNTS):
                # kept as (type, count, where) so only the entries a window needs are ever fetched
                where = at if size <= vsz else struct.unpack_from('<Q' if self.big else '<I', head, at)[0]
                self._arrays[tag] = (typ, count, where, size <= vsz)
                continue
            if typ not in _FMT:
                continue
            if size <= vsz:
                raw = head[at:at + size]
            else:
                p = struct.unpack_from('<Q' if self.big else '<I', head, at)[0]
                raw = head[p:p + size] if p + size <= len(head) else get(uri, p, size)
            if typ == 2:
                tags[tag] = raw.split(b'\0')[0].decode()
            else:
                v = struct.unpack('<%d%s' % (count, _FMT[typ]), raw)
                tags[tag] = v[0] if count == 1 else v
        if tags.get(_T_BITS) != 16 or tags.get(_T_FMT, 1) != 2:
            raise RuntimeError('%s: expected signed 16-bit samples' % uri)
        self.compression = tags.get(_T_COMP, 1)
        if self.compression not in (1, 8, 32946):
            raise RuntimeError('%s: compression %s not handled' % (uri, self.compression))
        self.predictor = tags.get(_T_PRED, 1)
        if self.predictor not in (1, 2):
            raise RuntimeError('%s: predictor %s not handled' % (uri, self.predictor))
        if _T_TW not in tags:
            raise RuntimeError('%s: expected a tiled TIFF' % uri)
        self.width, self.height = tags[_T_W], tags[_T_H]
        self.tw, self.th = tags[_T_TW], tags[_T_TH]
        self.across = -(-self.width // self.tw)
        sx, sy = tags[_T_SCALE][0], tags[_T_SCALE][1]
        tie = tags[_T_TIE]
        self.gt = (tie[3] - tie[0] * sx, sx, 0.0, tie[4] + tie[1] * sy, 0.0, -sy)
        nd = tags.get(_T_NODATA)
        self.nodata = int(float(nd)) if nd not in (None, '') else None

    def _entries(self, tag: int, i0: int, i1: int) -> np.ndarray:
        """Tile index entries [i0, i1) of TileOffsets or TileByteCounts."""
        typ, count, where, inline = self._arrays[tag]
        sz = _SIZE[typ]
        dt = {16: '<u8', 4: '<u4', 3: '<u2'}[typ]
        if inline:
            return np.frombuffer(self._head[where:where + count * sz], dt)[i0:i1].astype(np.int64)
        return np.frombuffer(self._get(self.uri, where + i0 * sz, (i1 - i0) * sz), dt).astype(np.int64)

    def _decode(self, raw: bytes) -> np.ndarray:
        if self.compression != 1:
            raw = zlib.decompress(raw)
        a = np.frombuffer(raw, '<i2', self.tw * self.th).reshape(self.th, self.tw)
        if self.predictor == 2:      # horizontal differencing, undone modulo 2**16
            a = a.view(np.uint16).cumsum(axis=1, dtype=np.uint16).view(np.int16)
        return a

    def read(self, c0: int, r0: int, c1: int, r1: int) -> np.ndarray:
        """Stored int16 for columns [c0, c1) and rows [r0, r1), NODATA where a tile is absent."""
        out = np.full((r1 - r0, c1 - c0), NODATA if self.nodata is None else self.nodata, np.int16)
        tc0, tc1 = c0 // self.tw, (c1 - 1) // self.tw + 1
        for tr in range(r0 // self.th, (r1 - 1) // self.th + 1):
            i0 = tr * self.across + tc0
            offs = self._entries(_T_OFFS, i0, i0 + tc1 - tc0)
            cnts = self._entries(_T_CNTS, i0, i0 + tc1 - tc0)
            j = 0
            while j < len(offs):
                if cnts[j] == 0:       # sparse tile: every cell nodata
                    j += 1
                    continue
                k = j + 1                  # neighbours stored together are one request (COG puts 8 B between tiles)
                while k < len(offs) and cnts[k] and 0 <= offs[k] - (offs[k - 1] + cnts[k - 1]) <= _GAP:
                    k += 1
                blob = self._get(self.uri, int(offs[j]), int(offs[k - 1] + cnts[k - 1] - offs[j]))
                for t in range(j, k):
                    s = int(offs[t] - offs[j])
                    tile = self._decode(blob[s:s + int(cnts[t])])
                    tx0, ty0 = (tc0 + t) * self.tw, tr * self.th
                    a0, a1 = max(c0, tx0), min(c1, tx0 + self.tw)
                    b0, b1 = max(r0, ty0), min(r1, ty0 + self.th)
                    out[b0 - r0:b1 - r0, a0 - c0:a1 - c0] = tile[b0 - ty0:b1 - ty0, a0 - tx0:a1 - tx0]
                j = k
        return out


def _open(uri: str, get) -> Cog:
    with _lock:
        hit = _cache.get((uri, get))
    if hit is None:
        hit = Cog(uri, get)
        with _lock:
            _cache[(uri, get)] = hit
    return hit


# ------------------------------------------------------------------ site ----
def site_window(lon_min: float, lat_min: float, lon_max: float, lat_max: float,
                base: str | None = None, layers: tuple = LAYERS, get=fetch) -> dict:
    """The four layers over every 30 m cell touching the bbox (lon/lat degrees), in m3/m3 with NaN for nodata.

    Returns {layer: float32 [rows, cols], 'geotransform', 'shape', 'window' (col0, row0, col1, row1 in the
    national grid), 'crs', 'units', 'source'}. Raises ValueError when the bbox misses the grid.
    """
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError('bbox must be (lon_min, lat_min, lon_max, lat_max)')
    base = (base or BASE).rstrip('/')
    cogs = {k: _open('%s/%s' % (base, FILES[k]), get) for k in layers}
    g = next(iter(cogs.values()))
    x0, dx, _, y0, _, dy = g.gt
    c0 = max(0, int(math.floor((lon_min - x0) / dx)))
    c1 = min(g.width, int(math.ceil((lon_max - x0) / dx)))
    r0 = max(0, int(math.floor((y0 - lat_max) / -dy)))
    r1 = min(g.height, int(math.ceil((y0 - lat_min) / -dy)))
    if c0 >= c1 or r0 >= r1:
        raise ValueError('bbox %s is outside the SMAP-HydroBlocks grid (CONUS)' % ((lon_min, lat_min, lon_max, lat_max),))
    for k, c in cogs.items():
        if c.gt != g.gt or (c.width, c.height) != (g.width, g.height):
            raise RuntimeError('layer %s is not on the grid of the others' % k)

    def one(k):
        raw = cogs[k].read(c0, r0, c1, r1)
        v = raw.astype(np.float32) * np.float32(SCALE)
        v[raw == NODATA] = np.nan
        return k, v

    with ThreadPoolExecutor(len(cogs)) as pool:
        out = dict(pool.map(one, cogs))
    out.update(geotransform=(x0 + c0 * dx, dx, 0.0, y0 + r0 * dy, 0.0, dy), shape=(r1 - r0, c1 - c0),
               window=(c0, r0, c1, r1), crs=CRS, units='m3/m3', source=SOURCE)
    return out
