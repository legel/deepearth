"""One zarr store per site: chip arrays per resolution group, UTC timestamps and per-chip metadata.

Layout of `<root>/sites/<site_id>.zarr` (zarr v2, Blosc zstd level 5 with bit shuffle):

    r10   (N, 4, 30, 30) uint16   B02 B03 B04 B08, DN
    r20   (N, 6, 15, 15) uint16   B05 B06 B07 B8A B11 B12, DN
    r60   (N, 2, 5, 5)   uint16   B01 B09, DN
    scl   (N, 15, 15)    uint8    Sen2Cor scene classification
    cld   (N, 15, 15)    uint8    Sen2Cor cloud probability (%), 255 where the source has none
    time_us (N,)         int64    acquisition time, microseconds since 1970-01-01T00:00:00 UTC
    meta/<field> (N,)             per-chip metadata (see META_FIELDS)

Resumability: the attribute `n_committed` counts the chips that were fully written. On open, arrays longer than
`n_committed` (a crash between writes) are truncated back to it. The committed `meta/acq_key` values are the
manifest of done acquisitions.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numcodecs
import numpy as np
import zarr

from bands import CLD_UNAVAILABLE, GROUPS

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
CHUNK = 256
META_FIELDS = {
    # name: dtype ("str" = variable-length UTF-8)
    "acq_key": "str", "item_id": "str", "source": "str", "platform": "str", "mgrs_tile": "str",
    "datatake_id": "str", "processing_baseline": "str", "generation_time": "str", "datetime_utc": "str",
    "epsg": "int32", "relative_orbit": "int16", "boa_add_offset": "int16",
    "sun_azimuth": "float32", "sun_elevation": "float32", "view_azimuth": "float32", "view_incidence": "float32",
    "scene_cloud_cover": "float32",
    "x_ul_r10": "float64", "y_ul_r10": "float64", "x_ul_r20": "float64", "y_ul_r20": "float64",
    "x_ul_r60": "float64", "y_ul_r60": "float64",
    "tower_dx_m": "float32", "tower_dy_m": "float32",
    "nodata_frac": "float32", "cloud_frac": "float32", "shadow_frac": "float32", "snow_frac": "float32",
    "invalid_frac": "float32", "clear_frac": "float32", "cld_mean": "float32", "has_cld": "bool",
}


def to_us(iso: str) -> int:
    """ISO 8601 UTC string ('2018-01-03T15:51:05.722000Z') to integer microseconds since the Unix epoch."""
    s = iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def from_us(us: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(us))


class SiteStore:
    def __init__(self, path: Path, site_id: str, lat: float, lon: float):
        self.path = Path(path)
        self.root = zarr.open_group(str(self.path), mode="a")
        if "r10" not in self.root:
            self._create(site_id, lat, lon)
        self._recover()

    def _create(self, site_id, lat, lon):
        g = self.root
        for grp in GROUPS:
            shape = (0, len(grp.bands), grp.size, grp.size) if grp.name in ("r10", "r20", "r60") else (0, grp.size,
                                                                                                         grp.size)
            g.create_dataset(grp.name, shape=shape, chunks=(CHUNK,) + shape[1:], dtype=grp.dtype,
                             compressor=COMPRESSOR, fill_value=CLD_UNAVAILABLE if grp.name == "cld" else 0)
            g[grp.name].attrs.update(bands=list(grp.bands), res_m=grp.res, units="DN" if grp.dtype == "uint16"
                                     else ("SCL class" if grp.name == "scl" else "percent"))
        g.create_dataset("time_us", shape=(0,), chunks=(4096,), dtype="int64", compressor=COMPRESSOR)
        g["time_us"].attrs.update(units="microseconds since 1970-01-01T00:00:00Z")
        m = g.require_group("meta")
        for name, dt in META_FIELDS.items():
            if dt == "str":
                m.create_dataset(name, shape=(0,), chunks=(4096,), dtype=object, object_codec=numcodecs.VLenUTF8(),
                                 compressor=COMPRESSOR)
            else:
                m.create_dataset(name, shape=(0,), chunks=(4096,), dtype=dt, compressor=COMPRESSOR)
        g.attrs.update(site_id=site_id, tower_lat=lat, tower_lon=lon, n_committed=0, chip_m=300,
                       format="fluxfill-sentinel2-v1")

    def arrays(self):
        yield from (self.root[g.name] for g in GROUPS)
        yield self.root["time_us"]
        yield from (self.root["meta"][k] for k in META_FIELDS)

    def _recover(self):
        n = int(self.root.attrs["n_committed"])
        for a in self.arrays():
            if a.shape[0] != n:
                a.resize((n,) + a.shape[1:])

    def __len__(self) -> int:
        return int(self.root.attrs["n_committed"])

    def done_keys(self) -> set:
        n = len(self)
        return set(self.root["meta"]["acq_key"][:n]) if n else set()

    def append(self, chips: list) -> None:
        """Append a batch of chips (dicts with group arrays, 'time_us' and the META_FIELDS) and commit."""
        if not chips:
            return
        n = len(self)
        for grp in GROUPS:
            self.root[grp.name].append(np.stack([c[grp.name] for c in chips]).astype(grp.dtype), axis=0)
        self.root["time_us"].append(np.array([c["time_us"] for c in chips], dtype="int64"))
        for k, dt in META_FIELDS.items():
            vals = [c[k] for c in chips]
            arr = np.array(vals, dtype=object) if dt == "str" else np.array(vals, dtype=dt)
            self.root["meta"][k].append(arr)
        self.root.attrs["n_committed"] = n + len(chips)

    def set_attrs(self, **kw):
        self.root.attrs.update(**kw)
