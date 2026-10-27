"""Clay v1.5 inputs from FluxFill Sentinel-2 chips.

Clay v1.5 (Clay Foundation 2024; code github.com/Clay-foundation/model, `configs/metadata.yaml`, the wall-to-wall
tutorial) takes for Sentinel-2 L2A:
- 10 bands in the order blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22
  (B02 B03 B04 B05 B06 B07 B08 B8A B11 B12) at 10 m GSD; B01 and B09 are not part of its Sentinel-2 input;
- pixels normalised per band as (DN - mean) / std with the metadata means and standard deviations;
- band central wavelengths in micrometres;
- time as (sin, cos) of week-of-year x 2 pi / 52 and (sin, cos) of hour x 2 pi / 24, in UTC;
- location as (sin, cos) of latitude and (sin, cos) of longitude, in radians;
- the GSD in metres, which scales the 2-D sine-cosine patch position encoding;
- a chip whose height and width are multiples of the patch size, 8.

FluxFill chip handling (decision recorded in README.md):
1. DN are harmonised across processing baselines: DN_h = DN + BOA_ADD_OFFSET = 10000 x reflectance.
2. 20 m bands are resampled to the 10 m grid by nearest neighbour (each 20 m pixel becomes 2 x 2), as in Clay's
   Sentinel-2 tutorials (stackstac, Resampling.nearest at 10 m).
3. The 30 x 30 px chip is edge-padded (the border pixel repeated) by one pixel on each side to 32 x 32 px (4 x 4
   patches of 8 px) at the native 10 m GSD. Of the modes of `to_size`, it gave the CLS closest to that of a true
   32 x 32 px chip read from the source (`sensitivity.py`; README.md).
4. No-data pixels (DN 0) are set to the band mean, i.e. 0 after normalisation.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np

CLAY_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
CLAY_NAMES = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")
# configs/metadata.yaml, sentinel-2-l2a, Clay commit f14e698f3c (2026-03-27); unchanged since the v1.5 release.
MEAN = np.array([1105., 1355., 1552., 1887., 2422., 2630., 2743., 2785., 2388., 1835.], np.float32)
STD = np.array([1809., 1757., 1888., 1870., 1732., 1697., 1742., 1648., 1470., 1379.], np.float32)
WAVES_UM = np.array([0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19], np.float32)
GSD = 10.0
PATCH = 8
R10 = ("B02", "B03", "B04", "B08")
R20 = ("B05", "B06", "B07", "B8A", "B11", "B12")


def upsample_r20(r20: np.ndarray, x_ul_r10, y_ul_r10, x_ul_r20, y_ul_r20) -> np.ndarray:
    """(N, 6, 15, 15) 20 m chips onto the (N, 6, 30, 30) 10 m chip grid, nearest neighbour, edge-clamped.

    The two windows come from one tile grid, so their corners differ by whole 10 m pixels (0 or +-10 m per axis).
    """
    n = r20.shape[0]
    out = np.empty((n, r20.shape[1], 30, 30), r20.dtype)
    centres = (np.arange(30) + 0.5) * 10.0
    for i in range(n):
        cols = np.clip(np.floor((x_ul_r10[i] + centres - x_ul_r20[i]) / 20.0).astype(int), 0, 14)
        rows = np.clip(np.floor((y_ul_r20[i] - (y_ul_r10[i] - centres)) / 20.0).astype(int), 0, 14)
        out[i] = r20[i][:, rows[:, None], cols[None, :]]
    return out


def clay_stack(r10: np.ndarray, r20_10m: np.ndarray, offset: np.ndarray) -> tuple:
    """Harmonised float32 (N, 10, 30, 30) in Clay band order, and the no-data mask (N, 30, 30)."""
    r10 = r10.astype(np.float32)
    r20 = r20_10m.astype(np.float32)
    # Clay order: B02 B03 B04 | B05 B06 B07 | B08 | B8A B11 B12
    x = np.concatenate([r10[:, 0:3], r20[:, 0:3], r10[:, 3:4], r20[:, 3:6]], axis=1)
    nodata = (r10 == 0).all(axis=1)
    bad = x == 0
    x = x + offset.astype(np.float32)[:, None, None, None]
    x = np.where(bad, MEAN[None, :, None, None], x)
    return x, nodata


def normalise(x: np.ndarray) -> np.ndarray:
    return (x - MEAN[None, :, None, None]) / STD[None, :, None, None]


def to_size(x: np.ndarray, mode: str) -> tuple:
    """A (N, C, 30, 30) normalised chip as a Clay input of side divisible by 8, and its GSD.

    reflect   1-px reflect pad to 32 x 32 (pixels mirrored about the edge pixel), GSD 10 m
    edge      1-px replicate pad to 32 x 32, GSD 10 m   [chosen]
    zero      1-px pad with the band mean (0 after normalisation) to 32 x 32, GSD 10 m
    bilinear  bilinear resampling 30 -> 32 px over the same 300 m, GSD 9.375 m
    crop24    central 24 x 24 px (240 m, 3 x 3 patches), GSD 10 m
    """
    if mode in ("reflect", "edge"):
        return np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode=mode), GSD
    if mode == "zero":
        return np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant"), GSD
    if mode == "crop24":
        return x[..., 3:27, 3:27], GSD
    if mode == "bilinear":
        import torch
        import torch.nn.functional as F
        t = F.interpolate(torch.from_numpy(np.ascontiguousarray(x)), size=(32, 32), mode="bilinear",
                          align_corners=False)
        return t.numpy(), 300.0 / 32
    raise ValueError(mode)


def time_features(time_us: np.ndarray) -> np.ndarray:
    """(N, 4): sin, cos of ISO week x 2 pi / 52, then sin, cos of UTC hour x 2 pi / 24 (Clay normalize_timestamp)."""
    out = np.empty((len(time_us), 4), np.float32)
    for i, us in enumerate(time_us):
        d = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(us))
        week = d.isocalendar().week * 2 * math.pi / 52
        hour = d.hour * 2 * math.pi / 24
        out[i] = (math.sin(week), math.cos(week), math.sin(hour), math.cos(hour))
    return out


def latlon_features(lat: float, lon: float, n: int) -> np.ndarray:
    """(N, 4): sin, cos of latitude, then sin, cos of longitude (Clay normalize_latlon)."""
    la, lo = math.radians(lat), math.radians(lon)
    return np.tile(np.array([math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)], np.float32), (n, 1))
