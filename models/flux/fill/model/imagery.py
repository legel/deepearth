"""Imagery tokens: Clay v1.5 CLS embeddings of the Sentinel-2 chips around a tower (embeddings/clay/README.md), as
one token per day of the window.

A day with one or more acquisitions gets the mean CLS of that day's chips (two chips on one day come from overlapping
tiles of one overpass) and their SCL fractions; a day without an acquisition is unobserved. Cloudy chips are kept,
with their cloud, shadow, snow and no-data fractions as token features, so the model learns how far to trust each
embedding. Acquisition times are UTC and are moved to the site's local standard time, the time base of the towers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CLAY = Path("/home/photon/4tb/fluxfill/clay/sites")
CLS_DIM = 1024
META_KEYS = ("cloud_frac", "shadow_frac", "snow_frac", "invalid_frac", "clear_frac", "nodata_frac", "cld_mean")
N_META = len(META_KEYS) + 1      # + number of chips that day (scaled)


@dataclass
class SiteImagery:
    cls: np.ndarray        # (N, 1024) float16
    day: np.ndarray        # (N,) int64 days since 1970-01-01 in local standard time
    meta: np.ndarray       # (N, len(META_KEYS)) float32


def load_imagery(site: str, utc_offset_h: float, root: Path = CLAY) -> Optional[SiteImagery]:
    p = root / f"{site}.zarr"
    if not p.exists():
        return None
    import zarr
    g = zarr.open(str(p), mode="r")
    n = int(g.attrs.get("n_committed", g["cls"].shape[0]))
    if n == 0:
        return None
    t = g["time_us"][:n].astype(np.int64)
    day = (t + int(utc_offset_h * 3600e6)) // 86_400_000_000
    meta = np.stack([np.asarray(g["meta"][k][:n], np.float32) for k in META_KEYS], 1)
    meta = np.nan_to_num(meta, nan=1.0)
    meta[:, META_KEYS.index("cld_mean")] /= 100.0        # cloud probability in percent -> fraction
    return SiteImagery(np.asarray(g["cls"][:n]), day, meta)


def window_imagery(img: Optional[SiteImagery], start: pd.Timestamp, n_days: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-day features of a window: CLS (D, 1024) float32, meta (D, N_META), and a has-acquisition flag (D,)."""
    cls = np.zeros((n_days, CLS_DIM), np.float32)
    meta = np.zeros((n_days, N_META), np.float32)
    has = np.zeros(n_days, bool)
    if img is None:
        return cls, meta, has
    d0 = int((pd.Timestamp(start) - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1))
    rel = img.day - d0
    sel = np.flatnonzero((rel >= 0) & (rel < n_days))
    if sel.size == 0:
        return cls, meta, has
    cnt = np.zeros(n_days)
    np.add.at(cnt, rel[sel], 1)
    np.add.at(cls, rel[sel], img.cls[sel].astype(np.float32))
    np.add.at(meta[:, :-1], rel[sel], img.meta[sel])
    has = cnt > 0
    cls[has] /= cnt[has, None]
    meta[has, :-1] /= cnt[has, None]
    meta[:, -1] = cnt / 4.0
    return cls, meta, has
