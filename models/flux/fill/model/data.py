"""FluxFill data: site records on a day-aligned half-hourly grid, the site split, standardisation, and the day-token
layout of a 180-day window.

A site record is a (V, T) float32 array of measured values (NaN where nothing was measured or a value failed a
quality rule), on a regular 30-minute grid of local standard time that starts at local midnight, so that the 48 steps
of a day-token are one solar day. Variables follow the benchmark (`bench/windows.py`): 15 tower variables, then 7
site-downscaled ERA5 reanalysis drivers (ONEFlux), then 8 "reanalysis-land" drivers from `data/drivers/`: ERA5-Land
TA, VPD, soil water and temperature at two depths and snow depth, and ERA5 PA moved to the tower elevation
(Copernicus DEM 30 m). The land drivers are hourly in UTC; each value is placed on the half-hour of local standard
time that it is valid at, and the half-hour between stays unobserved (as at hourly towers). Values are standardised
per variable with training-site statistics.

Sources: the final tables `towers/hh/<SITE>.parquet` once they exist, else the interim Shuttle tables
`towers/interim/shuttle/<SITE>.parquet` (data/towers/README.md). Both use `<VAR>_flag == 0` for a measured value.
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))

from masks import STEPS_PER_DAY, WINDOW_DAYS  # noqa: E402
from windows import EXTRA_TOWER_VARS, REANALYSIS_VARS, TOWER_VARS, lst_index, with_extra_columns  # noqa: E402

TOWERS = Path("/home/photon/4tb/fluxfill/towers")
DRIVERS = Path("/home/photon/4tb/fluxfill/drivers")
LAND_SOURCE = ("TA", "VPD", "SWC_1", "SWC_2", "TS_1", "TS_2", "D_SNOW", "PA")
LAND_VARS = tuple(f"LAND_{v}" for v in LAND_SOURCE)
# Channel set (E30): "base" (the rows above) or "extended" (plus windows.EXTRA_TOWER_VARS, further measured tower
# channels, appended after the land rows). Chosen per process before this module is imported, by the environment
# variable FLUXFILL_CHANNELS (train.py sets it from --extra-channels; a checkpoint records its set and method.py
# refuses a mismatch), because the row layout is fixed for the whole process.
CHANNELS = os.environ.get("FLUXFILL_CHANNELS", "base")
assert CHANNELS in ("base", "extended", "methane"), f"FLUXFILL_CHANNELS must be base, extended or methane, not {CHANNELS}"
# "methane" (E34): only the measured channels that control methane production and transport: water-table depth,
# soil temperature and soil water below the first layer.
METHANE_VARS = ("WTD", "TS_2", "TS_3", "SWC_2")
EXTRA_VARS: Tuple[str, ...] = (tuple(EXTRA_TOWER_VARS) if CHANNELS == "extended" else
                               METHANE_VARS if CHANNELS == "methane" else ())
V_BASE = len(TOWER_VARS) + len(REANALYSIS_VARS) + len(LAND_VARS)
VARS: Tuple[str, ...] = tuple(TOWER_VARS) + tuple(REANALYSIS_VARS) + LAND_VARS + EXTRA_VARS
V_TAB = len(VARS)
FLUX_VARS = ("NEE", "LE", "H", "FCH4")
PROVENANCE = {"tower": 0, "reanalysis": 1, "reanalysis-land": 2, "imagery": 3}
PROV_OF_VAR = np.array([0] * len(TOWER_VARS) + [1] * len(REANALYSIS_VARS) + [2] * len(LAND_VARS) + [0] * len(EXTRA_VARS),
                       np.int64)
IS_EXTRA = np.arange(V_TAB) >= V_BASE
IS_LAND = PROV_OF_VAR == 2
IS_FLUX = np.array([v in FLUX_VARS for v in VARS])
IS_TOWER = PROV_OF_VAR == 0
IS_TOWER_MET = IS_TOWER & ~IS_FLUX
IS_REAN = PROV_OF_VAR == 1
T_WIN = WINDOW_DAYS * STEPS_PER_DAY


def table_dir(root: Path = TOWERS) -> Path:
    """The final per-site tables when the build has written them, else the interim Shuttle tables."""
    hh = root / "hh"
    return hh if hh.exists() and any(hh.glob("*.parquet")) else root / "interim" / "shuttle"


# ---------------------------------------------------------------------------------------------------------- split

def provisional_is_val(site: str) -> bool:
    """Provisional 20 % validation hold-out by a seeded hash of the site id, used only until split.csv exists."""
    h = int.from_bytes(hashlib.sha256(f"fluxfill-provisional-v0|{site}".encode()).digest()[:8], "little")
    return h % 100 < 20


def load_split(sites: Sequence[str], root: Path = TOWERS) -> Tuple[Dict[str, str], str]:
    """{site: 'train' | 'val'} and the split's name. Test sites of the frozen split are dropped here, so no code
    downstream can read them."""
    # FLUXFILL_SPLIT selects another split file with the same test sites (split_v2.csv: 13 more methane validation
    # sites); the frozen split.csv itself is never changed (the test release manifest records its checksum).
    f = root / os.environ.get("FLUXFILL_SPLIT", "split.csv")
    if f.exists():
        s = pd.read_csv(f)
        col = "site_id" if "site_id" in s else s.columns[0]
        scol = "split" if "split" in s else [c for c in s.columns if c != col][0]
        m = dict(zip(s[col].astype(str), s[scol].astype(str)))
        return {k: m[k] for k in sites if m.get(k) in ("train", "val")}, "frozen" if f.name == "split.csv" else f.stem
    return {k: ("val" if provisional_is_val(k) else "train") for k in sites}, "provisional"


def scaling_order(train_sites: Sequence[str], root: Path = TOWERS, seed: str = "fluxfill-scaling-v1") -> List[str]:
    """Training sites in the order of the data-scaling study: within each stratum of the frozen split (IGBP group x
    Koppen main class x FCH4) sites are ranked by SHA-256 of `seed|site`; each site's key is (rank + 0.5) / stratum
    size, and sites are sorted by key (hash as tie-break). Every prefix of this list is a nested, stratified subset:
    the first n sites hold each stratum in proportion to its size, up to one site."""
    s = pd.read_csv(root / "split.csv").set_index("site_id")
    hs = lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest()
    keys = {}
    by = {}
    for site in train_sites:
        by.setdefault(str(s.loc[site, "stratum"]) if site in s.index else "?", []).append(site)
    for st, members in by.items():
        members = sorted(members, key=hs)
        for r, site in enumerate(members):
            keys[site] = ((r + 0.5) / len(members), hs(site))
    return sorted(train_sites, key=lambda x: keys[x])


# ---------------------------------------------------------------------------------------------------------- records

@dataclass
class SiteRecord:
    site: str
    t0: pd.Timestamp                      # local standard time of step 0 (a midnight)
    values: np.ndarray                    # (V_TAB, T) float32 raw units, NaN where not measured
    lat: float = np.nan
    lon: float = np.nan
    elev: float = np.nan
    utc_offset_h: float = 0.0
    heldout_year: Optional[int] = None    # R2 year of a training site, never trained on
    meta: Dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return self.values.shape[1] // STEPS_PER_DAY

    def day_of(self, ts) -> int:
        return int((pd.Timestamp(ts) - self.t0) / pd.Timedelta(days=1))


def measured_matrix(table: pd.DataFrame, t: pd.DatetimeIndex) -> np.ndarray:
    """(V_TAB, len(t)) float32: values where measured (flag 0 and finite; reanalysis where finite), NaN elsewhere."""
    tab = table.set_index(lst_index(table))
    tab = tab[~tab.index.duplicated()].reindex(t)
    out = np.full((V_TAB, len(t)), np.nan, np.float32)
    for i, v in enumerate(VARS):
        if v not in tab:
            continue
        x = tab[v].to_numpy(np.float32)
        ok = np.isfinite(x)
        if f"{v}_flag" in tab:
            ok &= tab[f"{v}_flag"].to_numpy() == 0
        out[i] = np.where(ok, x, np.nan)
    return out


_LAND_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def land_table(site: str) -> Optional[pd.DataFrame]:
    """The site's reanalysis-land driver table (data/drivers/load.py `baseline`), hourly, UTC; None if absent."""
    if site not in _LAND_CACHE:
        if not (DRIVERS / "era5_land" / f"{site}.parquet").exists():
            _LAND_CACHE[site] = None
        else:
            sys.path.insert(0, str(HERE.parent / "data" / "drivers"))
            from load import baseline
            try:
                _LAND_CACHE[site] = baseline(site)
            except Exception as e:  # a driver file for a site the driver build skipped
                print(f"no land drivers for {site}: {e}", flush=True)
                _LAND_CACHE[site] = None
    return _LAND_CACHE[site]


def land_matrix(site: str, t: pd.DatetimeIndex, utc_offset_h: float) -> np.ndarray:
    """(len(LAND_VARS), len(t)) float32 land drivers on the LST half-hour grid `t` (NaN where none)."""
    out = np.full((len(LAND_VARS), len(t)), np.nan, np.float32)
    lt = land_table(site)
    if lt is None:
        return out
    tu = pd.DatetimeIndex(lt["time_utc"])
    tu = tu.tz_convert(None) if tu.tz is not None else tu
    tl = tu + pd.Timedelta(hours=utc_offset_h)
    pos = t.get_indexer(tl)
    ok = pos >= 0
    for i, v in enumerate(LAND_SOURCE):
        if v in lt:
            out[i, pos[ok]] = lt[v].to_numpy(np.float32)[ok]
    return out


def record_from_table(tab: pd.DataFrame, site: str, sites_meta: Optional[pd.DataFrame] = None,
                      land: bool = True) -> SiteRecord:
    """A SiteRecord from a per-site table (final or interim format) on its own day-aligned grid."""
    if EXTRA_VARS:                                 # wind direction as sine and cosine rows (E30)
        tab = with_extra_columns(tab)
    idx = lst_index(tab)
    t0 = idx.min().normalize()
    t1 = idx.max().normalize() + pd.Timedelta(days=1)
    t = pd.date_range(t0, t1, freq="30min", inclusive="left")
    rec = SiteRecord(site, t0, measured_matrix(tab, t))
    if "t_utc" in tab and "t_lst" in tab:
        rec.utc_offset_h = float(np.median((tab["t_lst"].to_numpy() - tab["t_utc"].to_numpy())[:1000]) / 60.0)
    elif "time_utc" in tab:
        # The final tables store time_utc tz-aware (UTC) and time_lst naive (local standard time).
        utc = pd.DatetimeIndex(tab["time_utc"][:1])
        utc = utc.tz_convert(None) if utc.tz is not None else utc
        rec.utc_offset_h = float((pd.DatetimeIndex(tab["time_lst"][:1]) - utc)[0] / pd.Timedelta(hours=1))
    if sites_meta is not None and site in sites_meta.index:
        r = sites_meta.loc[site]
        rec.lat, rec.lon = float(r["lat"]), float(r["lon"])
        rec.elev = float(r["elevation_m"]) if np.isfinite(r.get("elevation_m", np.nan)) else np.nan
    if land:
        lm = land_matrix(site, t, rec.utc_offset_h)
        # Land drivers only on days the table covers: a context table without the target window's years must not
        # bring those years back through the drivers (memory.py builds site memory from such tables).
        covered = pd.DatetimeIndex(t.normalize()).isin(pd.DatetimeIndex(idx.normalize()).unique())
        rec.values[IS_LAND] = np.where(covered[None], lm, np.nan)
    return rec


def load_record(path: Path, sites_meta: Optional[pd.DataFrame] = None) -> SiteRecord:
    src = [v for v in VARS if not v.startswith("WD_")] + (["WD"] if "WD_SIN" in VARS else [])
    cols = ["t_lst", "time_lst", "t_utc", "time_utc"] + [c for v in src for c in (v, f"{v}_flag")]
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    tab = pd.read_parquet(path, columns=[c for c in cols if c in have])
    return record_from_table(tab, path.stem, sites_meta)


def sites_table(root: Path = TOWERS) -> pd.DataFrame:
    f = root / "sites.parquet"
    s = pd.read_parquet(f) if f.exists() else pd.read_csv(root / "sites.csv")
    return s.drop_duplicates("site_id").set_index("site_id")


def heldout_year(rec: SiteRecord) -> Optional[int]:
    """The last full calendar year of the record with a flux measured in at least 10 % of its steps (regime R2)."""
    t = rec.t0 + pd.to_timedelta(np.arange(rec.values.shape[1]) * 30, unit="m")
    any_flux = np.isfinite(rec.values[IS_FLUX]).any(0)
    years = pd.Series(any_flux, index=t.year)
    frac = years.groupby(level=0).mean()
    first, last = t[0], t[-1]
    full = [y for y in frac.index if pd.Timestamp(y, 1, 1) >= first and pd.Timestamp(y, 12, 31, 23, 30) <= last]
    ok = [y for y in full if frac[y] >= 0.10]
    return max(ok) if ok else None


def load_records(sites: Sequence[str], root: Path = TOWERS, verbose: bool = True) -> Dict[str, SiteRecord]:
    d = table_dir(root)
    meta = sites_table(root)
    out = {}
    for k, s in enumerate(sites):
        p = d / f"{s}.parquet"
        if not p.exists():
            continue
        try:
            out[s] = load_record(p, meta)
        except Exception as e:  # an interim table being written while read
            print(f"skip {s}: {e}", flush=True)
        if verbose and k % 20 == 0:
            print(f"loaded {k + 1}/{len(sites)} sites", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------- stats

@dataclass
class Stats:
    mean: np.ndarray       # (V_TAB,)
    std: np.ndarray        # (V_TAB,)

    def norm(self, x: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
        r = slice(None) if rows is None else rows
        return (x - self.mean[r, None]) / self.std[r, None]

    def denorm(self, z, rows: Optional[np.ndarray] = None):
        r = slice(None) if rows is None else rows
        return z * self.std[r, None] + self.mean[r, None]

    def to_dict(self):
        return {"vars": list(VARS), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d):
        assert tuple(d["vars"]) == VARS, (
            f"variable list changed: the checkpoint has {len(d['vars'])} rows, this process {len(VARS)} "
            f"(FLUXFILL_CHANNELS={CHANNELS}; a checkpoint trained with --extra-channels needs FLUXFILL_CHANNELS=extended)")
        return Stats(np.array(d["mean"], np.float64), np.array(d["std"], np.float64))


def fit_stats(records: Dict[str, SiteRecord], train_sites: Sequence[str]) -> Stats:
    """Mean and standard deviation of every variable over all measured values of the training sites, held-out years
    excluded."""
    s1, s2, n = np.zeros(V_TAB), np.zeros(V_TAB), np.zeros(V_TAB)
    for s in train_sites:
        r = records[s]
        x = r.values.astype(np.float64)
        if r.heldout_year is not None:
            t = r.t0 + pd.to_timedelta(np.arange(x.shape[1]) * 30, unit="m")
            x = x[:, t.year != r.heldout_year]
        ok = np.isfinite(x)
        s1 += np.where(ok, x, 0).sum(1); s2 += np.where(ok, x * x, 0).sum(1); n += ok.sum(1)
    mean = s1 / np.maximum(n, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
    std = np.where(std > 1e-6, std, 1.0)
    return Stats(mean, std)


# ---------------------------------------------------------------------------------------------------------- tokens

def site_features(lat: float, lon: float, elev: float) -> np.ndarray:
    """Location as a point on the unit sphere plus elevation in km (0 where unknown)."""
    if not np.isfinite(lat):
        return np.zeros(4, np.float32)
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la),
                     (elev / 1000.0) if np.isfinite(elev) else 0.0], np.float32)


def doy_features(start: pd.Timestamp, n_days: int = WINDOW_DAYS) -> np.ndarray:
    """(n_days, 4): sine and cosine of the first two annual harmonics at each day's midday."""
    d = pd.date_range(start + pd.Timedelta(hours=12), periods=n_days, freq="D")
    a = 2 * np.pi * (d.day_of_year.to_numpy() - 0.5) / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)], 1).astype(np.float32)


def window_tokens(values: np.ndarray, stats: Stats) -> Tuple[np.ndarray, np.ndarray]:
    """(V_TAB, T_WIN) raw values -> standardised day-tokens z (V, D, 48) with NaN -> 0, and measured (V, D, 48)."""
    meas = np.isfinite(values)
    z = np.where(meas, stats.norm(values.astype(np.float64)), 0.0).astype(np.float32)
    D = values.shape[1] // STEPS_PER_DAY
    return z.reshape(V_TAB, D, STEPS_PER_DAY), meas.reshape(V_TAB, D, STEPS_PER_DAY)


def flux_measured_days(rec: SiteRecord) -> np.ndarray:
    """(n_days,) number of half-hours per day with any flux measured, for the usable-window rule."""
    m = np.isfinite(rec.values[IS_FLUX]).any(0)
    return m[: rec.n_days * STEPS_PER_DAY].reshape(rec.n_days, STEPS_PER_DAY).sum(1)


def valid_starts(rec: SiteRecord, exclude_heldout: bool, min_flux_frac: float = 0.20) -> np.ndarray:
    """Day indices where a 180-day window lies in the record, has measured fluxes in at least `min_flux_frac` of its
    steps (the dataset's usable rule), and, for training, does not touch the held-out year."""
    D = rec.n_days
    if D < WINDOW_DAYS:
        return np.zeros(0, np.int64)
    c = np.concatenate(([0], np.cumsum(flux_measured_days(rec))))
    frac = (c[WINDOW_DAYS:] - c[:-WINDOW_DAYS]) / T_WIN
    ok = frac >= min_flux_frac
    if exclude_heldout and rec.heldout_year is not None:
        days = rec.t0 + pd.to_timedelta(np.arange(D), unit="D")
        in_y = np.concatenate(([0], np.cumsum(days.year == rec.heldout_year)))
        ok &= (in_y[WINDOW_DAYS:] - in_y[:-WINDOW_DAYS]) == 0
    return np.flatnonzero(ok)


def window_values(rec: SiteRecord, start_day: int) -> np.ndarray:
    a = start_day * STEPS_PER_DAY
    return rec.values[:, a:a + T_WIN]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
