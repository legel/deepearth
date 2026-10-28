"""The M3 donor library: real joint outages of all tower fluxes, from training-site records.

An outage is a run of half-hours in which none of NEE, LE, H and FCH4 has a measured value at a site that measures
at least one of them in that year. Runs come from training sites only (towers/split.csv), so a test window never
receives a pattern taken from its own record. A 180-day donor slice is drawn by `masks.cut_outage` from one donor
site-year chosen by the window's seed.

Usage: python outages.py <towers dir> <out.parquet>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FLUXES = ("NEE", "LE", "H", "FCH4")


def site_outages(table: pd.DataFrame) -> pd.DataFrame:
    """One row per site-year: the boolean all-fluxes-missing series packed as run starts and lengths."""
    t = pd.DatetimeIndex(table["time_lst"])
    meas = np.zeros(len(table), bool)
    for v in FLUXES:
        if v in table:
            meas |= table[v].notna().to_numpy() & (table.get(f"{v}_flag", 0) == 0)
    rows = []
    for y in np.unique(t.year):
        sel = t.year == y
        m = meas[sel]
        if not m.any() or sel.sum() < 48 * 300:
            continue                       # the year has no measured flux, or is too short to donate from
        miss = ~m
        e = np.flatnonzero(np.diff(np.concatenate(([0], miss.astype(np.int8), [0]))))
        rows.append({"year": int(y), "n": int(sel.sum()), "starts": e[::2].astype(np.int32),
                     "lengths": (e[1::2] - e[::2]).astype(np.int32), "missing_frac": float(miss.mean())})
    return pd.DataFrame(rows)


def series(row) -> np.ndarray:
    """The all-fluxes-missing boolean series of one donor site-year."""
    out = np.zeros(int(row["n"]), bool)
    for s, l in zip(row["starts"], row["lengths"]):
        out[s:s + l] = True
    return out


def main():
    towers, out = Path(sys.argv[1]), Path(sys.argv[2])
    split = pd.read_csv(towers / "split.csv")
    train = split[split.split == "train"].site_id
    frames = []
    for site in train:
        f = towers / "hh" / f"{site}.parquet"
        if not f.exists():
            continue
        import pyarrow.parquet as pq
        have = set(pq.read_schema(f).names)
        cols = [c for c in ["time_lst"] + [c for v in FLUXES for c in (v, f"{v}_flag")] if c in have]
        tab = pd.read_parquet(f, columns=cols)
        d = site_outages(tab)
        if len(d):
            d.insert(0, "site", site)
            frames.append(d)
    lib = pd.concat(frames, ignore_index=True)
    lib.to_parquet(out, index=False)
    print(f"{len(lib)} donor site-years from {lib.site.nunique()} training sites; "
          f"median all-flux missing fraction {lib.missing_frac.median():.2f}")


if __name__ == "__main__":
    main()
