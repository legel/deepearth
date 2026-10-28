"""M1: distribution-matched artificial gaps (Irvin et al. 2021, Agric. For. Meteorol. 308-309:108528).

The gap-length sampling distribution is learned per site so that artificial gaps merged with the site's real gaps
reproduce the observed gap-length distribution. This module calls the released implementation (FluxGapfill,
github.com/stanfordmlgroup/methane-gapfill-ml, Apache-2.0; Irvin et al. 2021, Zenodo 10.5281/zenodo.5515761; local
clone below), whose `sample_artificial_gaps` there is an exact, faster reformulation of the released sampler
(identical output and random state). M1 is drawn once over a site's whole record on its primary flux, then cut into
windows, so it keeps the paper's semantics: about `eval_frac` of the measured half-hours of the record are hidden.

FluxFill-Bench choices, each stated:
1. Primary flux: NEE where the site measures it, else the first of LE, H, FCH4 that it measures (the paper's target
   is FCH4; the benchmark's first target is NEE). The record is the site table on its native step (30 or 60 min)
   from its first to its last row, measured values only (flag 0).
2. The M1 positions hide every tower channel, as a station outage does (as M2 and M3).
3. Seed: per site, `seed_for(site, "record", "M1") mod 2**31`, for both the learning and the sampling.
4. Monte Carlo draws per grid point: N_MC = 5 in place of the paper's 50. The grid stays 10 x 10 (alpha, p) and the
   distance Cramer-von Mises. With 50 draws the learning costs about 37 core-hours over the 131 test-site records
   (measured: 0.05 s per draw on a 40,176-step record); with 5 draws about 4.
Learned record masks are cached in CACHE/<site>.npz (mask, record times, primary flux, parameters, pmf).
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HARNESS = "/home/photon/ecological/flux_gapfill/methane-gapfill-ml"
CACHE = Path("/home/photon/4tb/fluxfill/bench/m1")
EVAL_FRAC, N_GRID, N_MC, DIST = 0.1, 10, 5, "CramerVonMises"
PRIMARY = ("NEE", "LE", "H", "FCH4")


def learn_and_sample(flux_record: np.ndarray, eval_frac: float = 0.1, seed: int = 1000, dist: str = "CramerVonMises",
                     n_grid: int = 10, n_mc: int = 50):
    """Boolean mask over the record (True = artificial gap on a measured half-hour), and the learned pmf."""
    if HARNESS not in sys.path:
        sys.path.insert(0, HARNESS)
    from fluxgapfill.preprocess.artificial import learn_gap_dist, sample_artificial_gaps
    flux = np.asarray(flux_record, np.float64)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pmf = learn_gap_dist(flux_data=flux, dist=dist, n_grid=n_grid, n_mc=n_mc, seed=seed)
        masked = sample_artificial_gaps(flux_data=flux, sampling_pmf=pmf, eval_frac=eval_frac, seed=seed)
    return np.isnan(masked) & np.isfinite(flux), pmf


def primary_record(table):
    """(flux name, regular record times, measured values with NaN elsewhere) of a site table."""
    import pandas as pd
    from windows import lst_index
    idx = lst_index(table)
    order = np.argsort(idx.asi8, kind="stable")
    for k in PRIMARY:
        if k not in table:
            continue
        x = table[k].to_numpy(np.float64)[order]
        f = table.get(f"{k}_flag")
        ok = np.isfinite(x) & ((f.to_numpy()[order] == 0) if f is not None else True)
        if ok.sum() < 48 * 30:
            continue
        t = idx[order]
        step = pd.Timedelta(np.median(np.diff(t.asi8)), "ns")
        grid = pd.date_range(t[0], t[-1], freq=step)
        s = pd.Series(np.where(ok, x, np.nan), index=t)
        s = s[~s.index.duplicated()].reindex(grid)
        return k, grid, s.to_numpy(np.float64)
    return None, None, None


def m1_excluded(cache: Path = CACHE) -> dict:
    """Sites without an M1 mask, with the reason: {site: reason}, from <cache>/EXCLUDED.json. The released harness's
    sampler (sample_artificial_gaps) loops without end when the record is so fragmented that no run of unmasked
    half-hours is long enough for any gap length its learned distribution can draw before the eval fraction is met."""
    import json
    p = cache / "EXCLUDED.json"
    return json.loads(p.read_text()) if p.exists() else {}


def record_mask(site: str, table=None, cache: Path = CACHE):
    """The site's M1 record mask and its times, from the cache, else learned from `table` and cached."""
    from masks import seed_for
    p = cache / f"{site}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return np.unpackbits(z["mask"])[: int(z["n"])].astype(bool), z["t"].astype("datetime64[m]")
    if table is None:
        raise FileNotFoundError(f"no M1 record mask for {site} and no table to learn it from")
    k, grid, x = primary_record(table)
    if k is None:
        return np.zeros(0, bool), np.zeros(0, "datetime64[m]")
    seed = seed_for(site, "record", "M1") % 2 ** 31
    m, pmf = learn_and_sample(x, eval_frac=EVAL_FRAC, seed=seed, dist=DIST, n_grid=N_GRID, n_mc=N_MC)
    cache.mkdir(parents=True, exist_ok=True)
    t = np.asarray(grid.values, "datetime64[m]")
    np.savez_compressed(p, mask=np.packbits(m), n=m.size, t=t.astype("int64"), flux=k, seed=seed, pmf=pmf,
                        params=np.array([EVAL_FRAC, N_GRID, N_MC]), dist=DIST)
    return m, t


if __name__ == "__main__":
    # Learn and cache the M1 record masks of every site in a window list: python irvin_gaps.py <windows.json> [k/n]
    import json
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    specs = json.loads(Path(sys.argv[1]).read_text())
    specs = specs["windows"] if isinstance(specs, dict) else specs
    sites = sorted({(s["site"], s["table"]) for s in specs})
    if len(sys.argv) > 2:
        k, n = map(int, sys.argv[2].split("/"))
        sites = sites[k::n]
    import time
    for site, tab in sites:
        if (CACHE / f"{site}.npz").exists():
            continue
        t0 = time.time()
        m, t = record_mask(site, pd.read_parquet(tab))
        print(f"{site}: {m.size} steps, {int(m.sum())} masked, {time.time() - t0:.0f} s", flush=True)
