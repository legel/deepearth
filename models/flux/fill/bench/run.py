"""FluxFill-Bench runner: every method on every (window, mask) of a regime, scored per variable and gap class.

A method is any object with
    name: str
    predict(window: Window, mask: np.ndarray, context: dict) -> np.ndarray   # (S, V, T) joint samples
where `mask` (V, T) is True on hidden values; the method must not read `window.values[mask]` (the runner passes a
copy with those values set to NaN). `context` carries the site's full table for methods that use other years
(climatology, site memory); the runner removes the target window's calendar years from it, so no method sees
values from the year it fills.

Output: one parquet row per (method, regime, family, site, window, variable, gap class) with CRPS, MAE, RMSE, bias,
coverage, n and the truth sums sum_y, sum_y2, sum_abs_y (for pooled R2 and percentage error); and one row per
(method, ..., variable, sum block) with the CRPS and bias of daily, 30-day and window sums from the joint samples.
Point scores (MAE, RMSE, bias) use each method's predictive mean, the mean of its samples.

Regimes: R1 (default) as above. R3, the ungauged site (upscaling): the context is empty (no other year of the site,
so no climatology pool and no site memory) and only the M2-180d mask is scored, so every tower channel of the
window is hidden and the method has the window's reanalysis drivers, the site's location and its imagery only.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))

from masks import (BLOCK_DAYS, M5_MODALITIES, STEPS_PER_DAY, SUBDAILY_HOURS, contiguous_block,  # noqa: E402
                   contiguous_steps, cut_outage, mixed_blocks, modality_dropout, seed_for, transplanted_outage,
                   window_from_record)
import outages  # noqa: E402
from metrics import (GAP_CLASSES, block_sums, coverage_samples, crps_samples, gap_class, gap_runs)  # noqa: E402
from windows import EXTRA_TOWER_VARS, TOWER_VARS, Window, load_window, lst_index, with_extra_columns  # noqa: E402

FLUX_KEYS = ("NEE", "LE", "H", "FCH4", "G")
"""Tower fluxes: the rows M1 hides and the M5 \"fluxes\" modality."""

SUM_BLOCKS = {"daily": STEPS_PER_DAY, "30-day": 30 * STEPS_PER_DAY, "window": 180 * STEPS_PER_DAY}
FLUXES = ("NEE", "LE", "H", "FCH4")


def masks_for(w: Window, families: Iterable[str]) -> Dict[str, np.ndarray]:
    """The benchmark masks of one window, keyed by name (docs/BENCHMARK.md section 5).

    M1 hides the tower fluxes where the site's distribution-matched record mask (irvin_gaps.py) is set. M2 hides all
    tower channels for one block per length; M2-sub for one sub-daily block per length. M5 hides one block per tower
    and reanalysis row, each with its own log-uniform length and placement, and in addition one whole modality chosen
    by the seed (tower meteorology, reanalysis or all fluxes).

    Extra tower rows (--extra-tower, windows.EXTRA_TOWER_VARS) are tower rows for M2, M2-sub and M3 and tower
    meteorology for M5's modality; their M5 blocks come from a separate seed stream ("M5x"), so the masks of the
    release rows are those of the release whether or not extra rows are present."""
    V, T = w.values.shape
    ext = [i for i in w.rows("tower") if w.keys[i] in EXTRA_TOWER_VARS]
    tower, rean = [i for i in w.rows("tower") if i not in ext], w.rows("reanalysis")
    flux = [i for i in tower if w.keys[i] in FLUX_KEYS]
    met = [i for i in tower if w.keys[i] not in FLUX_KEYS]
    all_tower = tower + ext
    out = {}
    start = str(w.start.date())
    for fam in families:
        if fam == "M1" and flux:
            from irvin_gaps import m1_excluded, record_mask
            if w.site in m1_excluded():
                w.meta["M1"] = "excluded: " + m1_excluded()[w.site]
                continue
            rm, rt = record_mask(w.site)
            m = np.zeros((V, T), bool)
            if rm.size:
                m[flux] = window_from_record(rm, rt, w.t.values)[None, :]
            out["M1"] = m
        elif fam == "M2":
            for d in BLOCK_DAYS:
                out[f"M2-{d}d"] = contiguous_block(all_tower, V, T, d,
                                                   np.random.default_rng(seed_for(w.site, start, "M2", d)))
        elif fam == "M2-sub":
            for h in SUBDAILY_HOURS:
                out[f"M2-{h}h"] = contiguous_steps(all_tower, V, T, h * 2,
                                                   np.random.default_rng(seed_for(w.site, start, "M2h", h)))
        elif fam == "M5":
            rng = np.random.default_rng(seed_for(w.site, start, "M5"))
            m, _ = mixed_blocks(tower + rean, V, T, rng)
            groups = {"tower_met": met, "reanalysis": rean, "fluxes": flux}
            avail = [g for g in M5_MODALITIES if groups.get(g)]
            g = avail[int(rng.integers(len(avail)))]
            m[groups[g]] = True
            if ext:
                mx, _ = mixed_blocks(ext, V, T, np.random.default_rng(seed_for(w.site, start, "M5x")))
                m |= mx
                if g == "tower_met":
                    m[ext] = True
            w.meta["M5_modality"] = g
            out["M5"] = m
        elif fam == "M4-reanalysis" and rean:
            m = modality_dropout(rean, V, T)
            m |= contiguous_block(all_tower, V, T, 30, np.random.default_rng(seed_for(w.site, start, "M4r", 30)))
            out["M4-no-reanalysis+M2-30d"] = m
        elif fam == "M3":
            # A real joint outage of all tower fluxes from another (training) site-year, applied to every tower row.
            lib = outage_library()
            rng = np.random.default_rng(seed_for(w.site, start, "M3"))
            cand = lib[lib.site != w.site].reset_index(drop=True)
            for _ in range(20):                       # donors with no outage in any 180-day slice are redrawn
                row = cand.iloc[int(rng.integers(len(cand)))]
                try:
                    o = cut_outage(outages.series(row), T, rng)
                except ValueError:
                    continue
                out["M3"] = transplanted_outage(all_tower, V, o)
                w.meta["M3_donor"] = f"{row.site} {int(row.year)}"
                break
    return out


_OUTAGES = None


def outage_library(path: str = "/home/photon/4tb/fluxfill/bench/outages_train.parquet") -> pd.DataFrame:
    """The M3 donor library (bench/outages.py), loaded once."""
    global _OUTAGES
    if _OUTAGES is None:
        _OUTAGES = pd.read_parquet(path)
    return _OUTAGES


def score(method: str, regime: str, fam: str, w: Window, mask: np.ndarray, samples: np.ndarray,
          pit: np.ndarray = None) -> List[Dict]:
    """Rows of scores for one (window, mask). `pit` (V, T), optional: the method's exact predictive CDF at the truth;
    where given, interval coverage is computed from it (a central interval of level q contains y exactly when
    (1 - q) / 2 <= F(y) <= (1 + q) / 2) instead of from sample quantiles, and `cov_method` says which was used."""
    rows = []
    runs = gap_runs(mask)
    cls = gap_class(runs)
    for i, k in enumerate(w.keys):
        if w.provenance[i] != "tower":
            continue
        target = mask[i] & w.measured[i]
        scored = target & np.isfinite(samples[:, i]).all(axis=0)
        if not target.any():
            continue
        # fill_rate: share of the hidden measured values the method filled; scores cover only those it filled.
        base = dict(method=method, regime=regime, family=fam, site=w.site, window=str(w.start.date()), variable=k,
                    n_target=int(target.sum()), fill_rate=float(scored.sum() / target.sum()))
        if not scored.any():
            rows.append(dict(base, kind="values", gap_class="all", n=0))
            continue
        for c in range(-1, len(GAP_CLASSES)):
            sel = scored if c == -1 else scored & (cls[i] == c)
            if not sel.any():
                continue
            y, s = w.values[i, sel], samples[:, i, sel]
            mu = s.mean(axis=0)
            pv = pit[i, sel] if pit is not None else None
            if pv is not None and np.isfinite(pv).all():
                cov = {q: float(np.mean((pv >= (1 - q) / 2) & (pv <= (1 + q) / 2))) for q in (0.5, 0.8, 0.95)}
                cov_method = "exact"
            else:
                cov = coverage_samples(s, y, (0.5, 0.8, 0.95))
                cov_method = "samples"
            rows.append(dict(base, kind="values", gap_class="all" if c == -1 else f"{GAP_CLASSES[c][0]}-{GAP_CLASSES[c][1]}",
                             n=int(sel.sum()), crps=float(crps_samples(s, y).mean()), mae=float(np.abs(mu - y).mean()),
                             rmse=float(np.sqrt(((mu - y) ** 2).mean())), bias=float((mu - y).mean()),
                             cov50=cov[0.5], cov80=cov[0.8], cov95=cov[0.95], cov_method=cov_method, **truth_sums(y)))
        if k in FLUXES:
            for bname, blk in SUM_BLOCKS.items():
                ss, ts = block_sums(np.nan_to_num(samples[:, i]), w.values[i], scored, blk)
                if ts.size:
                    rows.append(dict(base, kind=f"sum-{bname}", gap_class="all", n=int(ts.size),
                                     crps=float(crps_samples(ss, ts).mean()), mae=float(np.abs(ss.mean(0) - ts).mean()),
                                     rmse=float(np.sqrt(((ss.mean(0) - ts) ** 2).mean())), bias=float((ss.mean(0) - ts).mean()),
                                     cov50=np.nan, cov80=np.nan, cov95=coverage_samples(ss, ts, (0.95,))[0.95],
                                     **truth_sums(ts)))
    return rows


def truth_sums(y: np.ndarray) -> Dict[str, float]:
    """Sums of the scored truth (sum_y, sum_y2, sum_abs_y). With n, rmse and mae of a row they give the pooled R2
    and the weighted absolute percentage error of any set of rows (bench/results_dashboard.py)."""
    y = np.asarray(y, np.float64)
    return dict(sum_y=float(y.sum()), sum_y2=float((y ** 2).sum()), sum_abs_y=float(np.abs(y).sum()))


def verify_masks(w: Window, masks: Dict[str, np.ndarray], manifest: Dict) -> None:
    """Refuse to score a frozen-release window whose recomputed masks differ from the manifest (BENCHMARK.md rule 1)."""
    import hashlib
    ref = manifest["masks"].get(f"{w.site}|{w.start.date()}")
    if ref is None:
        raise ValueError(f"{w.site} {w.start.date()} is not in the frozen release")
    for name, a in masks.items():
        h = hashlib.sha256(np.packbits(a).tobytes()).hexdigest()
        if ref.get(name) != h:
            raise ValueError(f"mask {name} of {w.site} {w.start.date()} differs from the frozen release")


def run(methods, windows: List[Dict], tables: Dict[str, pd.DataFrame], regime: str, families: Iterable[str],
        sites: Dict[str, Dict] = None, progress: bool = False, manifest: Dict = None,
        tower_vars=TOWER_VARS) -> pd.DataFrame:
    rows = []
    for spec in windows:
        site, start = spec["site"], spec["start"]
        tab = tables[site]
        w = load_window(tab, site, start, tower_vars=tower_vars)
        idx = lst_index(tab)
        years = set(range(w.t[0].year, w.t[-1].year + 1))
        context = {"table": tab[~idx.year.isin(list(years))], "index": idx[~idx.year.isin(list(years))]}
        if regime == "R3":                      # ungauged site: no other year of the site at all
            context = {"table": tab.iloc[:0], "index": idx[:0]}
        items = []
        ms = masks_for(w, families)
        if regime == "R3":
            ms = {k: v for k, v in ms.items() if k == "M2-180d"}
        if manifest is not None:
            verify_masks(w, ms, manifest)
        for name, mask in ms.items():
            visible = Window(w.site, w.start, w.t, w.keys, np.where(mask, np.nan, w.values), w.measured & ~mask,
                             w.provenance, w.meta)
            items.append((name, mask, visible))
        for m in methods:
            if hasattr(m, "lat") and sites and site in sites:
                m.lat, m.lon = sites[site]["lat"], sites[site]["lon"]
                m.tz = sites[site].get("utc_offset", round(sites[site]["lon"] / 15))
            # A method may fill all masks of a window in one call (`predict_many`, for batched training); each
            # mask's prediction still sees only that mask's visible copy, so the protocol is unchanged.
            # Wall-clock seconds of each mask's prediction (predict_s; for predict_many, the batch time split evenly
            # over the masks), recorded on every score row of that mask: the method's inference cost on this machine.
            t0 = time.perf_counter()
            if hasattr(m, "predict_many"):
                outs = m.predict_many([(v, mk) for _, mk, v in items], context)
                per = (time.perf_counter() - t0) / max(len(items), 1)
                outs = iter(outs)
            else:
                outs = (m.predict(v, mk, context) for _, mk, v in items)
                per = None
            for name, mask, _ in items:
                t1 = time.perf_counter()
                s = next(outs)
                dt = per if per is not None else time.perf_counter() - t1
                # Exact coverage for methods with a closed-form predictive CDF: the truth is handed over only after
                # this mask's prediction has returned (and never with predict_many, whose last state is not per mask).
                pit = m.pit(w.values) if hasattr(m, "pit") and not hasattr(m, "predict_many") else None
                if pit is not None and os.environ.get("FLUXFILL_PIT_DUMP"):    # PIT of every hidden value, for
                    d = Path(os.environ["FLUXFILL_PIT_DUMP"]); d.mkdir(parents=True, exist_ok=True)   # copula_fit.py
                    np.savez_compressed(d / f"{site}_{start}_{name.replace(' ', '').replace(':', '_').replace(',', '')}.npz",
                                        pit=np.where(mask, pit, np.nan).astype(np.float32), keys=np.array(w.keys))
                rows += [dict(r, predict_s=dt) for r in score(m.name, regime, name, w, mask, s, pit)]
        if progress:
            print(f"{site} {start}: {len(rows)} rows", flush=True)
    return pd.DataFrame(rows)


class Climatology:
    """Floor reference (baselines/climatology.py), as a benchmark method."""
    name = "climatology"

    def __init__(self, n_samples: int = 64):
        self.n = n_samples

    def predict(self, w: Window, mask: np.ndarray, context: Dict) -> np.ndarray:
        import climatology
        V, T = w.values.shape
        out = np.full((self.n, V, T), np.nan)
        tab, idx = context["table"], context["index"]
        for i, k in enumerate(w.keys):
            if not mask[i].any():
                continue
            src = k if k in tab else None
            if src is None:
                continue
            series = pd.Series(tab[src].to_numpy(np.float64), index=idx)
            flag = tab.get(f"{src}_flag")
            meas = pd.Series((flag.to_numpy() == 0) if flag is not None else np.isfinite(series.to_numpy()), index=idx)
            s = climatology.predict(series, meas & series.notna(), w.t, n_samples=self.n,
                                    seed=seed_for(w.site, str(w.start.date()), "clim", i))
            out[:, i] = np.where(mask[i][None], s, w.values[i][None])
        return out


def method_registry() -> Dict:
    """Benchmark methods by name (baselines/README.md). Imports are lazy: a method's dependencies load only when it
    is used."""
    def lazy(module, cls, **kw):
        def make():
            mod = __import__(module)
            return getattr(mod, cls)(**kw)
        return make
    return {"climatology": Climatology,
            "mds": lazy("mds", "MDS"),
            "irvin_rf": lazy("irvin_rf", "IrvinRF", n_jobs=int(os.environ.get("FLUXFILL_NJOBS", "1"))),
            "vekuri_de": lazy("vekuri_de", "VekuriDE", device=os.environ.get("FLUXFILL_DE_DEVICE", "cpu"),
                               threads=int(os.environ.get("FLUXFILL_NJOBS", "1"))),
            "liu_xgb": lazy("liu_xgb", "LiuXGB", device=os.environ.get("FLUXFILL_XGB_DEVICE", "cpu"),
                            n_jobs=int(os.environ.get("FLUXFILL_NJOBS", "1"))),
            "saits": lazy("pypots_imputers", "SAITSMethod"),
            "csdi": lazy("pypots_imputers", "CSDIMethod"),
            "chronos2": lazy("chronos2", "Chronos2"),
            "fluxfill": fluxfill_method}


def fluxfill_method():
    """FluxFill (models/flux/fill/model) as a runner method. Checkpoint from FLUXFILL_CKPT; FLUXFILL_MODE marginal (v0,
    exact coverage from its predictive CDF) or joint (v1); FLUXFILL_DEVICE (default cuda); FLUXFILL_SAMPLES (64);
    FLUXFILL_NAME, the method name in the rows (default fluxfill, or fluxfill-v1 in joint mode)."""
    sys.path.insert(0, str(HERE.parent / "model"))
    from method import FluxFill
    mode = os.environ.get("FLUXFILL_MODE", "marginal")
    return FluxFill(os.environ["FLUXFILL_CKPT"], mode=mode, n_samples=int(os.environ.get("FLUXFILL_SAMPLES", "64")),
                    device=os.environ.get("FLUXFILL_DEVICE", "cuda"),
                    name=os.environ.get("FLUXFILL_NAME", "fluxfill" if mode == "marginal" else "fluxfill-v1"),
                    copula=os.environ.get("FLUXFILL_COPULA") or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="JSON list of {site, start, table}")
    ap.add_argument("--regime", default="R1", choices=["R1", "R3"],
                    help="R1: other years as context; R3: ungauged site (empty context, M2-180d only)")
    ap.add_argument("--families", nargs="+", default=["M2", "M4-reanalysis"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["climatology"])
    ap.add_argument("--sites", default="/home/photon/4tb/fluxfill/towers/sites.csv")
    ap.add_argument("--manifest", default=None, help="frozen release MANIFEST.json; masks are verified against it")
    ap.add_argument("--extra-tower", action="store_true",
                    help="opt-in (E30, not release 0.2): windows also carry windows.EXTRA_TOWER_VARS as tower rows")
    a = ap.parse_args()
    if a.regime == "R3" and "M2" not in a.families:
        raise SystemExit("--regime R3 scores the M2-180d mask: include family M2")
    if a.extra_tower and a.manifest:
        raise SystemExit("--extra-tower windows are not those of a frozen release: no --manifest")
    specs = json.loads(Path(a.windows).read_text())
    if isinstance(specs, dict):            # val_windows.py writes {"split", "windows", ...}
        specs = specs["windows"]
    tables = {s["site"]: pd.read_parquet(s["table"]) for s in specs}
    if a.extra_tower:
        tables = {k: with_extra_columns(v) for k, v in tables.items()}
    meta = pd.read_csv(a.sites).set_index("site_id")
    sites = {k: {"lat": float(r.lat), "lon": float(r.lon), **({"utc_offset": float(r.utc_offset)} if "utc_offset" in meta and pd.notna(r.utc_offset) else {})}
             for k, r in meta.iterrows()}
    if os.environ.get("FLUXFILL_GPU_GB"):          # hard ceiling on the shared GPU for every method in this process
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["FLUXFILL_GPU_GB"]) / total), 0)
    registry = method_registry()
    methods = [registry[m]() for m in a.methods]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    parts = []
    if os.environ.get("FLUXFILL_RESUME") and Path(a.out).exists():
        # resume an interrupted run: keep the windows already scored, run the rest
        prev = pd.read_parquet(a.out)
        done = set(zip(prev.site, prev.window))
        parts.append(prev)
        specs = [s for s in specs if (s["site"], s["start"]) not in done]
        print(f"resume: {len(done)} windows kept, {len(specs)} to run", flush=True)
    for spec in specs:                       # written after every window, so a long run leaves usable partial output
        parts.append(run(methods, [spec], tables, a.regime, a.families, sites, progress=True,
                         manifest=json.loads(Path(a.manifest).read_text()) if a.manifest else None,
                         tower_vars=TOWER_VARS + (EXTRA_TOWER_VARS if a.extra_tower else ())))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(a.out, index=False)
    for m in methods:
        if getattr(m, "log", None):
            pd.DataFrame(m.log).to_csv(Path(a.out).with_suffix(f".{m.name}.log.csv"), index=False)
    print(df[(df.kind == "values") & (df.gap_class == "all")].groupby(["method", "family", "variable"])[["crps", "mae", "cov95"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
