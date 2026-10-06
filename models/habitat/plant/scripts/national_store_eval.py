#!/usr/bin/env python3
"""Evaluate a national map store exactly as stored (docs/joint_model.md).

usage:
  national_store_eval.py plots <store> --out DIR      AUC of the stored maps on independent plots (VegBank, BLM AIM,
                                                      FIA; CONUS), against Daru's published maps and the per-species
                                                      MaxEnt range cards at the same plots
  national_store_eval.py fidelity <store> --out DIR   what storing costs: per-species VegBank AUC of the full model
                                                      vs the stored field, and their Spearman correlation
  national_store_eval.py bench <store> [--n 20]       CPU decode time of 256² and 512² windows, bytes on disk
All take --config (default configs/conus.json) for the data, run, grid and plot paths.

In every AUC a plot outside the species' calibration ecoregions (or without climate) ranks lowest, as the map of the
environment model shows it; a store with a learned calibration penalty is also scored as it serves its maps
(``auc_joint_served``: every plot with climate, outside the area lowered by the species' penalty). Species with
>= min_presence presence plots are scored.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer  # noqa: I001  (before rasterio: its bundled PROJ can break pyproj's transforms)
import rasterio
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ranges import codec, config, validate  # noqa: E402
from ranges.joint.data import JointData  # noqa: E402
from ranges.joint.reader import ECOREGION_RASTER, Store  # noqa: E402
from ranges.joint.store import shared_features, species_matrix  # noqa: E402
from ranges.predictors import ConusStack  # noqa: E402


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def grid_cells(grid_dir: Path, lon, lat):
    """Rows and columns of the 240 m grid cells containing lon/lat (may fall off the grid)."""
    with rasterio.open(grid_dir / ECOREGION_RASTER) as r:
        t, crs, eco = r.transform, r.crs, r.read(1)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(np.asarray(lon, float), np.asarray(lat, float))
    if not np.isfinite(x[np.isfinite(lon)]).all():
        raise RuntimeError("non-finite coordinate transform (import pyproj before rasterio)")
    rr = np.floor((t.f - y) / -t.e).astype(np.int64)
    cc = np.floor((x - t.c) / t.a).astype(np.int64)
    return rr, cc, eco


def cmd_plots(a, cfg):
    ev, grid = cfg["evaluation"], cfg.path(cfg["store"]["grids"]["conus"])
    P = cfg.path
    store = Store(a.store)
    T = store.T
    names = [str(s).replace("_", " ") for s in T["species"]]
    calib = [[int(x) for x in str(c).split()] for c in T["calibration"]]
    stack = ConusStack(grid)
    daru = {p.stem for p in P(ev["daru_rasters"]).glob("*.tif")}
    truth = validate.PlotTruth(P(ev["aim_csv"]), P(ev["fia_dir"]), vegbank_dir=P(ev["vegbank_dir"]),
                               wcvp_dir=P(ev["wcvp_dir"]))
    H, W = store.shape("conus")
    rows = []
    for src in ("vegbank", "aim", "fia"):
        fn = getattr(truth, src)
        ref, sel = None, []
        for i, n in enumerate(names):
            tr = fn(n)
            if tr is None or tr.present.sum() < ev["min_presence"]:
                continue
            if ref is None:                                   # every species' truth lists the same plots, in order
                ref = tr[["lon", "lat"]].reset_index(drop=True)
            sel.append((i, tr.present.values.astype(bool)))
        if ref is None:
            continue
        lon, lat = ref.lon.values.astype(float), ref.lat.values.astype(float)
        rr, cc, eco = grid_cells(grid, lon, lat)
        on = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        G, valid = store.cells("conus", rr, cc)
        peco = np.full(len(lon), -1)
        peco[on] = eco[rr[on], cc[on]]
        idx = np.array([i for i, _ in sel])
        F = G @ T["codes"][idx].T + T["offsets"][idx]                    # plots x species
        for j, (i, y) in enumerate(sel):
            in_area = np.isin(peco, calib[i])
            rec = {"source": src, "species": names[i], "presences": int(y.sum()), "inferred": bool(T["inferred"][i]),
                   "auc_joint": auc(y, np.where(in_area & valid, F[:, j], -1e9))}
            if store.has_penalty:
                g, served = store.served(F[:, j], i, in_area, valid & on)
                rec["auc_joint_served"] = auc(y, np.where(served, g, -1e9))
            if names[i] in daru:
                d = validate._sample(P(ev["daru_raw_rasters"]) / f"{names[i]}.tif", lon, lat)
                rec["auc_daru"] = auc(y, np.nan_to_num(d))
            label = names[i].replace(" ", "_")
            for cards in ev["maxent_cards"]:
                card = P(cards) / label / f"{label}.rangecard"
                if card.exists():
                    v = codec.decode_cells(card, stack, rr.clip(0, H - 1), cc.clip(0, W - 1))
                    s = np.where(v["suitability"] > 0, (v["suitability"].astype(float) - 1) / 254, 0.0)
                    rec["auc_maxent"] = auc(y, np.where(on, s, 0.0))
                    break
            rows.append(rec)
        print(f"{src}: {len(sel)} species scored", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval_species.csv", index=False)
    summ = {}
    for src, g in df.groupby("source"):
        d = {"species": int(len(g)), "auc_joint": round(float(g.auc_joint.median()), 4)}
        if "auc_joint_served" in g:
            d["auc_joint_served"] = round(float(g.auc_joint_served.median()), 4)
        for other in ("daru", "maxent"):
            c = f"auc_{other}"
            if c in g:
                p = g.dropna(subset=["auc_joint", c])
                if len(p):
                    d[f"vs_{other}"] = {"species": int(len(p)), "joint": round(float(p.auc_joint.median()), 4),
                                        other: round(float(p[c].median()), 4),
                                        "joint_better": round(float((p.auc_joint > p[c]).mean()), 3)}
        summ[src] = d
    (out / "eval_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def cmd_fidelity(a, cfg):
    from national_store import load_run, stored_model
    m = stored_model(cfg, a.model)
    run = Path(a.run) if a.run else m["run"]
    data_dir = Path(a.data_dir) if a.data_dir else m["data_dir"]
    model, st = load_run(run, m["train"]["flag_variables"], a.device, data_dir / "field")
    data = JointData(data_dir)
    labels = list(data.species().species)
    store = Store(a.store)
    T = store.T
    j_of = {s: j for j, s in enumerate(T["species"])}
    PX = np.asarray(data["plot_X"])
    if m["train"].get("fill_plots"):                       # the plots as the model scores them (shoreline fill)
        PX = np.load(data_dir / "plot_fill_vegbank.npz")["X"]
    lat, lon = np.asarray(data["plot_lat"], np.float32), np.asarray(data["plot_lon"], np.float32)
    rc = np.load(data_dir / "field" / "rc_plots_vegbank.npy") if model.field is not None else None
    Hf = shared_features(model, st.transform(PX), np.stack([lat, lon], 1), rc, a.device)
    Wv, b, pen = species_matrix(model)
    rr, cc, _ = grid_cells(cfg.path(cfg["store"]["grids"]["conus"]), lon.astype(float), lat.astype(float))
    G, valid = store.cells("conus", rr, cc)
    peco = np.asarray(data["plot_eco"])
    Y = np.asarray(data["eval_y"]).astype(bool)
    rows = []
    for i, s in enumerate(np.asarray(data["eval_sid"])):
        y = Y[i]
        if y.sum() < cfg["evaluation"]["min_presence"]:
            continue
        j = j_of[labels[s]]
        in_area = np.isin(peco, [int(v) for v in str(T["calibration"][j]).split()])
        inside = in_area & valid
        full = (Hf @ Wv[s] + b[s]).cpu().numpy()
        dec = G @ T["codes"][j] + T["offsets"][j]
        rec = {"species": labels[s], "auc_full": roc_auc_score(y, np.where(inside, full, -1e9)),
               "auc_stored": roc_auc_score(y, np.where(inside, dec, -1e9)),
               "spearman_inside": (spearmanr(full[inside], dec[inside]).correlation if inside.sum() > 10 else np.nan)}
        if pen is not None:                                # as served: outside the area lowered by the penalty
            p_ = float(pen[s]) * ~in_area
            rec["auc_full_served"] = roc_auc_score(y, np.where(valid, full - p_, -1e9))
            rec["auc_stored_served"] = roc_auc_score(y, np.where(valid, dec - p_, -1e9))
        rows.append(rec)
    d = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "fidelity.csv", index=False)
    print(f"{len(d)} species: median AUC full {d.auc_full.median():.4f} vs stored {d.auc_stored.median():.4f}; "
          f"mean loss {(d.auc_full - d.auc_stored).mean():+.4f}; Spearman inside the calibration area: median "
          f"{d.spearman_inside.median():.4f}, 5th percentile {d.spearman_inside.quantile(0.05):.4f}")
    if "auc_full_served" in d:
        print(f"as served: median AUC full {d.auc_full_served.median():.4f} vs stored "
              f"{d.auc_stored_served.median():.4f}")


def cmd_bench(a, cfg):
    store = Store(a.store, grids={k: cfg.path(v) for k, v in cfg["store"]["grids"].items()}, cache_tiles=64)
    H, W = store.shape(a.region)
    rng = np.random.default_rng(0)
    S = len(store.species)
    store.ecoregions(a.region)                                          # load the ecoregion layer once
    res = {}
    for n in (256, 512):
        cold, warm, done = [], [], 0
        while done < a.n:
            r0, c0 = int(rng.integers(0, H - n)), int(rng.integers(0, W - n))
            if not store.valid(a.region, r0, r0 + n, c0, c0 + n).mean() > 0.5:
                continue
            store._cache.clear()                                         # first read: tiles not cached
            for times in (cold, warm):                                   # then another species, same window
                s = int(rng.integers(0, S))
                t = time.perf_counter()
                store.decode(a.region, r0, r0 + n, c0, c0 + n, s)
                times.append(time.perf_counter() - t)
            done += 1
        res[str(n)] = {"ms_median": round(1e3 * float(np.median(cold)), 1),
                       "ms_p90": round(1e3 * float(np.percentile(cold, 90)), 1),
                       "repeat_window_ms_median": round(1e3 * float(np.median(warm)), 1)}
    sizes = {f.name: f.stat().st_size for f in sorted(Path(a.store).iterdir()) if f.is_file()}
    res.update({"total_gb": round(sum(sizes.values()) / 1e9, 2), "species": S, "bytes": sizes})
    print(json.dumps({k: v for k, v in res.items() if k != "bytes"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plots", "fidelity", "bench"):
        p = sub.add_parser(name)
        p.add_argument("store")
        p.add_argument("--config")
        if name != "bench":
            p.add_argument("--out", required=True)
    f = sub.choices["fidelity"]
    f.add_argument("--run")
    f.add_argument("--model", help="environment, or a stage of joint.stages (default store.model)")
    f.add_argument("--data-dir")
    f.add_argument("--device", default="cuda")
    b = sub.choices["bench"]
    b.add_argument("--region", default="conus")
    b.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cfg = config.load(a.config)
    {"plots": cmd_plots, "fidelity": cmd_fidelity, "bench": cmd_bench}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
