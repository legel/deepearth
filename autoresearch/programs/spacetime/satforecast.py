"""RETIRED satellite compound probe.

This historical module fit satellite transforms over all rows and represented a
future+new-place test as ``future & held`` followed by ``train = ~test``. That is
not a valid scientific split. Use ``probe_sat.py`` instead; it embargoes the two
cross-quadrants and fits transforms on training rows only.

Two established wins to COMPOUND (Ensue tag spacetime):
  (1) AlphaEarth satellite (ae 64d, join by gbifID) BEATS coordinate-PE on held-out-block family.
  (2) LSTM past-state propagator forecasts phenology day-of-year + abundance (Earth4D itself non-additive).

THIS PROBE answers two questions, reusing the EXISTING probe.py loaders + phenology/dyntargets runners
(no core edit; encoders/*, core/fusion.py, evaluate.py untouched; probe.py/phenology.py NOT modified here):

  TASK A  CLAY vs AlphaEarth vs coord-PE on the held-out-block FAMILY task (spatial).
          Extends satprobe.py: adds CLAY 1024d alongside AlphaEarth 64d, raw, RFF. Does richer 1024d CLAY
          beat 64d AlphaEarth on the same task?

  TASK B  SATELLITE IN THE FORECASTER (the compound). Feed the per-location AlphaEarth `ae` (and CLAY)
          feature as the QUERY + NEIGHBOUR positional feature into the phenology day-of-year and abundance
          propagators, vs raw-coord / RFF / Earth4D positional features. Leak-guards inherited from the
          existing runners: query features carry NO timestamp; graph edges are spatial-only (no dt-to-query);
          neighbours carry only their OWN observed past state. Satellite is a STATIC per-location feature, so
          it slots exactly where the positional feature goes -- safe. Does real land-cover help forecast WHEN
          a species is active (phenology MAE / abundance R2), not just WHERE?

Reports absolute skill + deltas vs coordinate-PE controls, multi-seed, over the SAME query set.
"""
import argparse, glob, time, json
import numpy as np, torch
from pathlib import Path

from deepearth.autoresearch.programs.spacetime.probe import (
    load_obs, spatial_holdout, temporal_holdout)
from deepearth.autoresearch.programs.spacetime.phenology import run_phenology
from deepearth.autoresearch.programs.spacetime.dyntargets import run_abundance
from deepearth.encoders.spacetime.earth4d import Earth4D
import torch.nn as nn, torch.nn.functional as F


def join_sat(cache, gid, key_file, key, mask_key=None):
    """Join a satellite feature matrix to each obs by gbifID. Returns (feat[N_join,D], keep_bool[N])."""
    z = np.load(str(Path(cache) / key_file))
    ids = z["gbifID"]
    feat_all = z[key]
    if mask_key is not None:
        has = z[mask_key]
        ok = has.astype(bool)
        m = {int(g): i for i, g, h in zip(range(len(ids)), ids.tolist(), ok.tolist()) if h}
    else:
        m = {int(g): i for i, g in enumerate(ids.tolist())}
    idx = np.array([m.get(int(g), -1) for g in gid])
    keep = idx >= 0
    return feat_all[idx[keep]].astype(np.float32), keep


def linprobe(X, y, test, n_cls, dev, steps, seed):
    """Linear head X->family, held-out-block accuracy (top1, top5). Fresh seed per call."""
    torch.manual_seed(seed)
    X = torch.tensor(X); y = torch.tensor(y)
    tr = ~torch.tensor(test); te = torch.tensor(test)
    Xtr, ytr = X[tr].to(dev), y[tr].to(dev)
    Xte, yte = X[te].to(dev), y[te].to(dev)
    h = nn.Linear(X.shape[1], n_cls).to(dev)
    o = torch.optim.Adam(h.parameters(), 1e-2)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, Xtr.shape[0], (4096,), device=dev, generator=g)
        loss = F.cross_entropy(h(Xtr[i]), ytr[i]); o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():
        lo = h(Xte)
        return ((lo.argmax(-1) == yte).float().mean().item(),
                (lo.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--rec_k", type=int, default=32)
    ap.add_argument("--rec_hidden", type=int, default=256)
    ap.add_argument("--rec_block_deg", type=float, default=3.0)
    ap.add_argument("--gnn_hops", type=int, default=2)
    ap.add_argument("--pheno_tol", type=float, default=15.0)
    ap.add_argument("--abund_win", type=float, default=90.0)
    ap.add_argument("--task_a", action="store_true", help="CLAY vs AlphaEarth vs coord-PE, held-out-block family")
    ap.add_argument("--task_b_pheno", action="store_true", help="satellite-in-forecaster: phenology DOY")
    ap.add_argument("--task_b_abund", action="store_true", help="satellite-in-forecaster: abundance")
    ap.add_argument("--with_clay", action="store_true", help="also run CLAY 1024d variant in task B (slower)")
    ap.add_argument("--spatial_levels", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    raise SystemExit(
        "satforecast.py is retired because its per-seed transforms and joint "
        "split leak test information. Use probe_sat.py, whose strict split "
        "embargoes cross-quadrants and fits transforms on train only."
    )
    dev = a.device if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()

    # ---- obs + gbifID + event-time ----
    lat, lon, fam, n_fam, days, gid, _sp = load_obs(a.cache_dir, a.n_shards, with_time=True, with_gid=True)

    # ---- join AlphaEarth (and CLAY) by gbifID; restrict to intersection so all variants share the SAME obs ----
    AE, keep_ae = join_sat(a.cache_dir, gid, "gbif_alphaearth_tokens.npz", "ae")
    CLAY, keep_clay = join_sat(a.cache_dir, gid, "gbif_clay_tokens.npz", "clay", "has_clay")
    keep = keep_ae & keep_clay
    # re-index satellite feats onto the intersection
    ae_idx = np.cumsum(keep_ae) - 1
    clay_idx = np.cumsum(keep_clay) - 1
    lat, lon, fam, days, gid = lat[keep], lon[keep], fam[keep], days[keep], gid[keep]
    AE = AE[ae_idx[keep]]
    CLAY = CLAY[clay_idx[keep]]
    # standardize satellite features (linear-head / propagator conditioning stability)
    AE = ((AE - AE.mean(0)) / (AE.std(0) + 1e-6)).astype(np.float32)
    CLAY = ((CLAY - CLAY.mean(0)) / (CLAY.std(0) + 1e-6)).astype(np.float32)
    N = len(lat)
    rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)   # raw coords (spatial-only, no time leak)
    coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
    print(f"[data] N(join AE&CLAY)={N} families={n_fam} AE={AE.shape[1]}d CLAY={CLAY.shape[1]}d "
          f"AE_cov={keep_ae.sum()} CLAY_cov={keep_clay.sum()} {time.time()-t0:.0f}s")

    out = {"N": N, "n_fam": int(n_fam), "seeds": seeds}

    # ========================= TASK A: CLAY vs AlphaEarth vs coord-PE (spatial family) =========================
    if a.task_a:
        # Earth4D positional feature (spatial-block eval, t=0)
        enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.spatial_levels,
                      spatial_log2_hashmap_size=20, temporal_log2_hashmap_size=20, freq_log_scale_init=-2.5).to(dev)
        with torch.no_grad():
            e4d = enc(torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1)).to(dev)).cpu().numpy()
        rff_rng = np.random.default_rng(0)
        proj = rn @ rff_rng.normal(0, 8.0, (2, 64)).astype(np.float32)
        RFF = np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32)
        variants = {"raw-coords": rn, "RFF": RFF, "Earth4D": e4d,
                    "AlphaEarth-64": AE, "CLAY-1024": CLAY,
                    "raw+AlphaEarth": np.concatenate([rn, AE], 1).astype(np.float32),
                    "raw+CLAY": np.concatenate([rn, CLAY], 1).astype(np.float32)}
        res_a = {}
        print("=== TASK A | held-out-block FAMILY (multi-seed) ===")
        for name, X in variants.items():
            t1, t5 = [], []
            for s in seeds:
                test = spatial_holdout(lat, lon, a.holdout, seed=s)
                acc, top5 = linprobe(X, fam, test, n_fam, dev, a.steps, s)
                t1.append(acc); t5.append(top5)
            t1, t5 = np.array(t1), np.array(t5)
            res_a[name] = {"top1_mean": float(t1.mean()), "top1_std": float(t1.std()),
                           "top5_mean": float(t5.mean()), "top5_std": float(t5.std())}
            print(f"  {name:16s} top1 {t1.mean():.4f}+/-{t1.std():.4f}  top5 {t5.mean():.4f}+/-{t5.std():.4f}")
        out["task_a"] = res_a

    # ========================= TASK B: satellite-in-forecaster (phenology / abundance) =========================
    if a.task_b_pheno or a.task_b_abund:
        # Build forecast splits like probe.py: future time AND held-out spatial block (rule-1 strict).
        # Coordinate-PE controls: raw / RFF / Earth4D (t=0 so no time leak into the query positional feature).
        enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.spatial_levels,
                      spatial_log2_hashmap_size=20, temporal_log2_hashmap_size=20, freq_log_scale_init=-2.5).to(dev)
        with torch.no_grad():
            e4d_sp = torch.tensor(enc(torch.tensor(
                np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1).astype(np.float32)).to(dev)).cpu().numpy())
        raw_sp = torch.tensor(rn)
        _rng = np.random.default_rng(0)
        _proj = rn @ _rng.normal(0, 8.0, (2, e4d_sp.shape[1] // 2)).astype(np.float32)
        rff_sp = torch.tensor(np.concatenate([np.sin(_proj), np.cos(_proj)], 1).astype(np.float32))
        ae_sp = torch.tensor(AE)
        raw_ae = torch.tensor(np.concatenate([rn, AE], 1).astype(np.float32))
        clay_sp = torch.tensor(CLAY)

        feat_variants = [("raw", raw_sp), ("rff", rff_sp), ("e4d", e4d_sp),
                         ("AlphaEarth", ae_sp), ("raw+AlphaEarth", raw_ae)]
        if a.with_clay:
            feat_variants.append(("CLAY", clay_sp))

        def run_task_b(runner, label, unit, extra):
            print(f"=== TASK B | {label} (satellite-in-forecaster, future+newplace, multi-seed) ===")
            agg = {}
            for name, feat in feat_variants:
                st_mae, gnn_mae, lstm_mae, gnn_acc, lstm_acc, st_acc = [], [], [], [], [], []
                fdim = feat.shape[1]
                for s in seeds:
                    test = temporal_holdout(days, a.holdout) & spatial_holdout(lat, lon, a.holdout, seed=s)
                    torch.manual_seed(s)
                    r = runner(feat, fdim, days, coords_ll, test, dev,
                               K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, **extra)
                    st_mae.append(r["static_mae"]); gnn_mae.append(r["gnn_mae"]); lstm_mae.append(r["lstm_mae"])
                    gnn_acc.append(r["gnn_acc"]); lstm_acc.append(r["lstm_acc"]); st_acc.append(r["static_acc"])
                d = {k: (float(np.mean(v)), float(np.std(v))) for k, v in
                     dict(static_mae=st_mae, gnn_mae=gnn_mae, lstm_mae=lstm_mae,
                          gnn_acc=gnn_acc, lstm_acc=lstm_acc, static_acc=st_acc).items()}
                best_prop = max(d["static_mae"][0] - d["gnn_mae"][0], d["static_mae"][0] - d["lstm_mae"][0])
                agg[name] = d | {"best_prop_gain": best_prop}
                print(f"  {name:16s} static {unit} {d['static_mae'][0]:7.3f} | GNN {unit} {d['gnn_mae'][0]:7.3f} "
                      f"(prop {d['static_mae'][0]-d['gnn_mae'][0]:+.3f}) acc/R2 {d['gnn_acc'][0]:+.4f}+/-{d['gnn_acc'][1]:.4f} "
                      f"| LSTM {unit} {d['lstm_mae'][0]:7.3f} (prop {d['static_mae'][0]-d['lstm_mae'][0]:+.3f}) "
                      f"acc/R2 {d['lstm_acc'][0]:+.4f}+/-{d['lstm_acc'][1]:.4f}")
            # deltas: satellite forecaster vs best coord-PE forecaster
            coord_best_gnn = min(agg[c]["gnn_mae"][0] for c in ("raw", "rff", "e4d"))
            coord_best_lstm = min(agg[c]["lstm_mae"][0] for c in ("raw", "rff", "e4d"))
            ae_gnn = agg["AlphaEarth"]["gnn_mae"][0]; ae_lstm = agg["AlphaEarth"]["lstm_mae"][0]
            print(f"  >> DELTA {label}: best-coord-PE GNN {unit} {coord_best_gnn:.3f} vs AlphaEarth {ae_gnn:.3f} "
                  f"(sat {coord_best_gnn-ae_gnn:+.3f} {unit}; +=sat better) | LSTM coord {coord_best_lstm:.3f} vs AE {ae_lstm:.3f} "
                  f"(sat {coord_best_lstm-ae_lstm:+.3f})")
            agg["_delta_sat_vs_coord_gnn_mae"] = coord_best_gnn - ae_gnn
            agg["_delta_sat_vs_coord_lstm_mae"] = coord_best_lstm - ae_lstm
            return agg

        if a.task_b_pheno:
            out["task_b_phenology"] = run_task_b(run_phenology, "PHENOLOGY(day-of-year)", "MAEd",
                                                 dict(tol_days=a.pheno_tol, block_deg=a.rec_block_deg))
        if a.task_b_abund:
            out["task_b_abundance"] = run_task_b(run_abundance, "ABUNDANCE(log-count)", "MAE",
                                                 dict(win=a.abund_win))

    out["seconds"] = time.time() - t0
    print(f"[done] {out['seconds']:.0f}s")
    print("RESULT_JSON " + json.dumps(out))
    return out


if __name__ == "__main__":
    main()
