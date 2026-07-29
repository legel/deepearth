"""Satellite-channel spacetime probe -- SATELLITE remote-sensing embeddings as the real environment->biology
channel (science.md rule 24). STANDALONE module: imports the shared probe machinery (Earth4D, load_obs,
evaluate, spatial/temporal holdout) and the phenology propagator; touches NO other file. Additive, flag-gated,
commits nothing. Runs on the SAME box the contended probe.py runs on without editing it.

WHY: prior rounds (Ensue tag spacetime, LOOP-spacetime-env_covariates) found Earth4D's hash and worldclim/soil
env all TIE a coordinate PE (RFF) at held-out biology -- bare coordinates + smooth climate carry no more than
location. The MISSING channel is high-resolution remote-sensing embeddings AT each location, joinable by gbifID:
  data/deepcal/gbif_alphaearth_tokens.npz  key `ae`   (621558, 64)   Google AlphaEarth, FULL coverage
  data/deepcal/gbif_clay_tokens.npz        key `clay` (215356,1024)  CLAY, partial (has_clay mask)

Q1 (--sat, spatial-block held-out): family acc from raw / RFF / Earth4D (coordinate-PE controls) vs AlphaEarth
    vs CLAY (has_clay subset) vs Earth4D+AlphaEarth fused. Does satellite >> any coordinate PE at held-out
    LOCATIONS? st_gain = satellite MINUS best coord-PE.
Q2 (--sat --phenology --forecast --forecast_spatial): add the satellite channel to the day-of-year phenology
    propagator. Leak-guards intact: query features are SPACE+SATELLITE only (the satellite embedding is a static
    land-cover code, no query timestamp), neighbours carry only their observed past DOY, edges spatial-only (no
    dt-to-query). Does AlphaEarth at the query improve forecast over coords-only, and does Earth4D+satellite beat
    satellite-alone? Reports forecast MAE + within-tol delta vs the best coordinate-PE.

  python -m deepearth.autoresearch.programs.spacetime.probe_sat --sat --sat_clay --cache_dir data/deepcal
  python -m deepearth.autoresearch.programs.spacetime.probe_sat --sat --phenology --forecast --forecast_spatial ...
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from deepearth.encoders.spacetime.earth4d import Earth4D
from deepearth.autoresearch.programs.spacetime.probe import (
    load_obs, evaluate, spatial_holdout, temporal_holdout,
)


def load_sat(cache: str, gid, want_clay: bool = False):
    """Join high-resolution REMOTE-SENSING embeddings to each observation by gbifID.

    Returns (ae, clay, has_clay):
      ae        [N, 64]  float32, standardized per column; obs with no ae imputed to 0 (post-std mean).
      clay      [N,1024] float32, standardized per column over has_clay obs; missing imputed to 0. (None if not want_clay)
      has_clay  [N]      bool, True where a real CLAY embedding was joined. (None if not want_clay)
    Standardization matches probe.load_env: zero-mean/unit-std over PRESENT values, NaN->0 = neutral impute."""
    cachep = Path(cache)
    az = np.load(cachep / "gbif_alphaearth_tokens.npz")
    ae_map = {int(g): j for j, g in enumerate(az["gbifID"].tolist())}
    ae_src = az["ae"]
    N = len(gid)
    ae = np.full((N, ae_src.shape[1]), np.nan, np.float32)
    for i, g in enumerate(gid):
        j = ae_map.get(int(g))
        if j is not None:
            ae[i] = ae_src[j]
    mu = np.nanmean(ae, 0); sd = np.nanstd(ae, 0); sd[sd < 1e-6] = 1.0
    ae = np.nan_to_num((ae - mu) / sd, nan=0.0).astype(np.float32)

    clay = has_clay = None
    if want_clay:
        cz = np.load(cachep / "gbif_clay_tokens.npz")
        cmask = cz["has_clay"]; cids = cz["gbifID"]; csrc = cz["clay"].astype(np.float32)
        clay_map = {int(cids[k]): k for k in range(len(cids)) if cmask[k]}
        clay = np.full((N, csrc.shape[1]), np.nan, np.float32)
        has_clay = np.zeros(N, bool)
        for i, g in enumerate(gid):
            k = clay_map.get(int(g))
            if k is not None:
                clay[i] = csrc[k]; has_clay[i] = True
        cmu = np.nanmean(clay, 0); csd = np.nanstd(clay, 0); csd[csd < 1e-6] = 1.0
        clay = np.nan_to_num((clay - cmu) / csd, nan=0.0).astype(np.float32)
    return ae, clay, has_clay


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_levels", type=int, default=18)
    ap.add_argument("--temporal_levels", type=int, default=18)
    ap.add_argument("--log2_hashmap", type=int, default=20)
    ap.add_argument("--head_hidden", type=int, default=0)
    ap.add_argument("--sat", action="store_true")
    ap.add_argument("--sat_clay", action="store_true")
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--forecast_spatial", action="store_true")
    ap.add_argument("--phenology", action="store_true")
    ap.add_argument("--pheno_attn", action="store_true")
    ap.add_argument("--pheno_tol", type=float, default=15.0)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--attn_layers", type=int, default=2)
    ap.add_argument("--rec_k", type=int, default=16)
    ap.add_argument("--rec_hidden", type=int, default=256)
    ap.add_argument("--gnn_hops", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    t0 = time.time()
    lat, lon, fam, n_fam, days, gid, _sp = load_obs(a.cache_dir, a.n_shards, with_time=a.forecast, with_gid=True)
    fam_t = torch.tensor(fam)

    if a.forecast:
        test = temporal_holdout(days, a.holdout)
        if a.forecast_spatial:
            test = test & spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        tmin, tspan = np.nanmin(days), max(np.nanmax(days) - np.nanmin(days), 1e-6)
        tnorm = ((days - tmin) / tspan).astype(np.float32)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), tnorm], 1))
    else:
        test = spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        coords = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1))

    enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,
                  spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap,
                  freq_log_scale_init=-2.5).to(dev)

    rn = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
    if a.forecast:
        rn = np.concatenate([rn, tnorm[:, None]], 1)

    ae, clay, has_clay = load_sat(a.cache_dir, gid, want_clay=a.sat_clay)

    # ================= Q2: FORECAST phenology + satellite propagator =================
    if a.phenology:
        assert a.forecast, "--phenology requires --forecast (live event-time + past->future split)"
        from deepearth.autoresearch.programs.spacetime.phenology import run_phenology
        coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))
        # SPACE-ONLY query features (t stripped -> no time leak): the satellite embedding is a static land-cover
        # code with no query timestamp, so it is a legitimate space+satellite query feature exactly like lat/lon.
        rn_sp = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
        raw_sp = torch.tensor(rn_sp)
        _proj = rn_sp @ (np.random.default_rng(0).normal(0, 8.0, (2, 32)).astype(np.float32))
        rff_sp = torch.tensor(np.concatenate([np.sin(_proj), np.cos(_proj)], 1).astype(np.float32))
        coords_sp = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1).astype(np.float32))
        with torch.no_grad():
            e4d_sp = enc(coords_sp.to(dev)).cpu()
        ae_sp = torch.tensor(ae)
        sat_fused = torch.cat([e4d_sp, ae_sp], 1)
        feats = {"raw": raw_sp, "rff": rff_sp, "e4d": e4d_sp, "sat": ae_sp, "e4dsat": sat_fused}
        kw = dict(K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops,
                  tol_days=a.pheno_tol, attn=a.pheno_attn, attn_heads=a.attn_heads, attn_layers=a.attn_layers)
        r = {}
        for name, f in feats.items():
            r[name] = run_phenology(f, f.shape[1], days, coords_ll, test, dev, **kw)
        dt = time.time() - t0
        props = ["static", "gnn", "lstm"] + (["attn"] if a.pheno_attn else [])
        def best_mae(ft): return min(r[ft][p + "_mae"] for p in props)
        def best_acc(ft): return max(r[ft][p + "_acc"] for p in props)
        n_te = r["raw"]["n_te"]
        best_coord_mae = min(best_mae("raw"), best_mae("rff"), best_mae("e4d"))
        best_coord_acc = max(best_acc("raw"), best_acc("rff"), best_acc("e4d"))
        sat_mae, e4dsat_mae = best_mae("sat"), best_mae("e4dsat")
        sat_acc, e4dsat_acc = best_acc("sat"), best_acc("e4dsat")
        print(f"=== SPACETIME SAT probe | mode=PHENOLOGY-FORECAST(future+newplace) obs={len(lat)} queries={n_te} tol=+/-{a.pheno_tol:.0f}d K={a.rec_k} attn={a.pheno_attn} seed={a.seed} ===")
        for ft in ("raw", "rff", "e4d", "sat", "e4dsat"):
            d = r[ft]
            attn_s = f" | ATTN MAE {d.get('attn_mae', float('nan')):6.2f}d acc {d.get('attn_acc', float('nan')):.4f}" if a.pheno_attn else ""
            print(f"  {ft:>7} | static MAE {d['static_mae']:6.2f}d acc {d['static_acc']:.4f} | GNN MAE {d['gnn_mae']:6.2f}d acc {d['gnn_acc']:.4f} | LSTM MAE {d['lstm_mae']:6.2f}d acc {d['lstm_acc']:.4f}{attn_s}  (best {best_mae(ft):.2f}d/{best_acc(ft):.4f})")
        print(f"  SATELLITE forecast skill (best-of-heads MAE, lower=better) | best-coord-PE {best_coord_mae:6.2f}d | AlphaEarth {sat_mae:6.2f}d (dMAE {best_coord_mae-sat_mae:+.2f}d) | Earth4D+AE {e4dsat_mae:6.2f}d (dMAE {best_coord_mae-e4dsat_mae:+.2f}d)  (E4D+AE vs AE-alone {sat_mae-e4dsat_mae:+.2f}d)")
        print(f"  SATELLITE within-tol acc | best-coord-PE {best_coord_acc:.4f} | AlphaEarth {sat_acc:.4f} (d {sat_acc-best_coord_acc:+.4f}) | Earth4D+AE {e4dsat_acc:.4f} (d {e4dsat_acc-best_coord_acc:+.4f})")
        print(f"  [profile] queries={n_te} K={a.rec_k} hidden={a.rec_hidden} hops={a.gnn_hops} steps={a.steps} seed={a.seed} {dt:.1f}s")
        return {"sat_forecast_mae": sat_mae, "e4dsat_forecast_mae": e4dsat_mae, "best_coord_forecast_mae": best_coord_mae,
                "sat_forecast_mae_delta": best_coord_mae - sat_mae, "e4dsat_forecast_mae_delta": best_coord_mae - e4dsat_mae,
                "sat_forecast_acc": sat_acc, "e4dsat_forecast_acc": e4dsat_acc, "best_coord_forecast_acc": best_coord_acc,
                "sat_forecast_acc_delta": sat_acc - best_coord_acc, "e4dsat_forecast_acc_delta": e4dsat_acc - best_coord_acc,
                "n_te": n_te, "seconds": dt, "phenology": True}

    # ================= Q1: spatial-block family from satellite vs coordinate-PE =================
    with torch.no_grad():
        e4d = enc(coords.to(dev)).cpu()
    raw = torch.tensor(rn)
    proj = rn @ (np.random.default_rng(0).normal(0, 8.0, (rn.shape[1], e4d.shape[1] // 2)).astype(np.float32))
    rff = torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))
    ae_t = torch.tensor(ae)
    fused = torch.cat([e4d, ae_t], 1)
    raw_acc, raw_t5 = evaluate(raw, fam_t, test, n_fam, dev, a.steps, a.lr, "raw", a.head_hidden)
    rff_acc, rff_t5 = evaluate(rff, fam_t, test, n_fam, dev, a.steps, a.lr, "rff", a.head_hidden)
    e4d_acc, e4d_t5 = evaluate(e4d, fam_t, test, n_fam, dev, a.steps, a.lr, "earth4d", a.head_hidden)
    ae_acc, ae_t5 = evaluate(ae_t, fam_t, test, n_fam, dev, a.steps, a.lr, "alphaearth", a.head_hidden)
    fus_acc, fus_t5 = evaluate(fused, fam_t, test, n_fam, dev, a.steps, a.lr, "fused", a.head_hidden)
    best_coord = max(raw_acc, rff_acc, e4d_acc)

    clay_line = ""; clay_res = {}
    if a.sat_clay and has_clay is not None and int(has_clay.sum()) > 0:
        sub = torch.tensor(has_clay)
        test_sub = test[has_clay]
        fam_sub = fam_t[sub]
        if int((~test_sub).sum()) > 0 and int(test_sub.sum()) > 0:
            clay_t = torch.tensor(clay[has_clay]); e4d_sub = e4d[sub]; ae_sub = ae_t[sub]
            c_clay, _ = evaluate(clay_t, fam_sub, test_sub, n_fam, dev, a.steps, a.lr, "clay", a.head_hidden)
            c_e4d, _ = evaluate(e4d_sub, fam_sub, test_sub, n_fam, dev, a.steps, a.lr, "e4d_sub", a.head_hidden)
            c_ae, _ = evaluate(ae_sub, fam_sub, test_sub, n_fam, dev, a.steps, a.lr, "ae_sub", a.head_hidden)
            clay_line = (f"  [CLAY subset] obs={int(has_clay.sum())} held={int(test_sub.sum())} | Earth4D {c_e4d:.4f} | AlphaEarth {c_ae:.4f} || CLAY {c_clay:.4f}"
                         f"   st_gain(CLAY vs E4D) {c_clay-c_e4d:+.4f}  (AE vs E4D) {c_ae-c_e4d:+.4f}")
            clay_res = {"clay_acc": c_clay, "clay_e4d_acc": c_e4d, "clay_ae_acc": c_ae,
                        "st_gain_clay": c_clay - c_e4d, "clay_n_held": int(test_sub.sum())}
    dt = time.time() - t0
    mode = ("FORECAST(future+newplace)" if a.forecast_spatial else "FORECAST(past->future)") if a.forecast else "spatial-block"
    print(f"=== SPACETIME SAT probe | mode=SAT({mode}) obs={len(lat)} held-out={int(test.sum())} families={n_fam} ae_dim={ae.shape[1]} seed={a.seed} ===")
    print(f"  held-out family acc | raw {raw_acc:.4f} | RFF {rff_acc:.4f} | Earth4D {e4d_acc:.4f} || AlphaEarth {ae_acc:.4f} | Earth4D+AE {fus_acc:.4f}")
    print(f"    st_gain(AlphaEarth vs best-coord-PE) {ae_acc-best_coord:+.4f}   st_gain(fused vs best-coord-PE) {fus_acc-best_coord:+.4f}   (best-coord-PE={best_coord:.4f})")
    print(f"  held-out top5 acc   | raw {raw_t5:.4f} | RFF {rff_t5:.4f} | Earth4D {e4d_t5:.4f} || AlphaEarth {ae_t5:.4f} | Earth4D+AE {fus_t5:.4f}")
    if clay_line: print(clay_line)
    print(f"  [profile] ae_dim={ae.shape[1]} frac_held={test.mean():.3f} head_hidden={a.head_hidden} steps={a.steps} forecast={a.forecast} seed={a.seed} {dt:.1f}s")
    return {"st_gain": ae_acc - best_coord, "st_gain_fused": fus_acc - best_coord,
            "ae_acc": ae_acc, "fused_acc": fus_acc, "earth4d_acc": e4d_acc, "rff_acc": rff_acc,
            "raw_acc": raw_acc, "best_coord_pe": best_coord, "obs": len(lat), "seconds": dt,
            "sat": True, "n_held": int(test.sum())} | clay_res


if __name__ == "__main__":
    main()
