"""Factorized conditional-INTENSITY probe (Cox point process) for the Earth4D encoder.

WHY THIS EXISTS. Every classification probe here asks a coordinate function for `f(x,t) -> species`, a
deterministic map. The data is a stochastic point process: ~180 species are co-suitable at a location, so
H(species | x, t) dominates a top-1 loss and encoder capacity has nowhere to show up. That is the observed
signature across the scorecard -- fair-gain positive (the encoder carries real signal) but the absolute score
pinned near a floor. Two further mis-specifications ride along:

  * presence-only records with NO effort term make lambda(x) and observer-effort(x) non-identifiable, so the
    model fits where people looked. (This is why failure clusters in under-sampled geography.)
  * a 1364-way softmax over a shared feature gives the long tail no statistical strength.

THE MODEL. Grid to cells x period, count records, and fit a Poisson intensity with a FACTORIZED score and an
effort offset:

    log lambda[c,s] = <z_c, e_s> + b_s + log E_c + g          z_c = W f(x_c, t)   (f = the encoder under test)

`e_s` is a learned species embedding (statistical strength shared across the tail), `E_c` is observer effort,
and `g` a global scale. Evaluated by held-out POISSON DEVIANCE EXPLAINED against an intercept-only null
(prevalence x effort) -- a proper scoring rule, not accuracy.

LEAK GUARDS.
  * Split is TEMPORAL (past -> future), the same `temporal_holdout` the forecast probes use.
  * Effort E_c is built from TRAIN rows ONLY and reused unchanged at test time, so it never carries test counts
    (test-period effort would sum the test labels).
  * The cell universe is TRAIN cells; features come from cell centre + period mid-time, never from labels.

NOT COMPARABLE to the species_from_spacetime record (0.0474). That is per-OBSERVATION top-1 accuracy; the
cell-level top-1 printed here has a different denominator and is auxiliary only. Deviance explained is the
result this probe is for.

  python -m deepearth.autoresearch.programs.spacetime.intensity_probe --n_shards 12 --feature earth4d
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from deepearth.autoresearch.programs.spacetime.probe import load_obs, temporal_holdout
from deepearth.encoders.spacetime.earth4d import Earth4D


def _cells(lat, lon, block):
    ci = np.floor(lat / block).astype(np.int64)
    cj = np.floor(lon / block).astype(np.int64)
    return ci * 100000 + cj


def poisson_deviance(y, lam):
    """2 * sum[ y log(y/lam) - (y - lam) ], with y log y := 0 at y = 0. Lower = better; 0 = saturated fit."""
    lam = np.clip(lam, 1e-9, None)
    t = np.where(y > 0, y * np.log(np.clip(y, 1e-9, None) / lam), 0.0)
    return float(2.0 * (t - (y - lam)).sum())


def fit_intensity(Ztr, Zte, Ytr, off_tr, off_te, d, steps, lr, dev, seed=0, use_feat=True, wd=1e-4):
    """Fit log lambda[c,s] = <W z_c, e_s> + b_s + logE_c + g by Poisson NLL on the TRAIN period.

    Returns (lam_train, lam_test): the SAME fitted parameters applied to the train-period and test-period
    features. Only the time coordinate differs between Ztr and Zte, so the test prediction is the model
    extrapolating forward in time -- never a refit on test features. use_feat=False -> intercept-only null."""
    torch.manual_seed(seed)
    C, S = Ytr.shape
    Ztr_t = torch.as_tensor(Ztr, dtype=torch.float32, device=dev)
    Zte_t = torch.as_tensor(Zte, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    # feature standardization on TRAIN stats (an unnormalized 256-d RFF diverges at any usable lr)
    mu, sd = Ztr_t.mean(0, keepdim=True), Ztr_t.std(0, keepdim=True).clamp_min(1e-6)
    Ztr_t = (Ztr_t - mu) / sd; Zte_t = (Zte_t - mu) / sd
    o_tr = torch.as_tensor(off_tr, dtype=torch.float32, device=dev)[:, None]
    o_te = torch.as_tensor(off_te, dtype=torch.float32, device=dev)[:, None]
    E = nn.Parameter(torch.randn(S, d, device=dev) * 0.01)
    b = nn.Parameter(torch.zeros(S, device=dev))
    g = nn.Parameter(torch.zeros((), device=dev))
    # LEARNED effort exponent. Fixing the offset coefficient at 1.0 assumes records scale exactly linearly with
    # effort; the data rejects that (a fixed-1.0 offset made even the intercept null WORSE than no offset).
    # alpha is shared by every model INCLUDING the null, so the comparison stays fair.
    alpha = nn.Parameter(torch.ones((), device=dev))
    W = nn.Linear(Ztr_t.shape[1], d).to(dev)
    params = [E, b, g, alpha] + (list(W.parameters()) if use_feat else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def loglam(Z, o):
        h = W(Z) @ E.T if use_feat else torch.zeros(Z.shape[0], S, device=dev)
        return (h + b + alpha * o + g).clamp(-20, 8)   # e^8 per cell-bin-species is already far above any real count

    for _ in range(steps):
        opt.zero_grad()
        ll = loglam(Ztr_t, o_tr)
        (ll.exp() - Y * ll).mean().backward()              # Poisson NLL up to a constant
        opt.step()
    with torch.no_grad():
        print(f"      [effort exponent alpha={float(alpha):.3f}]" if use_feat else
              f"      [null effort exponent alpha={float(alpha):.3f}]")
        return loglam(Ztr_t, o_tr).exp().cpu().numpy(), loglam(Zte_t, o_te).exp().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--n_shards", type=int, default=12)
    ap.add_argument("--feature", default="all", choices=["all", "earth4d", "rff", "raw"])
    ap.add_argument("--block", type=float, default=0.1)        # cell size (deg)
    ap.add_argument("--holdout", type=float, default=0.2)      # fraction of TIME BINS held out as the future
    ap.add_argument("--n_bins", type=int, default=24)          # time bins (rows are cell x bin)
    ap.add_argument("--emb_dim", type=int, default=32)         # species embedding dim
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--no_effort", action="store_true")        # ABLATION: drop the effort offset
    ap.add_argument("--spatial_levels", type=int, default=24)
    ap.add_argument("--temporal_levels", type=int, default=24)
    ap.add_argument("--log2_hashmap", type=int, default=22)
    ap.add_argument("--fourier", type=int, default=0)
    ap.add_argument("--time_harmonics", type=int, default=0)
    ap.add_argument("--spatial_siren", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    lat, lon, fam, n_fam, days, _gid, sp_obs = load_obs(a.cache_dir, a.n_shards, with_time=True)
    _u, sp = np.unique(sp_obs, return_inverse=True); sp = sp.astype(np.int64); S = int(sp.max()) + 1
    cid = _cells(lat, lon, a.block)
    # TIME BINS: a single train period identifies NO temporal variation, so a shifted time coordinate at test
    # is extrapolation along an unidentified direction (this is what blew up the first build). Bin time and
    # train across MANY past bins -> the encoder can actually learn how intensity moves with t, and the future
    # bins are a genuine forecast.
    tmin_d, tmax_d = float(np.nanmin(days)), float(np.nanmax(days))
    edges = np.linspace(tmin_d, tmax_d + 1e-6, a.n_bins + 1)
    tb = np.clip(np.digitize(days, edges) - 1, 0, a.n_bins - 1)
    n_tr_bins = max(int(round(a.n_bins * (1.0 - a.holdout))), 1)
    test = tb >= n_tr_bins                                     # LEAK GUARD: past bins -> future bins
    tr_cells = np.unique(cid[~test])                           # LEAK GUARD: cell universe from TRAIN only
    cmap = {c: i for i, c in enumerate(tr_cells.tolist())}
    C = len(tr_cells)

    def counts(bins):
        Y = np.zeros((C, len(bins), S), np.float32)
        bpos = {b: k for k, b in enumerate(bins)}
        for c, s, b in zip(cid, sp, tb):
            k = bpos.get(int(b)); i = cmap.get(int(c))
            if k is not None and i is not None:
                Y[i, k, s] += 1.0
        return Y.reshape(C * len(bins), S)

    tr_bins = list(range(n_tr_bins)); te_bins = list(range(n_tr_bins, a.n_bins))
    Ytr, Yte = counts(tr_bins), counts(te_bins)
    # LEAK GUARD: effort is the per-cell mean records-per-bin over TRAIN bins only, tiled across bins. Using
    # each test bin's own effort would sum that bin's test labels.
    eff_cell = Ytr.reshape(C, len(tr_bins), S).sum(2).mean(1)
    logE_cell = np.zeros(C, np.float32) if a.no_effort else np.log(np.clip(eff_cell, 0.5, None)).astype(np.float32)
    off_tr = np.repeat(logE_cell[:, None], len(tr_bins), 1).reshape(-1).astype(np.float32)
    off_te = np.repeat(logE_cell[:, None], len(te_bins), 1).reshape(-1).astype(np.float32)
    log_rho = 0.0                                              # bins are equal-width -> no exposure ratio needed

    # cell centres + the mid-time of each period as the time coordinate
    clat = (tr_cells // 100000).astype(np.float32) * a.block + a.block / 2
    clon = (tr_cells % 100000).astype(np.float32) * a.block + a.block / 2

    bin_mid = ((edges[:-1] + edges[1:]) / 2 - tmin_d) / max(tmax_d - tmin_d, 1e-6)

    def coords_for(bins):
        la = np.repeat(clat[:, None], len(bins), 1).reshape(-1)
        lo = np.repeat(clon[:, None], len(bins), 1).reshape(-1)
        tt = np.tile(bin_mid[bins].astype(np.float32), C)
        return (np.stack([la, lo, np.zeros_like(la), tt], 1).astype(np.float32),
                np.stack([la / 90.0, lo / 180.0, tt], 1).astype(np.float32))

    rn_tr, raw_tr = coords_for(tr_bins)
    rn_te, raw_te = coords_for(te_bins)
    rng = np.random.default_rng(0); B = rng.normal(0, 8.0, (3, 128)).astype(np.float32)
    rff_tr = np.concatenate([np.sin(raw_tr @ B), np.cos(raw_tr @ B)], 1)
    rff_te = np.concatenate([np.sin(raw_te @ B), np.cos(raw_te @ B)], 1)

    enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,
                  spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap,
                  freq_log_scale_init=-2.5, fourier_features=a.fourier, time_harmonics=a.time_harmonics,
                  spatial_siren=a.spatial_siren).to(dev)
    with torch.no_grad():
        e4d_tr = enc(torch.tensor(rn_tr).to(dev)).cpu().numpy()
        e4d_te = enc(torch.tensor(rn_te).to(dev)).cpu().numpy()

    feats = {"earth4d": (e4d_tr, e4d_te), "rff": (rff_tr, rff_te), "raw": (raw_tr, raw_te)}
    want = list(feats) if a.feature == "all" else [a.feature]

    _, lam_null = fit_intensity(raw_tr, raw_te, Ytr, off_tr, off_te, a.emb_dim, a.steps, a.lr, dev,
                                a.seed, use_feat=False)
    # the null predicts the same rate for the future period (prevalence x effort), so score it on test as-is
    D_null = poisson_deviance(Yte, lam_null)
    D_sat = 0.0
    print(f"=== SPACETIME | mode=INTENSITY(Cox point process, factorized) split=TEMPORAL(future) "
          f"cells={C} bins={a.n_bins}(train {len(tr_bins)}/test {len(te_bins)}) species={S} block={a.block}deg effort={'OFF' if a.no_effort else 'ON'} emb={a.emb_dim} ===")
    print(f"  train records={int(Ytr.sum())} test records={int(Yte.sum())} log_rho={log_rho:+.3f}  "
          f"null(prevalence x effort) test deviance={D_null:.0f}")

    out = {}
    for name in want:
        ftr, fte = feats[name]
        # fit on the TRAIN period, predict the TEST period with the SAME parameters (only time moves)
        _, lam_te = fit_intensity(ftr, fte, Ytr, off_tr, off_te, a.emb_dim, a.steps, a.lr, dev,
                                  a.seed, use_feat=True)
        D = poisson_deviance(Yte, lam_te)
        dev_expl = 1.0 - (D - D_sat) / max(D_null - D_sat, 1e-9)
        top1 = float((lam_te.argmax(1) == Yte.argmax(1))[Yte.sum(1) > 0].mean())
        out[name] = (D, dev_expl, top1)
        print(f"  {name:>8} | test deviance {D:12.0f} | DEVIANCE EXPLAINED {dev_expl:+.4f} | "
              f"cell-top1 {top1:.4f} (auxiliary, NOT the per-obs record metric)")

    if "earth4d" in out and "rff" in out:
        print(f"  st_gain(Earth4D vs RFF, deviance explained) {out['earth4d'][1] - out['rff'][1]:+.4f}   "
              f"(Earth4D {out['earth4d'][1]:+.4f}  RFF {out['rff'][1]:+.4f})")
    print(f"  [profile] {int(Ytr.sum())} train recs, {C} cells x {S} species, {a.steps} steps in {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
