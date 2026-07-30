"""Two more TEMPORALLY-DYNAMIC forecast targets for the spacetime propagator (science.md rule 1+2b),
to test whether the phenology unlock (LOOP-spacetime-nonstationary-phenology-dayofyear: on day-of-year the
propagator cut error nearly in half, +69.6d MAE, SNR 9.4, while Earth4D stayed neutral) GENERALIZES beyond
mean phenology -- and thereby to SCOPE which dynamic heads the recurrence unlocks for the forecasting PR.

Both targets are things a STATIC spatial climatology cannot forecast (measured on-box):
  * FIRST-ARRIVAL / seasonal onset -- earliest DOY of a species at a place; per (0.5deg cell, species) the
    onset DOY has std ~76d (a static coord map cannot pin it). DISTINCT from the mean-DOY phenology target:
    it is the leading EDGE of the season, not its centre.
  * ABUNDANCE / activity level -- log observation count in the query cell over a trailing time window; per
    (cell, quarter) log-count std ~1.66 and a static per-cell mean leaves 65% of that variance (it is 65%
    NON-stationary in time) -- a dynamic quantity a static climatology cannot forecast.

Same rigor + SAME leak-guards as phenology.py:
  * the QUERY-point positional feature is SPACE-ONLY (lat,lon,t=0); the query timestamp is never a feature.
  * the graph EDGE carries only the SPATIAL offset (dlat,dlon), never dt-to-query (elapsed days to the query),
    so the model cannot reconstruct the answer by time-arithmetic; it must propagate the PATTERN of nearby
    past state.
  * neighbours carry their OWN observed state (past DOY for onset; past recency for abundance) = the PAST
    state to propagate; the query has none of its own.

Three heads per target on the IDENTICAL future+new-place query set, each over Earth4D / RFF / raw features:
StaticX (no propagation floor) vs GNNX (message passing) vs LSTMX (causal rollout). propagator_gain = the
propagator's skill minus the static floor. Additive + flag-gated (--first_arrival / --abundance); imported
only when probe.py is called with the flag; default path never touches this module.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import build_causal_windows

_DOY = 365.25


# ---------------------------------------------------------------------------------------------------------
# circular helpers (shared with phenology's day-of-year encoding)
# ---------------------------------------------------------------------------------------------------------
def doy_of(days):
    return np.mod(np.asarray(days, dtype=np.float64), _DOY).astype(np.float32)


def doy_to_vec(doy):
    ang = 2.0 * np.pi * (np.asarray(doy, np.float32) / _DOY)
    return np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32)


def vec_to_doy(v):
    ang = torch.remainder(torch.atan2(v[..., 1], v[..., 0]), 2 * np.pi)
    return ang / (2 * np.pi) * _DOY


def circ_err_days(pred_doy, true_doy):
    d = torch.abs(pred_doy - true_doy)
    return torch.minimum(d, _DOY - d)


# ---------------------------------------------------------------------------------------------------------
# heads -- circular (onset) reuse the phenology (cos,sin) decode; abundance is a scalar regressor
# ---------------------------------------------------------------------------------------------------------






# ---------------------------------------------------------------------------------------------------------
# window / tensor assembly (SPATIAL-only edge leak-guard, identical to phenology.py)
# ---------------------------------------------------------------------------------------------------------
def _windows(lat, lon, days, q_idx, pool_idx, K):
    qi, _ = build_causal_windows(lat[q_idx], lon[q_idx], days[q_idx],
                                 lat[pool_idx], lon[pool_idx], days[pool_idx], K)
    gi = np.where(qi >= 0, pool_idx[np.clip(qi, 0, None)], -1)
    return gi, qi >= 0


def _assemble(qfeat_all, nstate_all, days, lat, lon, q_idx, gidx, valid, target, K, out_dim):
    """Build neighbour (feat, state), query feat, SPATIAL edge (dlat,dlon), mask, lengths, and target.

    LEAK-GUARD (same as phenology): edge EXCLUDES dt-to-query. Neighbour state carries only its OWN observed
    quantity, never anything computed relative to the query timestamp."""
    B = len(q_idx)
    N, F_ = qfeat_all.shape
    S = nstate_all.shape[1]
    gsafe = np.clip(gidx, 0, N - 1)
    vmask = torch.tensor(valid)
    nfeat = qfeat_all[torch.tensor(gsafe.reshape(-1))].reshape(B, K, F_) * vmask.unsqueeze(-1)
    nstate = torch.tensor(nstate_all[gsafe.reshape(-1)]).reshape(B, K, S) * vmask.unsqueeze(-1)
    dlat = np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)
    dlon = np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)
    edge = torch.tensor(np.stack([dlat / 90.0, dlon / 180.0], -1)).float()
    qfeat = qfeat_all[torch.tensor(q_idx)]
    ok = vmask.any(1)
    lengths = vmask.sum(1)
    if out_dim == 2:                                                  # circular target passed as DOY [Nq]
        ytrue = torch.tensor(doy_of(target[q_idx]))
        yvec = torch.tensor(doy_to_vec(doy_of(target[q_idx])))
        return nfeat[ok], nstate[ok], qfeat[ok], edge[ok], vmask[ok], lengths[ok], yvec[ok], ytrue[ok]
    y = torch.tensor(target[q_idx].astype(np.float32)).unsqueeze(-1)  # scalar regression target [Nq,1]
    return nfeat[ok], nstate[ok], qfeat[ok], edge[ok], vmask[ok], lengths[ok], y[ok], y[ok]


def _fit_eval(qfeat_all, feat_dim, nstate_all, days, lat, lon, test, dev, target, out_dim,
              state_dim, K, steps, lr, hidden, hops, skill_fn, loss_fn):
    N = qfeat_all.shape[0]
    tr_idx = np.where(~test)[0]
    te_idx = np.where(test)[0]
    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat, lon, days, q_train, tr_idx, K)
    g_te, v_te = _windows(lat, lon, days, te_idx, tr_idx, K)
    tr = _assemble(qfeat_all, nstate_all, days, lat, lon, q_train, g_tr, v_tr, target, K, out_dim)
    te = _assemble(qfeat_all, nstate_all, days, lat, lon, te_idx, g_te, v_te, target, K, out_dim)
    to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, ytr, yttr = to(tr)
    nfte, nste, qfte, ete, mte, lte, yte, ytte = to(te)
    n_te = int(nfte.shape[0])
    keys = ("static", "gnn", "lstm")
    out = {"n_te": n_te}
    if nftr.shape[0] == 0 or n_te == 0:
        return {f"{k}_{m}": float("nan") for k in keys for m in ("mae", "acc")} | {"n_te": n_te}
    Btr = nftr.shape[0]
    bs = min(2048, Btr)

    def train(model, fwd_tr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            s = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(fwd_tr(model, s), ytr[s])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

    sh = StaticVec(feat_dim, hidden, out_dim).to(dev)
    train(sh, lambda m, s: m(qftr[s]))
    with torch.no_grad():
        out["static_mae"], out["static_acc"] = skill_fn(sh(qfte), yte, ytte)

    gnn = GNNVec(feat_dim, hidden, hops, state_dim, out_dim).to(dev)
    train(gnn, lambda m, s: m(nftr[s], nstr[s], qftr[s], etr[s], mtr[s]))
    with torch.no_grad():
        out["gnn_mae"], out["gnn_acc"] = skill_fn(gnn(nfte, nste, qfte, ete, mte), yte, ytte)

    lstm = LSTMVec(feat_dim, hidden, state_dim, out_dim).to(dev)
    train(lstm, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
    with torch.no_grad():
        out["lstm_mae"], out["lstm_acc"] = skill_fn(lstm(nfte, nste, ete, lte), yte, ytte)
    return out


# ---------------------------------------------------------------------------------------------------------
# TARGET 1 -- FIRST-ARRIVAL / seasonal onset (per (0.5deg cell, species) earliest DOY)
# ---------------------------------------------------------------------------------------------------------
def _first_arrival_doy(lat, lon, days, sp, block=0.5):
    """Onset target: for each obs, the EARLIEST day-of-year observed for its species in its 0.5deg cell.

    This is the leading edge of the season (distinct from the mean-DOY phenology target). Circular DOY."""
    doy = doy_of(days)
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    onset = defaultdict(lambda: 1e9)
    keys = list(zip(ci.tolist(), cj.tolist(), sp.tolist()))
    for kk, d in zip(keys, doy):
        if d < onset[kk]:
            onset[kk] = float(d)
    return np.array([onset[kk] for kk in keys], dtype=np.float32)


def _circ_skill(pred_vec, yvec, ytrue, tol_days=15.0):
    err = circ_err_days(vec_to_doy(pred_vec), ytrue)
    return err.mean().item(), (err <= tol_days).float().mean().item()




def _nan_dyn():
    keys = ("static", "gnn", "lstm")
    return {f"{k}_{m}": float("nan") for k in keys for m in ("mae", "acc")} | {"n_te": 0}




# ---------------------------------------------------------------------------------------------------------
# TARGET 2 -- ABUNDANCE / activity level (log obs count in the query cell over a trailing window)
# ---------------------------------------------------------------------------------------------------------
def _abundance_target(lat, lon, days, block=0.5, win=90.0, lead=0.0, delta=False):
    """Per-obs activity = log(1 + #obs in the SAME 0.5deg cell within the trailing `win` days ending at the
    obs day). A dynamic count a static climatology cannot forecast (65% of its variance is non-stationary).

    Also returns each obs's OWN local recency density in the win BEFORE it -- neighbours carry this as their
    past state (never anything relative to the query)."""
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    cell_days = defaultdict(list)
    keys = list(zip(ci.tolist(), cj.tolist()))
    for kk, d in zip(keys, days):
        cell_days[kk].append(float(d))
    for kk in cell_days:
        cell_days[kk] = np.sort(np.array(cell_days[kk]))
    cnt = np.empty(len(days), np.float32)
    for i, (kk, d) in enumerate(zip(keys, days)):
        arr = cell_days[kk]
        # FORECAST-AHEAD lever: target counts activity in the window ENDING lead days in the FUTURE
        # ([d+lead-win, d+lead]); neighbours (build_causal_windows) still see only the past up to d, so the
        # propagator must forecast forward. lead=0 -> nowcast (established); lead>0 -> lead-time forecast.
        lo = np.searchsorted(arr, d + lead - win, "left")
        hi = np.searchsorted(arr, d + lead, "right")
        fut = np.log1p(hi - lo)                                      # future-window log-activity
        if delta:
            # DELTA-DYNAMICS target: future log-activity MINUS trailing-past log-activity ([d-win, d]).
            # Removes the stationary seasonal-mean; a pure forward CHANGE a static map cannot represent.
            plo = np.searchsorted(arr, d - win, "left")
            phi = np.searchsorted(arr, d, "right")
            cnt[i] = fut - np.log1p(phi - plo)
        else:
            cnt[i] = fut                                             # >=0
    return cnt


def _reg_skill(pred, y, _yt):
    pred = pred.squeeze(-1); y = y.squeeze(-1)
    mae = (pred - y).abs().mean().item()
    ss_res = ((pred - y) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item() + 1e-9
    return mae, 1.0 - ss_res / ss_tot                                # (MAE, R2) -- acc slot holds R2






# ---------------------------------------------------------------------------------------------------------
# TARGET-BREADTH map (LOOP-spacetime-target-breadth) -- more dynamic per-(cell[,species]) LEVEL targets on
# the SAME leak-guarded propagator recipe (space-only query feat; SPATIAL-only edge; neighbours carry OWN
# past state). Each returns (target[N] float32, nstate[N,S] float32) so probe can reuse _windows/_assemble.
# All are forecast-ahead: the window is [d+lead-win, d+lead]; build_causal_windows still sees only past<=d.
# ---------------------------------------------------------------------------------------------------------
def _occupancy_target(lat, lon, days, sp, block=0.5, win=180.0, lead=180.0, sub=30.0):
    """OCCUPANCY-LEVEL: fraction of `sub`-day sub-bins within the future window [d+lead-win, d+lead] in which
    the query's (cell,species) is DETECTED (>=1 obs). A [0,1] detection-rate level a static map cannot pin.
    Neighbour past state = its OWN trailing occupancy fraction over [d-win, d] (never relative to query)."""
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    key_days = defaultdict(list)
    keys = list(zip(ci.tolist(), cj.tolist(), sp.tolist()))
    for kk, d in zip(keys, days):
        key_days[kk].append(float(d))
    for kk in key_days:
        key_days[kk] = np.sort(np.array(key_days[kk]))
    nb = max(1, int(round(win / sub)))
    tgt = np.empty(len(days), np.float32); past = np.empty(len(days), np.float32)
    for i, (kk, d) in enumerate(zip(keys, days)):
        arr = key_days[kk]
        f0 = d + lead - win
        occ = 0
        for b in range(nb):
            lo = np.searchsorted(arr, f0 + b * sub, "left")
            hi = np.searchsorted(arr, f0 + (b + 1) * sub, "left")
            occ += 1 if hi > lo else 0
        tgt[i] = occ / nb
        p0 = d - win; pocc = 0
        for b in range(nb):
            lo = np.searchsorted(arr, p0 + b * sub, "left")
            hi = np.searchsorted(arr, p0 + (b + 1) * sub, "left")
            pocc += 1 if hi > lo else 0
        past[i] = pocc / nb
    return tgt, past.reshape(-1, 1).astype(np.float32)


def _richness_target(lat, lon, days, sp, block=0.5, win=180.0, lead=180.0):
    """COMMUNITY RICHNESS-LEVEL (rule 10-12 community, dynamic): log(1 + #distinct species) observed in the
    query CELL over the future window [d+lead-win, d+lead]. Per-cell dynamic diversity a static climatology
    cannot forecast. Neighbour past state = its OWN trailing log-richness over [d-win, d]."""
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    cell = defaultdict(list)
    keys = list(zip(ci.tolist(), cj.tolist()))
    for kk, d, s in zip(keys, days, sp.tolist()):
        cell[kk].append((float(d), int(s)))
    for kk in cell:
        cell[kk].sort()
    tgt = np.empty(len(days), np.float32); past = np.empty(len(days), np.float32)
    for i, (kk, d) in enumerate(zip(keys, days)):
        recs = cell[kk]
        ds = np.array([r[0] for r in recs])
        f0, f1 = d + lead - win, d + lead
        lo = np.searchsorted(ds, f0, "left"); hi = np.searchsorted(ds, f1, "right")
        tgt[i] = np.log1p(len({recs[j][1] for j in range(lo, hi)}))
        plo = np.searchsorted(ds, d - win, "left"); phi = np.searchsorted(ds, d, "right")
        past[i] = np.log1p(len({recs[j][1] for j in range(plo, phi)}))
    return tgt, past.reshape(-1, 1).astype(np.float32)


def _community_activity_target(lat, lon, days, block=0.5, win=180.0, lead=180.0):
    """COMMUNITY TOTAL-ACTIVITY-LEVEL (rule 10-12, dynamic): log(1 + total obs of ALL species) in the query
    cell over the future window. Cell-level dynamic throughput. Neighbour past = OWN trailing log-activity."""
    tgt = _abundance_target(lat, lon, days, block=block, win=win, lead=lead, delta=False)
    return tgt, tgt.reshape(-1, 1).astype(np.float32)


# =========================================================================================================
# LOOP-spacetime NEW DIRECTIONS on the mean-DOY phenology graduation target (additive, flag-gated).
# The propagator forte is NON-STATIONARY DISTRIBUTIONAL TIMING (community mean-DOY). These probe where new
# science remains: (2) env-conditioning as a propagator input; (3) other distributional-timing targets;
# (4) per-taxon breakdown. Spatial generalization (1) is driven by the `test` mask handed in from probe.py
# (spatial_holdout instead of temporal_holdout) -- no new head. All keep phenology.py leak-guards: query
# feature space-only (t=0); edge SPATIAL offset only (no dt-to-query); neighbours carry OWN observed DOY.
# =========================================================================================================






def run_pheno_env(qfeat_all, feat_dim, days, coords_ll, env, test, dev,
                  K=16, steps=4000, lr=3e-3, hidden=256, tol_days=15.0):
    """ENV-conditioning ablation for mean-DOY: neighbour-only vs neighbour+env vs env-only(static) + plain static
    floor. Per-obs env [N,D] joined by caller. Returns per-mode {mae,acc} + n_te + env_dim. Leak-guards intact."""
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    N, F_ = qfeat_all.shape
    E = env.shape[1]
    tr_idx = np.where(~test)[0]; te_idx = np.where(test)[0]
    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat, lon, days, q_train, tr_idx, K)
    g_te, v_te = _windows(lat, lon, days, te_idx, tr_idx, K)
    doyv = doy_to_vec(doy_of(days))

    def pack(q_idx, gidx, valid):
        B = len(q_idx); gsafe = np.clip(gidx, 0, N - 1); vmask = torch.tensor(valid)
        nfeat = qfeat_all[torch.tensor(gsafe.reshape(-1))].reshape(B, K, F_) * vmask.unsqueeze(-1)
        ndoy = torch.tensor(doyv[gsafe.reshape(-1)]).reshape(B, K, 2) * vmask.unsqueeze(-1)
        nenv = torch.tensor(env[gsafe.reshape(-1)]).reshape(B, K, E).float() * vmask.unsqueeze(-1)
        dlat = np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)
        dlon = np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)
        edge = torch.tensor(np.stack([dlat / 90.0, dlon / 180.0], -1)).float()
        qfeat = qfeat_all[torch.tensor(q_idx)]
        qenv = torch.tensor(env[q_idx]).float()
        ytrue = torch.tensor(doy_of(days[q_idx])); yvec = torch.tensor(doyv[q_idx])
        ok = vmask.any(1); lengths = vmask.sum(1)
        return [t[ok] for t in (nfeat, ndoy, qfeat, edge, vmask, lengths, nenv, qenv, yvec, ytrue)]

    to = lambda ts: [t.to(dev) for t in ts]
    nftr, ndtr, qftr, etr, mtr, ltr, netr, qetr, yvtr, yttr = to(pack(q_train, g_tr, v_tr))
    nfte, ndte, qfte, ete, mte, lte, nete, qete, yvte, ytte = to(pack(te_idx, g_te, v_te))
    n_te = int(nfte.shape[0])
    out = {"n_te": n_te, "env_dim": E}
    if nftr.shape[0] == 0 or n_te == 0:
        return {"static_mae": float("nan"), "static_acc": float("nan"), "neighbour_mae": float("nan"),
                "neighbour_acc": float("nan"), "neighbourenv_mae": float("nan"), "neighbourenv_acc": float("nan"),
                "envonly_mae": float("nan"), "envonly_acc": float("nan"), "n_te": n_te, "env_dim": E}
    Btr = nftr.shape[0]; bs = min(2048, Btr)
    vloss = lambda pred, tgt: (1.0 - (pred * tgt).sum(-1)).mean()

    def run(model, fwd_tr, fwd_te):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            s = torch.randint(0, Btr, (bs,), device=dev)
            opt.zero_grad(); vloss(fwd_tr(s), yvtr[s]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            return _circ_skill(fwd_te(), yvte, ytte, tol_days)

    base = LSTMVec(F_, hidden, 2, 2).to(dev)
    out["neighbour_mae"], out["neighbour_acc"] = run(
        base, lambda s: base(nftr[s], ndtr[s], etr[s], ltr[s]), lambda: base(nfte, ndte, ete, lte))

    ne = LSTMVecEnv(F_, E, hidden, use_nenv=True).to(dev)
    out["neighbourenv_mae"], out["neighbourenv_acc"] = run(
        ne, lambda s: ne(nftr[s], ndtr[s], etr[s], netr[s], qetr[s], ltr[s]),
        lambda: ne(nfte, ndte, ete, nete, qete, lte))

    se = StaticVecEnv(F_, E, hidden).to(dev)
    out["envonly_mae"], out["envonly_acc"] = run(
        se, lambda s: se(qftr[s], qetr[s]), lambda: se(qfte, qete))

    sh = StaticVec(F_, hidden, 2).to(dev)
    out["static_mae"], out["static_acc"] = run(sh, lambda s: sh(qftr[s]), lambda: sh(qfte))
    return out


# ------- (3) DISTRIBUTIONAL-TIMING TARGET CLASS: distributional (phase centroid / peak week) vs mean-DOY -------
def _phase_centroid_doy(lat, lon, days, block=0.5, train_mask=None):
    """Per-obs target = circular-mean DOY of the query's 0.5deg cell (community phase centroid).

    LEAK GUARD (train_mask): the cell statistic is built ONLY from train rows. Aggregating over ALL obs put
    each test row's own DOY into its own label, and since every row in a cell then shares one label and the
    split is temporal (train/test share cells), a spatial feature could memorize cell->label. Cells with no
    train rows return NaN and the caller drops them."""
    doy = doy_of(days); ang = 2.0 * np.pi * doy / _DOY
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0.0]); keys = list(zip(ci.tolist(), cj.tolist()))
    src = np.ones(len(doy), bool) if train_mask is None else np.asarray(train_mask, bool)
    for kk, a, ok in zip(keys, ang, src):
        if ok:
            acc[kk][0] += np.cos(a); acc[kk][1] += np.sin(a)
    cen = {kk: (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * _DOY for kk, (c, s) in acc.items()}
    return np.array([cen.get(kk, np.nan) for kk in keys], dtype=np.float32)


def _peak_week_doy(lat, lon, days, block=0.5, win=14.0, train_mask=None):
    """Per-obs target = DOY of the densest `win`-day activity window in the query cell (the seasonal PEAK).

    LEAK GUARD (train_mask): built ONLY from train rows -- see _phase_centroid_doy. Aggregating over ALL obs
    made this a cell->label lookup that scored 0.68 vs a 0.067 record (static field 26.5d MAE with propagation
    adding +1.6d, the tell). Cells with no train rows return NaN and the caller drops them."""
    doy = doy_of(days)
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    by = defaultdict(list); keys = list(zip(ci.tolist(), cj.tolist()))
    src = np.ones(len(doy), bool) if train_mask is None else np.asarray(train_mask, bool)
    for kk, d, ok in zip(keys, doy, src):
        if ok:
            by[kk].append(float(d))
    grid = np.arange(0, 365, 7.0); peak = {}
    for kk, ds in by.items():
        arr = np.asarray(ds); best_c, best_g = -1, 0.0
        for g in grid:
            dd = np.abs(arr - g); dd = np.minimum(dd, _DOY - dd)
            c = int((dd <= win / 2).sum())
            if c > best_c:
                best_c, best_g = c, g
        peak[kk] = best_g
    return np.array([peak.get(kk, np.nan) for kk in keys], dtype=np.float32)


def run_pheno_disttarget(qfeat_all, feat_dim, days, coords_ll, test, dev, target="phase_centroid",
                         K=16, steps=4000, lr=3e-3, hidden=256, hops=2, tol_days=15.0):
    """Distributional-timing target class: static floor vs GNN vs LSTM on phase_centroid / peak_week / mean_doy."""
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    # HARD LEAK GUARD for CELL-AGGREGATE targets. phase_centroid / peak_week assign ONE value per 0.5deg cell,
    # so if any cell contains both train and test rows the label is directly readable from the train rows in
    # that cell and a spatial feature just memorizes cell->label. Building the statistic from train rows only
    # does NOT fix this (verified: it scored 0.672 with static MAE 17.6d, propagation NEGATIVE). The only sound
    # split for these targets is SPATIAL -- test cells disjoint from train cells.
    if target in ("phase_centroid", "peak_week"):
        _tst = np.asarray(test, bool)
        _cid = np.floor(lat / 0.5).astype(np.int64) * 100000 + np.floor(lon / 0.5).astype(np.int64)
        _shared = np.intersect1d(np.unique(_cid[_tst]), np.unique(_cid[~_tst]))
        if len(_shared):
            raise SystemExit(
                f"[dyntargets] --pheno_disttarget {target} is a CELL-AGGREGATE target and {len(_shared)} cells "
                f"contain both train and test rows -> every test label is readable off the train rows in its own "
                f"cell (cell->label lookup, not a forecast). Re-run with a SPATIAL holdout (--pheno_spatial) so "
                f"test cells are disjoint from train cells, or use --pheno_disttarget mean_doy (per-obs target).")
    train_mask = ~np.asarray(test, bool)                     # LEAK GUARD: cell stats from TRAIN rows only
    if target == "phase_centroid":
        tgt = _phase_centroid_doy(lat, lon, days, train_mask=train_mask)
    elif target == "peak_week":
        tgt = _peak_week_doy(lat, lon, days, train_mask=train_mask)
    else:
        tgt = doy_of(days)
    valid = ~np.isnan(tgt)                                   # drop rows whose cell has no train support
    if not valid.all():
        import torch as _t
        vt = _t.as_tensor(valid)
        qfeat_all = qfeat_all[vt]; coords_ll = coords_ll[vt]
        days = days[valid]; lat = lat[valid]; lon = lon[valid]
        test = np.asarray(test, bool)[valid]; tgt = tgt[valid]
    nstate = doy_to_vec(doy_of(days))
    skill = lambda p, yv, yt: _circ_skill(p, yv, yt, tol_days)
    loss = lambda pred, tgt_: (1.0 - (pred * tgt_).sum(-1)).mean()
    return _fit_eval(qfeat_all, feat_dim, nstate, days, lat, lon, test, dev, tgt, 2, 2,
                     K, steps, lr, hidden, hops, skill, loss)


# ------- (4) PER-TAXON breakdown of the mean-DOY propagator gain -------
def run_pheno_by_taxon(qfeat_all, feat_dim, days, coords_ll, group, test, dev,
                       K=16, steps=4000, lr=3e-3, hidden=256, tol_days=15.0, min_te=200):
    """group[N] int per-obs taxon-group id. Trains static + LSTM on the full mean-DOY set; reports per-group
    (static_mae, lstm_mae, gain, n_te) on each group's test subset. Leak-guards identical to phenology."""
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    N, F_ = qfeat_all.shape
    tr_idx = np.where(~test)[0]; te_idx = np.where(test)[0]
    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat, lon, days, q_train, tr_idx, K)
    g_te, v_te = _windows(lat, lon, days, te_idx, tr_idx, K)
    doyv = doy_to_vec(doy_of(days))

    def pack(q_idx, gidx, valid):
        B = len(q_idx); gsafe = np.clip(gidx, 0, N - 1); vmask = torch.tensor(valid)
        nfeat = qfeat_all[torch.tensor(gsafe.reshape(-1))].reshape(B, K, F_) * vmask.unsqueeze(-1)
        ndoy = torch.tensor(doyv[gsafe.reshape(-1)]).reshape(B, K, 2) * vmask.unsqueeze(-1)
        dlat = np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)
        dlon = np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)
        edge = torch.tensor(np.stack([dlat / 90.0, dlon / 180.0], -1)).float()
        qfeat = qfeat_all[torch.tensor(q_idx)]
        ytrue = torch.tensor(doy_of(days[q_idx])); yvec = torch.tensor(doyv[q_idx])
        ok = vmask.any(1); lengths = vmask.sum(1); grp = torch.tensor(group[q_idx])
        return [t[ok] for t in (nfeat, ndoy, qfeat, edge, vmask, lengths, yvec, ytrue, grp)]

    to = lambda ts: [t.to(dev) for t in ts[:-1]] + [ts[-1]]
    tr = pack(q_train, g_tr, v_tr); te = pack(te_idx, g_te, v_te)
    nftr, ndtr, qftr, etr, mtr, ltr, yvtr, yttr, _ = to(tr)
    nfte, ndte, qfte, ete, mte, lte, yvte, ytte, gte = to(te)
    n_te = int(nfte.shape[0])
    if nftr.shape[0] == 0 or n_te == 0:
        return {"groups": [], "n_te": n_te}
    Btr = nftr.shape[0]; bs = min(2048, Btr)
    vloss = lambda pred, tgt: (1.0 - (pred * tgt).sum(-1)).mean()

    sh = StaticVec(F_, hidden, 2).to(dev); opt = torch.optim.Adam(sh.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        opt.zero_grad(); vloss(sh(qftr[s]), yvtr[s]).backward(); opt.step()
    lstm = LSTMVec(F_, hidden, 2, 2).to(dev); opt = torch.optim.Adam(lstm.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        opt.zero_grad(); vloss(lstm(nftr[s], ndtr[s], etr[s], ltr[s]), yvtr[s]).backward(); opt.step()
    sh.eval(); lstm.eval()
    with torch.no_grad():
        s_err = circ_err_days(vec_to_doy(sh(qfte)), ytte).cpu().numpy()
        l_err = circ_err_days(vec_to_doy(lstm(nfte, ndte, ete, lte)), ytte).cpu().numpy()
    gnp = gte.cpu().numpy(); rows = []
    for gid in np.unique(gnp):
        m = gnp == gid
        if int(m.sum()) < min_te:
            continue
        rows.append({"group": int(gid), "n_te": int(m.sum()),
                     "static_mae": float(s_err[m].mean()), "lstm_mae": float(l_err[m].mean()),
                     "gain": float(s_err[m].mean() - l_err[m].mean())})
    rows.sort(key=lambda d: -d["n_te"])
    return {"groups": rows, "n_te": n_te}

# ---------------------------------------------------------------------------------------------------------
# LOOP-spacetime rule-24 DENSE-FIELD interpolation on the mean-DOY graduation target (additive, default-off)
# ---------------------------------------------------------------------------------------------------------
# The characterized mean-DOY graduation reads neighbouring OCCUPIED cells. rule-24's true claim is denser:
# infer phenology-DOY at a cell x time that has NO observation of the query species/point in the query's own
# cell -- pure spatial interpolation from SURROUNDING occupied cells. Here we (a) forbid any pool neighbour
# in the query's OWN `block`-deg cell (strict leak-guard: the query cell contributes nothing to itself), and
# (b) partition test queries by whether the query cell is genuinely EMPTY in the pool (no obs at all) vs
# OCCUPIED (has obs, but excluded from its own window). Reports MAE and gain-over-static in each regime.
# `drop_cell_frac` additionally thins the pool by whole cells to stress how far interpolation degrades.
def _cellid(lat, lon, block):
    return np.floor(lat / block).astype(np.int64) * 1000003 + np.floor(lon / block).astype(np.int64)


def _windows_nocell(lat, lon, days, q_idx, pool_idx, K, block):
    """Same-cell-excluded causal windows: over-query, then drop any neighbour whose block-cell == query cell.
    Pure spatial interpolation from SURROUNDING cells (the query cell contributes nothing to itself)."""
    qcell = _cellid(lat[q_idx], lon[q_idx], block)
    # over-query K*8 causal nearest, then filter same-cell and keep first K survivors (nearest-first).
    qi, _ = build_causal_windows(lat[q_idx], lon[q_idx], days[q_idx],
                                 lat[pool_idx], lon[pool_idx], days[pool_idx], K * 8)
    gi = np.where(qi >= 0, pool_idx[np.clip(qi, 0, None)], -1)          # [B, K*8] pool indices (-1 pad)
    pcell = np.where(gi >= 0, _cellid(lat[np.clip(gi, 0, None)], lon[np.clip(gi, 0, None)], block), -2)
    same = (pcell == qcell[:, None]) | (gi < 0)                          # drop same-cell AND pads
    out = np.full((len(q_idx), K), -1, dtype=np.int64)
    for r in range(len(q_idx)):
        keep = gi[r][~same[r]]
        out[r, :min(K, len(keep))] = keep[:K]
    return out, out >= 0


def run_pheno_densefield(qfeat_all, feat_dim, days, coords_ll, test, dev, block=0.5, drop_cell_frac=0.0,
                         K=16, steps=800, lr=3e-3, hidden=256, hops=2, tol_days=15.0, seed=0):
    """rule-24 dense-field: mean-DOY interpolation at query points whose OWN cell is excluded from the window.
    Static floor vs LSTM propagator; MAE-gain reported for EMPTY-cell vs OCCUPIED-cell query subsets."""
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    target = doy_of(days)
    nstate_all = doy_to_vec(target)
    N = qfeat_all.shape[0]
    tr_idx = np.where(~test)[0]; te_idx = np.where(test)[0]
    rng = np.random.default_rng(seed)
    pool = tr_idx
    if drop_cell_frac > 0:                                               # thin the pool by whole cells
        pcells = np.unique(_cellid(lat[tr_idx], lon[tr_idx], block))
        rng.shuffle(pcells)
        drop = set(pcells[: int(len(pcells) * drop_cell_frac)].tolist())
        keepm = np.array([c not in drop for c in _cellid(lat[tr_idx], lon[tr_idx], block)])
        pool = tr_idx[keepm]
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows_nocell(lat, lon, days, q_train, pool, K, block)
    g_te, v_te = _windows_nocell(lat, lon, days, te_idx, pool, K, block)
    tr = _assemble(qfeat_all, nstate_all, days, lat, lon, q_train, g_tr, v_tr, target, K, 2)
    # keep the ok-mask alignment for test so we can label empty/occupied on the SAME rows _assemble kept
    okte = torch.tensor(v_te).any(1).numpy()
    te = _assemble(qfeat_all, nstate_all, days, lat, lon, te_idx, g_te, v_te, target, K, 2)
    to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, yvtr, yttr = to(tr)
    nfte, nste, qfte, ete, mte, lte, yvte, ytte = to(te)
    n_te = int(nfte.shape[0])
    if nftr.shape[0] == 0 or n_te == 0:
        return {"n_te": n_te, "empty_n": 0, "occ_n": 0}
    # EMPTY vs OCCUPIED label per kept test row: is the query's cell present anywhere in the (thinned) pool?
    poolcells = set(np.unique(_cellid(lat[pool], lon[pool], block)).tolist())
    te_kept = te_idx[okte]
    qcell_te = _cellid(lat[te_kept], lon[te_kept], block)
    is_empty = np.array([c not in poolcells for c in qcell_te])          # True = genuinely empty query cell
    Btr = nftr.shape[0]; bs = min(2048, Btr)
    vloss = lambda pred, tgt_: (1.0 - (pred * tgt_).sum(-1)).mean()

    def fit(model, fwd):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            s = torch.randint(0, Btr, (bs,), device=dev)
            opt.zero_grad(); vloss(fwd(model, s), yvtr[s]).backward(); opt.step()
        model.eval()

    sh = StaticVec(feat_dim, hidden, 2).to(dev); fit(sh, lambda m, s: m(qftr[s]))
    lstm = LSTMVec(feat_dim, hidden, 2, 2).to(dev); fit(lstm, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
    with torch.no_grad():
        s_err = circ_err_days(vec_to_doy(sh(qfte)), ytte).cpu().numpy()
        l_err = circ_err_days(vec_to_doy(lstm(nfte, nste, ete, lte)), ytte).cpu().numpy()
    def stat(mask):
        if int(mask.sum()) == 0:
            return {"n": 0, "static_mae": float("nan"), "lstm_mae": float("nan"), "gain": float("nan")}
        sm = float(s_err[mask].mean()); lm = float(l_err[mask].mean())
        return {"n": int(mask.sum()), "static_mae": sm, "lstm_mae": lm, "gain": sm - lm}
    return {"n_te": n_te, "block": block, "drop_cell_frac": drop_cell_frac, "pool_n": int(len(pool)),
            "all": stat(np.ones(n_te, bool)), "empty": stat(is_empty), "occ": stat(~is_empty)}


# =====================================================================================================
# CROSS-ENCODER ROUTING TEST (additive, flag-gated; the two loops connect):
# The BIOLOGICAL loop found co-occurrence/community is a spatial-niche axis the PHYLO graph does NOT serve
# (graph_gain negative). By the routing law it should be served HERE by the spacetime/env encoder. This
# helper builds a per-species co-occurrence PARTNER-SET target on the SAME 0.5deg grid the bio loop used
# (derived/cooccur_count_005.npy, species_local==vocab index==cooccur row), and predicts a held-out
# species' partner-set from ENVIRONMENT + SPACE vs a non-spatial (prevalence) baseline. micro-AP + gain.
# All leak-guards reported per run.
# =====================================================================================================
def cooccur_routing(cache, thresh=2, min_deg=5, seed=0, mechanism="env", n_shards_space=None,
                    cooccur_file="cooccur_count_005.npy", env_channels="all"):
    """Predict a species co-occurrence PARTNER-SET from per-species ENV/SPACE features (held-out species).

    Target: binary partner matrix P[S,S], P[i,j]=1 iff species i and j co-occur >= `thresh` times in a 0.5deg
    cell (derived/cooccur_count_005.npy, symmetric, diagonal removed). We hold out a random subset of species
    as QUERIES; the candidate partner set is TRAIN species only (held-out queries can never be candidates and
    their own cooccur row is never used as a feature -> no target leakage).

    Model = bilinear niche-similarity: score(query i, partner j) = f(feat_i) . g(feat_j), f,g small MLPs,
    trained on TRAIN x TRAIN partner labels (BCE). At test we score each held-out query i against all TRAIN
    candidates j and rank; micro-AP over the flattened held-out (query x train-candidate) label matrix.

    mechanism:
      'env'   : per-species niche features = worldclim(19)+AlphaEarth(64) mean (shared-niche hypothesis).
      'space' : per-species GEOGRAPHIC features = cell-centroid (lat,lon mean) + spatial spread (lat,lon std)
                + a spatial-neighbour partner-propagation feature (mean partner-prevalence of the species that
                share the query's occupied cells, computed over TRAIN species only -> no query-row leakage).
      'both'  : concat(env, space).
    Baseline (non-spatial, features-blind) = partner PREVALENCE: rank candidates by how often each train
    species is a partner of ANY train species. gain = micro_AP(features) - micro_AP(prevalence).
    """
    import numpy as _np, glob as _g
    from pathlib import Path as _P
    import torch as _t, torch.nn as _nn, torch.nn.functional as _F
    cachep = _P(cache)
    C = _np.load(cachep / "derived" / cooccur_file).astype(_np.int64)
    S = C.shape[0]
    _np.fill_diagonal(C, 0)
    P = (C >= thresh).astype(_np.float32)                        # [S,S] binary partner-set

    # ---- per-species features ----
    wc = _np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(x): i for i, x in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = _np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(x): i for i, x in enumerate(ae["gbifID"])}; AE = ae["ae"]
    Dsum = _np.zeros((S, 83), _np.float64); Dn = _np.zeros(S, _np.int64)
    latsum = _np.zeros(S); lonsum = _np.zeros(S); lat2 = _np.zeros(S); lon2 = _np.zeros(S); geon = _np.zeros(S, _np.int64)
    cells_by_sp = [set() for _ in range(S)]
    files = sorted(_g.glob(str(cachep / "gbif_tokens" / "*.npz")))
    if n_shards_space is not None:
        files = files[:n_shards_space]
    for f in files:
        z = _np.load(f); sl = z["species_local"].astype(_np.int64); gid = z["gbifID"]; la = z["lat"]; lo = z["lon"]
        for s, gg, a, o in zip(sl, gid, la, lo):
            s = int(s); gg = int(gg)
            if gg in wcm and gg in aem:
                Dsum[s, :19] += WC[wcm[gg]]; Dsum[s, 19:] += AE[aem[gg]]; Dn[s] += 1
            latsum[s] += a; lonsum[s] += o; lat2[s] += a * a; lon2[s] += o * o; geon[s] += 1
            cells_by_sp[s].add((int(_np.floor(a / 0.5)), int(_np.floor(o / 0.5))))
    ENV = _np.full((S, 83), _np.nan, _np.float32)
    ok = Dn > 0; ENV[ok] = (Dsum[ok] / Dn[ok, None]).astype(_np.float32)
    GEO = _np.full((S, 4), _np.nan, _np.float32)
    okg = geon > 0
    GEO[okg, 0] = latsum[okg] / geon[okg]; GEO[okg, 1] = lonsum[okg] / geon[okg]
    GEO[okg, 2] = _np.sqrt(_np.maximum(lat2[okg] / geon[okg] - (latsum[okg] / geon[okg]) ** 2, 0))
    GEO[okg, 3] = _np.sqrt(_np.maximum(lon2[okg] / geon[okg] - (lonsum[okg] / geon[okg]) ** 2, 0))

    deg = (P > 0).sum(1)
    elig = ok & okg & (deg >= min_deg)
    idx = _np.where(elig)[0]
    rng = _np.random.default_rng(seed); rng.shuffle(idx)
    cut = len(idx) // 5
    te = idx[:cut]; tr = idx[cut:]

    def _z(X, tr_):
        m = _np.nanmean(X[tr_], 0); sd = _np.nanstd(X[tr_], 0); sd[sd < 1e-6] = 1.0
        return _np.nan_to_num((X - m) / sd, nan=0.0).astype(_np.float32)
    ENVz = _z(ENV, tr); GEOz = _z(GEO, tr)
    if env_channels == "worldclim":
        ENVz = ENVz[:, :19]
    elif env_channels == "alphaearth":
        ENVz = ENVz[:, 19:83]


    prevalence_tr = P[tr][:, tr].mean(0)                        # [len(tr)]
    prev_full = _np.zeros(S, _np.float32); prev_full[tr] = prevalence_tr
    sp_of_cell = {}
    for s in tr:
        for c in cells_by_sp[s]:
            sp_of_cell.setdefault(c, []).append(s)
    prop = _np.zeros(S, _np.float32)
    for s in range(S):
        neigh = []
        for c in cells_by_sp[s]:
            neigh += [x for x in sp_of_cell.get(c, []) if x != s]
        if neigh:
            prop[s] = prev_full[neigh].mean()
    PROP = _z(prop[:, None], tr)
    SPACE = _np.concatenate([GEOz, PROP], 1)

    if mechanism == "env":
        FEAT = ENVz
    elif mechanism == "space":
        FEAT = SPACE
    else:
        FEAT = _np.concatenate([ENVz, SPACE], 1)

    dev = "cuda" if _t.cuda.is_available() else "cpu"
    Ftr = _t.tensor(FEAT[tr], device=dev); Fte = _t.tensor(FEAT[te], device=dev)
    Ptr = _t.tensor(P[tr][:, tr], device=dev)
    Pte = P[te][:, tr]

    class Bilin(_nn.Module):
        def __init__(s, d, h=64, emb=32):
            super().__init__()
            s.f = _nn.Sequential(_nn.Linear(d, h), _nn.ReLU(), _nn.Linear(h, emb))
            s.g = _nn.Sequential(_nn.Linear(d, h), _nn.ReLU(), _nn.Linear(h, emb))
            s.b = _nn.Parameter(_t.zeros(1))
        def forward(s, qi, cj):
            return s.f(qi) @ s.g(cj).T + s.b
    net = Bilin(FEAT.shape[1]).to(dev)
    opt = _t.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    Btr = len(tr); pos_w = _t.tensor([(Ptr.numel() - Ptr.sum()) / Ptr.sum().clamp_min(1)], device=dev)
    eye = _t.eye(Btr, device=dev, dtype=_t.bool)
    for _ in range(1200):
        qb = _t.randint(0, Btr, (min(256, Btr),), device=dev)
        logit = net(Ftr[qb], Ftr)
        lab = Ptr[qb]
        m = ~eye[qb]
        loss = _F.binary_cross_entropy_with_logits(logit[m], lab[m], pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with _t.no_grad():
        scores = net(Fte, Ftr).cpu().numpy()

    from sklearn.metrics import average_precision_score as _ap
    y_flat = Pte.reshape(-1)
    ap_feat = float(_ap(y_flat, scores.reshape(-1))) if y_flat.sum() > 0 else float("nan")
    prev_score = _np.broadcast_to(prevalence_tr[None, :], Pte.shape).reshape(-1)
    ap_prev = float(_ap(y_flat, prev_score)) if y_flat.sum() > 0 else float("nan")
    base_rate = float(y_flat.mean())
    return {
        "mechanism": mechanism, "thresh": thresh, "cooccur_file": cooccur_file, "env_channels": env_channels,
        "n_query_sp": int(len(te)), "n_cand_sp": int(len(tr)), "feat_dim": int(FEAT.shape[1]),
        "micro_AP_feat": ap_feat, "micro_AP_prevalence": ap_prev, "micro_AP_baserate": base_rate,
        "gain_over_prevalence": ap_feat - ap_prev, "lift_over_baserate": ap_feat / max(base_rate, 1e-9),
        "leak_guard": "held-out query species excluded from candidates+features; own cooccur row never a feature; prevalence+propagation from TRAIN species only; self-candidate diagonal masked in training",
    }


# =====================================================================================================
# SDM ENV->BIOLOGY (science.md rules 1-6, B1/B5/B6/B8): predict which SPECIES occur at a held-out 0.5deg
# CELL from the cell's ENVIRONMENT (+ spatial position). Isolated niche/propagation mechanism at the probe
# level (older full-model sweeps called this "saturated"). env-only vs space-only vs both.
# =====================================================================================================
def sdm_presence(cache, seed=0, mechanism="both", min_cell_obs=3, cooccur_file="cooccur_count_005.npy"):
    """Multi-label species-presence at a CELL from env+space (held-out cells).

    Target: for each occupied 0.5deg cell, the binary species-presence vector Y[cell, species] over the 2141
    vocab. Cell feature = mean env (worldclim19+AE64) of the obs in that cell, plus its (lat,lon) centroid.
    Hold out a random 20% of CELLS; a bilinear cell->species head trained on TRAIN cells predicts held-out
    cell presence. micro-AP over held-out (cell x species). Baseline = species PREVALENCE across train cells.

    mechanism: 'env' = cell env only; 'space' = cell centroid (lat,lon) only; 'both' = concat.
    Leak-guards: held-out cells contribute no obs to any feature/prevalence; species-prevalence from TRAIN
    cells only; a cell's own species vector is never a feature.
    """
    import numpy as _np, glob as _g
    from pathlib import Path as _P
    import torch as _t, torch.nn as _nn, torch.nn.functional as _F
    cachep = _P(cache)
    S = _np.load(cachep / "derived" / cooccur_file).shape[0]
    wc = _np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(x): i for i, x in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = _np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(x): i for i, x in enumerate(ae["gbifID"])}; AE = ae["ae"]
    from collections import defaultdict
    cell_env = defaultdict(lambda: [_np.zeros(83, _np.float64), 0])
    cell_ll = defaultdict(lambda: [0.0, 0.0, 0])
    cell_sp = defaultdict(set)
    for f in sorted(_g.glob(str(cachep / "gbif_tokens" / "*.npz"))):
        z = _np.load(f); sl = z["species_local"].astype(_np.int64); gid = z["gbifID"]; la = z["lat"]; lo = z["lon"]
        for s, gg, a, o in zip(sl, gid, la, lo):
            gg = int(gg)
            c = (int(_np.floor(a / 0.5)), int(_np.floor(o / 0.5)))
            cell_sp[c].add(int(s))
            cell_ll[c][0] += a; cell_ll[c][1] += o; cell_ll[c][2] += 1
            if gg in wcm and gg in aem:
                ce = cell_env[c]; ce[0][:19] += WC[wcm[gg]]; ce[0][19:] += AE[aem[gg]]; ce[1] += 1
    cells = [c for c in cell_sp if cell_env[c][1] >= min_cell_obs and len(cell_sp[c]) >= 1]
    cells = sorted(cells)
    Nc = len(cells)
    ENV = _np.zeros((Nc, 83), _np.float32); LL = _np.zeros((Nc, 2), _np.float32); Y = _np.zeros((Nc, S), _np.float32)
    for i, c in enumerate(cells):
        ce = cell_env[c]; ENV[i] = (ce[0] / ce[1]).astype(_np.float32)
        cl = cell_ll[c]; LL[i] = [cl[0] / cl[2], cl[1] / cl[2]]
        for s in cell_sp[c]:
            Y[i, s] = 1.0
    rng = _np.random.default_rng(seed); order = _np.arange(Nc); rng.shuffle(order)
    cut = Nc // 5; te = order[:cut]; tr = order[cut:]
    def _z(X, tr_):
        import numpy as __np
        m = __np.nanmean(X[tr_], 0); sd = __np.nanstd(X[tr_], 0); sd[sd < 1e-6] = 1.0
        return __np.nan_to_num((X - m) / sd, nan=0.0).astype(_np.float32)
    ENVz = _z(ENV, tr); LLz = _z(LL, tr)
    if mechanism == "env": FEAT = ENVz
    elif mechanism == "space": FEAT = LLz
    else: FEAT = _np.concatenate([ENVz, LLz], 1)
    # restrict candidate species to those present in >=1 train cell (leak-safe prevalence)
    prev = Y[tr].mean(0)
    cand = _np.where(prev > 0)[0]
    Ytr = Y[tr][:, cand]; Yte = Y[te][:, cand]; prev_c = prev[cand]
    dev = "cuda" if _t.cuda.is_available() else "cpu"
    Ft = _t.tensor(FEAT[tr], device=dev); Fe = _t.tensor(FEAT[te], device=dev)
    Yt = _t.tensor(Ytr, device=dev)
    net = _nn.Sequential(_nn.Linear(FEAT.shape[1], 256), _nn.ReLU(), _nn.Linear(256, len(cand))).to(dev)
    opt = _t.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    pw = _t.tensor((1 - prev_c) / _np.maximum(prev_c, 1e-3), device=dev, dtype=_t.float32).clamp(max=50.0)
    for _ in range(1500):
        b = _t.randint(0, len(tr), (min(512, len(tr)),), device=dev)
        loss = _F.binary_cross_entropy_with_logits(net(Ft[b]), Yt[b], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with _t.no_grad():
        sc = net(Fe).cpu().numpy()
    from sklearn.metrics import average_precision_score as _ap
    yf = Yte.reshape(-1)
    ap_feat = float(_ap(yf, sc.reshape(-1))) if yf.sum() > 0 else float("nan")
    prev_score = _np.broadcast_to(prev_c[None, :], Yte.shape).reshape(-1)
    ap_prev = float(_ap(yf, prev_score)) if yf.sum() > 0 else float("nan")
    br = float(yf.mean())
    return {
        "mechanism": mechanism, "cooccur_file": cooccur_file,
        "n_query_cells": int(len(te)), "n_cand_sp": int(len(cand)), "feat_dim": int(FEAT.shape[1]),
        "micro_AP_feat": ap_feat, "micro_AP_prevalence": ap_prev, "micro_AP_baserate": br,
        "gain_over_prevalence": ap_feat - ap_prev, "lift_over_baserate": ap_feat / max(br, 1e-9),
        "leak_guard": "held-out cells give no obs to features/prevalence; species-prevalence from TRAIN cells only; cell's own species vector never a feature",
    }


# =====================================================================================================
# HARDENED SDM ENV->PRESENCE (LOOP-spacetime this-turn): fixes the 40-held-out-cell weakness of
# sdm_presence via (a) configurable finer grid cell_deg, (b) proper spatial-block cross-validation
# (contiguous super-blocks held out, geographically separated from train), (c) per-channel env
# decomposition worldclim/alphaearth/soil/elev, (d) optional per-cell seasonal TIME features (phenology)
# appended to env to test whether WHEN informs WHERE-WHO over env-only. Additive; original sdm_presence
# untouched. Deterministic single-seed; caller loops seeds for CI.
# =====================================================================================================
def sdm_presence_hard(cache, seed=0, mechanism="env", min_cell_obs=3, cell_deg=0.1,
                      holdout_mode="block", block_deg=2.0, holdout_frac=0.2,
                      env_channels="all", add_time=False, cooccur_file="cooccur_count_005.npy"):
    """Hardened multi-label species-presence@cell. See module header for the design.

    holdout_mode:
      'random' : random `holdout_frac` of cells (old behaviour, but on the finer grid).
      'block'  : hold out whole contiguous SUPER-BLOCKS of width `block_deg` (>> cell_deg) so held-out
                 cells are geographically separated from train cells (true spatial generalization, no
                 adjacent-cell leakage). ~holdout_frac of super-blocks held out.
    env_channels: 'all'(wc19+ae64) | 'worldclim'(19) | 'alphaearth'(64) | 'soil'(9) | 'elev'(1).
    add_time: append per-cell [sin(mean_doy), cos(mean_doy), R] (circular seasonal timing) to FEAT.
    Leak-guards preserved: held-out cells give no obs to any feature/prevalence; prevalence from TRAIN
    cells only; a cell's own species vector is never a feature; block holdout separates geography.
    """
    import numpy as _np, glob as _g
    from pathlib import Path as _P
    import torch as _t, torch.nn as _nn, torch.nn.functional as _F
    from collections import defaultdict
    cachep = _P(cache)
    S = _np.load(cachep / "derived" / cooccur_file).shape[0]
    wc = _np.load(cachep / "gbif_worldclim_tokens.npz"); wcm = {int(x): i for i, x in enumerate(wc["gbifID"])}; WC = wc["worldclim"]
    ae = _np.load(cachep / "gbif_alphaearth_tokens.npz"); aem = {int(x): i for i, x in enumerate(ae["gbifID"])}; AE = ae["ae"]
    so = _np.load(cachep / "gbif_soil_features.npz"); som = {int(x): i for i, x in enumerate(so["gbifID"])}; SO = so["feat"]
    el = _np.load(cachep / "gbif_elev.npz"); elm = {int(x): i for i, x in enumerate(el["gbifID"])}; EL = el["elev"]
    tt = _np.load(cachep / "gbif_eventtime.npz"); ttm = {int(x): i for i, x in enumerate(tt["gbifID"])}; DAYS = tt["days"]
    # env layout: [wc19 | ae64 | soil9 | elev1] = 93 dims; per-channel counts tracked separately
    ED = 93
    cell_env = defaultdict(lambda: [_np.zeros(ED, _np.float64), _np.zeros(ED, _np.int64)])
    cell_ll = defaultdict(lambda: [0.0, 0.0, 0])
    cell_sp = defaultdict(set)
    cell_tt = defaultdict(lambda: [0.0, 0.0, 0])   # sum sin, sum cos, n  (circular DOY over 365)
    for f in sorted(_g.glob(str(cachep / "gbif_tokens" / "*.npz"))):
        z = _np.load(f); sl = z["species_local"].astype(_np.int64); gid = z["gbifID"]; la = z["lat"]; lo = z["lon"]
        for s, gg, a, o in zip(sl, gid, la, lo):
            gg = int(gg)
            c = (int(_np.floor(a / cell_deg)), int(_np.floor(o / cell_deg)))
            cell_sp[c].add(int(s))
            cell_ll[c][0] += a; cell_ll[c][1] += o; cell_ll[c][2] += 1
            ce = cell_env[c]
            if gg in wcm: ce[0][:19] += WC[wcm[gg]]; ce[1][:19] += 1
            if gg in aem: ce[0][19:83] += AE[aem[gg]]; ce[1][19:83] += 1
            if gg in som: ce[0][83:92] += SO[som[gg]]; ce[1][83:92] += 1
            if gg in elm: ce[0][92] += EL[elm[gg]]; ce[1][92] += 1
            if gg in ttm:
                ang = 2 * _np.pi * (DAYS[ttm[gg]] % 365.0) / 365.0
                ct = cell_tt[c]; ct[0] += _np.sin(ang); ct[1] += _np.cos(ang); ct[2] += 1
    # a cell qualifies if it has >=min_cell_obs obs with worldclim (the always-present env anchor)
    cells = [c for c in cell_sp if cell_env[c][1][0] >= min_cell_obs and len(cell_sp[c]) >= 1]
    cells = sorted(cells)
    Nc = len(cells)
    ENV = _np.zeros((Nc, ED), _np.float32); LL = _np.zeros((Nc, 2), _np.float32)
    TT = _np.zeros((Nc, 3), _np.float32); Y = _np.zeros((Nc, S), _np.float32)
    for i, c in enumerate(cells):
        ce = cell_env[c]; n = _np.maximum(ce[1], 1); ENV[i] = (ce[0] / n).astype(_np.float32)
        cl = cell_ll[c]; LL[i] = [cl[0] / cl[2], cl[1] / cl[2]]
        ct = cell_tt[c]
        if ct[2] > 0:
            sb = ct[0] / ct[2]; cb = ct[1] / ct[2]; R = _np.hypot(sb, cb)
            TT[i] = [sb, cb, R]
        for s in cell_sp[c]:
            Y[i, s] = 1.0
    # ---- holdout ----
    rng = _np.random.default_rng(seed)
    if holdout_mode == "block":
        blk = _np.array([(int(_np.floor(c[0] * cell_deg / block_deg)),
                          int(_np.floor(c[1] * cell_deg / block_deg))) for c in cells])
        ublk = list({tuple(b) for b in blk}); rng.shuffle(ublk)
        ncut = max(1, int(len(ublk) * holdout_frac)); held = set(ublk[:ncut])
        te = _np.array([i for i in range(Nc) if tuple(blk[i]) in held], dtype=_np.int64)
        tr = _np.array([i for i in range(Nc) if tuple(blk[i]) not in held], dtype=_np.int64)
    else:
        order = _np.arange(Nc); rng.shuffle(order)
        cut = int(Nc * holdout_frac); te = order[:cut]; tr = order[cut:]

    # ---- channel selection ----
    ch = {"all": slice(0, 83), "worldclim": slice(0, 19), "alphaearth": slice(19, 83),
          "soil": slice(83, 92), "elev": slice(92, 93)}[env_channels]
    ENV = ENV[:, ch]

    def _z(X, tr_):
        import numpy as __np
        m = __np.nanmean(X[tr_], 0); sd = __np.nanstd(X[tr_], 0); sd[sd < 1e-6] = 1.0
        return __np.nan_to_num((X - m) / sd, nan=0.0).astype(__np.float32)
    ENVz = _z(ENV, tr); LLz = _z(LL, tr); TTz = _z(TT, tr)
    if mechanism == "env": FEAT = ENVz
    elif mechanism == "space": FEAT = LLz
    else: FEAT = _np.concatenate([ENVz, LLz], 1)
    if add_time: FEAT = _np.concatenate([FEAT, TTz], 1)

    prev = Y[tr].mean(0); cand = _np.where(prev > 0)[0]
    Ytr = Y[tr][:, cand]; Yte = Y[te][:, cand]; prev_c = prev[cand]
    dev = "cuda" if _t.cuda.is_available() else "cpu"
    _t.manual_seed(seed)
    Ft = _t.tensor(FEAT[tr], device=dev); Fe = _t.tensor(FEAT[te], device=dev)
    Yt = _t.tensor(Ytr, device=dev)
    net = _nn.Sequential(_nn.Linear(FEAT.shape[1], 256), _nn.ReLU(), _nn.Linear(256, len(cand))).to(dev)
    opt = _t.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    pw = _t.tensor((1 - prev_c) / _np.maximum(prev_c, 1e-3), device=dev, dtype=_t.float32).clamp(max=50.0)
    for _ in range(1500):
        b = _t.randint(0, len(tr), (min(512, len(tr)),), device=dev)
        loss = _F.binary_cross_entropy_with_logits(net(Ft[b]), Yt[b], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with _t.no_grad():
        sc = net(Fe).cpu().numpy()
    from sklearn.metrics import average_precision_score as _ap
    yf = Yte.reshape(-1)
    ap_feat = float(_ap(yf, sc.reshape(-1))) if yf.sum() > 0 else float("nan")
    prev_score = _np.broadcast_to(prev_c[None, :], Yte.shape).reshape(-1)
    ap_prev = float(_ap(yf, prev_score)) if yf.sum() > 0 else float("nan")
    br = float(yf.mean())
    return {
        "mechanism": mechanism, "cell_deg": cell_deg, "holdout_mode": holdout_mode, "block_deg": block_deg,
        "env_channels": env_channels, "add_time": bool(add_time),
        "n_query_cells": int(len(te)), "n_train_cells": int(len(tr)), "n_cand_sp": int(len(cand)),
        "feat_dim": int(FEAT.shape[1]),
        "micro_AP_feat": ap_feat, "micro_AP_prevalence": ap_prev, "micro_AP_baserate": br,
        "gain_over_prevalence": ap_feat - ap_prev, "lift_over_baserate": ap_feat / max(br, 1e-9),
        "leak_guard": "held-out cells give no obs to features/prevalence; prevalence from TRAIN cells only; cell's own species vector never a feature; block holdout geographically separates held-out super-blocks (width block_deg) from train",
    }
