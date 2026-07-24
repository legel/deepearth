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

from deepearth.autoresearch.programs.spacetime.recurrence import build_causal_windows

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
class StaticVec(nn.Module):
    """No-propagation floor: predict a (cos,sin) target from the QUERY point's OWN space-only feature."""

    def __init__(self, feat_dim, hidden=256, out=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.out = out

    def forward(self, qfeat):
        y = self.net(qfeat)
        return F.normalize(y, dim=-1) if self.out == 2 else y


class GNNVec(nn.Module):
    """Message-passing propagator. Neighbour node state = [pos-feat || neighbour observed state]; the query
    node has NO state of its own. Edge = SPATIAL offset (dlat,dlon) only (no dt leak). T hops, gated agg."""

    def __init__(self, feat_dim, hidden=256, hops=2, state_dim=2, out=2):
        super().__init__()
        self.hops = hops
        self.state_dim = state_dim
        self.out = out
        self.node_enc = nn.Sequential(nn.Linear(feat_dim + state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_enc = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))     # (dlat,dlon) only
        self.msg = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.upd = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, out)

    def forward(self, nfeat, nstate_in, qfeat, edge, mask):
        B, K, _ = nfeat.shape
        nstate = self.node_enc(torch.cat([nfeat, nstate_in], -1))
        qzero = torch.zeros(B, self.state_dim, device=nfeat.device, dtype=nfeat.dtype)
        qstate = self.node_enc(torch.cat([qfeat, qzero], -1))
        e = self.edge_enc(edge)
        m = mask.unsqueeze(-1).float()
        neg = torch.finfo(nstate.dtype).min
        for _ in range(self.hops):
            qb = qstate.unsqueeze(1).expand(-1, K, -1)
            cat = torch.cat([nstate, e, qb], -1)
            g = self.gate(cat).squeeze(-1).masked_fill(~mask, neg)
            w = torch.softmax(g, dim=1).unsqueeze(-1) * m
            qstate = self.upd((w * self.msg(cat)).sum(1), qstate)
        y = self.head(qstate)
        return F.normalize(y, dim=-1) if self.out == 2 else y


class LSTMVec(nn.Module):
    """1-D causal rollout over K past neighbours; each step = [pos-feat || neighbour state || dlat || dlon]
    (SPATIAL offset only, no dt-to-query). Final hidden -> target."""

    def __init__(self, feat_dim, hidden=256, state_dim=2, out=2):
        super().__init__()
        self.out = out
        self.lstm = nn.LSTM(feat_dim + state_dim + 2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out)

    def forward(self, nfeat, nstate_in, edge, lengths):
        x = torch.cat([nfeat, nstate_in, edge], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        y = self.head(h[-1])
        return F.normalize(y, dim=-1) if self.out == 2 else y


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


def run_first_arrival(qfeat_all, feat_dim, days, coords_ll, sp, test, dev,
                      K=16, steps=4000, lr=3e-3, hidden=256, hops=2, tol_days=15.0):
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    onset = _first_arrival_doy(lat, lon, days, sp)
    nstate = doy_to_vec(doy_of(days))                                # neighbour carries its OWN observed DOY
    skill = lambda p, yv, yt: _circ_skill(p, yv, yt, tol_days)
    loss = lambda pred, tgt: (1.0 - (pred * tgt).sum(-1)).mean()     # circular cosine regression
    return _fit_eval(qfeat_all, feat_dim, nstate, days, lat, lon, test, dev, onset, 2, 2,
                     K, steps, lr, hidden, hops, skill, loss)


def run_first_arrival_all(e4d, rff, raw, fd, days, coords_ll, sp, test, dev, **kw):
    return {ft: run_first_arrival(f, fd[ft], days, coords_ll, sp, test, dev, **kw)
            for ft, f in (("e4d", e4d), ("rff", rff), ("raw", raw))}


# ---------------------------------------------------------------------------------------------------------
# TARGET 2 -- ABUNDANCE / activity level (log obs count in the query cell over a trailing window)
# ---------------------------------------------------------------------------------------------------------
def _abundance_target(lat, lon, days, block=0.5, win=90.0):
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
        lo = np.searchsorted(arr, d - win, "left")
        hi = np.searchsorted(arr, d, "right")
        cnt[i] = np.log1p(hi - lo)                                   # includes self; >=log1p(1)
    return cnt


def _reg_skill(pred, y, _yt):
    pred = pred.squeeze(-1); y = y.squeeze(-1)
    mae = (pred - y).abs().mean().item()
    ss_res = ((pred - y) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item() + 1e-9
    return mae, 1.0 - ss_res / ss_tot                                # (MAE, R2) -- acc slot holds R2


def run_abundance(qfeat_all, feat_dim, days, coords_ll, test, dev,
                  K=16, steps=4000, lr=3e-3, hidden=256, hops=2, win=90.0):
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    tgt = _abundance_target(lat, lon, days, win=win)
    nstate = tgt.reshape(-1, 1)                                      # neighbour carries its OWN log-activity
    loss = lambda pred, y: F.smooth_l1_loss(pred, y)
    return _fit_eval(qfeat_all, feat_dim, nstate, days, lat, lon, test, dev, tgt, 1, 1,
                     K, steps, lr, hidden, hops, _reg_skill, loss)


def run_abundance_all(e4d, rff, raw, fd, days, coords_ll, test, dev, **kw):
    return {ft: run_abundance(f, fd[ft], days, coords_ll, test, dev, **kw)
            for ft, f in (("e4d", e4d), ("rff", rff), ("raw", raw))}
