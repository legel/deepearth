"""Non-stationary PHENOLOGY target for the spacetime forecaster (science.md rule 1+2b, decisive control).

Prior rounds (Ensue tag spacetime, LOOP-spacetime-gnn_forecaster): forward-in-time FAMILY-PRESENCE forecasting
is demonstrable (~23x chance) BUT a real propagator (GNN/LSTM) does NOT beat a no-propagation static head
(propagator_gain ~+0.002, inside the +/-0.008 floor) and Earth4D HURTS vs raw coords (-0.093). DIAGNOSIS: the
held-out-future family-presence target is dominated by STATIONARY spatial climatology (where a family CAN
occur) -- a static (lat,lon) map fits it directly, so propagating past state buys nothing.

This module runs the DECISIVE control on a TEMPORALLY NON-STATIONARY target a static climatology CANNOT win:
the DAY-OF-YEAR (phenology / seasonal timing) of an observation. Measured on-box: overall DOY std ~87 days but
a 0.5-deg spatial block-mean only reduces it to ~85 (a static coord map explains ~3% of timing); the signal is
carried by WHEN nearby species were recently observed -- exactly the past-state a propagator can carry and a
static coord map cannot. If propagation still does not help HERE, the standalone forecast signal is
climatological and the recurrence's value must be sought in the full fusion model. If it DOES help here, the
propagator (and Earth4D's live time axis) carries real forecasting value the family target simply could not
surface.

Target = day-of-year theta in [0,365). Predicted as a UNIT VECTOR (cos theta, sin theta) so the loss is
circular (Dec 31 near Jan 1); skill reported as circular MAE in days + within-+/-N-days accuracy.

Three heads on the IDENTICAL future+new-place query set, each over Earth4D / RFF / raw features:
  * StaticDOY   -- no propagation: predict DOY from the QUERY point's OWN positional feature only. Because a
                   static coord map explains ~3% of timing, this is a near-climatology FLOOR.
  * GNNDOY      -- GraphCast-style message passing: K strictly-earlier past neighbours carry their OBSERVED
                   (cos,sin) DOY as node state + learned space-time edges (dlat,dlon,dt); the query decodes its
                   DOY from the AGGREGATED past state. This is the propagator that CAN carry seasonal state.
  * LSTMDOY     -- 1-D causal rollout over the same K neighbours (recurrence.py-style) carrying neighbour DOY.

propagator_gain = GNN(or LSTM) circular skill MINUS the static floor, per feature type. POSITIVE here (beyond
the noise floor) is the decisive evidence that past-state propagation matters when the target is dynamic.

Additive + flag-gated: imported only when probe.py is called with --phenology; default path never touches it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.programs.spacetime.recurrence import build_causal_windows

_DOY = 365.25


def doy_of(days):
    """Day-of-year in [0, 365.25) from absolute event day."""
    return np.mod(np.asarray(days, dtype=np.float64), _DOY).astype(np.float32)


def doy_to_vec(doy):
    """DOY -> unit (cos, sin) so the target is circular (Dec 31 ~ Jan 1)."""
    ang = 2.0 * np.pi * (np.asarray(doy, np.float32) / _DOY)
    return np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32)


def vec_to_doy(v):
    """Unit-vector prediction [*,2] -> DOY in [0, 365.25)."""
    ang = torch.atan2(v[..., 1], v[..., 0])                       # [-pi, pi]
    ang = torch.remainder(ang, 2 * np.pi)
    return ang / (2 * np.pi) * _DOY


def circ_err_days(pred_doy, true_doy):
    """Absolute circular error in days (min of forward/backward around the 365-day ring)."""
    d = torch.abs(pred_doy - true_doy)
    return torch.minimum(d, _DOY - d)


class StaticDOY(nn.Module):
    """No-propagation floor: predict (cos,sin) DOY from the QUERY point's OWN positional feature only."""

    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, qfeat):
        return F.normalize(self.net(qfeat), dim=-1)


class GNNDOY(nn.Module):
    """Message-passing propagator predicting DOY. Neighbour node state carries the neighbour's OBSERVED
    (cos,sin) DOY (the PAST seasonal state to propagate); the query has NO DOY of its own (it is the future we
    predict). Learned space-time edges (dlat,dlon,dt), T hops, attention-gated aggregation, decode -> (cos,sin)."""

    def __init__(self, feat_dim, hidden=256, hops=2):
        super().__init__()
        self.hops = hops
        # neighbour node = [ pos-feat || observed (cos,sin) DOY ]; query node = [ pos-feat || zeros (no DOY) ]
        self.node_enc = nn.Sequential(nn.Linear(feat_dim + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_enc = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))   # (dlat,dlon) spatial-only (no dt leak)
        self.msg = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.upd = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, nfeat, ndoy, qfeat, edge, mask):
        # nfeat[B,K,F]; ndoy[B,K,2] observed neighbour DOY vec; qfeat[B,F]; edge[B,K,3]; mask[B,K]
        B, K, _ = nfeat.shape
        nstate = self.node_enc(torch.cat([nfeat, ndoy], -1))      # [B,K,H] neighbour carries its past DOY
        qzero = torch.zeros(B, 2, device=nfeat.device, dtype=nfeat.dtype)
        qstate = self.node_enc(torch.cat([qfeat, qzero], -1))     # [B,H] query has NO DOY
        e = self.edge_enc(edge)
        m = mask.unsqueeze(-1).float()
        neg = torch.finfo(nstate.dtype).min
        for _ in range(self.hops):
            qb = qstate.unsqueeze(1).expand(-1, K, -1)
            cat = torch.cat([nstate, e, qb], -1)
            msg = self.msg(cat)
            g = self.gate(cat).squeeze(-1).masked_fill(~mask, neg)
            w = torch.softmax(g, dim=1).unsqueeze(-1) * m
            agg = (w * msg).sum(1)
            qstate = self.upd(agg, qstate)
        return F.normalize(self.head(qstate), dim=-1)


class LSTMDOY(nn.Module):
    """1-D causal rollout: LSTM over K past neighbours (past->present), each step = [pos-feat || neighbour DOY
    vec || dt || dlat || dlon]; final hidden -> query (cos,sin) DOY."""

    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim + 2 + 2, hidden, batch_first=True)   # +neighbour DOY vec +spatial offset (no dt)
        self.head = nn.Linear(hidden, 2)

    def forward(self, nfeat, ndoy, edge, lengths):
        x = torch.cat([nfeat, ndoy, edge], -1)                    # [B,K, F+2+2]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return F.normalize(self.head(h[-1]), dim=-1)


class AttnDOY(nn.Module):
    """ROUND-1 STRUCTURAL PROPAGATOR: temporal self-attention over the K-neighbour causal past-context window.

    Where LSTMDOY compresses the window sequentially into one recurrent state, this propagator lets every past
    observation attend to every other (multi-head self-attention) so long-range seasonal structure in the
    window is mixed directly, then a learned QUERY token attention-pools the refined past state -> (cos,sin) DOY.
    A learned RECENCY (position-in-window) embedding gives the attention temporal order without leaking dt to the
    query (the window is ordered past->present; index = ordinal recency, NOT elapsed days). Leak-guards intact:
    per-step token = [pos-feat || neighbour DOY vec || SPATIAL offset (dlat,dlon)]; NO dt-to-query anywhere; the
    query token carries pos-feat + zero DOY only. Padded steps are masked out of attention."""

    def __init__(self, feat_dim, hidden=256, heads=4, layers=2):
        super().__init__()
        self.tok = nn.Linear(feat_dim + 2 + 2, hidden)            # neighbour step -> token
        self.qtok = nn.Linear(feat_dim, hidden)                   # query token from its own pos-feat (no DOY)
        self.pos = nn.Parameter(torch.zeros(1, 4096, hidden))     # learned recency (ordinal) embedding
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.pool = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

    def forward(self, nfeat, ndoy, qfeat, edge, mask):
        # nfeat[B,K,F]; ndoy[B,K,2]; qfeat[B,F]; edge[B,K,2]; mask[B,K] bool (True=valid past step)
        B, K, _ = nfeat.shape
        x = self.tok(torch.cat([nfeat, ndoy, edge], -1)) + self.pos[:, :K]
        pad = ~mask                                               # True where padded -> excluded from attention
        x = self.enc(x, src_key_padding_mask=pad)                # self-attention over the past window
        q = self.qtok(qfeat).unsqueeze(1)                        # [B,1,H] query token (no DOY of its own)
        pooled, _ = self.pool(q, x, x, key_padding_mask=pad)     # query attention-pools refined past state
        return F.normalize(self.head(pooled.squeeze(1)), dim=-1)


def _windows(lat, lon, days, q_idx, pool_idx, K):
    qi, vi = build_causal_windows(lat[q_idx], lon[q_idx], days[q_idx],
                                  lat[pool_idx], lon[pool_idx], days[pool_idx], K)
    gi = np.where(qi >= 0, pool_idx[np.clip(qi, 0, None)], -1)
    return gi, vi


def _tensors(qfeat_all, doyvec, days, lat, lon, q_idx, gidx, valid, K):
    """Assemble neighbour (feat, DOY vec), query feat, SPATIAL edge (dlat,dlon), mask, and target DOY vec + true DOY.

    LEAK-GUARD: the edge deliberately EXCLUDES dt (elapsed days to the query). The query's day-of-year IS the
    target, so handing the model the exact (query_day - neighbour_day) lets it reconstruct query_DOY =
    neighbour_DOY + dt arithmetically (smoke test: raw-GNN MAE 1.4d, a pure time-arithmetic identity, not
    seasonal-state propagation). With only the SPATIAL offset, the propagator must infer the query's timing
    from the PATTERN of nearby past DOYs -- the genuine phenology signal a static coord map cannot carry."""
    B = len(q_idx)
    N = qfeat_all.shape[0]
    F_ = qfeat_all.shape[1]
    gsafe = np.clip(gidx, 0, N - 1)
    vmask = torch.tensor(valid)
    nfeat = qfeat_all[torch.tensor(gsafe.reshape(-1))].reshape(B, K, F_) * vmask.unsqueeze(-1)
    ndoy = torch.tensor(doyvec[gsafe.reshape(-1)]).reshape(B, K, 2) * vmask.unsqueeze(-1)   # 0 where padded
    dlat = np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)
    dlon = np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)
    edge = torch.tensor(np.stack([dlat / 90.0, dlon / 180.0], -1)).float()   # SPATIAL only; no dt (see leak-guard)
    qfeat = qfeat_all[torch.tensor(q_idx)]
    ytrue = torch.tensor(doy_of(days[q_idx]))
    yvec = torch.tensor(doyvec[q_idx])
    ok = vmask.any(1)
    lengths = vmask.sum(1)
    return (nfeat[ok], ndoy[ok], qfeat[ok], edge[ok], vmask[ok], lengths[ok], yvec[ok], ytrue[ok])


def _skill(pred_vec, ytrue, tol_days=15.0):
    pdoy = vec_to_doy(pred_vec)
    err = circ_err_days(pdoy, ytrue)
    return err.mean().item(), (err <= tol_days).float().mean().item()


def run_phenology(qfeat_all, feat_dim, days, coords_ll, test, dev, K=16, steps=4000, lr=3e-3,
                  hidden=256, hops=2, tol_days=15.0, attn=False, attn_heads=4, attn_layers=2, warmup=200):
    """Train static / GNN / LSTM DOY heads on PAST queries; evaluate circular DOY skill on future+new-place.

    Returns dict of {static_mae, static_acc, gnn_mae, gnn_acc, lstm_mae, lstm_acc, n_te} in DAYS / within-tol.
    attn=True additionally trains AttnDOY (temporal self-attention propagator) -> {attn_mae, attn_acc}."""
    N = qfeat_all.shape[0]
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    doyvec = doy_to_vec(doy_of(days))
    tr_idx = np.where(~test)[0]
    te_idx = np.where(test)[0]
    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat, lon, days, q_train, tr_idx, K)
    g_te, v_te = _windows(lat, lon, days, te_idx, tr_idx, K)

    tr = _tensors(qfeat_all, doyvec, days, lat, lon, q_train, g_tr, v_tr, K)
    te = _tensors(qfeat_all, doyvec, days, lat, lon, te_idx, g_te, v_te, K)
    to = lambda ts: [t.to(dev) for t in ts]
    nftr, ndtr, qftr, etr, mtr, ltr, yvtr, yttr = to(tr)
    nfte, ndte, qfte, ete, mte, lte, yvte, ytte = to(te)

    n_te = int(nfte.shape[0])
    out = {"n_te": n_te}
    if nftr.shape[0] == 0 or n_te == 0:
        return {k: float("nan") for k in ("static_mae", "static_acc", "gnn_mae", "gnn_acc",
                                          "lstm_mae", "lstm_acc")} | {"n_te": n_te}
    Btr = nftr.shape[0]
    bs = min(2048, Btr)

    def vec_loss(pred, tgt):
        # 1 - cos(pred,tgt) circular regression loss on the unit circle
        return (1.0 - (pred * tgt).sum(-1)).mean()

    # ---- No-propagation static DOY floor ----
    sh = StaticDOY(feat_dim, hidden).to(dev)
    opt = torch.optim.Adam(sh.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        loss = vec_loss(sh(qftr[s]), yvtr[s])
        opt.zero_grad(); loss.backward(); opt.step()
    sh.eval()
    with torch.no_grad():
        out["static_mae"], out["static_acc"] = _skill(sh(qfte), ytte, tol_days)

    # ---- GNN propagator carrying past DOY ----
    gnn = GNNDOY(feat_dim, hidden, hops).to(dev)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        loss = vec_loss(gnn(nftr[s], ndtr[s], qftr[s], etr[s], mtr[s]), yvtr[s])
        opt.zero_grad(); loss.backward(); opt.step()
    gnn.eval()
    with torch.no_grad():
        out["gnn_mae"], out["gnn_acc"] = _skill(gnn(nfte, ndte, qfte, ete, mte), ytte, tol_days)

    # ---- LSTM rollout carrying past DOY ----
    lstm = LSTMDOY(feat_dim, hidden).to(dev)
    opt = torch.optim.Adam(lstm.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        loss = vec_loss(lstm(nftr[s], ndtr[s], etr[s], ltr[s]), yvtr[s])
        opt.zero_grad(); loss.backward(); opt.step()
    lstm.eval()
    with torch.no_grad():
        out["lstm_mae"], out["lstm_acc"] = _skill(lstm(nfte, ndte, ete, lte), ytte, tol_days)

    # ---- ROUND-1: temporal self-attention propagator carrying past DOY ----
    if attn:
        att = AttnDOY(feat_dim, hidden, heads=attn_heads, layers=attn_layers).to(dev)
        opt = torch.optim.Adam(att.parameters(), lr=lr)
        # LR warmup then cosine decay: self-attention is init-fragile (prior GNN one-seed collapse), warmup
        # stabilizes it -- a flat-LR Adam on a transformer over a short noisy window can diverge early.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: min(1.0, (t + 1) / max(1, warmup)) *
            (0.5 * (1 + np.cos(np.pi * max(0, t - warmup) / max(1, steps - warmup)))))
        for _ in range(steps):
            s = torch.randint(0, Btr, (bs,), device=dev)
            loss = vec_loss(att(nftr[s], ndtr[s], qftr[s], etr[s], mtr[s]), yvtr[s])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(att.parameters(), 1.0)
            opt.step(); sched.step()
        att.eval()
        with torch.no_grad():
            out["attn_mae"], out["attn_acc"] = _skill(att(nfte, ndte, qfte, ete, mte), ytte, tol_days)

    return out


def run_phenology_all(e4d, rff, raw, feat_dims, days, coords_ll, test, dev, K=16, steps=4000, lr=3e-3,
                      hidden=256, hops=2, tol_days=15.0, attn=False, attn_heads=4, attn_layers=2):
    """Run the three heads over Earth4D / RFF / raw features; return per-feature dicts."""
    kw = dict(attn=attn, attn_heads=attn_heads, attn_layers=attn_layers)
    r = {}
    r["e4d"] = run_phenology(e4d, feat_dims["e4d"], days, coords_ll, test, dev, K, steps, lr, hidden, hops, tol_days, **kw)
    r["rff"] = run_phenology(rff, feat_dims["rff"], days, coords_ll, test, dev, K, steps, lr, hidden, hops, tol_days, **kw)
    r["raw"] = run_phenology(raw, feat_dims["raw"], days, coords_ll, test, dev, K, steps, lr, hidden, hops, tol_days, **kw)
    return r
