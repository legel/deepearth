"""ADDITIVE probe (H1 crux): train the Earth4D hash END-TO-END through the LSTM propagator on a CAUSAL
forecast objective, and compare to the fair coordinate baselines. NOTHING here edits core/fusion.py,
evaluate.py, encoders/*, earth4d.py, or existing probe default paths. Scratch-only, flag-gated.

THE GAP THIS FILLS
------------------
Every existing propagator path (recurrence.run_recurrence, dyntargets.run_pheno_*, run_abundance) takes a
PRECOMPUTED positional featurization `qfeat_all` -- Earth4D is only ever a FROZEN lookup. The one end-to-end
trainer (recurrence.run_field_decode) is a STATIC per-point decode with NO propagator. So the science.md
rule-1 object -- "train the 4D field JOINTLY with a causal auto-regressive forecaster" -- has never been run.
This probe runs it.

TASK (real non-stationary signal, +52d propagator gain known): phenology mean-DOY forecast.
  * split: temporal (train = past, test = latest `holdout` frac by event day) -- causal, rule 1.
    optional --forecast_spatial: test must ALSO be a held-out 0.5deg block (new place AND future time).
  * per query q (a held-out future obs): gather its K spatial-nearest CAUSAL (strictly-earlier-day) train
    neighbours (recurrence.build_causal_windows, leak-safe: p_day<q_day only), order past->present, run an
    LSTM that propagates their observed-DOY node-state forward; head regresses the query's mean-DOY vector
    (sin,cos). MAE in circular days + within-tol accuracy vs a STATIC no-propagation floor.
  * LEAK GUARD (identical to probe.py --phenology): the query-POINT positional feature is SPACE-ONLY
    (encoder at t=0 / raw lat,lon). The query's own timestamp never enters any feature. Neighbours carry
    their OBSERVED past DOY as explicit node state (that IS the propagation) + spatial offset (dlat,dlon).
    dt is NOT emitted (matches build_causal_windows contract: spatial offset only).

FOUR ARMS -- identical LSTM propagator, identical windows, identical head width/steps/lr; ONLY the query-point
positional featurizer and whether its encoder trains differ:
  (a) raw          query feat = (lat/90, lon/180)                       fixed          [fair coord baseline]
  (b) e4d_frozen   query feat = Earth4D(lat,lon,elev0,t0)              encoder FROZEN  [static-hash control]
  (c) e4d_e2e      query feat = Earth4D(lat,lon,elev0,t0)              encoder TRAINED [THE ENGINE]
  (d) mlp_e2e      query feat = coordMLP(lat/90,lon/180)               MLP TRAINED     [matched-capacity fair ctrl:
                                                                        a generic learned PE that ALSO trains, so
                                                                        any (c) win is the HASH FIELD, not merely
                                                                        "the query encoder got to learn"]

st_gain      = acc(c) - acc(a)      MAE_gain      = MAE(a) - MAE(c)     (engine vs fair coordinate baseline)
st_gain_hash = acc(c) - acc(d)      MAE_gain_hash = MAE(d) - MAE(c)     (isolates the hash vs a trained PE)
Also reports absolute skill (MAE days, within-tol acc) and the static floor per arm.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace python3.12 \
    deepearth/autoresearch/programs/spacetime/earth4d_engine.py \
    --cache_dir data/deepcal --n_shards 8 --steps 1500 --rec_k 24 --seed 0 [--forecast_spatial]
"""
import argparse, glob, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/workspace")
from deepearth.encoders.spacetime.earth4d import Earth4D
from deepearth.autoresearch.programs.spacetime.recurrence import build_causal_windows


# ----- data (mirror probe.load_obs with_time; no import to avoid pulling probe's heavy argparse) -----
def load_obs_time(cache, n_shards):
    cachep = Path(cache)
    vocab = np.load(cachep / "gbif_vocab.npz", allow_pickle=True)
    gidx = vocab["global_idx"]
    rows = list(csv.DictReader(open(cachep / "derived/species_index.csv")))
    family = np.array([rows[i]["family"] for i in gidx])
    fam_id = np.unique(family, return_inverse=True)[1]
    et = np.load(cachep / "gbif_eventtime.npz")
    id2day = dict(zip(et["gbifID"].tolist(), et["days"].tolist()))
    lat, lon, sp, day = [], [], [], []
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f)
        lat.append(z["lat"]); lon.append(z["lon"]); sp.append(z["species_local"])
        day.append(np.array([id2day.get(int(i), np.nan) for i in z["gbifID"]], dtype=np.float32))
    lat = np.concatenate(lat).astype(np.float32)
    lon = np.concatenate(lon).astype(np.float32)
    days = np.concatenate(day).astype(np.float32)
    # drop obs with no event time (cannot forecast without it)
    ok = ~np.isnan(days)
    return lat[ok], lon[ok], days[ok]


def load_elev(cache, n_shards):
    """Per-obs elevation (m) aligned EXACTLY to load_obs_time order/mask. Joins gbif_elev.npz by gbifID over the
    same shard glob + same NaN-day drop. Returns float32 elev with np.nan where unknown. Leak-free: elevation is a
    STATIC spatial attribute of the query location (like lat/lon), never derived from the target DOY."""
    from pathlib import Path as _P
    cachep = _P(cache)
    ez = np.load(cachep / "gbif_elev.npz", allow_pickle=True)
    id2elev = dict(zip(ez["gbifID"].tolist(), ez["elev"].tolist()))
    et = np.load(cachep / "gbif_eventtime.npz")
    id2day = dict(zip(et["gbifID"].tolist(), et["days"].tolist()))
    elev, day = [], []
    for f in sorted(glob.glob(str(cachep / "gbif_tokens/*.npz")))[:n_shards]:
        z = np.load(f)
        ids = z["gbifID"]
        elev.append(np.array([id2elev.get(int(i), np.nan) for i in ids], dtype=np.float32))
        day.append(np.array([id2day.get(int(i), np.nan) for i in ids], dtype=np.float32))
    elev = np.concatenate(elev).astype(np.float32)
    day = np.concatenate(day).astype(np.float32)
    ok = ~np.isnan(day)                                        # identical mask to load_obs_time
    return elev[ok]


def temporal_holdout(days, frac):
    thr = np.nanquantile(days, 1.0 - frac)
    return days >= thr


def spatial_holdout(lat, lon, frac, seed=0, block=0.5):
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    cells = np.unique(np.stack([ci, cj], 1), axis=0)
    rng = np.random.default_rng(seed)
    held = set(map(tuple, cells[rng.random(len(cells)) < frac].tolist()))
    return np.array([(int(a), int(b)) in held for a, b in zip(ci, cj)])


# ----- targets: query mean-DOY (circular) -----
_DOY = 365.25

def doy_of(days):
    return np.mod(days, 365.25).astype(np.float32)

def phase_centroid_doy(lat, lon, days, block=0.5):
    """Per-obs target = circular-mean DOY of ALL obs in the query's 0.5deg cell (community seasonal phase).
    Strong spatially-structured target (each cell has a characteristic phase) -- the honest test for whether a
    positional field carries seasonal spatial structure raw coords lack. Mirrors dyntargets._phase_centroid_doy."""
    doy = doy_of(days); ang = 2.0 * np.pi * doy / _DOY
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    acc = defaultdict(lambda: [0.0, 0.0]); keys = list(zip(ci.tolist(), cj.tolist()))
    for kk, a in zip(keys, ang):
        acc[kk][0] += np.cos(a); acc[kk][1] += np.sin(a)
    cen = {kk: (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * _DOY for kk, (c, s) in acc.items()}
    return np.array([cen[kk] for kk in keys], dtype=np.float32)

def doy_to_vec(doy):
    a = doy / 365.25 * 2 * np.pi
    return np.stack([np.sin(a), np.cos(a)], -1).astype(np.float32)

def vec_to_doy(v):
    a = np.arctan2(v[..., 0], v[..., 1]) % (2 * np.pi)
    return a / (2 * np.pi) * 365.25

def circ_err_days(pred_doy, true_doy):
    d = np.abs(pred_doy - true_doy) % 365.25
    return np.minimum(d, 365.25 - d)


def abundance_target(lat, lon, days, block=0.5, win=90.0, lead=180.0, delta=True):
    """Per-obs FUTURE log-activity (delta): log1p(#obs in query's cell in [d+lead-win, d+lead]) minus trailing
    past log1p(#obs in [d-win, d]). A genuine forward FORECAST (lead>0) of a NON-stationary activity change --
    a static climatology cannot represent it. Neighbours (causal, day<d) carry only their OWN trailing-past
    log-activity as node state (never anything about the future). Mirrors dyntargets._abundance_target."""
    ci = np.floor(lat / block).astype(np.int64); cj = np.floor(lon / block).astype(np.int64)
    from collections import defaultdict
    cell_days = defaultdict(list); keys = list(zip(ci.tolist(), cj.tolist()))
    for kk, d in zip(keys, days):
        cell_days[kk].append(float(d))
    for kk in cell_days:
        cell_days[kk] = np.sort(np.array(cell_days[kk]))
    tgt = np.empty(len(days), np.float32); past = np.empty(len(days), np.float32)
    for i, (kk, d) in enumerate(zip(keys, days)):
        arr = cell_days[kk]
        lo = np.searchsorted(arr, d + lead - win, "left"); hi = np.searchsorted(arr, d + lead, "right")
        fut = np.log1p(hi - lo)
        plo = np.searchsorted(arr, d - win, "left"); phi = np.searchsorted(arr, d, "right")
        p = np.log1p(phi - plo)
        past[i] = p
        tgt[i] = (fut - p) if delta else fut
    return tgt, past


# ----- coord MLP fair control (matched-capacity trainable PE) -----
class CoordMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ----- the joint propagator: LSTM over causal neighbour DOY-state; query positional feat from `qenc` -----
class JointForecaster(nn.Module):
    """LSTM propagator whose per-step token carries the neighbour's observed-DOY node state + spatial offset,
    fused with the QUERY-point positional feature (space-only). The query encoder (`qenc`) is trained jointly
    when it has parameters (e4d_e2e / mlp_e2e) and frozen otherwise (raw / e4d_frozen)."""
    def __init__(self, qfeat_dim, hidden=256):
        super().__init__()
        step_in = 2 + 2          # neighbour DOY (sin,cos) + spatial offset (dlat,dlon)
        self.lstm = nn.LSTM(step_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + qfeat_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, qfeat, ndoy_vec, doff, lengths):
        # ndoy_vec[B,K,2]  doff[B,K,2]  qfeat[B,qfeat_dim]  lengths[B]
        x = torch.cat([ndoy_vec, doff], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-1], qfeat], -1))          # [B,2] predicted mean-DOY vector


def _assemble(tgt_doy, lat, lon, q_idx, gidx, valid, K):
    """Build neighbour target-DOY-state + spatial offset tensors for a query set. Leak-safe: uses only past
    neighbours' own target-DOY and their spatial offset to the query; never the query's own time."""
    B = gidx.shape[0]
    gsafe = np.clip(gidx, 0, len(tgt_doy) - 1)
    vmask = torch.tensor(valid)
    ndoy = tgt_doy[gsafe]                                        # neighbour target DOY (per-obs or cell-phase)
    ndoy_vec = torch.tensor(np.where(valid[..., None], doy_to_vec(ndoy), 0.0)).float()
    dlat = torch.tensor(np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)).float()
    dlon = torch.tensor(np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)).float()
    doff = torch.stack([dlat / 90.0, dlon / 180.0], -1)
    lengths = vmask.sum(1).long()
    ok = vmask.any(1)
    return ndoy_vec, doff, lengths, ok


def _static_floor(tgt_doy, days, lat, lon, q_te, g_te, v_te, tol_days):
    """No-propagation baseline: predict each query's target-DOY as the circular mean of its causal neighbours'
    OWN target-DOY (pure spatial nowcast, no learned forward propagation). Reports the floor MAE + within-tol acc."""
    gsafe = np.clip(g_te, 0, len(days) - 1)
    v = doy_to_vec(tgt_doy[gsafe])                              # [B,K,2] neighbour target-DOY vectors
    w = v_te[..., None].astype(np.float32)
    s = (v * w).sum(1); n = w.sum(1) + 1e-6
    pred = vec_to_doy(s / n)
    err = circ_err_days(pred, tgt_doy[q_te])
    return float(np.mean(err)), float(np.mean(err <= tol_days))


def run_arm(arm, tgt_doy, lat, lon, q_tr, g_tr, v_tr, q_te, g_te, v_te, dev, enc=None,
            steps=1500, lr=3e-3, hidden=256, tol_days=15.0, K=24, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    # neighbour tensors
    nd_tr, off_tr, len_tr, ok_tr = _assemble(tgt_doy, lat, lon, q_tr, g_tr, v_tr, K)
    nd_te, off_te, len_te, ok_te = _assemble(tgt_doy, lat, lon, q_te, g_te, v_te, K)
    # query targets
    ytr = torch.tensor(doy_to_vec(tgt_doy[q_tr]))
    yte_doy = tgt_doy[q_te]

    # query-point positional feature builder (SPACE-ONLY -- leak guard)
    lat_t = torch.tensor(lat); lon_t = torch.tensor(lon)
    def qfeat_of(q_idx, grad):
        if arm == "raw":
            f = torch.stack([lat_t[q_idx] / 90.0, lon_t[q_idx] / 180.0], -1).to(dev)
            return f.detach()
        if arm in ("e4d_frozen", "e4d_e2e"):
            coords = torch.stack([lat_t[q_idx], lon_t[q_idx], torch.zeros(len(q_idx)),
                                  torch.zeros(len(q_idx))], -1).to(dev)             # t=0 -> no time leak
            if grad and arm == "e4d_e2e":
                return enc(coords)
            with torch.no_grad():
                return enc(coords)
        if arm == "mlp_e2e":
            f = torch.stack([lat_t[q_idx] / 90.0, lon_t[q_idx] / 180.0], -1).to(dev)
            return enc(f)                                                            # coordMLP, trainable
        raise ValueError(arm)

    with torch.no_grad():
        qfeat_dim = qfeat_of(q_tr[:4], grad=False).shape[1]
    model = JointForecaster(qfeat_dim, hidden=hidden).to(dev)

    params = list(model.parameters())
    if arm in ("e4d_e2e", "mlp_e2e"):
        params += list(enc.parameters())                        # END-TO-END: encoder trains through the loss
    opt = torch.optim.Adam(params, lr=lr)

    nd_tr, off_tr, ytr = nd_tr[ok_tr].to(dev), off_tr[ok_tr].to(dev), ytr[ok_tr].to(dev)
    len_tr = len_tr[ok_tr]; q_tr_ok = q_tr[ok_tr.numpy()]
    nd_te, off_te = nd_te[ok_te].to(dev), off_te[ok_te].to(dev)
    len_te = len_te[ok_te]; q_te_ok = q_te[ok_te.numpy()]; yte_doy_ok = yte_doy[ok_te.numpy()]
    Btr = nd_tr.shape[0]
    if Btr == 0 or nd_te.shape[0] == 0:
        return dict(mae=float("nan"), acc=float("nan"), n_te=0)
    if arm in ("e4d_e2e", "mlp_e2e"): enc.train()
    bs = min(2048, Btr)
    for _ in range(steps):
        si = torch.randint(0, Btr, (bs,))
        qf = qfeat_of(q_tr_ok[si.numpy()], grad=True)
        pred = model(qf, nd_tr[si], off_tr[si], len_tr[si])
        loss = F.mse_loss(pred, ytr[si])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    if enc is not None: enc.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, nd_te.shape[0], 8192):
            qf = qfeat_of(q_te_ok[s:s + 8192], grad=False)
            preds.append(model(qf, nd_te[s:s + 8192], off_te[s:s + 8192], len_te[s:s + 8192]).cpu().numpy())
        pv = np.concatenate(preds)
        pdoy = vec_to_doy(pv)
        err = circ_err_days(pdoy, yte_doy_ok)
    return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)))


# ================= ABUNDANCE FORECAST MODE (regression, R2) =================
class AbundForecaster(nn.Module):
    """LSTM over causal neighbours' past-activity state + spatial offset, fused with the query positional feat;
    regresses the query cell's FUTURE log-activity delta. Same 4-arm design as the DOY forecaster."""
    def __init__(self, qfeat_dim, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(1 + 2, hidden, batch_first=True)     # neighbour past-activity (1) + offset (2)
        self.head = nn.Sequential(nn.Linear(hidden + qfeat_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, qfeat, nact, doff, lengths):
        x = torch.cat([nact.unsqueeze(-1), doff], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-1], qfeat], -1)).squeeze(-1)


def _assemble_abund(past_act, lat, lon, q_idx, gidx, valid, K):
    gsafe = np.clip(gidx, 0, len(past_act) - 1)
    vmask = torch.tensor(valid)
    nact = torch.tensor(np.where(valid, past_act[gsafe], 0.0)).float()
    dlat = torch.tensor(np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)).float()
    dlon = torch.tensor(np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)).float()
    doff = torch.stack([dlat / 90.0, dlon / 180.0], -1)
    return nact, doff, vmask.sum(1).long(), vmask.any(1)


def _r2(pred, y):
    ssr = float(((pred - y) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum()) + 1e-9
    return 1.0 - ssr / sst, float(np.abs(pred - y).mean())


def run_arm_abund(arm, tgt, past_act, lat, lon, q_tr, g_tr, v_tr, q_te, g_te, v_te, dev, enc=None,
                  steps=1500, lr=3e-3, hidden=256, K=24, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    nd_tr, off_tr, len_tr, ok_tr = _assemble_abund(past_act, lat, lon, q_tr, g_tr, v_tr, K)
    nd_te, off_te, len_te, ok_te = _assemble_abund(past_act, lat, lon, q_te, g_te, v_te, K)
    ytr = torch.tensor(tgt[q_tr]).float(); yte = tgt[q_te]
    lat_t = torch.tensor(lat); lon_t = torch.tensor(lon)
    def qfeat_of(q_idx, grad):
        if arm == "raw":
            return torch.stack([lat_t[q_idx] / 90.0, lon_t[q_idx] / 180.0], -1).to(dev).detach()
        if arm in ("e4d_frozen", "e4d_e2e"):
            coords = torch.stack([lat_t[q_idx], lon_t[q_idx], torch.zeros(len(q_idx)), torch.zeros(len(q_idx))], -1).to(dev)
            if grad and arm == "e4d_e2e":
                return enc(coords)
            with torch.no_grad():
                return enc(coords)
        if arm == "mlp_e2e":
            return enc(torch.stack([lat_t[q_idx] / 90.0, lon_t[q_idx] / 180.0], -1).to(dev))
        raise ValueError(arm)
    with torch.no_grad():
        qfeat_dim = qfeat_of(q_tr[:4], grad=False).shape[1]
    model = AbundForecaster(qfeat_dim, hidden=hidden).to(dev)
    params = list(model.parameters())
    if arm in ("e4d_e2e", "mlp_e2e"): params += list(enc.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    nd_tr, off_tr, ytr = nd_tr[ok_tr].to(dev), off_tr[ok_tr].to(dev), ytr[ok_tr].to(dev)
    len_tr = len_tr[ok_tr]; q_tr_ok = q_tr[ok_tr.numpy()]
    nd_te, off_te = nd_te[ok_te].to(dev), off_te[ok_te].to(dev)
    len_te = len_te[ok_te]; q_te_ok = q_te[ok_te.numpy()]; yte_ok = yte[ok_te.numpy()]
    Btr = nd_tr.shape[0]
    if Btr == 0 or nd_te.shape[0] == 0:
        return dict(r2=float("nan"), mae=float("nan"), n_te=0)
    if arm in ("e4d_e2e", "mlp_e2e"): enc.train()
    bs = min(2048, Btr)
    for _ in range(steps):
        si = torch.randint(0, Btr, (bs,))
        qf = qfeat_of(q_tr_ok[si.numpy()], grad=True)
        pred = model(qf, nd_tr[si], off_tr[si], len_tr[si])
        loss = F.smooth_l1_loss(pred, ytr[si])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    if enc is not None: enc.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, nd_te.shape[0], 8192):
            qf = qfeat_of(q_te_ok[s:s + 8192], grad=False)
            preds.append(model(qf, nd_te[s:s + 8192], off_te[s:s + 8192], len_te[s:s + 8192]).cpu().numpy())
        pv = np.concatenate(preds)
    r2, mae = _r2(pv, yte_ok)
    return dict(r2=r2, mae=mae, n_te=int(len(yte_ok)))


# ============ H2: TIME-CONDITIONED end-to-end arm (rule 2b + rule 1 combined) ============
class TimeCondForecaster(nn.Module):
    """Per rollout step the token = [neighbour-DOY(2) || spatial-offset(2) || query-cell Earth4D re-encoded AT
    the neighbour's day (qfeat_dim)]. The encoder's OWN time axis is thus marched forward through the window and
    trained end-to-end -- the only design that jointly exercises rule-2b recurrence-through-the-encoder AND
    rule-1 causal training. If Earth4D carries propagatable dynamics, THIS is where they surface."""
    def __init__(self, qfeat_dim, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(2 + 2 + qfeat_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
    def forward(self, ndoy_vec, doff, qfeat_seq, lengths):
        x = torch.cat([ndoy_vec, doff, qfeat_seq], -1)          # qfeat_seq[B,K,qfeat_dim]
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])


def run_arm_timecond(tgt_doy, days, lat, lon, q_tr, g_tr, v_tr, q_te, g_te, v_te, dev, enc,
                     steps=1500, lr=3e-3, hidden=256, tol_days=15.0, K=24, seed=0, tmin=0.0, tspan=1.0):
    """Earth4D end-to-end, TIME-CONDITIONED: query cell re-encoded at each neighbour's day, marched through the
    LSTM. Leak-safe: uses the NEIGHBOUR's past day (< query day) as the encoder time slot, never the query's day."""
    torch.manual_seed(seed); np.random.seed(seed)
    lat_t = torch.tensor(lat); lon_t = torch.tensor(lon)
    def build(q_idx, gidx, valid):
        B, Kk = gidx.shape
        gsafe = np.clip(gidx, 0, len(days) - 1)
        vmask = torch.tensor(valid)
        ndoy = torch.tensor(np.where(valid[..., None], doy_to_vec(tgt_doy[gsafe]), 0.0)).float()
        dlat = torch.tensor(np.where(valid, lat[gsafe] - lat[q_idx][:, None], 0.0)).float()
        dlon = torch.tensor(np.where(valid, lon[gsafe] - lon[q_idx][:, None], 0.0)).float()
        doff = torch.stack([dlat / 90.0, dlon / 180.0], -1)
        # per-step encoder time slot = neighbour's day (normalized), broadcast query lat/lon (leak-safe: past day)
        step_day = np.where(valid, days[gsafe], 0.0)
        tnorm = ((step_day - tmin) / tspan).astype(np.float32)
        y = torch.tensor(doy_to_vec(tgt_doy[q_idx]))
        return ndoy, doff, vmask.sum(1).long(), vmask.any(1), q_idx, tnorm, y
    nd_tr, off_tr, len_tr, ok_tr, qi_tr, tn_tr, ytr = build(q_tr, g_tr, v_tr)
    nd_te, off_te, len_te, ok_te, qi_te, tn_te, yte = build(q_te, g_te, v_te)
    with torch.no_grad():
        qdim = enc(torch.zeros(1, 4, device=dev)).shape[1]
    model = TimeCondForecaster(qdim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(enc.parameters()), lr=lr)

    def qseq(q_idx, tnorm, valid_shape):
        B, Kk = valid_shape
        qlat = np.broadcast_to(lat_t[q_idx].numpy()[:, None], (B, Kk)).reshape(-1)
        qlon = np.broadcast_to(lon_t[q_idx].numpy()[:, None], (B, Kk)).reshape(-1)
        coords = torch.tensor(np.stack([qlat, qlon, np.zeros_like(qlat), tnorm.reshape(-1)], 1)).float().to(dev)
        return enc(coords).reshape(B, Kk, -1)

    def to(*t): return [x.to(dev) for x in t]
    nd_tr, off_tr, ytr = to(nd_tr[ok_tr], off_tr[ok_tr], ytr[ok_tr])
    len_tr = len_tr[ok_tr]; qi_tr_ok = qi_tr[ok_tr.numpy()]; tn_tr = tn_tr[ok_tr.numpy()]
    nd_te, off_te = to(nd_te[ok_te], off_te[ok_te]); len_te = len_te[ok_te]
    qi_te_ok = qi_te[ok_te.numpy()]; tn_te = tn_te[ok_te.numpy()]; yte_doy = tgt_doy[q_te][ok_te.numpy()]
    Btr = nd_tr.shape[0]
    if Btr == 0 or nd_te.shape[0] == 0:
        return dict(mae=float("nan"), acc=float("nan"), n_te=0)
    enc.train(); bs = min(1024, Btr)
    for _ in range(steps):
        si = torch.randint(0, Btr, (bs,)); sn = si.numpy()
        qs = qseq(qi_tr_ok[sn], tn_tr[sn], (bs, nd_tr.shape[1]))
        pred = model(nd_tr[si], off_tr[si], qs, len_tr[si])
        loss = F.mse_loss(pred, ytr[si])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); enc.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, nd_te.shape[0], 4096):
            e = slice(s, s + 4096)
            qs = qseq(qi_te_ok[e], tn_te[e], (min(4096, nd_te.shape[0] - s), nd_te.shape[1]))
            preds.append(model(nd_te[e], off_te[e], qs, len_te[e]).cpu().numpy())
        pv = np.concatenate(preds); err = circ_err_days(vec_to_doy(pv), yte_doy)
    return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)))


# =========================================================================================
# PHYSICS-INFORMED PROPAGATOR (rule 2b). Baseline = vanilla LSTM propagator (arm "lstm", the
# +52d engine). Physics variants test whether PHYSICAL STRUCTURE (seasonal harmonic forcing,
# spatial-diffusion Green's-function kernel, Hopkins latitude/longitude advection cline,
# continuous-time ODE) beats or interpretably-matches the black-box LSTM on mean-DOY forecast,
# and yields recoverable physical parameters (length-scale in deg, cline in days/deg).
# ALL arms: same causal windows, same static floor, same leak guard (query feat space-only,
# neighbours carry only OWN past DOY + spatial offset), same eval (circular-MAE, within-tol acc).
# =========================================================================================

def _assemble_phys(tgt_doy, lat, lon, days, q_idx, gidx, valid, K, elev=None):
    """Rich neighbour tensors for physics arms. Returns, per query, its K causal neighbours':
      ndoy_vec [B,K,2]  observed-DOY (sin,cos) node-state (the propagated quantity)
      ndoy     [B,K]    observed-DOY (days, raw) -- for kernel/advection circular math
      doff     [B,K,2]  spatial offset (dlat/90, dlon/180) query->neighbour
      dlat_deg [B,K]    RAW dlat in degrees (q_lat - n_lat), signed  -- advection cline axis
      dlon_deg [B,K]    RAW dlon in degrees (q_lon - n_lon), signed
      ddist    [B,K]    great-circle-ish planar distance (deg) query<->neighbour  -- kernel radius
      ndt      [B,K]    day-gap = q_day - n_day (>0, strictly causal)  -- ODE integration step
      valid mask, lengths, ok. Leak-safe: query's OWN day used only to form ndt (a POSITIVE lag to
      each past neighbour); never the query DOY/target. Matches build_causal_windows contract."""
    gsafe = np.clip(gidx, 0, len(tgt_doy) - 1)
    vmask = torch.tensor(valid)
    ndoy = tgt_doy[gsafe]
    ndoy_vec = torch.tensor(np.where(valid[..., None], doy_to_vec(ndoy), 0.0)).float()
    ndoy_t = torch.tensor(np.where(valid, ndoy, 0.0)).float()
    dlat = np.where(valid, lat[q_idx][:, None] - lat[gsafe], 0.0).astype(np.float32)   # q - n (signed)
    dlon = np.where(valid, lon[q_idx][:, None] - lon[gsafe], 0.0).astype(np.float32)
    doff = torch.stack([torch.tensor(-dlat / 90.0), torch.tensor(-dlon / 180.0)], -1)  # keep n->q sign as before
    ddist = torch.tensor(np.sqrt(dlat ** 2 + (dlon * np.cos(np.deg2rad(lat[q_idx][:, None]))) ** 2)).float()
    ndt = torch.tensor(np.where(valid, np.maximum(days[q_idx][:, None] - days[gsafe], 0.0), 0.0)).float()
    if elev is not None:
        eq = elev[q_idx][:, None]; en = elev[gsafe]
        delev = (eq - en) / 100.0                              # q - n, in 100m units (Hopkins elevation axis)
        delev = np.where(valid & np.isfinite(delev), delev, 0.0).astype(np.float32)
    else:
        delev = np.zeros_like(dlat, dtype=np.float32)
    delev_t = torch.tensor(delev)
    return (ndoy_vec, ndoy_t, doff, torch.tensor(dlat), torch.tensor(dlon), ddist, ndt,
            delev_t, vmask.sum(1).long(), vmask.any(1))


def _circ_wmean_doy(ndoy, w, valid):
    """Weighted circular mean of neighbour DOYs. ndoy[B,K] days, w[B,K] weights, valid[B,K] mask -> pred DOY[B]."""
    a = ndoy / 365.25 * 2 * np.pi
    wv = w * valid
    s = (torch.sin(a) * wv).sum(1); c = (torch.cos(a) * wv).sum(1)
    return (torch.atan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 365.25


class HarmonicForecaster(nn.Module):
    """LSTM propagator with explicit SEASONAL HARMONIC forcing: each step token carries the neighbour's DOY at
    1st AND 2nd annual harmonic (sin/cos of 2*pi*doy and 4*pi*doy) + spatial offset. Head predicts (sin,cos) of
    the annual cycle -> a phase+amplitude of the fundamental. Physics: phenology is periodic; 2nd harmonic
    captures skew/bimodality a single sinusoid misses."""
    def __init__(self, qfeat_dim, hidden=256):
        super().__init__()
        self.lstm = nn.LSTM(4 + 2, hidden, batch_first=True)      # doy@h1(2)+doy@h2(2)+offset(2)
        self.head = nn.Sequential(nn.Linear(hidden + qfeat_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
    def forward(self, qfeat, ndoy_vec, ndoy2_vec, doff, lengths):
        x = torch.cat([ndoy_vec, ndoy2_vec, doff], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(torch.cat([h[-1], qfeat], -1))


class ODEForecaster(nn.Module):
    """Continuous-time propagator: a learned vector field f(h) integrated by explicit Euler over the ACTUAL
    day-gaps of the causal window (dh = f(h, token) * dt_norm). Contrasts the discrete LSTM: dynamics respect
    real elapsed time, not step index. dt_norm = day-gap / 365 (annual units)."""
    def __init__(self, qfeat_dim, hidden=128):
        super().__init__()
        self.enc = nn.Linear(2 + 2, hidden)
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden + qfeat_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.hidden = hidden
    def forward(self, qfeat, ndoy_vec, doff, ndt, lengths):
        B, K, _ = ndoy_vec.shape
        tok = self.enc(torch.cat([ndoy_vec, doff], -1))           # [B,K,H] per-neighbour drive
        dtn = (ndt / 365.25).clamp(0, 3).unsqueeze(-1)            # annual units, capped
        h = torch.zeros(B, self.hidden, device=ndoy_vec.device)
        # integrate present->... over ordered causal steps (windows are past->present already)
        for k in range(K):
            drive = tok[:, k, :]
            m = (k < lengths).float().unsqueeze(-1).to(h.device)
            h = h + m * (self.f(h + drive) * dtn[:, k, :])
        return self.head(torch.cat([h, qfeat], -1))


def run_arm_phys(arm, tgt_doy, lat, lon, days, q_tr, g_tr, v_tr, q_te, g_te, v_te, dev,
                 steps=1500, lr=3e-3, hidden=256, tol_days=15.0, K=24, seed=0, elev=None):
    """One physics arm. Returns MAE, acc, and (for kernel/advection) recovered physical params."""
    torch.manual_seed(seed); np.random.seed(seed)
    P_tr = _assemble_phys(tgt_doy, lat, lon, days, q_tr, g_tr, v_tr, K, elev=elev)
    P_te = _assemble_phys(tgt_doy, lat, lon, days, q_te, g_te, v_te, K, elev=elev)
    (ndv_tr, ndoy_tr, off_tr, dla_tr, dlo_tr, dd_tr, ndt_tr, dev_tr, len_tr, ok_tr) = P_tr
    (ndv_te, ndoy_te, off_te, dla_te, dlo_te, dd_te, ndt_te, dev_te, len_te, ok_te) = P_te
    ytr = torch.tensor(doy_to_vec(tgt_doy[q_tr])); ytr_doy = torch.tensor(tgt_doy[q_tr]).float()
    yte_doy = tgt_doy[q_te]
    lat_t = torch.tensor(lat); lon_t = torch.tensor(lon)

    def qfeat(q_idx):
        return torch.stack([lat_t[q_idx] / 90.0, lon_t[q_idx] / 180.0], -1).to(dev).detach()

    # subset to valid queries
    def sub(P, ok):
        return [t[ok] for t in P[:-2]] + [P[-2][ok]]
    ndv_tr, ndoy_tr, off_tr, dla_tr, dlo_tr, dd_tr, ndt_tr, dev_tr, len_tr = \
        [t[ok_tr] for t in (ndv_tr, ndoy_tr, off_tr, dla_tr, dlo_tr, dd_tr, ndt_tr, dev_tr, len_tr)]
    q_tr_ok = q_tr[ok_tr.numpy()]; ytr = ytr[ok_tr]; ytr_doy = ytr_doy[ok_tr]
    valid_tr = torch.tensor(v_tr[ok_tr.numpy()]).float()
    ndv_te, ndoy_te, off_te, dla_te, dlo_te, dd_te, ndt_te, dev_te, len_te = \
        [t[ok_te] for t in (ndv_te, ndoy_te, off_te, dla_te, dlo_te, dd_te, ndt_te, dev_te, len_te)]
    q_te_ok = q_te[ok_te.numpy()]; yte_doy_ok = yte_doy[ok_te.numpy()]
    valid_te = torch.tensor(v_te[ok_te.numpy()]).float()
    Btr = ndv_tr.shape[0]
    if Btr == 0 or ndv_te.shape[0] == 0:
        return dict(mae=float("nan"), acc=float("nan"), n_te=0)

    to = lambda t: t.to(dev)
    extra = {}

    # ---------- KERNEL / ADVECTION: interpretable, near-parameter-free physical models ----------
    if arm in ("kernel", "advection"):
        log_ell = torch.zeros(1, device=dev, requires_grad=True)          # log length-scale (deg)
        bias = torch.zeros(1, device=dev, requires_grad=True)
        beta_lat = torch.zeros(1, device=dev, requires_grad=True)         # Hopkins cline days/deg lat
        beta_lon = torch.zeros(1, device=dev, requires_grad=True)
        params = [log_ell, bias]
        if arm == "advection": params += [beta_lat, beta_lon]
        opt = torch.optim.Adam(params, lr=5e-2)
        ndoy_trd, dd_trd, dla_trd, dlo_trd, vtrd = to(ndoy_tr), to(dd_tr), to(dla_tr), to(dlo_tr), to(valid_tr)
        ytr_doyd = to(ytr_doy)
        def predict(ndoy, dd, dla, dlo, valid):
            ell = torch.exp(log_ell).clamp(1e-2, 50.0)
            logit = -(dd / ell) + bias                                   # Green's-function distance decay
            logit = logit.masked_fill(valid < 0.5, -1e9)
            w = torch.softmax(logit, dim=1)                             # stable normalized kernel weights
            adj = ndoy
            if arm == "advection":
                # advect each neighbour's DOY to the query location along lat/lon cline (days/deg)
                adj = ndoy + beta_lat * dla + beta_lon * dlo             # dla=q_lat-n_lat (signed)
            return _circ_wmean_doy(adj, w, valid)
        for _ in range(max(steps, 1200)):
            si = torch.randint(0, Btr, (min(4096, Btr),), device=dev)
            pd = predict(ndoy_trd[si], dd_trd[si], dla_trd[si], dlo_trd[si], vtrd[si])
            # circular MAE-ish smooth loss via (sin,cos) mse
            a = pd / 365.25 * 2 * np.pi; ay = ytr_doyd[si] / 365.25 * 2 * np.pi
            loss = ((torch.sin(a) - torch.sin(ay)) ** 2 + (torch.cos(a) - torch.cos(ay)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pd = predict(to(ndoy_te), to(dd_te), to(dla_te), to(dlo_te), to(valid_te)).cpu().numpy()
        err = circ_err_days(pd, yte_doy_ok)
        extra = {"length_scale_deg": float(torch.exp(log_ell).detach().cpu()),
                 "bias": float(bias.detach().cpu())}
        if arm == "advection":
            extra["cline_days_per_deg_lat"] = float(beta_lat.detach().cpu())
            extra["cline_days_per_deg_lon"] = float(beta_lon.detach().cpu())
        return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)), **extra)

    # ---------- PERSISTENCE: temporally-most-recent causal neighbour's DOY (parameter-free physics) ----------
    # Hypothesis for WHY the LSTM beats the circular-mean floor: seasonal persistence. The present-most (smallest
    # positive lag) past neighbour is the best DOY predictor; a circular MEAN washes this out. Selecting the most
    # recent neighbour uses ndt only for ARGMIN (which obs is newest) -- it predicts that obs's OWN past DOY, never
    # reconstructs the query day from the gap -> leak-free (validated by comparing to persist_shuf).
    if arm in ("persist", "persist_shuf", "persist_spadv", "persist_clines"):
        beta_lat = torch.zeros(1, device=dev, requires_grad=True)
        beta_lon = torch.zeros(1, device=dev, requires_grad=True)
        beta_elev = torch.zeros(1, device=dev, requires_grad=True)   # Hopkins elevation cline days/100m
        _adv = arm in ("persist_spadv", "persist_clines")
        def most_recent(ndoy, ndt, dla, dlo, dev_, valid, shuf=False):
            lag = ndt.clone(); lag[valid < 0.5] = 1e9
            if shuf:
                lag = lag[torch.randperm(lag.shape[0])]                 # break recency->query alignment
            idx = lag.argmin(1)                                         # newest causal neighbour
            r = torch.arange(ndoy.shape[0])
            d = ndoy[r, idx]
            if _adv:
                d = d + beta_lat * dla[r, idx] + beta_lon * dlo[r, idx] # spatial-only advection (leak-free)
            if arm == "persist_clines":
                d = d + beta_elev * dev_[r, idx]                        # + elevation cline (days/100m)
            return d
        if _adv:
            plist = [beta_lat, beta_lon] + ([beta_elev] if arm == "persist_clines" else [])
            opt = torch.optim.Adam(plist, lr=5e-2)
            ndoy_trd, ndt_trd, dla_trd, dlo_trd, dev_trd, vtrd = to(ndoy_tr), to(ndt_tr), to(dla_tr), to(dlo_tr), to(dev_tr), to(valid_tr)
            ytr_doyd = to(ytr_doy)
            for _ in range(1500):
                si = torch.randint(0, Btr, (min(4096, Btr),), device=dev)
                pd = most_recent(ndoy_trd[si], ndt_trd[si], dla_trd[si], dlo_trd[si], dev_trd[si], vtrd[si])
                a = pd / 365.25 * 2 * np.pi; ay = ytr_doyd[si] / 365.25 * 2 * np.pi
                loss = ((torch.sin(a) - torch.sin(ay)) ** 2 + (torch.cos(a) - torch.cos(ay)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pd = most_recent(to(ndoy_te), to(ndt_te), to(dla_te), to(dlo_te), to(dev_te), to(valid_te),
                             shuf=(arm == "persist_shuf")).cpu().numpy()
        err = circ_err_days(pd, yte_doy_ok)
        ex = {}
        if _adv:
            ex = {"cline_days_per_deg_lat": float(beta_lat.detach().cpu()),
                  "cline_days_per_deg_lon": float(beta_lon.detach().cpu())}
        if arm == "persist_clines":
            ex["cline_days_per_100m_elev"] = float(beta_elev.detach().cpu())
        return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)), **ex)

    # ---------- RECENCY KERNEL: interpretable version of the LSTM. Circular mean of neighbour DOYs weighted by
    # exp(-lag/tau) (learnable recency timescale tau, days) + optional Hopkins lat/lon advection cline. Uses lag
    # ONLY as a monotone recency weight (not to reconstruct query day); leak-audited vs recency_shuf. ----------
    if arm in ("recency", "recency_adv", "recency_shuf", "recency_clines"):
        log_tau = torch.zeros(1, device=dev, requires_grad=True)          # log recency timescale (days)
        beta_lat = torch.zeros(1, device=dev, requires_grad=True)
        beta_lon = torch.zeros(1, device=dev, requires_grad=True)
        beta_elev = torch.zeros(1, device=dev, requires_grad=True)        # Hopkins elevation cline days/100m
        _radv = arm in ("recency_adv", "recency_clines")
        params = [log_tau] + ([beta_lat, beta_lon] if _radv else []) + ([beta_elev] if arm == "recency_clines" else [])
        opt = torch.optim.Adam(params, lr=5e-2)
        ndoy_trd, ndt_trd, dla_trd, dlo_trd, dev_trd, vtrd = to(ndoy_tr), to(ndt_tr), to(dla_tr), to(dlo_tr), to(dev_tr), to(valid_tr)
        ytr_doyd = to(ytr_doy)
        def predict(ndoy, ndt, dla, dlo, dev_, valid, shuf=False):
            tau = torch.exp(log_tau).clamp(1.0, 3650.0)
            lag = ndt
            if shuf: lag = lag[torch.randperm(lag.shape[0])]
            logit = (-lag / tau).masked_fill(valid < 0.5, -1e9)
            w = torch.softmax(logit, 1)
            adj = ndoy + (beta_lat * dla + beta_lon * dlo if _radv else 0.0) \
                       + (beta_elev * dev_ if arm == "recency_clines" else 0.0)
            return _circ_wmean_doy(adj, w, valid)
        if arm != "recency_shuf":
            for _ in range(1500):
                si = torch.randint(0, Btr, (min(4096, Btr),), device=dev)
                pd = predict(ndoy_trd[si], ndt_trd[si], dla_trd[si], dlo_trd[si], dev_trd[si], vtrd[si])
                a = pd / 365.25 * 2 * np.pi; ay = ytr_doyd[si] / 365.25 * 2 * np.pi
                loss = ((torch.sin(a) - torch.sin(ay)) ** 2 + (torch.cos(a) - torch.cos(ay)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pd = predict(to(ndoy_te), to(ndt_te), to(dla_te), to(dlo_te), to(dev_te), to(valid_te),
                         shuf=(arm == "recency_shuf")).cpu().numpy()
        err = circ_err_days(pd, yte_doy_ok)
        ex = {"tau_days": float(torch.exp(log_tau).detach().cpu())}
        if _radv:
            ex["cline_days_per_deg_lat"] = float(beta_lat.detach().cpu())
            ex["cline_days_per_deg_lon"] = float(beta_lon.detach().cpu())
        if arm == "recency_clines":
            ex["cline_days_per_100m_elev"] = float(beta_elev.detach().cpu())
        return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)), **ex)

    # ---------- ECOGRADIENT: direct ecological phenology cline (Hopkins' Bioclimatic Law). Fit the query's OWN
    # mean-DOY as a circular function of its ABSOLUTE latitude and elevation (query location attributes only --
    # leak-free, no neighbour DOY, no target reuse). Recovers days/deg-lat and days/100m-elev in the SIGN the
    # ecology literature reports (spring phenology advances/delays with lat/altitude). de-persistenced variant
    # (arm ecogradient_resid) fits the residual after removing the freshest-neighbour persistence prediction. ----
    if arm in ("ecogradient", "ecogradient_resid"):
        alat_tr = torch.tensor(np.abs(lat[q_tr_ok]) / 10.0).float().to(dev)   # |lat| in 10-deg units
        elev_tr = torch.tensor((elev[q_tr_ok] if elev is not None else np.zeros(len(q_tr_ok))) / 100.0).float().to(dev)
        alat_te = torch.tensor(np.abs(lat[q_te_ok]) / 10.0).float().to(dev)
        elev_te = torch.tensor((elev[q_te_ok] if elev is not None else np.zeros(len(q_te_ok))) / 100.0).float().to(dev)
        fin_tr = torch.isfinite(elev_tr); elev_tr = torch.nan_to_num(elev_tr)
        elev_te = torch.nan_to_num(elev_te)
        y_tr = to(ytr_doy); y_te = yte_doy_ok
        base_tr = torch.zeros_like(y_tr); base_te = np.zeros(len(q_te_ok), dtype=np.float32)
        if arm == "ecogradient_resid":
            # remove freshest-neighbour persistence, fit clines on the seasonal residual
            lag_tr = to(ndt_tr).clone(); lag_tr[to(valid_tr) < 0.5] = 1e9
            idx_tr = lag_tr.argmin(1); r_tr = torch.arange(ndoy_tr.shape[0])
            base_tr = to(ndoy_tr)[r_tr, idx_tr]
            lag_te = to(ndt_te).clone(); lag_te[to(valid_te) < 0.5] = 1e9
            idx_te = lag_te.argmin(1); r_te = torch.arange(ndoy_te.shape[0])
            base_te = to(ndoy_te)[r_te, idx_te].cpu().numpy()
        c0 = torch.zeros(1, device=dev, requires_grad=True)
        b_lat = torch.zeros(1, device=dev, requires_grad=True)
        b_elev = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([c0, b_lat, b_elev], lr=5e-2)
        Btr2 = alat_tr.shape[0]
        def pred(alat, elev_, base):
            return base + c0 + b_lat * alat + b_elev * elev_
        for _ in range(2000):
            si = torch.randint(0, Btr2, (min(4096, Btr2),), device=dev)
            pd = pred(alat_tr[si], elev_tr[si], base_tr[si])
            a = pd / 365.25 * 2 * np.pi; ay = y_tr[si] / 365.25 * 2 * np.pi
            loss = ((torch.sin(a) - torch.sin(ay)) ** 2 + (torch.cos(a) - torch.cos(ay)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pd = pred(alat_te, elev_te, to(torch.tensor(base_te).float())).cpu().numpy()
        err = circ_err_days(pd, y_te)
        # coefficients back in per-1-unit terms: b_lat is per-10deg -> /10 for per-deg; b_elev per-100m already
        ex = {"eco_days_per_deg_abslat": float(b_lat.detach().cpu()) / 10.0,
              "eco_days_per_100m_elev": float(b_elev.detach().cpu()),
              "eco_intercept": float(c0.detach().cpu())}
        return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)), **ex)

    # ---------- LSTM / HARMONIC / ODE: learned propagators ----------
    qfeat_dim = 2
    if arm == "lstm":
        model = JointForecaster(qfeat_dim, hidden=hidden).to(dev)
    elif arm == "harmonic":
        model = HarmonicForecaster(qfeat_dim, hidden=hidden).to(dev)
    elif arm in ("ode", "ode_shuf", "ode_relgap"):
        model = ODEForecaster(qfeat_dim, hidden=min(hidden, 128)).to(dev)
    else:
        raise ValueError(arm)
    # LEAK AUDIT for the ODE dt channel. ndt = q_day - n_day carries the query's absolute day; combined with
    # neighbour DOY it can reconstruct the query DOY (target leak). Two nulls:
    #   ode_shuf   -> ndt shuffled across queries (destroys query-day info, keeps gap magnitude distribution)
    #   ode_relgap -> replace absolute-lag ndt by RELATIVE gaps between consecutive neighbours (leak-free dt)
    if arm == "ode_shuf":
        perm = torch.randperm(ndt_tr.shape[0]); ndt_tr = ndt_tr[perm]
        perm2 = torch.randperm(ndt_te.shape[0]); ndt_te = ndt_te[perm2]
    if arm == "ode_relgap":
        # neighbour days recoverable from tgt? no -- use ndt to derive consecutive gaps, drop the last-to-query lag.
        # ndt is q_day - n_day (descending as k->present). consecutive neighbour gap = |ndt[k] - ndt[k+1]|; the
        # final present->query lag (min ndt) is REMOVED so the query day cannot be reconstructed.
        gap = torch.zeros_like(ndt_tr); gap[:, 1:] = (ndt_tr[:, :-1] - ndt_tr[:, 1:]).abs(); ndt_tr = gap
        gap2 = torch.zeros_like(ndt_te); gap2[:, 1:] = (ndt_te[:, :-1] - ndt_te[:, 1:]).abs(); ndt_te = gap2
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ndv_trd, off_trd, ndt_trd, len_trd, ytrd = to(ndv_tr), to(off_tr), to(ndt_tr), len_tr, to(ytr)
    # 2nd-harmonic neighbour DOY vec (for harmonic arm)
    def h2(ndoy_days):
        a2 = ndoy_days / 365.25 * 4 * np.pi
        return torch.stack([torch.sin(a2), torch.cos(a2)], -1)
    nd2_trd = to(h2(ndoy_tr)); nd2_ted = None
    for _ in range(steps):
        si = torch.randint(0, Btr, (min(2048, Btr),))
        qf = qfeat(q_tr_ok[si.numpy()])
        if arm == "lstm":
            pred = model(qf, ndv_trd[si], off_trd[si], len_trd[si])
        elif arm == "harmonic":
            pred = model(qf, ndv_trd[si], nd2_trd[si], off_trd[si], len_trd[si])
        else:
            pred = model(qf, ndv_trd[si], off_trd[si], ndt_trd[si], len_trd[si])
        loss = F.mse_loss(pred, ytrd[si])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    ndv_ted, off_ted, ndt_ted = to(ndv_te), to(off_te), to(ndt_te)
    nd2_ted = to(h2(ndoy_te))
    with torch.no_grad():
        preds = []
        for s in range(0, ndv_ted.shape[0], 8192):
            e = slice(s, s + 8192); qf = qfeat(q_te_ok[e])
            if arm == "lstm":
                p = model(qf, ndv_ted[e], off_ted[e], len_te[e])
            elif arm == "harmonic":
                p = model(qf, ndv_ted[e], nd2_ted[e], off_ted[e], len_te[e])
            else:
                p = model(qf, ndv_ted[e], off_ted[e], ndt_ted[e], len_te[e])
            preds.append(p.cpu().numpy())
        err = circ_err_days(vec_to_doy(np.concatenate(preds)), yte_doy_ok)
    return dict(mae=float(np.mean(err)), acc=float(np.mean(err <= tol_days)), n_te=int(len(err)))


def run_physprop(a, lat, lon, days, tgt_doy, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev, split, floor_mae, floor_acc):
    import json as _json
    K = a.rec_k
    arms = [x for x in a.arms.split(",") if x]
    elev = None
    if any(("clines" in ar) or ("ecogradient" in ar) for ar in arms):
        elev = load_elev(a.cache_dir, a.n_shards)
        _fin = np.isfinite(elev)
        print(f"  ELEV loaded: {int(_fin.sum())}/{len(elev)} obs have elevation "
              f"(range {np.nanmin(elev[_fin]):.0f}..{np.nanmax(elev[_fin]):.0f} m)")
    res = {}
    for arm in arms:
        res[arm] = run_arm_phys(arm, tgt_doy, lat, lon, days, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev,
                                steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.tol_days, K=K, seed=a.seed, elev=elev)
    n_te = res[arms[0]]["n_te"]
    print(f"=== EARTH4D-ENGINE PHYSPROP | physics-informed propagator vs vanilla LSTM | target={a.target} ===")
    print(f"  split={split} obs={len(days)} q_te={n_te} K={K} steps={a.steps} seed={a.seed} tol=+/-{a.tol_days:.0f}d")
    print(f"  STATIC FLOOR (neighbour circular-mean, no propagation): MAE {floor_mae:.1f}d  acc {floor_acc:.4f}")
    for arm in arms:
        r = res[arm]
        line = f"    {arm:10s}  MAE {r['mae']:.1f}d  acc {r['acc']:.4f}   (MAE_gain vs floor {floor_mae - r['mae']:+.1f}d)"
        if "length_scale_deg" in r: line += f"   ell={r['length_scale_deg']:.3f}deg"
        if "tau_days" in r: line += f"   tau={r['tau_days']:.1f}d"
        if "cline_days_per_deg_lat" in r:
            line += f"  cline_lat={r['cline_days_per_deg_lat']:+.2f}d/deg  cline_lon={r['cline_days_per_deg_lon']:+.2f}d/deg"
        if "cline_days_per_100m_elev" in r:
            line += f"  cline_elev={r['cline_days_per_100m_elev']:+.2f}d/100m"
        if "eco_days_per_deg_abslat" in r:
            line += f"  eco_lat={r['eco_days_per_deg_abslat']:+.2f}d/deg|lat|  eco_elev={r['eco_days_per_100m_elev']:+.2f}d/100m"
        print(line)
    if "lstm" in res:
        base = res["lstm"]
        for arm in arms:
            if arm == "lstm": continue
            r = res[arm]
            print(f"  {arm:10s} vs vanilla-LSTM:  d_acc {r['acc'] - base['acc']:+.4f}   d_MAE {base['mae'] - r['mae']:+.1f}d")
    print("RESULT_JSON " + _json.dumps({"mode": "physprop", "split": split, "target": a.target, "n_te": n_te,
          "floor_mae": floor_mae, "floor_acc": floor_acc, "arms": res, "seed": a.seed, "K": K, "steps": a.steps}))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--rec_k", type=int, default=24)
    ap.add_argument("--rec_hidden", type=int, default=256)
    ap.add_argument("--tol_days", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial_levels", type=int, default=16)
    ap.add_argument("--temporal_levels", type=int, default=16)
    ap.add_argument("--log2_hashmap", type=int, default=19)
    ap.add_argument("--forecast_spatial", action="store_true")
    ap.add_argument("--mode", default="doy", choices=["doy", "abundance", "physprop"])
    ap.add_argument("--target", default="mean_doy", choices=["mean_doy", "phase_centroid"])
    ap.add_argument("--abund_win", type=float, default=90.0)
    ap.add_argument("--abund_lead", type=float, default=180.0)   # forward horizon (days); >0 = genuine lead-time forecast
    ap.add_argument("--abund_delta", action="store_true")        # future minus trailing-past (remove stationary mean)
    ap.add_argument("--arms", default="raw,e4d_frozen,e4d_e2e,mlp_e2e")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = a.device

    lat, lon, days = load_obs_time(a.cache_dir, a.n_shards)
    test = temporal_holdout(days, a.holdout)
    split = "future"
    if a.forecast_spatial:
        test = test & spatial_holdout(lat, lon, a.holdout, seed=a.seed)
        split = "future+newplace"
    tr_idx = np.where(~test)[0]; te_idx = np.where(test)[0]
    K = a.rec_k
    # causal windows: query -> K spatial-nearest strictly-earlier-day TRAIN neighbours (leak-safe)
    def win(q, pool):
        qi, vi = build_causal_windows(lat[q], lon[q], days[q], lat[pool], lon[pool], days[pool], K, fast=True)
        gi = np.where(qi >= 0, pool[np.clip(qi, 0, None)], -1)
        return gi, vi
    rng = np.random.default_rng(a.seed)
    q_tr = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = win(q_tr, tr_idx)
    g_te, v_te = win(te_idx, tr_idx)

    arms = [x for x in a.arms.split(",") if x]
    def mk_enc(arm):
        if arm in ("e4d_frozen", "e4d_e2e"):
            return Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,
                           spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap,
                           freq_log_scale_init=-2.5).to(dev)
        if arm == "mlp_e2e":
            return CoordMLP(2, 64).to(dev)
        return None

    if a.mode == "abundance":
        tgt, past_act = abundance_target(lat, lon, days, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta)
        # static floor: predict the query's future-delta as the mean of causal neighbours' PAST activity delta-proxy
        # (no-propagation nowcast). Report R2 vs the target.
        gsafe = np.clip(g_te, 0, len(past_act) - 1)
        w = v_te.astype(np.float32); num = (past_act[gsafe] * w).sum(1); den = w.sum(1) + 1e-6
        floor_pred = num / den
        ok_f = v_te.any(1)
        floor_r2, floor_mae = _r2(floor_pred[ok_f], tgt[te_idx][ok_f])
        res = {}
        for arm in arms:
            res[arm] = run_arm_abund(arm, tgt, past_act, lat, lon, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev,
                                     enc=mk_enc(arm), steps=a.steps, lr=a.lr, hidden=a.rec_hidden, K=K, seed=a.seed)
        dt = time.time() - t0; n_te = res[arms[0]]["n_te"]
        print(f"=== EARTH4D-ENGINE H1 | causal end-to-end ABUNDANCE forecast (lead={a.abund_lead:.0f}d win={a.abund_win:.0f}d delta={a.abund_delta}) ===")
        print(f"  split={split} obs={len(days)} q_te={n_te} K={K} steps={a.steps} seed={a.seed}  {dt:.0f}s")
        print(f"  STATIC FLOOR (neighbour past-activity mean, no propagation): R2 {floor_r2:+.4f}  MAE {floor_mae:.3f}")
        for arm in arms:
            r = res[arm]; print(f"    {arm:12s}  R2 {r['r2']:+.4f}  MAE {r['mae']:.3f}   (R2_gain vs floor {r['r2'] - floor_r2:+.4f})")
        if "raw" in res and "e4d_e2e" in res:
            print(f"  ENGINE st_gain (e4d_e2e - raw):      R2 {res['e4d_e2e']['r2'] - res['raw']['r2']:+.4f}")
        if "e4d_frozen" in res and "e4d_e2e" in res:
            print(f"  end-to-end vs frozen hash:           R2 {res['e4d_e2e']['r2'] - res['e4d_frozen']['r2']:+.4f}")
        if "mlp_e2e" in res and "e4d_e2e" in res:
            print(f"  HASH-ISOLATE st_gain (e4d_e2e - mlp): R2 {res['e4d_e2e']['r2'] - res['mlp_e2e']['r2']:+.4f}")
        print("RESULT_JSON " + __import__("json").dumps({"mode": "abundance", "split": split, "lead": a.abund_lead,
              "delta": a.abund_delta, "n_te": n_te, "floor_r2": floor_r2, "arms": res, "seconds": dt, "seed": a.seed, "K": K, "steps": a.steps}))
        return res

    # ---- DOY mode ----
    if a.target == "phase_centroid":
        tgt_doy = phase_centroid_doy(lat, lon, days)            # cell-level seasonal phase (strong spatial signal)
    else:
        tgt_doy = doy_of(days)                                  # per-obs mean-DOY
    floor_mae, floor_acc = _static_floor(tgt_doy, days, lat, lon, te_idx, g_te, v_te, a.tol_days)
    if a.mode == "physprop":
        return run_physprop(a, lat, lon, days, tgt_doy, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev, split, floor_mae, floor_acc)
    tmin = float(np.nanmin(days)); tspan = max(float(np.nanmax(days) - np.nanmin(days)), 1e-6)
    res = {}
    for arm in arms:
        if arm == "e4d_e2e_tc":                                 # H2: time-conditioned end-to-end (rule 2b + rule 1)
            enc = Earth4D(verbose=False, spatial_levels=a.spatial_levels, temporal_levels=a.temporal_levels,
                          spatial_log2_hashmap_size=a.log2_hashmap, temporal_log2_hashmap_size=a.log2_hashmap,
                          freq_log_scale_init=-2.5).to(dev)
            res[arm] = run_arm_timecond(tgt_doy, days, lat, lon, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev, enc,
                                        steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.tol_days, K=K,
                                        seed=a.seed, tmin=tmin, tspan=tspan)
            continue
        res[arm] = run_arm(arm, tgt_doy, lat, lon, q_tr, g_tr, v_tr, te_idx, g_te, v_te, dev, enc=mk_enc(arm),
                           steps=a.steps, lr=a.lr, hidden=a.rec_hidden, tol_days=a.tol_days, K=K, seed=a.seed)
    dt = time.time() - t0
    n_te = res[arms[0]]["n_te"]
    print(f"=== EARTH4D-ENGINE H1 | causal end-to-end forecast | target={a.target} ===")
    print(f"  split={split} obs={len(days)} q_te={n_te} K={K} steps={a.steps} seed={a.seed} tol=+/-{a.tol_days:.0f}d  {dt:.0f}s")
    print(f"  STATIC FLOOR (neighbour circular-mean, no propagation): MAE {floor_mae:.1f}d  acc {floor_acc:.4f}")
    for arm in arms:
        r = res[arm]
        print(f"    {arm:12s}  MAE {r['mae']:.1f}d  acc {r['acc']:.4f}   (MAE_gain vs floor {floor_mae - r['mae']:+.1f}d)")
    if "raw" in res and "e4d_e2e" in res:
        a_raw, a_e2e = res["raw"], res["e4d_e2e"]
        print(f"  ENGINE st_gain (e4d_e2e - raw):      acc {a_e2e['acc'] - a_raw['acc']:+.4f}   MAE {a_raw['mae'] - a_e2e['mae']:+.1f}d")
    if "e4d_frozen" in res and "e4d_e2e" in res:
        af, a2 = res["e4d_frozen"], res["e4d_e2e"]
        print(f"  end-to-end vs frozen hash:           acc {a2['acc'] - af['acc']:+.4f}   MAE {af['mae'] - a2['mae']:+.1f}d")
    if "mlp_e2e" in res and "e4d_e2e" in res:
        am, a2 = res["mlp_e2e"], res["e4d_e2e"]
        print(f"  HASH-ISOLATE st_gain (e4d_e2e - mlp): acc {a2['acc'] - am['acc']:+.4f}   MAE {am['mae'] - a2['mae']:+.1f}d")
    print("RESULT_JSON " + __import__("json").dumps({"mode": "doy", "split": split, "target": a.target, "n_te": n_te, "floor_mae": floor_mae,
          "floor_acc": floor_acc, "arms": res, "seconds": dt, "seed": a.seed, "K": K, "steps": a.steps}))
    return res


if __name__ == "__main__":
    main()
