"""Physics-inspired 4D recurrence propagator for the spacetime encoder (science.md rule 2b).

The forecast probe showed a static positional lookup (Earth4D) CANNOT forecast biology to a new place at a
future time (strict st_gain < 0 across seeds): indexing a 4D grid cell has no mechanism to PROPAGATE state
past->future. This module supplies that mechanism.

Rollout (per query point at a NEW place and FUTURE time):
  1. gather its K spatial-nearest observations from the TRAIN/PAST pool whose event day is strictly earlier
     (a causal local context window going back in time -- science.md rule 2b);
  2. order the window past->present and run an LSTM that PROPAGATES a hidden state forward through it;
  3. the final hidden state is the state rolled forward to the query location/time -> classify family.

Per-step token = [ positional-feat(neighbor)  ||  learned family embedding of the neighbor's observed family
                   ||  dt = (query_day - neighbor_day) normalized  ||  spatial offset (dlat,dlon) to query ].
The positional-feat is Earth4D for the mechanism-ON run and raw/RFF for the fair controls (identical rollout,
only the positional featurization swapped) so st_gain isolates whether Earth4D's 4D field carries structure
that PROPAGATES, not just structure that indexes.

Additive + flag-gated: imported only when probe.py is called with --recurrence; the default probe path never
touches this file.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TIME_HORIZON = 2.0


def normalize_time_from_train(days, train_mask, horizon=1.0):
    """Normalize event time using training rows only.

    Fitting the origin or span on validation/test dates would leak the evaluation period's extent into every
    coordinate feature, so both come from train rows alone.

    ``horizon`` reserves headroom for the held-out future INSIDE the encoder's representable time range.
    Without it, train occupies [0,1] and held-out rows land above 1.0 -- which is leak-free but hits a hard
    encoder limit: Earth4D's hash grid SATURATES past t~1.1 (measured: t=1.2, 1.5, 2.0 and 3.0 all return a
    byte-identical feature vector), so every test row becomes temporally indistinguishable and the forecast
    probe silently loses its time axis. Dividing the span by ``horizon`` compresses train into [0, 1/horizon]
    and leaves the remainder for the future. ``horizon`` must be a DESIGN constant (e.g. 1/(1-holdout)) --
    deriving it from test dates would reintroduce exactly the leak this function exists to remove.
    """
    days = np.asarray(days)
    train_mask = np.asarray(train_mask, dtype=bool)
    if days.ndim != 1 or train_mask.ndim != 1 or len(days) != len(train_mask):
        raise ValueError("days and train_mask must be aligned 1D arrays")
    if not np.isfinite(days).all():
        raise ValueError("event days must be finite before time normalization")
    if not train_mask.any():
        raise ValueError("time normalization needs at least one training row")
    if not np.isfinite(horizon) or horizon < 1.0:
        raise ValueError("horizon must be a finite design constant >= 1.0")
    train_days = days[train_mask]
    origin = float(train_days.min())
    span = max(float(train_days.max()) - origin, 1e-6) * float(horizon)
    normalized = ((days - origin) / span).astype(np.float32)
    return normalized, origin, span


def normalize_forecast_time(days, test_mask, horizon=DEFAULT_TIME_HORIZON):
    """Fit time on training rows and reserve predeclared future headroom."""
    return normalize_time_from_train(
        days, ~np.asarray(test_mask, dtype=bool), horizon=horizon
    )


def phenology_mode(forecast_spatial=False, pheno_spatial=False):
    """Exact evaluation-design label consumed by the trace record gate."""
    if forecast_spatial:
        return "PHENOLOGY-FUTURE-HELD"
    if pheno_spatial:
        return "PHENOLOGY-HELD"
    return "PHENOLOGY-FUTURE"


def phenology_feature_set(spec, nofair=False):
    """Requested phenology features plus mandatory raw/RFF controls for Earth4D."""
    feats = [x for x in spec.split(",") if x]
    if "e4d" in feats and not nofair:
        for control in ("raw", "rff"):
            if control not in feats:
                feats.append(control)
    return tuple(feats)


def validate_dynamic_target_causality(
    *,
    ar_rollout=False,
    ar_cond_lead=False,
    abundance=False,
    abund_prop_arch=False,
    breadth_target="",
    lead=0.0,
):
    """Fail closed for audited paths whose inputs or labels cross the origin."""
    unsafe = []
    if ar_rollout:
        unsafe.append("--ar_rollout")
    if ar_cond_lead:
        unsafe.append("--ar_cond_lead")
    if float(lead) > 0 and abundance:
        unsafe.append("--abundance with --abund_lead > 0")
    if float(lead) > 0 and abund_prop_arch:
        unsafe.append("--abund_prop_arch with --abund_lead > 0")
    if float(lead) > 0 and breadth_target:
        unsafe.append("--breadth_target with --abund_lead > 0")
    if unsafe:
        raise ValueError(
            "causality audit quarantine: "
            + ", ".join(unsafe)
            + " can expose post-origin observations through neighbor state or "
              "uncensored target windows; quarantined until future-sentinel, "
              "horizon-purge, and right-censoring tests pass"
        )



def strict_spatiotemporal_masks(lat, lon, days, future, held_place, block=0.5):
    """Construct and validate past+seen train / future+held test masks.

    The remaining past+held and future+seen quadrants are embargoed. ``held_place``
    must hold complete spatial blocks, not selected rows within a block.
    """
    arrays = tuple(np.asarray(x) for x in (lat, lon, days, future, held_place))
    if any(x.ndim != 1 for x in arrays) or len({len(x) for x in arrays}) != 1:
        raise ValueError("space-time split inputs must be aligned 1D arrays")
    lat, lon, days = arrays[:3]
    future, held_place = (arrays[3].astype(bool), arrays[4].astype(bool))
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or not np.isfinite(days).all():
        raise ValueError("space-time split coordinates and days must be finite")
    if not np.isfinite(block) or block <= 0:
        raise ValueError("spatial block size must be positive and finite")

    train = ~future & ~held_place
    test = future & held_place
    embargo = ~(train | test)
    if not train.any() or not test.any():
        raise ValueError("strict space-time split produced an empty train or test set")
    if not days[train].max() < days[test].min():
        raise ValueError("strict space-time split is not chronological")

    blocks = np.stack(
        [np.floor(lat / block), np.floor(lon / block)], axis=1
    ).astype(np.int64)
    train_blocks = set(map(tuple, blocks[train].tolist()))
    test_blocks = set(map(tuple, blocks[test].tolist()))
    if train_blocks & test_blocks:
        raise ValueError("strict space-time split reuses a held-out spatial block")
    return train, test, embargo


def build_causal_windows_kdtree(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K, over=8):
    """Exact causal-window builder backed by a cKDTree.

    The bucketed builder searches a FIXED 3x3 ring of `block_deg` cells; to widen the receptive field you must
    grow block_deg, whose candidate count -- and per-query Python cost -- scales with the block AREA (K128 at
    block 5deg took 2271s, CPU-bound on the neighbour loop, per LOOP-spacetime-window-breadth-K128-push). This
    builder replaces the ring scan with a single cKDTree over the pool's (lat,lon): it returns the TRUE K
    spatial-nearest CAUSAL (strictly-earlier-day) neighbours directly, with NO ring-boundary truncation and NO
    block-area scan.

    ``K*over`` is only the initial query breadth. Rows that do not yet contain all of their required causal
    neighbours are re-queried with a geometrically increasing breadth until the answer is exact. Resolved rows
    leave the active batch, so the full pool is queried only for rows whose causal history actually requires
    it. This matters when more than ``K*over`` spatially closer observations are future or simultaneous: a
    fixed over-query would silently return padding instead of the true past neighbours.

    Leak-safe: the causal filter is strictly ``p_day < q_day``. The selected K-nearest-in-space set is ordered
    past-to-present, with deterministic spatial-distance/pool-index tie breaking. Returns an index array
    ``[Nq,K]`` into the pool (padded with -1) and a matching valid mask."""
    from scipy.spatial import cKDTree
    q_lat = np.asarray(q_lat)
    q_lon = np.asarray(q_lon)
    q_day = np.asarray(q_day)
    p_lat = np.asarray(p_lat)
    p_lon = np.asarray(p_lon)
    p_day = np.asarray(p_day)
    query_arrays = (q_lat, q_lon, q_day)
    pool_arrays = (p_lat, p_lon, p_day)
    if any(array.ndim != 1 for array in query_arrays + pool_arrays):
        raise ValueError("causal-window coordinates and days must be 1D arrays")
    if len({len(array) for array in query_arrays}) != 1:
        raise ValueError("query latitude, longitude, and day lengths differ")
    if len({len(array) for array in pool_arrays}) != 1:
        raise ValueError("pool latitude, longitude, and day lengths differ")
    if any(not np.isfinite(array).all() for array in query_arrays + pool_arrays):
        raise ValueError("causal-window coordinates and days must be finite")
    Nq = len(q_lat)
    Np = len(p_lat)
    K = int(K)
    if K < 0:
        raise ValueError("K must be non-negative")
    idx = np.full((Nq, K), -1, dtype=np.int64)
    if Np == 0 or Nq == 0 or K == 0:
        return idx, idx >= 0

    query_xy = np.stack([q_lat, q_lon], axis=-1)
    pool_xy = np.stack([p_lat, p_lon], axis=-1)
    tree = cKDTree(pool_xy)

    # Knowing the total available history lets rows with no history terminate without a tree query and lets
    # rows with fewer than K past observations terminate as soon as every one of those observations is found.
    sorted_pool_days = np.sort(p_day)
    causal_total = np.searchsorted(sorted_pool_days, q_day, side="left")
    need = np.minimum(K, causal_total).astype(np.int64, copy=False)
    pending = np.flatnonzero(need > 0)
    if len(pending) == 0:
        return idx, idx >= 0

    initial_breadth = max(K, int(np.ceil(K * max(float(over), 1.0))))
    breadth = min(Np, initial_breadth)

    def spatial_order(query_i, candidates):
        """Sort pool indices by exact squared distance, then pool index for deterministic ties."""
        d2 = ((p_lat[candidates] - q_lat[query_i]) ** 2
              + (p_lon[candidates] - q_lon[query_i]) ** 2)
        order = np.lexsort((candidates, d2))
        return candidates[order], d2[order]

    while len(pending):
        dist, nn_idx = tree.query(query_xy[pending], k=breadth, workers=-1)
        if breadth == 1:                                          # cKDTree squeezes the k axis for k=1
            dist = dist[:, None]
            nn_idx = nn_idx[:, None]
        causal = p_day[nn_idx] < q_day[pending, None]
        ready = causal.sum(axis=1) >= need[pending]

        for local_i in np.flatnonzero(ready):
            query_i = pending[local_i]
            candidates = nn_idx[local_i, causal[local_i]]
            candidates, candidate_d2 = spatial_order(query_i, candidates)
            target = int(need[query_i])
            cutoff_d2 = candidate_d2[target - 1]

            # If the causal cutoff coincides with the outer edge of this k-NN query, cKDTree may have omitted
            # other points at exactly the same distance. Pull that radius once so pool-index tie breaking is
            # exact and independent of cKDTree's internal ordering.
            furthest_d2 = float(dist[local_i, -1]) ** 2
            edge_tol = 1e-12 * max(1.0, abs(furthest_d2))
            if breadth < Np and cutoff_d2 >= furthest_d2 - edge_tol:
                radius = np.nextafter(np.sqrt(cutoff_d2), np.inf)
                tied = np.asarray(tree.query_ball_point(query_xy[query_i], radius), dtype=np.int64)
                tied = tied[p_day[tied] < q_day[query_i]]
                candidates, candidate_d2 = spatial_order(query_i, tied)

            selected = candidates[:target]
            selected_d2 = candidate_d2[:target]
            # Stable past-to-present order. Same-day neighbours retain spatial-distance/pool-index ordering.
            temporal_order = np.lexsort((selected, selected_d2, p_day[selected]))
            idx[query_i, :target] = selected[temporal_order]

        pending = pending[~ready]
        if len(pending) == 0:
            break
        if breadth == Np:
            # ``causal_total`` and the strict filter should make this unreachable unless the day values have
            # unsupported ordering semantics (for example NaN query days).
            raise RuntimeError("failed to resolve causal neighbours after querying the full pool")
        breadth = min(Np, max(breadth + 1, breadth * 2))

    return idx, idx >= 0


def build_causal_windows(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K, block_deg=2.0, fast=False):
    """For each query, K nearest PAST train obs (strictly earlier day), ordered past->present.

    fast=True dispatches to the adaptive cKDTree builder (build_causal_windows_kdtree) -- same output
    contract and a receptive field NOT limited to the 3x3 block ring.
    Neighbour search is bucketed into coarse `block_deg` spatial cells (query cell + 8 ring) so it is O(N)
    not O(Nq*Np). Returns index array [Nq, K] into the pool (padded with -1) and a valid mask [Nq, K]."""
    if fast:
        return build_causal_windows_kdtree(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K)
    q_lat = np.asarray(q_lat); q_lon = np.asarray(q_lon); q_day = np.asarray(q_day)
    p_lat = np.asarray(p_lat); p_lon = np.asarray(p_lon); p_day = np.asarray(p_day)
    # bucket the pool
    def cell(la, lo):
        return (np.floor(la / block_deg).astype(np.int64), np.floor(lo / block_deg).astype(np.int64))
    pci, pcj = cell(p_lat, p_lon)
    from collections import defaultdict
    buckets = defaultdict(list)
    for k, (ci, cj) in enumerate(zip(pci.tolist(), pcj.tolist())):
        buckets[(ci, cj)].append(k)
    for key in buckets:
        buckets[key] = np.asarray(buckets[key], dtype=np.int64)
    qci, qcj = cell(q_lat, q_lon)
    idx = np.full((len(q_lat), K), -1, dtype=np.int64)
    for n in range(len(q_lat)):
        cand = []
        ci, cj = int(qci[n]), int(qcj[n])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                b = buckets.get((ci + di, cj + dj))
                if b is not None:
                    cand.append(b)
        if not cand:
            continue
        cand = np.concatenate(cand)
        past = cand[p_day[cand] < q_day[n]]                       # strictly-earlier -> causal
        if len(past) == 0:
            continue
        d2 = (p_lat[past] - q_lat[n]) ** 2 + (p_lon[past] - q_lon[n]) ** 2
        order = past[np.argsort(d2)][:K]                          # K nearest in space
        order = order[np.argsort(p_day[order])]                   # then order past->present in time
        idx[n, : len(order)] = order
    valid = idx >= 0
    return idx, valid


class Rollout(nn.Module):
    """LSTM that propagates local past state forward to the query. Head classifies query family."""

    def __init__(self, feat_dim, n_fam, hidden=256, fam_emb=32):
        super().__init__()
        self.fam_emb = nn.Embedding(n_fam + 1, fam_emb)           # +1 = pad id
        step_in = feat_dim + fam_emb + 3                          # +dt +dlat +dlon
        self.lstm = nn.LSTM(step_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_fam)

    def forward(self, feats, fam_ids, dt, doff, lengths):
        # feats[B,K,F] positional feat of neighbours; fam_ids[B,K]; dt[B,K]; doff[B,K,2]; lengths[B]
        fe = self.fam_emb(fam_ids)
        x = torch.cat([feats, fe, dt.unsqueeze(-1), doff], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                   batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1])                                   # [B, n_fam] logits at the query


def run_recurrence(pos_feats, fam, days, coords_ll, test, n_fam, dev, K=16,
                   steps=4000, lr=3e-3, hidden=256, tag="", pad_fam=None):
    """Train the rollout on PAST queries, evaluate on the future+new-place held-out queries.

    pos_feats [N,F]  : positional featurization (Earth4D for mechanism-ON; raw/RFF for controls)
    fam [N]          : family id per obs           days [N] : event day
    coords_ll [N,2]  : (lat,lon)                    test [N] bool : held-out (future+new place)
    Returns (acc, top5) for the rolled-forward forecast at held-out queries."""
    N = pos_feats.shape[0]
    train = ~test
    tr_idx = np.where(train)[0]
    te_idx = np.where(test)[0]
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy(); day = days
    pad_fam = n_fam if pad_fam is None else pad_fam

    # Training queries: a subset of TRAIN obs, each forecast from its OWN past train-pool neighbours.
    # Validation queries: the held-out future+new-place obs, forecast from the whole train pool's past.
    def make_windows(query_idx, pool_idx):
        qi, vi = build_causal_windows(lat[query_idx], lon[query_idx], day[query_idx],
                                      lat[pool_idx], lon[pool_idx], day[pool_idx], K)
        # remap pool-local indices to global
        gi = np.where(qi >= 0, pool_idx[np.clip(qi, 0, None)], -1)
        return gi, vi

    # cap training queries for speed; they draw context from the full train pool
    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = make_windows(q_train, tr_idx)
    g_te, v_te = make_windows(te_idx, tr_idx)

    F_ = pos_feats.shape[1]

    def tensors(query_idx, gidx, valid):
        B, Kk = gidx.shape
        gsafe = np.clip(gidx, 0, N - 1)
        feats = pos_feats[torch.tensor(gsafe.reshape(-1))].reshape(B, Kk, F_)
        vmask = torch.tensor(valid)
        feats = feats * vmask.unsqueeze(-1)                       # zero padded steps
        fam_ids = torch.tensor(np.where(valid, fam[gsafe], pad_fam)).long()
        qd = day[query_idx][:, None]
        dt = torch.tensor(np.where(valid, (qd - day[gsafe]) / 365.0, 0.0)).float()
        dlat = torch.tensor(np.where(valid, lat[gsafe] - lat[query_idx][:, None], 0.0)).float()
        dlon = torch.tensor(np.where(valid, lon[gsafe] - lon[query_idx][:, None], 0.0)).float()
        doff = torch.stack([dlat / 90.0, dlon / 180.0], -1)
        lengths = vmask.sum(1).long()
        y = torch.tensor(fam[query_idx]).long()
        return feats, fam_ids, dt, doff, lengths, y, vmask.any(1)

    ftr, famtr, dttr, offtr, lentr, ytr, oktr = tensors(q_train, g_tr, v_tr)
    fte, famte, dtte, offte, lente, yte, okte = tensors(te_idx, g_te, v_te)
    # keep only queries that have >=1 causal neighbour
    def sel(t, m): return t[m]
    ftr, famtr, dttr, offtr, lentr, ytr = [sel(t, oktr) for t in (ftr, famtr, dttr, offtr, lentr, ytr)]
    fte, famte, dtte, offte, lente, yte = [sel(t, okte) for t in (fte, famte, dtte, offte, lente, yte)]

    ftr, famtr, dttr, offtr, lentr, ytr = ftr.to(dev), famtr.to(dev), dttr.to(dev), offtr.to(dev), lentr, ytr.to(dev)
    fte, famte, dtte, offte, lente, yte = fte.to(dev), famte.to(dev), dtte.to(dev), offte.to(dev), lente, yte.to(dev)

    model = Rollout(F_, n_fam, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Btr = ftr.shape[0]
    if Btr == 0 or fte.shape[0] == 0:
        return float("nan"), float("nan"), int(fte.shape[0])
    bs = min(2048, Btr)
    for _ in range(steps):
        sel_i = torch.randint(0, Btr, (bs,))
        logits = model(ftr[sel_i], famtr[sel_i], dttr[sel_i], offtr[sel_i], lentr[sel_i])
        loss = F.cross_entropy(logits, ytr[sel_i])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(fte, famte, dtte, offte, lente)
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5, int(fte.shape[0])


class _CoordMLP(nn.Module):
    """Trainable coord->feature encoder (fair matched-capacity control for a TRAINED Earth4D: a generic PE
    that also gets to learn from the decode loss, so any Earth4D win is the 4D hash field, not just training)."""
    def __init__(self, in_dim, out_dim, hidden=512, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)



def run_recurrence_timecond(featurize, feat_dim, fam, days, coords_ll, test, n_fam, dev, K=16,
                            steps=4000, lr=3e-3, hidden=256, tag="", pad_fam=None):
    """Time-CONDITIONED rollout (science.md rule 24 + 2b): the positional token at each rollout step is the
    QUERY cell's field state re-encoded FORWARD IN TIME, not the neighbour's fixed static code.

    Default `run_recurrence` feeds each causal neighbour its OWN static featurization pos_feats[neighbour] --
    a fixed code the LSTM merely reads. Prior rounds showed Earth4D and RFF are then interchangeable: the hash
    adds no *propagatable* structure because nothing is ever propagated THROUGH the encoder's time axis.

    Here, for query q with causal window at past days d_1<...<d_K, step k's token is
        featurize(lat_q, lon_q, t = d_k)      # the query CELL, marched forward through its own time axis
    so the LSTM integrates the query cell's Earth4D STATE trajectory (slot-3 = live event day). The fair
    control passes the identical (lat_q, lon_q, t=d_k) triples through RFF/raw featurize -- same rollout, same
    query-cell-forward-in-time inputs, only the encoder swapped. st_gain then isolates whether Earth4D's 4D
    field carries structure that PROPAGATES across time, distinct from a generic space-time PE.

    `featurize(lat[M], lon[M], day[M]) -> Tensor[M, feat_dim]` re-encodes arbitrary space-time points.
    """
    N = coords_ll.shape[0]
    train = ~test
    tr_idx = np.where(train)[0]
    te_idx = np.where(test)[0]
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy(); day = days
    pad_fam = n_fam if pad_fam is None else pad_fam

    def make_windows(query_idx, pool_idx):
        qi, vi = build_causal_windows(lat[query_idx], lon[query_idx], day[query_idx],
                                      lat[pool_idx], lon[pool_idx], day[pool_idx], K)
        gi = np.where(qi >= 0, pool_idx[np.clip(qi, 0, None)], -1)
        return gi, vi

    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = make_windows(q_train, tr_idx)
    g_te, v_te = make_windows(te_idx, tr_idx)

    def tensors(query_idx, gidx, valid):
        B, Kk = gidx.shape
        gsafe = np.clip(gidx, 0, N - 1)
        vmask = torch.tensor(valid)
        # KEY DIFFERENCE vs run_recurrence: the per-step positional token is the QUERY cell (lat_q, lon_q)
        # re-encoded at the NEIGHBOUR's day -> the encoder's own time axis carries the propagated state.
        qlat = np.broadcast_to(lat[query_idx][:, None], (B, Kk))
        qlon = np.broadcast_to(lon[query_idx][:, None], (B, Kk))
        step_day = np.where(valid, day[gsafe], 0.0)               # neighbour's past day per step
        feats = featurize(qlat.reshape(-1), qlon.reshape(-1), step_day.reshape(-1)).reshape(B, Kk, feat_dim)
        feats = feats * vmask.unsqueeze(-1)                       # zero padded steps
        fam_ids = torch.tensor(np.where(valid, fam[gsafe], pad_fam)).long()
        qd = day[query_idx][:, None]
        dt = torch.tensor(np.where(valid, (qd - day[gsafe]) / 365.0, 0.0)).float()
        dlat = torch.tensor(np.where(valid, lat[gsafe] - lat[query_idx][:, None], 0.0)).float()
        dlon = torch.tensor(np.where(valid, lon[gsafe] - lon[query_idx][:, None], 0.0)).float()
        doff = torch.stack([dlat / 90.0, dlon / 180.0], -1)
        lengths = vmask.sum(1).long()
        y = torch.tensor(fam[query_idx]).long()
        return feats, fam_ids, dt, doff, lengths, y, vmask.any(1)

    ftr, famtr, dttr, offtr, lentr, ytr, oktr = tensors(q_train, g_tr, v_tr)
    fte, famte, dtte, offte, lente, yte, okte = tensors(te_idx, g_te, v_te)
    def sel(t, m): return t[m]
    ftr, famtr, dttr, offtr, lentr, ytr = [sel(t, oktr) for t in (ftr, famtr, dttr, offtr, lentr, ytr)]
    fte, famte, dtte, offte, lente, yte = [sel(t, okte) for t in (fte, famte, dtte, offte, lente, yte)]

    ftr, famtr, dttr, offtr, ytr = ftr.to(dev), famtr.to(dev), dttr.to(dev), offtr.to(dev), ytr.to(dev)
    fte, famte, dtte, offte, yte = fte.to(dev), famte.to(dev), dtte.to(dev), offte.to(dev), yte.to(dev)

    model = Rollout(feat_dim, n_fam, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Btr = ftr.shape[0]
    if Btr == 0 or fte.shape[0] == 0:
        return float("nan"), float("nan"), int(fte.shape[0])
    bs = min(2048, Btr)
    for _ in range(steps):
        sel_i = torch.randint(0, Btr, (bs,))
        logits = model(ftr[sel_i], famtr[sel_i], dttr[sel_i], offtr[sel_i], lentr[sel_i])
        loss = F.cross_entropy(logits, ytr[sel_i])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(fte, famte, dtte, offte, lente)
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5, int(fte.shape[0])


def run_field_decode(kind, coords4, rn_in, fam, test, n_fam, dev, enc=None, feat_dim=96,
                     steps=4000, lr=3e-3, head_hidden=256, wd=0.0):
    """rule 24 -- DENSE FIELD DECODE. Train an encoder END-TO-END to decode the family field from space-time,
    fitting the field between sparse TRAIN obs, then forecast the strict held-out (future+new-place) set.

    kind='earth4d' : trainable Earth4D encoder (enc) -> head            (the encoder learns field structure)
    kind='mlp'     : trainable coord-MLP on (lat/90,lon/180,t) -> head  (generic learned PE, matched capacity)
    kind='rff'     : FIXED random-Fourier features -> trainable head    (no learned encoder, positional control)

    Fair test: does a TRAINED Earth4D field-decoder generalize the field to held-out space-time better than a
    generic trainable PE (mlp) or a fixed PE (rff)? If not, the encoder's field carries no structure a plain
    learned coordinate map lacks. All share head width, steps, lr, batch; only the encoder differs."""
    coords4 = coords4.to(dev)
    rn = torch.tensor(rn_in).to(dev)
    y = torch.tensor(fam).long().to(dev)
    tr = torch.tensor(~test); te = torch.tensor(test)
    tr_i = torch.where(tr)[0].to(dev); te_i = torch.where(te)[0].to(dev)

    if kind == "earth4d":
        encoder = enc; enc_in = coords4; in_dim = feat_dim
    elif kind == "mlp":
        encoder = _CoordMLP(rn.shape[1], feat_dim).to(dev); enc_in = rn; in_dim = feat_dim
    elif kind == "rff":
        P = torch.tensor(np.random.default_rng(0).normal(0, 8.0, (rn.shape[1], feat_dim // 2)).astype(np.float32)).to(dev)
        proj = rn @ P
        rff_feats = torch.cat([torch.sin(proj), torch.cos(proj)], 1)
        encoder = None; enc_in = rff_feats; in_dim = rff_feats.shape[1]
    else:
        raise ValueError(kind)

    head = nn.Sequential(nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, n_fam)).to(dev)
    params = list(head.parameters()) + (list(encoder.parameters()) if encoder is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def feats_of(idx):
        if kind == "rff":
            return enc_in[idx]
        return encoder(enc_in[idx])

    Ntr = tr_i.shape[0]
    for _ in range(steps):
        sel = tr_i[torch.randint(0, Ntr, (4096,), device=dev)]
        logits = head(feats_of(sel))
        loss = F.cross_entropy(logits, y[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    if encoder is not None: encoder.eval()
    head.eval()
    with torch.no_grad():
        # chunk held-out eval to bound memory
        accs, t5s, tot = 0, 0, 0
        for s in range(0, te_i.shape[0], 8192):
            b = te_i[s:s + 8192]
            logits = head(feats_of(b)); yy = y[b]
            accs += (logits.argmax(-1) == yy).sum().item()
            t5s += (logits.topk(5, -1).indices == yy[:, None]).any(-1).sum().item()
            tot += b.shape[0]
    return accs / tot, t5s / tot, tot
class LocalCrossEraHead(nn.Module):
    """Classifier preserving local, rather than globally collapsed, range modes."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        if hidden <= 0:
            raise ValueError("local cross-era alignment requires a positive hidden width")
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.classifier = nn.Linear(hidden, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))

    def loss(
        self,
        anchor: torch.Tensor,
        labels: torch.Tensor,
        positive: torch.Tensor,
        positive_rows: torch.Tensor,
        temperature: float = 0.1,
        max_pairs: int = 1024,
    ) -> torch.Tensor:
        h = self.embed(anchor)
        classification = F.cross_entropy(self.classifier(h), labels)
        take = min(int(max_pairs), positive.shape[0])
        if take == 0:
            return classification
        rows = positive_rows[:take]
        z = F.normalize(h[rows], dim=-1)
        zp = F.normalize(self.embed(positive[:take]), dim=-1)
        pair_labels = labels[rows]
        similarity = (z @ zp.t()) / temperature
        diagonal = torch.eye(take, dtype=torch.bool, device=anchor.device)
        # Disconnected populations of one species are neither positives nor negatives.
        same_species = pair_labels[:, None] == pair_labels[None, :]
        allowed = (~same_species) | diagonal
        target = torch.arange(take, device=anchor.device)
        row_loss = F.cross_entropy(similarity.masked_fill(~allowed, -torch.inf), target)
        col_loss = F.cross_entropy(similarity.t().masked_fill(~allowed.t(), -torch.inf), target)
        return classification + 0.5 * (row_loss + col_loss)


class OrthogonalTemporalHead(nn.Module):
    """Classifier with norm-preserving temporal transport in its hidden state."""

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        if hidden <= 0 or hidden % 2:
            raise ValueError("orthogonal temporal transport requires a positive even hidden width")
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.classifier = nn.Linear(hidden, n_classes)
        # Identity initialization preserves the promoted static head at step zero.
        self.angular_velocity = nn.Parameter(torch.zeros(hidden // 2))

    def transport(self, hidden: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        pairs = hidden.reshape(hidden.shape[0], -1, 2)
        centered_time = phase.to(hidden.dtype).reshape(-1, 1) - 0.5
        angle = torch.pi * centered_time * torch.tanh(self.angular_velocity).reshape(1, -1)
        c, s = torch.cos(angle), torch.sin(angle)
        real, imag = pairs[..., 0], pairs[..., 1]
        return torch.stack((c * real - s * imag, s * real + c * imag), dim=-1).flatten(1)

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.transport(self.trunk(x), phase))


def build_probe_readout(
    in_dim: int,
    hidden: int,
    n_classes: int,
    *,
    cross_era: bool = False,
    temporal: bool = False,
) -> nn.Module:
    """Build the production Earth4D readout behind the fixed probe contract.

    The probe decides only which validated tensors are available. Scientific
    experiments may change the implementation selected here without changing
    benchmark data, optimization, or scoring code.
    """
    if cross_era and temporal:
        raise ValueError("cross-era and temporal readouts are mutually exclusive")
    if cross_era:
        return LocalCrossEraHead(in_dim, hidden, n_classes)
    if temporal:
        return OrthogonalTemporalHead(in_dim, hidden, n_classes)
    if hidden > 0:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    return nn.Linear(in_dim, n_classes)


def nearest_dated_conspecific(
    current_coords: torch.Tensor,
    current_labels: torch.Tensor,
    support_coords: torch.Tensor,
    support_labels: torch.Tensor,
    dated_rows: int,
    chunk: int = 2048,
) -> torch.Tensor:
    """Map current rows to their geographically nearest dated conspecific support."""
    if dated_rows <= 0 or dated_rows > len(support_coords):
        raise ValueError("dated support tail is required for local cross-era pairing")
    current = current_coords.detach().cpu().numpy()
    current_y = current_labels.detach().cpu().numpy()
    split = len(support_coords) - dated_rows
    support = support_coords[split:].detach().cpu().numpy()
    support_y = support_labels[split:].detach().cpu().numpy()

    def unit(x: np.ndarray) -> np.ndarray:
        lat = np.deg2rad(x[:, 0])
        lon = np.deg2rad(x[:, 1])
        return np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
        )

    current_unit, support_unit = unit(current), unit(support)
    partners = np.full(len(current), -1, dtype=np.int64)
    for cls in np.unique(current_y):
        current_idx = np.flatnonzero(current_y == cls)
        support_idx = np.flatnonzero(support_y == cls)
        if not len(support_idx):
            continue
        for start in range(0, len(current_idx), chunk):
            rows = current_idx[start : start + chunk]
            nearest = np.argmax(current_unit[rows] @ support_unit[support_idx].T, axis=1)
            partners[rows] = split + support_idx[nearest]
    return torch.tensor(partners, dtype=torch.long)
