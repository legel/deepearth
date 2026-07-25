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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_windows_kdtree(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K, over=8):
    """VECTORIZED causal-window builder (cKDTree) -- same contract as build_causal_windows.

    The bucketed builder searches a FIXED 3x3 ring of `block_deg` cells; to widen the receptive field you must
    grow block_deg, whose candidate count -- and per-query Python cost -- scales with the block AREA (K128 at
    block 5deg took 2271s, CPU-bound on the neighbour loop, per LOOP-spacetime-window-breadth-K128-push). This
    builder replaces the ring scan with a single cKDTree over the pool's (lat,lon): it returns the TRUE K
    spatial-nearest CAUSAL (strictly-earlier-day) neighbours directly, with NO ring-boundary truncation and NO
    per-query Python loop, so receptive-field breadth (K) is decoupled from an O(block_area) cost.

    Because ~half the spatial-nearest neighbours may be non-causal (later day), we over-query K*over candidates
    from the tree, drop the future ones, then keep the K spatially-closest survivors and order them past->present
    in time -- identical output semantics to the bucketed builder (K nearest-in-space among strictly-earlier
    obs). Leak-safe: the causal filter uses ONLY p_day<q_day; no dt is emitted here (edges add spatial offset
    only downstream). Returns index array [Nq,K] into the pool (padded with -1) and a valid mask [Nq,K]."""
    from scipy.spatial import cKDTree
    q_lat = np.asarray(q_lat); q_lon = np.asarray(q_lon); q_day = np.asarray(q_day)
    p_lat = np.asarray(p_lat); p_lon = np.asarray(p_lon); p_day = np.asarray(p_day)
    Nq = len(q_lat); Np = len(p_lat)
    idx = np.full((Nq, K), -1, dtype=np.int64)
    if Np == 0 or Nq == 0:
        return idx, idx >= 0
    tree = cKDTree(np.stack([p_lat, p_lon], -1))
    kq = min(Np, K * over)                                        # over-query so enough survive the causal filter
    dist, nn_idx = tree.query(np.stack([q_lat, q_lon], -1), k=kq, workers=-1)
    if kq == 1:                                                   # cKDTree squeezes the last axis when k==1
        dist = dist[:, None]; nn_idx = nn_idx[:, None]
    # nn_idx[Nq,kq] already sorted by increasing spatial distance. Mask the non-causal (future/simultaneous)
    # neighbours, then take the first K valid per row (still nearest-first), preserving the K-nearest-in-space set.
    causal = p_day[nn_idx] < q_day[:, None]                       # [Nq,kq] True where neighbour strictly earlier
    order_in_row = np.argsort(~causal, axis=1, kind="stable")     # valid (causal) entries float to the front, nearest-first
    nn_sorted = np.take_along_axis(nn_idx, order_in_row, axis=1)
    causal_sorted = np.take_along_axis(causal, order_in_row, axis=1)
    take = min(K, nn_sorted.shape[1])
    sel = nn_sorted[:, :take]                                     # [Nq,take] K spatial-nearest causal (padded logically by mask)
    selc = causal_sorted[:, :take]
    kept = np.where(selc, sel, -1)                               # [Nq,take], -1 = pad
    # order each kept window past->present in time (match the bucketed builder). Vectorized: pad rows get +inf
    # day so they sort to the end; argsort by day keeps valid neighbours contiguous at the front, earliest-first.
    key = np.where(kept >= 0, p_day[np.clip(kept, 0, None)], np.inf)   # [Nq,take]
    torder = np.argsort(key, axis=1, kind="stable")
    kept = np.take_along_axis(kept, torder, axis=1)
    idx[:, :take] = kept
    return idx, idx >= 0


def build_causal_windows(q_lat, q_lon, q_day, p_lat, p_lon, p_day, K, block_deg=2.0, fast=False):
    """For each query, K nearest PAST train obs (strictly earlier day), ordered past->present.

    fast=True dispatches to the vectorized cKDTree builder (build_causal_windows_kdtree) -- same output
    contract, receptive field NOT limited to the 3x3 block ring, and no per-query Python neighbour loop.
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
