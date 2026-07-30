"""GraphCast/GenCast-style GNN message-passing propagator for the spacetime forecaster (science.md rule 1+2b).

Prior rounds (Ensue tag spacetime): a static positional lookup (Earth4D or any PE) CANNOT forecast biology to
a new place at a future time -- indexing a 4D cell has no mechanism to PROPAGATE state past->future. The
4D-LSTM rollout (recurrence.py) supplied *a* propagation mechanism and lifted absolute forecast skill above a
static head, but a 1-D LSTM over K neighbours orders them arbitrarily in space and cannot weight an edge by its
geometry. This module upgrades the propagator to learned graph message passing (GraphCast/GenCast style):

  nodes  = the K strictly-earlier past observations in a query's spatiotemporal neighbourhood (+ the query node)
  edges  = neighbour -> query, with a LEARNED edge function e = phi_e([ spatial offset (dlat,dlon) || dt ]) so the
           propagator weights each past observation by its geometry to the future query (physics-inspired kernel)
  msg    = phi_m([ neighbour-node-state || edge-embedding ]);  query gathers sum_k gate_k * msg_k  (attention gate)
  hops   = T rounds of message passing (multi-hop: query state feeds back into the next round's messages)
  decode = phi_out(query-node-state-after-T-hops) -> family logits at the FUTURE query point.

This is a genuine propagator, not a lookup: the future query has NO features of its own beyond position and its
prediction is assembled ENTIRELY from causally-earlier observations weighted by learned space-time edges.

Fair controls (identical graph, identical query set, only the node featurization swapped):
  * Earth4D positional features  (mechanism-ON)
  * RFF positional features       (does the hash carry propagatable structure a generic PE lacks?)
  * a NO-PROPAGATION static head  (same query set, same positional feature of the QUERY point only, no neighbour
                                   aggregation) -> the absolute-skill floor the propagator must beat.

Additive + flag-gated: imported only when probe.py is called with --gnn; the default probe path never touches
this file.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.spacetime.editable_files.lib.recurrence import build_causal_windows


class GNNPropagator(nn.Module):
    """Message-passing propagator: K causal past nodes -> future query node, learned space-time edges, T hops.

    Node input = [ positional-feat(node) || family-embedding(node's observed family) ].  The query node has no
    observed family (it is the future we forecast) so it uses a dedicated learned <query> family token and its
    own positional feature. Messages flow neighbour->query each hop; the query state is re-broadcast so multiple
    hops let the query re-weight neighbours conditioned on its running estimate (GenCast-style processor)."""

    def __init__(self, feat_dim, n_fam, hidden=256, fam_emb=32, hops=2):
        super().__init__()
        self.hops = hops
        self.fam_emb = nn.Embedding(n_fam + 2, fam_emb)          # +1 pad, +1 dedicated <query> token
        self.query_fam = n_fam + 1                               # id of the <query> token
        node_in = feat_dim + fam_emb
        self.node_enc = nn.Sequential(nn.Linear(node_in, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_enc = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))   # (dlat,dlon,dt)
        # message = f(neighbour_state, edge, query_state); gate = scalar attention over neighbours
        self.msg = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.upd = nn.GRUCell(hidden, hidden)                    # query node update per hop
        self.head = nn.Linear(hidden, n_fam)

    def forward(self, nfeat, nfam, qfeat, edge, mask):
        # nfeat[B,K,F] neighbour pos-feat; nfam[B,K] neighbour family; qfeat[B,F] query pos-feat;
        # edge[B,K,3] (dlat,dlon,dt); mask[B,K] valid-neighbour bool.
        B, K, _ = nfeat.shape
        fe = self.fam_emb(nfam)                                  # [B,K,E]
        nstate = self.node_enc(torch.cat([nfeat, fe], -1))       # [B,K,H]
        qtok = self.fam_emb(torch.full((B,), self.query_fam, dtype=torch.long, device=nfeat.device))
        qstate = self.node_enc(torch.cat([qfeat, qtok], -1))     # [B,H] query node state
        e = self.edge_enc(edge)                                  # [B,K,H]
        m = mask.unsqueeze(-1).float()
        neg = torch.finfo(nstate.dtype).min
        for _ in range(self.hops):
            qb = qstate.unsqueeze(1).expand(-1, K, -1)           # broadcast query to each edge
            cat = torch.cat([nstate, e, qb], -1)                 # [B,K,3H]
            msg = self.msg(cat)                                  # [B,K,H]
            g = self.gate(cat).squeeze(-1)                       # [B,K]
            g = g.masked_fill(~mask, neg)
            w = torch.softmax(g, dim=1).unsqueeze(-1) * m        # attention over VALID neighbours only
            agg = (w * msg).sum(1)                               # [B,H] gathered message at the query
            qstate = self.upd(agg, qstate)                       # GRU update of the query node
        return self.head(qstate)


class StaticHead(nn.Module):
    """No-propagation baseline: predict the future query family from the QUERY point's OWN positional feature
    only (no neighbour aggregation). Same capacity family as the GNN's decode path. This is the absolute-skill
    floor a real propagator must beat -- if the GNN does not beat this, propagation carries nothing."""

    def __init__(self, feat_dim, n_fam, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_fam))

    def forward(self, qfeat):
        return self.net(qfeat)


def _build_tensors(featurize_static, feat_dim, fam, days, lat, lon, query_idx, gidx, valid, pad_fam,
                   qfeat_all):
    """Assemble neighbour + query + edge tensors for a set of queries (CPU), keeping only queries with >=1
    causal neighbour. `qfeat_all[N,F]` = precomputed positional feature per obs (used for both the query node
    and, indexed by neighbour, the neighbour node) so GNN/static share identical featurization."""
    B, K = gidx.shape
    N = qfeat_all.shape[0]
    gsafe = np.clip(gidx, 0, N - 1)
    vmask = torch.tensor(valid)
    nfeat = qfeat_all[torch.tensor(gsafe.reshape(-1))].reshape(B, K, feat_dim) * vmask.unsqueeze(-1)
    nfam = torch.tensor(np.where(valid, fam[gsafe], pad_fam)).long()
    qd = days[query_idx][:, None]
    dt = np.where(valid, (qd - days[gsafe]) / 365.0, 0.0)
    dlat = np.where(valid, lat[gsafe] - lat[query_idx][:, None], 0.0)
    dlon = np.where(valid, lon[gsafe] - lon[query_idx][:, None], 0.0)
    edge = torch.tensor(np.stack([dlat / 90.0, dlon / 180.0, dt], -1)).float()
    qfeat = qfeat_all[torch.tensor(query_idx)]                   # [B,F] query point's own positional feature
    y = torch.tensor(fam[query_idx]).long()
    ok = vmask.any(1)
    return nfeat[ok], nfam[ok], qfeat[ok], edge[ok], vmask[ok], y[ok]


def run_gnn(qfeat_all, feat_dim, fam, days, coords_ll, test, n_fam, dev, K=16, steps=4000, lr=3e-3,
            hidden=256, hops=2, also_static=True, pad_fam=None):
    # `also_static` retained for API compat; the static head is always computed now (cheap) so every feature
    # type gets an apples-to-apples no-propagation floor of its OWN featurization.
    """Train the GNN propagator on PAST queries, evaluate on the future+new-place held-out queries.

    qfeat_all[N,F] : positional featurization per obs (Earth4D for mechanism-ON; raw/RFF for controls).
    Returns dict {gnn_acc, gnn_top5, static_acc, static_top5, n_te} -- absolute forecast skill for the GNN
    propagator and (if also_static) the no-propagation static head on the IDENTICAL query set."""
    N = qfeat_all.shape[0]
    tr_idx = np.where(~test)[0]
    te_idx = np.where(test)[0]
    lat = coords_ll[:, 0].numpy(); lon = coords_ll[:, 1].numpy()
    pad_fam = n_fam if pad_fam is None else pad_fam

    def make_windows(qi_set, pool):
        qi, vi = build_causal_windows(lat[qi_set], lon[qi_set], days[qi_set],
                                      lat[pool], lon[pool], days[pool], K)
        gi = np.where(qi >= 0, pool[np.clip(qi, 0, None)], -1)
        return gi, vi

    rng = np.random.default_rng(0)
    q_train = tr_idx if len(tr_idx) <= 6000 else rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = make_windows(q_train, tr_idx)
    g_te, v_te = make_windows(te_idx, tr_idx)

    nftr, nfamtr, qftr, etr, mtr, ytr = _build_tensors(None, feat_dim, fam, days, lat, lon, q_train, g_tr, v_tr, pad_fam, qfeat_all)
    nfte, nfamte, qfte, ete, mte, yte = _build_tensors(None, feat_dim, fam, days, lat, lon, te_idx, g_te, v_te, pad_fam, qfeat_all)

    to = lambda *ts: [t.to(dev) for t in ts]
    nftr, nfamtr, qftr, etr, mtr, ytr = to(nftr, nfamtr, qftr, etr, mtr, ytr)
    nfte, nfamte, qfte, ete, mte, yte = to(nfte, nfamte, qfte, ete, mte, yte)

    out = {"n_te": int(nfte.shape[0])}
    if nftr.shape[0] == 0 or nfte.shape[0] == 0:
        return {"gnn_acc": float("nan"), "gnn_top5": float("nan"),
                "static_acc": float("nan"), "static_top5": float("nan"), "n_te": int(nfte.shape[0])}
    Btr = nftr.shape[0]
    bs = min(2048, Btr)

    # ---- GNN propagator ----
    gnn = GNNPropagator(feat_dim, n_fam, hidden=hidden, hops=hops).to(dev)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        logits = gnn(nftr[s], nfamtr[s], qftr[s], etr[s], mtr[s])
        loss = F.cross_entropy(logits, ytr[s])
        opt.zero_grad(); loss.backward(); opt.step()
    gnn.eval()
    with torch.no_grad():
        lg = gnn(nfte, nfamte, qfte, ete, mte)
        out["gnn_acc"] = (lg.argmax(-1) == yte).float().mean().item()
        out["gnn_top5"] = (lg.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()

    # ---- No-propagation static head on the IDENTICAL query set, SAME featurization (apples-to-apples floor) ----
    sh = StaticHead(feat_dim, n_fam, hidden=hidden).to(dev)
    opt = torch.optim.Adam(sh.parameters(), lr=lr)
    for _ in range(steps):
        s = torch.randint(0, Btr, (bs,), device=dev)
        loss = F.cross_entropy(sh(qftr[s]), ytr[s])
        opt.zero_grad(); loss.backward(); opt.step()
    sh.eval()
    with torch.no_grad():
        lg = sh(qfte)
        out["static_acc"] = (lg.argmax(-1) == yte).float().mean().item()
        out["static_top5"] = (lg.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return out
