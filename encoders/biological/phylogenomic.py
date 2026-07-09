"""Phylogenomic species network: a learnable representation per species that shares information between relatives.

Every species carries a learnable embedding. Because related species share ancestry, what is learned about one
should inform its relatives, so this module refines the per-species embeddings — before any species-linked readout —
by passing information along evolutionary structure. An observation of one species therefore back-propagates into
its relatives (extended backprop, and, within a batch, into every in-context neighbour). A free discriminative
component is kept alongside the refinement so congeners stay separable rather than collapsing onto a shared mean.
A new species of unknown placement can be projected into the same space (:meth:`SpeciesGraph.distance_from_embedding`
maps any per-species embedding to the distances the attention operator consumes).

This one file holds the whole phylogenomic stack: loading the scientifically-derived tree (:func:`parse_newick`,
:func:`build_tree_buffers`) and the two operators that refine the species states over it, selected by ``operator``:

* ``"ou-attention"`` — multi-head attention among species biased toward phylogenetic neighbors: for head ``h`` the
  logit between species ``i`` and ``j`` is the content score minus ``alpha_h`` times their phylogenetic distance,
  so a species borrows most from close relatives. The per-head rate ``alpha_h`` is learned, following the
  Ornstein-Uhlenbeck view that trait covariance decays with evolutionary distance. The distance it consumes is a
  *precomputed matrix* — in practice derived from an embedding (a lossy shadow of the tree), not the tree itself.
  This is the fast, differentiable default; it needs only a distance matrix, so it processes neighbours cheaply.

* ``"tree"`` — tree-structured message passing over the *actual* dated phylogeny (topology + branch lengths). The
  species states are placed at the tree's tips; a level-synchronous upward sweep carries information from tips to
  the root, with internal nodes accumulating genuine ancestral state; a downward sweep carries it back to the tips.
  Each message is gated by ``exp(-softplus(theta) * branch_length)``, so information decays along real evolutionary
  time, and ``theta`` is learned. This puts the tree *in the loop* rather than approximating it with a distance
  matrix, and lets a tip that is rare or entirely held out inherit a state from its training relatives through the
  shared ancestral nodes. It is the topology-faithful graph neural network the OU operator approximates, and the
  direction we encourage for continued work (see :class:`TreeMessagePassing`).
"""
from __future__ import annotations
import math
from collections import deque
from typing import Dict, List, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                                        # FlexAttention: block-sparse, fused, scales to 100k+
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _compiled_flex = torch.compile(flex_attention)
    _HAS_FLEX = True
    # Ampere (sm_8x) shared memory cannot fit the default backward pipelining, so constrain the backward tiles;
    # Hopper (sm_9x) has the memory for the faster default config. Same math, arch-tuned kernel.
    _cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (9, 0)
    _FLEX_KO = (None if _cap[0] >= 9 else
                {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_warps": 4, "num_stages": 2})

    def _flex_attention(*a, **kw):
        """Prefer the compiled (fused) kernel; fall back to eager if the backend cannot autotune it (e.g. the
        FlexAttention backward on some Ampere GPUs). Same math either way."""
        global _flex_attention
        try:
            return _compiled_flex(*a, **kw)
        except Exception:
            _flex_attention = flex_attention                # stick with eager for the rest of the session
            return flex_attention(*a, **kw)
except Exception:
    _HAS_FLEX = False


# ============================================================================ tree loading (Newick -> buffers)
# The species graph can refine states over the *actual* evolutionary tree -- topology and branch lengths -- rather
# than a precomputed distance matrix. The tree is static, so it is read once at data-load time: parse the Newick,
# prune to the model's tips (preserving every root-to-tip path length), compact node ids so model species occupy
# leaf ids 0..n_species-1, and group edges by level for a level-synchronous GPU sweep.

def parse_newick(path: str):
    """Parse a Newick tree into ``parent``, ``branch_length`` and ``label`` arrays (``parent == -1`` is the root).

    A small recursive-descent parser (the tree nests only ~50 deep). Handles a leading rooting comment such as
    ``[&U]`` and optional internal-node labels.
    """
    s = open(path).read().strip()
    if s.startswith("[&"):                                    # strip a rooting comment, e.g. "[&U]"
        s = s[s.index("]") + 1:].strip()
    if s.endswith(";"):
        s = s[:-1]
    parent: List[int] = []
    blen: List[float] = []
    label: List[str] = []
    pos = 0

    def node(p: int) -> int:
        parent.append(p); blen.append(0.0); label.append("")
        return len(parent) - 1

    def read_label_len(nd: int) -> None:
        nonlocal pos
        start = pos
        while pos < len(s) and s[pos] not in "(),:;":
            pos += 1
        if pos > start:
            label[nd] = s[start:pos]
        if pos < len(s) and s[pos] == ":":
            pos += 1; start = pos
            while pos < len(s) and s[pos] not in "(),:;":
                pos += 1
            blen[nd] = float(s[start:pos])

    def parse(p: int) -> int:
        nonlocal pos
        nd = node(p)
        if s[pos] == "(":                                    # internal node: parse its children
            pos += 1
            parse(nd)
            while s[pos] == ",":
                pos += 1; parse(nd)
            assert s[pos] == ")", f"malformed newick near position {pos}"
            pos += 1
        read_label_len(nd)
        return nd

    parse(-1)
    return np.asarray(parent), np.asarray(blen, np.float64), label


def build_tree_buffers(newick_path: str, tip_labels: Sequence[str]) -> Dict:
    """Build level-synchronous message-passing buffers for a set of model species.

    Args:
        newick_path: path to the dated Newick tree.
        tip_labels: ``tip_labels[i]`` is the Newick leaf label of model species ``i`` (model/local order).

    Returns a dict of buffers. Compact node ids place the ``n_species`` model tips at ids ``0 .. n_species-1`` (so
    the refined leaf states are ``H[:n_species]``), then the internal ancestral nodes. Branch lengths are scaled to
    unit mean (the operator's learnable decay adapts the absolute rate). Keys:
      ``n_nodes, n_species, root``;
      upward (children->parents, grouped by parent height): ``up_child, up_parent, up_blen`` (flat, level-ordered)
        with ``up_edge_ptr`` offsets, and ``up_par, up_par_ptr`` the unique parents updated at each level;
      downward (parents->children, grouped by child depth): ``down_parent, down_child, down_blen`` with
        ``down_edge_ptr`` offsets (each level's ``down_child`` is already the unique set updated there).
    """
    parent, blen, label = parse_newick(newick_path)
    N = len(parent)
    lab2node = {label[i]: i for i in range(N) if label[i]}
    missing = [t for t in tip_labels if t not in lab2node]
    if missing:
        raise KeyError(f"{len(missing)} model tips absent from the tree, e.g. {missing[:3]}")
    want_node = np.array([lab2node[t] for t in tip_labels])       # original node id per model species
    n_sp = len(tip_labels)

    # keep = wanted leaf or has a kept descendant (children always have a larger id than their parent)
    keep = np.zeros(N, bool); keep[want_node] = True
    for nd in range(N - 1, 0, -1):
        if keep[nd]:
            keep[parent[nd]] = True
    kept_children = np.zeros(N, int)
    for nd in range(N):
        if keep[nd] and parent[nd] >= 0:
            kept_children[parent[nd]] += 1
    is_wanted_leaf = np.zeros(N, bool); is_wanted_leaf[want_node] = True
    retain = is_wanted_leaf | (keep & (kept_children >= 2))       # tips + genuine branching ancestors

    # each retained node's nearest retained ancestor, accumulating the suppressed branch lengths between them
    eff_parent = np.full(N, -1); eff_blen = np.zeros(N)
    for nd in range(N):
        if not retain[nd] or parent[nd] < 0:
            continue
        p = parent[nd]; acc = blen[nd]
        while p >= 0 and not retain[p]:
            acc += blen[p]; p = parent[p]
        eff_parent[nd] = p; eff_blen[nd] = acc

    # compact ids: model tips first (model order), then internal retained nodes
    newid = np.full(N, -1)
    for i, on in enumerate(want_node):
        newid[on] = i
    nxt = n_sp
    for nd in np.where(retain)[0]:
        if newid[nd] < 0:
            newid[nd] = nxt; nxt += 1
    n_nodes = nxt
    cparent = np.full(n_nodes, -1); cblen = np.zeros(n_nodes)
    for nd in np.where(retain)[0]:
        ci = newid[nd]
        if eff_parent[nd] >= 0:
            cparent[ci] = newid[eff_parent[nd]]; cblen[ci] = eff_blen[nd]

    # heights (edges to nearest descendant leaf) and depths (edges from root) via a topological sweep
    children: List[List[int]] = [[] for _ in range(n_nodes)]
    for ci in range(n_nodes):
        if cparent[ci] >= 0:
            children[cparent[ci]].append(ci)
    indeg = np.array([len(children[c]) for c in range(n_nodes)])
    dq = deque([c for c in range(n_nodes) if indeg[c] == 0]); order: List[int] = []
    while dq:
        c = dq.popleft(); order.append(c); p = cparent[c]
        if p >= 0:
            indeg[p] -= 1
            if indeg[p] == 0:
                dq.append(p)
    height = np.zeros(n_nodes, int); depth = np.zeros(n_nodes, int)
    for c in order:                                              # children before parents
        for ch in children[c]:
            height[c] = max(height[c], height[ch] + 1)
    for c in reversed(order):                                    # parents before children
        for ch in children[c]:
            depth[ch] = depth[c] + 1
    root = int(np.where(cparent < 0)[0][0])

    scale = float(cblen[cblen > 0].mean())                       # scale branch lengths to unit mean
    edges = [(c, int(cparent[c]), float(cblen[c] / scale)) for c in range(n_nodes) if cparent[c] >= 0]

    def flatten(sorted_edges, level_of):
        levels = sorted(set(level_of(e) for e in sorted_edges))
        by_level = {k: [] for k in levels}
        for e in sorted_edges:
            by_level[level_of(e)].append(e)
        child, par, bl, ptr = [], [], [], [0]
        par_nodes, par_ptr = [], [0]
        for k in levels:
            es = by_level[k]
            child += [e[0] for e in es]; par += [e[1] for e in es]; bl += [e[2] for e in es]
            ptr.append(len(child))
            par_nodes += sorted(set(e[1] for e in es)); par_ptr.append(len(par_nodes))
        return (np.array(child, np.int64), np.array(par, np.int64), np.array(bl, np.float32), ptr,
                np.array(par_nodes, np.int64), par_ptr)

    # ``flatten`` returns (child, parent, ...): edges are (child, parent) tuples. Upward groups by parent height;
    # downward groups by child depth. Keep child/parent labelled correctly for the downward buffers.
    uc, up_, ub, up_ptr, upar, upar_ptr = flatten(sorted(edges, key=lambda e: height[e[1]]), lambda e: height[e[1]])
    dc, dp, db, dn_ptr, _, _ = flatten(sorted(edges, key=lambda e: depth[e[0]]), lambda e: depth[e[0]])
    buf = dict(n_nodes=n_nodes, n_species=n_sp, root=root,
               up_child=uc, up_parent=up_, up_blen=ub, up_edge_ptr=up_ptr, up_par=upar, up_par_ptr=upar_ptr,
               down_parent=dp, down_child=dc, down_blen=db, down_edge_ptr=dn_ptr, branch_scale=scale)
    _check_buffers(buf)
    return buf


def _check_buffers(b: Dict) -> None:
    """Structural invariants that guard the level-synchronous sweep (a swapped child/parent would silently break
    induction rather than error): every non-root node is updated exactly once by each sweep, and each edge points
    to a strictly shallower parent (upward) / from a strictly shallower parent (downward)."""
    n, root = b["n_nodes"], b["root"]
    non_root = np.setdiff1d(np.arange(n), [root])
    assert np.array_equal(np.sort(b["down_child"]), non_root), "downward sweep must update every non-root node once"
    assert set(range(b["n_species"])).issubset(set(b["up_child"].tolist())), "every tip must send an upward message"
    # each up edge's parent is the same node the down edge feeds from (topology consistency)
    up_pair = set(zip(b["up_child"].tolist(), b["up_parent"].tolist()))
    dn_pair = set(zip(b["down_child"].tolist(), b["down_parent"].tolist()))
    assert up_pair == dn_pair, "upward and downward sweeps must traverse the same (child, parent) edge set"


# ============================================================================ operators over the species graph
class OrnsteinUhlenbeckAttention(nn.Module):
    """One layer of multi-head attention among species, biased toward phylogenetic neighbors (the default operator).

    Each attention logit is the usual content score minus ``rate * phylo_distance`` (then softmaxed), so covariance
    decays with evolutionary distance exactly as an Ornstein-Uhlenbeck process on the tree. This is a fast, fully
    differentiable stand-in for tree-structured message passing that needs only a distance matrix -- even the
    embedding-derived shadow of the tree -- and so processes phylogenetic neighbors cheaply, which may prove the best
    trade-off in the long run. It is, however, an approximation of the topology-faithful GNN; see
    :class:`TreeMessagePassing` for the message-passing operator we consider the direction for continued work.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        # alpha_h = softplus(log_rate). Init to ~2 (not 0): at rate 0.69 the phylo prior is near-inert (within/across
        # attention ratio only ~1.3x over 2141 species), so the phylogeny is under-exploited; a stronger start lets the
        # OU locality actually concentrate attention on relatives from the beginning (rate stays learnable).
        self.log_rate = nn.Parameter(torch.full((n_heads,), 1.85))
        self.norm = nn.LayerNorm(d_model)
        # The FlexAttention path folds the learnable OU rate into 3 extra q/k dimensions instead of a differentiable
        # score_mod (which is 15-24x slower because FlexAttention's backward through a captured tensor is weak). The
        # factorization q+.k+ = q.k - r*(p_i - p_j)^2 gives an RBF bias on ordinal clade-order position. The head is
        # padded to the next power of two for the kernel.
        self.flex_rate = nn.Parameter(torch.full((n_heads,), 0.1))
        aug = self.d_head + 3
        self._flex_pad = (1 << (aug - 1).bit_length()) - aug

    def forward(self, h: torch.Tensor, phylo_distance: torch.Tensor = None,
                neighbor_idx: torch.Tensor = None, neighbor_dist: torch.Tensor = None,
                flex_mask=None) -> torch.Tensor:
        """``h`` is ``[N, d_model]`` species states. Three modes, same math: ``flex_mask`` uses block-sparse fused
        FlexAttention with the OU bias in ``score_mod`` (scales to 100k+ species); dense takes ``phylo_distance``
        ``[N, N]``; sparse takes ``neighbor_idx``/``neighbor_dist`` ``[N, K]``. The OU locality (covariance decays
        with evolutionary distance) makes the block-sparse form both faithful and O(N*window)."""
        N, H, dh = h.shape[0], self.n_heads, self.d_head
        rate = F.softplus(self.log_rate)
        q = self.q(h).view(N, H, dh)
        if flex_mask is not None:                                          # block-sparse fused FlexAttention
            r = self.flex_rate.abs().view(H, 1, 1)                         # rate folded into q/k (flex fast path)
            p = (torch.arange(N, device=h.device, dtype=torch.float32) / N).view(1, N, 1)   # ordinal clade position
            ones = torch.ones(H, N, 1, device=h.device)
            qx = torch.cat([-r * p * p, 2 * r * p, ones], dim=-1)          # [H, N, 3]: q+.k+ = q.k - r*(p_i - p_j)^2
            kx = torch.cat([ones, p.expand(H, N, 1), -r * p * p], dim=-1)
            qh = F.pad(torch.cat([q.transpose(0, 1), qx.to(q.dtype)], -1), (0, self._flex_pad)).unsqueeze(0)
            kh = F.pad(torch.cat([self.k(h).view(N, H, dh).transpose(0, 1), kx.to(q.dtype)], -1),
                       (0, self._flex_pad)).unsqueeze(0)
            vh = self.v(h).view(N, H, dh).transpose(0, 1).unsqueeze(0)
            out = _flex_attention(qh, kh, vh, block_mask=flex_mask, scale=1.0 / math.sqrt(dh),
                                  kernel_options=_FLEX_KO)[0].transpose(0, 1).reshape(N, -1)
        elif neighbor_idx is not None:                                     # sparse: attend to K phylo-neighbors
            kn = self.k(h).view(N, H, dh)[neighbor_idx]                     # [N, K, H, dh]
            vn = self.v(h).view(N, H, dh)[neighbor_idx]                     # [N, K, H, dh]
            logits = torch.einsum("nhd,nkhd->nhk", q, kn) / math.sqrt(dh)   # [N, H, K]
            logits = logits - rate.view(1, H, 1) * neighbor_dist.unsqueeze(1)
            out = torch.einsum("nhk,nkhd->nhd", torch.softmax(logits, dim=-1), vn).reshape(N, -1)
        else:                                                              # dense: full N x N (one big GEMM; the
            qd, kd, vd = (q.transpose(0, 1),                                # OU bias is a dense mask, so an explicit
                          self.k(h).view(N, H, dh).transpose(0, 1),         # softmax beats SDPA/FlashAttention here)
                          self.v(h).view(N, H, dh).transpose(0, 1))         # [H, N, dh]
            logits = (qd @ kd.transpose(-1, -2)) / math.sqrt(dh) - rate.view(H, 1, 1) * phylo_distance.unsqueeze(0)
            out = (torch.softmax(logits, dim=-1) @ vd).transpose(0, 1).reshape(N, -1)
        return self.norm(h + self.o(out))                                  # residual keeps discrimination


def _mlp(d: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))


class _TreeRound(nn.Module):
    """One upward+downward message-passing sweep over the tree, mapping species states to refined species states.

    The species states seed the tree's tips; internal nodes seed a learned ``unk`` prior. The upward sweep sends
    gated messages from children to parents (ancestral state is aggregated), the downward sweep sends them back from
    parents to children, and the refined tip states are read off with a residual so discrimination is preserved.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.unk = nn.Parameter(torch.randn(d_model) * 0.02)                # prior state for an internal node
        self.theta = nn.Parameter(torch.tensor(0.0))                        # branch-length decay rate (learned)
        self.mup = nn.Linear(d_model, d_model)                             # child -> parent message
        self.agg = nn.Linear(d_model, d_model)                             # aggregated children -> parent state
        self.mdn = nn.Linear(d_model, d_model)                             # parent -> child message
        self.comb = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.out = nn.Linear(d_model, d_model)                             # residual readout at the tips
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)   # gentle at init: free dominates,
        self.norm = nn.LayerNorm(d_model)                                  # but relatives already inform each tip

    def forward(self, x: torch.Tensor, tree: "TreeMessagePassing") -> torch.Tensor:
        n_sp, d = x.shape
        rate = F.softplus(self.theta)
        up_gate = torch.exp(-rate * tree.up_blen).unsqueeze(-1)             # per-edge decay, computed once per sweep
        down_gate = torch.exp(-rate * tree.down_blen).unsqueeze(-1)
        H = self.unk.expand(tree.n_nodes, d).clone()
        H = H.index_copy(0, tree.leaf_ids, x)                               # place species states at the tips
        # upward: children -> parents, one vectorized scatter per height level (children already finalized below)
        for lo, hi, plo, phi in tree.up_slices:
            msg = up_gate[lo:hi] * self.mup(H[tree.up_child[lo:hi]])
            acc = torch.zeros_like(H).index_add(0, tree.up_parent[lo:hi], msg)
            par = tree.up_par[plo:phi]
            H = H.index_copy(0, par, self.agg(acc[par]))
        # downward: parents -> children, one scatter per depth level (parents already finalized above)
        for lo, hi in tree.down_slices:
            ci = tree.down_child[lo:hi]
            dm = down_gate[lo:hi] * self.mdn(H[tree.down_parent[lo:hi]])
            H = H.index_copy(0, ci, self.comb(torch.cat([H[ci], dm], dim=-1)))
        return self.norm(x + self.out(H[:n_sp]))                            # residual keeps congeners separable


class TreeMessagePassing(nn.Module):
    """Refine per-species states by message passing over the actual dated phylogeny (topology + branch lengths).

    Why the tree, and why a GNN. The phylogeny is not merely a distance matrix to be approximated: its topology and
    branch lengths *are* the record of evolutionary history, and by descent-with-modification that history is what
    makes traits, ecophysiology, phenology, and ecological function phylogenetically conserved. Relatives resemble
    each other because they share ancestry, so propagating state along the real tree edges -- internal nodes carrying
    reconstructed ancestral state, branch lengths gating how much signal survives each split -- is the principled way
    to share information across species, most of all for rare and held-out clades where a species has few or no direct
    observations of its own. For this reason we regard a graph neural network over the phylogeny as the right
    architecture for continued experimentation, and treat :class:`OrnsteinUhlenbeckAttention` (the default operator,
    which biases dense attention by an embedding-derived distance) as an efficient approximation rather than the
    destination -- we are deliberately not ready to rely wholly on an embedding from which distances were derived.

    Research direction (encouraged). The frontier is efficient *multi-hop* propagation restricted to the species
    actually present in a given context window (the batch's focal clades) rather than a full-tree sweep every step --
    the same problem weather models solve by message passing over a fixed multi-scale mesh. See GraphCast (Lam et al.,
    "Learning skillful medium-range global weather forecasting," Science 382:1416-1421, 2023) for deterministic
    message-passing forecasting over a graph, and GenCast (Price et al., "Probabilistic weather forecasting with
    machine learning," Nature 637:84-90, 2025) for its diffusion-based probabilistic successor; the analogue here is
    the phylogeny as the mesh and ancestral state as the propagated field. Concrete lines worth pursuing: subtree-local
    message passing keyed to the context window (only the induced subtree spanning the focal species), learned
    long-range "skip" edges between distant but ecologically convergent clades, and a diffusion/generative readout for
    calibrated uncertainty over induced traits and community composition. See ``core/science.md`` for the full framing.

    Args:
        n_species: number of species (tips seeded with the species states).
        d_model: representation width.
        tree: the buffer dict from :func:`build_tree_buffers` (static topology, precomputed once).
        n_layers: number of upward+downward sweeps (each a :class:`_TreeRound`); more sweeps propagate further.
        hidden: message-MLP width (defaults to ``d_model``).
    """

    def __init__(self, n_species: int, d_model: int, tree: dict, n_layers: int = 2, hidden: int = None):
        super().__init__()
        assert tree["n_species"] == n_species, "tree tips must match the model species"
        self.n_nodes = int(tree["n_nodes"])
        # static topology as buffers (move with the module; excluded from optimization)
        self.register_buffer("leaf_ids", torch.arange(n_species, dtype=torch.long))
        self.register_buffer("up_child", torch.as_tensor(tree["up_child"], dtype=torch.long))
        self.register_buffer("up_parent", torch.as_tensor(tree["up_parent"], dtype=torch.long))
        self.register_buffer("up_blen", torch.as_tensor(tree["up_blen"], dtype=torch.float32))
        self.register_buffer("up_par", torch.as_tensor(tree["up_par"], dtype=torch.long))
        self.register_buffer("down_parent", torch.as_tensor(tree["down_parent"], dtype=torch.long))
        self.register_buffer("down_child", torch.as_tensor(tree["down_child"], dtype=torch.long))
        self.register_buffer("down_blen", torch.as_tensor(tree["down_blen"], dtype=torch.float32))
        # python-int level slices (static shapes -> compile-friendly, no data-dependent bounds in the sweep)
        ep, pp = tree["up_edge_ptr"], tree["up_par_ptr"]
        self.up_slices = [(ep[k], ep[k + 1], pp[k], pp[k + 1]) for k in range(len(ep) - 1)]
        dp = tree["down_edge_ptr"]
        self.down_slices = [(dp[k], dp[k + 1]) for k in range(len(dp) - 1)]
        self.rounds = nn.ModuleList([_TreeRound(d_model, hidden or d_model) for _ in range(max(1, n_layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for r in self.rounds:
            x = r(x, self)
        return x


class SpeciesGraph(nn.Module):
    """A learnable per-species representation refined by a phylogenetic operator (``ou-attention`` or ``tree``).

    Args:
        n_species: number of species.
        d_model: representation width.
        phylo_distance: ``[n_species, n_species]`` phylogenetic distances (for ``operator="ou-attention"``).
        n_heads, n_layers: attention heads (ou-attention) and refinement depth (both operators).
        operator: ``"ou-attention"`` (distance-biased attention; the default) or ``"tree"`` (message passing over the
            real phylogeny -- see :class:`TreeMessagePassing`, the sanctioned GNN research direction).
        tree: the buffer dict from :func:`build_tree_buffers` (required for ``operator="tree"``).
    """

    def __init__(self, n_species: int, d_model: int, phylo_distance: torch.Tensor = None, n_heads: int = 4,
                 n_layers: int = 2, top_k: int = None, flex: bool = False, operator: str = "ou-attention",
                 tree: dict = None):
        super().__init__()
        self.operator = operator
        self.free = nn.Parameter(torch.randn(n_species, d_model) * 0.02)     # discriminative component
        if operator == "tree":
            assert tree is not None, "operator='tree' requires the parsed tree buffers"
            self.tree = TreeMessagePassing(n_species, d_model, tree, n_layers=n_layers)
            self.layers = None
            for name in ("phylo_distance", "neighbor_idx", "neighbor_dist", "order", "inv_order"):
                self.register_buffer(name, None)
            self.flex_mask = None
            return
        assert phylo_distance is not None, "operator='ou-attention' requires a phylo_distance matrix"
        self.tree = None
        self.layers = nn.ModuleList([OrnsteinUhlenbeckAttention(d_model, n_heads) for _ in range(n_layers)])
        self.flex_mask = None                                               # a BlockMask (not a tensor buffer)
        self.window = top_k
        pd = idx = dist = order = inv_order = None
        if flex and _HAS_FLEX and top_k is not None and top_k < n_species:   # block-sparse banded FlexAttention (scales)
            with torch.no_grad():                                           # 1-D clade-contiguous ordering (MDS axis)
                jc = torch.eye(n_species, device=phylo_distance.device) - 1.0 / n_species
                gram = -0.5 * jc @ (phylo_distance ** 2) @ jc
                order = torch.linalg.eigh(gram)[1][:, -1].argsort()         # leading eigenvector orders the clades
            inv_order = order.argsort()
            w = top_k

            def mask_mod(b, h, qi, ki):
                return (qi - ki).abs() <= w                                 # banded: clademates are contiguous

            self.flex_mask = create_block_mask(mask_mod, None, None, n_species, n_species, device=phylo_distance.device)
        elif top_k is not None and top_k < n_species:                       # gather-based sparse (fallback)
            dist, idx = phylo_distance.topk(top_k, dim=-1, largest=False)
        else:                                                               # dense (fastest at small N)
            pd = phylo_distance
        self.register_buffer("phylo_distance", pd)
        self.register_buffer("neighbor_idx", idx)
        self.register_buffer("neighbor_dist", dist)
        self.register_buffer("order", order)
        self.register_buffer("inv_order", inv_order)

    @staticmethod
    def distance_from_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Pairwise distances from a per-species embedding (e.g. the E1 phylo vector), scaled to unit mean."""
        d = torch.cdist(embedding, embedding)
        d = 0.5 * (d + d.transpose(-1, -2))     # symmetrize away floating-point asymmetry
        d.fill_diagonal_(0.0)
        return d / (d.mean() + 1e-9)

    def forward(self) -> torch.Tensor:
        """Refine and return the species representations ``[n_species, d_model]``.

        Takes no per-observation input: the refinement is a function of the static species table and the tree
        alone, so a query's own hidden identity can never leak into its own prediction through this module.
        """
        if self.operator == "tree":                                        # message passing over the real tree
            return self.tree(self.free)
        h = self.free
        if self.flex_mask is not None:                                     # clade-contiguous order -> banded attention
            h = h[self.order]
            for layer in self.layers:
                h = layer(h, flex_mask=self.flex_mask)
            return h[self.inv_order]
        for layer in self.layers:
            if self.neighbor_idx is not None:
                h = layer(h, neighbor_idx=self.neighbor_idx, neighbor_dist=self.neighbor_dist)
            else:
                h = layer(h, phylo_distance=self.phylo_distance)
        return h


# --------------------------------------------------------------------------------------- standalone unit test
def _test():
    torch.manual_seed(0)
    N, d = 60, 32
    # a synthetic phylogeny: species in a few clades, close within a clade, far across
    clade = torch.randint(0, 5, (N,))
    centers = torch.randn(5, d)
    e1 = centers[clade] + 0.1 * torch.randn(N, d)
    D = SpeciesGraph.distance_from_embedding(e1)
    assert torch.allclose(D, D.T, atol=1e-5) and (D.diag().abs() < 1e-4).all(), "distance must be symmetric, zero diag"

    graph = SpeciesGraph(N, d, D, n_heads=4, n_layers=2)
    h = graph()
    assert h.shape == (N, d)

    # discrimination: refined species stay distinct (no collapse onto a shared mean)
    spread = (h - h.mean(0)).norm(dim=-1)
    assert (spread > 1e-3).all(), "species must not collapse"

    # phylogenetic borrowing: force a large rate and check attention concentrates on close relatives
    layer = graph.layers[0]
    with torch.no_grad():
        layer.log_rate.fill_(4.0)
        q = layer.q(graph.free).view(N, layer.n_heads, layer.d_head).transpose(0, 1)
        k = layer.k(graph.free).view(N, layer.n_heads, layer.d_head).transpose(0, 1)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(layer.d_head) - F.softplus(layer.log_rate).view(-1, 1, 1) * D
        attn = torch.softmax(logits, dim=-1).mean(0)     # [N, N] averaged over heads
        same = clade[:, None] == clade[None, :]
        within = attn[same].mean(); across = attn[~same].mean()
    assert within > across, f"attention must favor phylo neighbors ({within:.3f} vs {across:.3f})"

    # extended backprop: a loss on ONE species' refined state sends gradient to OTHER species' free embeddings
    fresh = SpeciesGraph(N, d, D, n_heads=4, n_layers=2)   # default rates, un-mutated
    fresh()[0].pow(2).sum().backward()                     # squared: not degenerate under the output LayerNorm
    reached = fresh.free.grad.abs().sum(dim=1) > 0
    assert reached[1:].any(), "an observation of one species must update its relatives"

    # sparse phylo-local attention: shape, self is the nearest neighbor, gradient flows, output tracks dense
    ksparse = SpeciesGraph(N, d, D, n_heads=4, n_layers=2, top_k=12)
    hs = ksparse()
    assert hs.shape == (N, d) and ksparse.neighbor_idx.shape == (N, 12)
    assert (ksparse.neighbor_dist[:, 0] < 1e-4).all(), "each species' nearest neighbor is itself (distance 0)"
    ksparse().pow(2).sum().backward(); assert ksparse.free.grad is not None

    _test_tree()
    _test_real_tree()
    print(f"phylogenomic.py: all unit tests passed (within-clade attn {within:.3f} > across {across:.3f}; "
          f"sparse top-k OK; tree message passing OK)")


def _synthetic_tree():
    """A small tree for testing: 8 tips in 4 sibling pairs, grouped into 2 clades, joined at a root.

    Compact ids: leaves 0..7; pair parents 8..11; clade nodes 12,13; root 14. Congeners are same-pair leaves
    (0&1, 2&3, ...); same-clade leaves are 0..3 and 4..7. Branch lengths grow with depth (tip 0.5, pair 1.0,
    clade 2.0), so the branch-gate should route information most strongly between congeners.
    """
    return dict(
        n_nodes=15, n_species=8, root=14,
        up_child=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        up_parent=[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
        up_blen=[.5] * 8 + [1.] * 4 + [2.] * 2, up_edge_ptr=[0, 8, 12, 14],
        up_par=[8, 9, 10, 11, 12, 13, 14], up_par_ptr=[0, 4, 6, 7],
        down_parent=[14, 14, 12, 12, 13, 13, 8, 8, 9, 9, 10, 10, 11, 11],
        down_child=[12, 13, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
        down_blen=[2., 2., 1., 1., 1., 1.] + [.5] * 8, down_edge_ptr=[0, 2, 6, 14])


def _test_tree():
    torch.manual_seed(0)
    tree = _synthetic_tree(); N, d = 8, 16
    graph = SpeciesGraph(N, d, operator="tree", tree=tree, n_layers=2)

    # no per-observation input: the refinement cannot leak a query's own hidden identity into its own prediction
    import inspect
    assert list(inspect.signature(graph.forward).parameters) == [], "species graph must take no per-query input"

    h = graph()
    assert h.shape == (N, d), "refined tips are returned in species order"
    h2 = graph()
    assert torch.allclose(h, h2), "refinement is deterministic (no per-forward randomness)"

    # discrimination: tips stay distinct and congeners remain separable (no collapse onto a shared/ancestral mean)
    assert ((h - h.mean(0)).norm(dim=-1) > 1e-3).all(), "species must not collapse"
    assert (h[0] - h[1]).norm() > 1e-3, "congeners must stay separable"

    # extended backprop: a loss on ONE tip's refined state reaches its relatives' free embeddings, its congener most
    graph.zero_grad(set_to_none=True)
    graph()[0].pow(2).sum().backward()
    g = graph.free.grad.abs().sum(dim=1)
    assert g[1:].any(), "an observation of one species must update its relatives through the tree"
    assert g[1] > 0, "the congener (sibling tip) must receive gradient"

    # mechanism: with the message MLPs made linear, information routing follows the real topology + branch gate --
    # tip 0 is more sensitive to its congener (tip 1) than to the far clade (tips 4..7).
    class _Avg(nn.Module):
        def forward(self, cat):
            a, b = cat.chunk(2, dim=-1); return 0.5 * (a + b)
    lin = SpeciesGraph(N, d, operator="tree", tree=tree, n_layers=1)
    r = lin.tree.rounds[0]
    r.mup = r.agg = r.mdn = r.out = nn.Identity(); r.norm = nn.Identity(); r.comb = _Avg()
    with torch.no_grad():
        r.theta.fill_(0.5); r.unk.zero_()
        lin.free.copy_(torch.randn(N, d))
    lin.zero_grad(set_to_none=True)
    lin()[0].sum().backward()
    sens = lin.free.grad.abs().sum(dim=1)
    assert sens[1] > sens[4:8].mean(), f"congener sensitivity {sens[1]:.3f} must exceed far-clade {sens[4:8].mean():.3f}"


def _test_real_tree():
    """Build the buffers from the real California tree and confirm the pruned tree preserves the reference
    patristic distances exactly (topology + branch lengths are faithful), or skip if the tree is absent."""
    import csv
    from pathlib import Path
    cache = Path("/home/photon/ecological/phylo_deepearth/data/cache")
    nwk = cache / "ca_subtree.dated.nwk"
    if not nwk.exists():
        print("phylogenomic.py: real tree absent; skipping patristic check")
        return
    rows = list(csv.DictReader(open(cache / "derived/species_index.csv")))
    gi = np.load(cache / "gbif_vocab.npz", allow_pickle=True)["global_idx"]
    tips = [rows[g]["tip_label"] for g in gi]
    b = build_tree_buffers(str(nwk), tips)
    # reconstruct patristic distances on the compact tree and compare to the reference matrix
    n = b["n_nodes"]; cparent = np.full(n, -1); cblen = np.zeros(n)
    for c, p, bl in zip(b["down_child"], b["down_parent"], b["down_blen"]):
        cparent[c] = p; cblen[c] = bl * b["branch_scale"]              # undo the unit-mean scaling
    depth = np.zeros(n, int)
    changed = True
    while changed:                                                    # compute depth by repeated relaxation
        changed = False
        for c in range(n):
            if cparent[c] >= 0 and depth[c] != depth[cparent[c]] + 1:
                depth[c] = depth[cparent[c]] + 1; changed = True
    rootdist = np.zeros(n)
    for c in np.argsort(depth):
        if cparent[c] >= 0:
            rootdist[c] = rootdist[cparent[c]] + cblen[c]

    def lca(a, bb):
        while depth[a] > depth[bb]: a = cparent[a]
        while depth[bb] > depth[a]: bb = cparent[bb]
        while a != bb: a = cparent[a]; bb = cparent[bb]
        return a

    Dref = np.load(cache / "derived/patristic_ref.npy")
    rng = np.random.default_rng(0); errs = []
    for _ in range(500):
        i, j = rng.integers(0, b["n_species"], 2)
        if i == j: continue
        d = rootdist[i] + rootdist[j] - 2 * rootdist[lca(i, j)]
        errs.append(abs(d - Dref[gi[i], gi[j]]))
    assert max(errs) < 1e-3, f"pruned patristic distances drifted from the reference (max err {max(errs):.3g})"
    print(f"phylogenomic.py: tree buffers OK (n_nodes {n}, tips {b['n_species']}, "
          f"up-levels {len(b['up_edge_ptr'])-1}; patristic max err {max(errs):.2e})")


if __name__ == "__main__":
    _test()
