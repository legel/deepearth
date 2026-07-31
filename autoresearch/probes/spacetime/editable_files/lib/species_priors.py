"""Species-level priors and functional traits, for THIS loop only.

Deliberately a copy. These two loaders also exist in `autoresearch/probes/biological/`, and the spacetime probe
used to import them across loop boundaries -- which made a spacetime result silently depend on another
loop's code. Each autoresearch loop must stand alone: its own probe, its own data, its own evals, so a
change in one loop can never move another loop's numbers without anyone noticing.

If the underlying cache format changes, both copies need updating. That is the cost of independence, and
it is cheaper than a cross-loop coupling nobody can see.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# build_tree_buffers is copied in below rather than imported from the biological encoder. Under the
# dependency DAG a probe loop may depend on its OWN leaf and on nothing sideways: importing the
# biological encoder from the spacetime loop is a sibling edge, and a change there would silently move


def load_species(cache: str):
    """species prior (E1) + family labels + latent-clade tree buffers -- all species-level, ~seconds."""
    cachep = Path(cache)
    vocab = np.load(cachep / "gbif_vocab.npz", allow_pickle=True)
    gidx = vocab["global_idx"]
    E1 = torch.tensor(vocab["E1"].astype(np.float32))                     # [N, 2048] frozen species prior
    rows = list(csv.DictReader(open(cachep / "derived/species_index.csv")))
    tip_labels = [rows[i]["tip_label"] for i in gidx]
    family = np.array([rows[i]["family"] for i in gidx])
    fam_id = torch.tensor(np.unique(family, return_inverse=True)[1], dtype=torch.long)
    nwk = cachep / "ca_subtree.dated.nwk"                                 # latent-clade tree buffers (in-tree tips)
    toks = set(re.findall(r"[^(),:;\s]+", open(nwk).read()))
    pairs = [(i, tl) for i, tl in enumerate(tip_labels) if tl in toks]
    tree = build_tree_buffers(str(nwk), [tl for _, tl in pairs])
    tip_row = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    return E1, fam_id, tree, tip_row, gidx

def load_trait(cache: str, gidx, key: str, dev):
    """Load a functional-trait target aligned to the 2141-species vocab.

    Returns (kind, Y, obs, nclass): kind in {'cat','multi'}. `cat` -> Y int64 [N] single label (full
    coverage), metric = trait-NN accuracy. `multi` -> Y float32 [N,L] + obs bool [N] (partial coverage),
    metric = trait-NN micro-AP over observed held-out species. This is the biology axis the phylo-seed
    may NOT already saturate (family is redundant; a trait can be additive)."""
    tr = np.load(Path(cache) / "derived/traits_syn.npz", allow_pickle=True)
    if key.startswith("cat_"):
        y = torch.tensor(tr[key][gidx].astype(np.int64)).to(dev)
        nclass = max(len(tr["catvocab_" + key[4:]]), int(tr[key][gidx].max()) + 1)  # data has a label beyond vocab len (cat_form) -> size head to observed max
        obs = (y >= 0).to(dev)                                  # cat traits here are full-coverage
        return "cat", y, obs, nclass
    if key.startswith("multi_"):
        base = key[6:]
        Y = torch.tensor(tr[key][gidx].astype(np.float32)).to(dev)     # [N, L] 0/1
        obs = torch.tensor(tr["multiobs_" + base][gidx]).to(dev)        # [N] species with a real observation
        return "multi", Y, obs, Y.shape[1]
    if key.startswith("num_"):
        base = key[4:]
        y = torch.tensor(tr[key][gidx].astype(np.float32)).to(dev)     # [N] continuous morphology/climate trait
        obs = torch.tensor(tr["numobs_" + base][gidx]).to(dev)         # [N] species with a real measurement
        # log1p-compress heavy-tailed size/rain/elev traits so NN correlation is not dominated by a few giants
        if base in ("height_max", "width_max", "rain_max", "rain_min", "elev_max", "elev_min", "lep_support"):
            y = torch.sign(y) * torch.log1p(y.abs())
        return "num", y, obs, 1
    raise ValueError(f"unknown trait key {key} (use cat_*, multi_* or num_*)")


def parse_newick(path: str):
    """Parse a Newick tree into ``parent``, ``branch_length``, ``label`` arrays (``parent == -1`` is the root).

    Recursive-descent (the tree nests only ~50 deep); handles a leading rooting comment (e.g. ``[&U]``) and
    optional internal-node labels.
    """
    s = open(path).read().strip()
    if s.startswith("[&"): s = s[s.index("]") + 1:].strip()   # strip a rooting comment, e.g. "[&U]"
    if s.endswith(";"): s = s[:-1]
    parent: List[int] = []; blen: List[float] = []; label: List[str] = []; pos = 0

    def node(p: int) -> int:
        parent.append(p); blen.append(0.0); label.append("")
        return len(parent) - 1

    def read_label_len(nd: int) -> None:
        nonlocal pos
        start = pos
        while pos < len(s) and s[pos] not in "(),:;": pos += 1
        if pos > start: label[nd] = s[start:pos]
        if pos < len(s) and s[pos] == ":":
            pos += 1; start = pos
            while pos < len(s) and s[pos] not in "(),:;": pos += 1
            blen[nd] = float(s[start:pos])

    def parse(p: int) -> int:
        nonlocal pos
        nd = node(p)
        if s[pos] == "(":                                    # internal node: parse its children
            pos += 1; parse(nd)
            while s[pos] == ",":
                pos += 1; parse(nd)
            assert s[pos] == ")", f"malformed newick near position {pos}"
            pos += 1
        read_label_len(nd)
        return nd

    parse(-1)
    return np.asarray(parent), np.asarray(blen, np.float64), label


def _check_buffers(b: Dict) -> None:
    """Structural invariants guarding the sweep (a swapped child/parent breaks induction silently rather than erroring):
    every non-root node updated exactly once per sweep, and up/down traverse the same (child, parent) edge set."""
    n, root = b["n_nodes"], b["root"]
    non_root = np.setdiff1d(np.arange(n), [root])
    assert np.array_equal(np.sort(b["down_child"]), non_root), "downward sweep must update every non-root node once"
    assert set(range(b["n_species"])).issubset(set(b["up_child"].tolist())), "every tip must send an upward message"
    # each up edge's parent is the same node the down edge feeds from (topology consistency)
    up_pair = set(zip(b["up_child"].tolist(), b["up_parent"].tolist()))
    dn_pair = set(zip(b["down_child"].tolist(), b["down_parent"].tolist()))
    assert up_pair == dn_pair, "upward and downward sweeps must traverse the same (child, parent) edge set"


def build_tree_buffers(newick_path: str, tip_labels: Sequence[str]) -> Dict:
    """Build level-synchronous message-passing buffers for model species (``tip_labels[i]`` = Newick label of species i).

    Compact ids place the ``n_species`` model tips at ``0..n_species-1`` (refined leaf states = ``H[:n_species]``),
    then internal nodes; branch lengths scaled to unit mean (the operator's learnable decay adapts the absolute rate).
    Keys: ``n_nodes, n_species, root``; upward (children->parents, grouped by parent height) ``up_child, up_parent,
    up_blen`` + ``up_edge_ptr`` offsets, ``up_par, up_par_ptr`` (unique parents updated per level); downward
    (parents->children, grouped by child depth) ``down_parent, down_child, down_blen`` + ``down_edge_ptr`` offsets.
    """
    parent, blen, label = parse_newick(newick_path)
    N = len(parent)
    lab2node = {label[i]: i for i in range(N) if label[i]}
    missing = [t for t in tip_labels if t not in lab2node]
    if missing:
        raise KeyError(f"{len(missing)} model tips absent from the tree, e.g. {missing[:3]}")
    want_node = np.array([lab2node[t] for t in tip_labels])       # original node id per model species
    n_sp = len(tip_labels)
    # keep = wanted leaf or has a kept descendant (a child always has a larger id than its parent)
    keep = np.zeros(N, bool); keep[want_node] = True
    for nd in range(N - 1, 0, -1):
        if keep[nd]: keep[parent[nd]] = True
    kept_children = np.zeros(N, int)
    for nd in range(N):
        if keep[nd] and parent[nd] >= 0: kept_children[parent[nd]] += 1
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
        if newid[nd] < 0: newid[nd] = nxt; nxt += 1
    n_nodes = nxt
    cparent = np.full(n_nodes, -1); cblen = np.zeros(n_nodes)
    for nd in np.where(retain)[0]:
        ci = newid[nd]
        if eff_parent[nd] >= 0: cparent[ci] = newid[eff_parent[nd]]; cblen[ci] = eff_blen[nd]
    # heights (edges to nearest descendant leaf) and depths (edges from root) via a topological sweep
    children: List[List[int]] = [[] for _ in range(n_nodes)]
    for ci in range(n_nodes):
        if cparent[ci] >= 0: children[cparent[ci]].append(ci)
    indeg = np.array([len(children[c]) for c in range(n_nodes)])
    dq = deque([c for c in range(n_nodes) if indeg[c] == 0]); order: List[int] = []
    while dq:
        c = dq.popleft(); order.append(c); p = cparent[c]
        if p >= 0:
            indeg[p] -= 1
            if indeg[p] == 0: dq.append(p)
    height = np.zeros(n_nodes, int); depth = np.zeros(n_nodes, int)
    for c in order:                                              # children before parents
        for ch in children[c]: height[c] = max(height[c], height[ch] + 1)
    for c in reversed(order):                                    # parents before children
        for ch in children[c]: depth[ch] = depth[c] + 1
    root = int(np.where(cparent < 0)[0][0])
    scale = float(cblen[cblen > 0].mean())                       # scale branch lengths to unit mean
    edges = [(c, int(cparent[c]), float(cblen[c] / scale)) for c in range(n_nodes) if cparent[c] >= 0]

    def flatten(sorted_edges, level_of):
        levels = sorted(set(level_of(e) for e in sorted_edges))
        by_level = {k: [] for k in levels}
        for e in sorted_edges: by_level[level_of(e)].append(e)
        child, par, bl, ptr = [], [], [], [0]
        par_nodes, par_ptr = [], [0]
        for k in levels:
            es = by_level[k]
            child += [e[0] for e in es]; par += [e[1] for e in es]; bl += [e[2] for e in es]
            ptr.append(len(child))
            par_nodes += sorted(set(e[1] for e in es)); par_ptr.append(len(par_nodes))
        return (np.array(child, np.int64), np.array(par, np.int64), np.array(bl, np.float32), ptr,
                np.array(par_nodes, np.int64), par_ptr)

    # edges are (child, parent) tuples: upward groups by parent height, downward by child depth.
    uc, up_, ub, up_ptr, upar, upar_ptr = flatten(sorted(edges, key=lambda e: height[e[1]]), lambda e: height[e[1]])
    dc, dp, db, dn_ptr, _, _ = flatten(sorted(edges, key=lambda e: depth[e[0]]), lambda e: depth[e[0]])
    buf = dict(n_nodes=n_nodes, n_species=n_sp, root=root,
               up_child=uc, up_parent=up_, up_blen=ub, up_edge_ptr=up_ptr, up_par=upar, up_par_ptr=upar_ptr,
               down_parent=dp, down_child=dc, down_blen=db, down_edge_ptr=dn_ptr, branch_scale=scale)
    _check_buffers(buf)
    return buf
