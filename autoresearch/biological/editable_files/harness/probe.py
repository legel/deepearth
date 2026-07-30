"""Standalone biological-encoder probe -- train + evaluate the species-graph IN ISOLATION.

No fusion model, no 621k observations, no full benchmark suite -- just the ~2141-species table + the
phylogeny. Measures the encoder's science (science.md rules 9, 25, 27): does refining a species through
its phylogenetic relatives let you impute its biology (family) for held-out species? Seconds, not minutes.

Objective (standalone `bio_gain`): family-NN accuracy of the graph-refined species embedding MINUS the raw
seed, on held-out species reconstructed from relatives (rule-25 mask). >0 ⟹ the phylogeny adds
family-discriminative structure the seed lacks. Reuses SpeciesGraph unchanged (no core edit).

  python -m deepearth.autoresearch.biological.editable_files.harness.probe --cache_dir data/deepcal --steps 400
"""
import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deepearth.encoders.biological.phylogenomic import SpeciesGraph, build_tree_buffers


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


def nn_family_acc(emb: torch.Tensor, fam_id: torch.Tensor, test: torch.Tensor) -> float:
    """held-out species -> nearest TRAIN species (cosine) -> predict its family. accuracy over held-out."""
    train = ~test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[test], dim=-1)
    nn = (ett @ et.t()).argmax(-1)
    return (fam_id[train][nn] == fam_id[test]).float().mean().item()


# --- VISION seed (rule-26 root cause; flag-gated; default probe path never calls this) -----------
def load_vision_seed(cache: str, gidx, fuse: bool):
    """Per-species VISION seed = mean DINOv3 (1024-d) over each species' observations, aligned to the
    2141-species vocab (token shards key `species_local` == vocab row index). VISION is ORTHOGONAL to the
    dated-tree topology in a way the E1 text prior is not: the champion E1 is ~0.89 family-NN (itself
    phylo-derived), so the tree re-encodes what E1 already has (redundancy law). Appearance is only weakly
    family-coherent (mean-DINO ~0.47 family-NN), leaving real headroom for the phylo operator to ADD family
    structure. `fuse` -> concat mean-DINO with mean-BioCLIP-image (768-d), each L2-normed (vision(+)text).
    Species with zero observations keep a zero seed (small minority; random holdout averages over them)."""
    z = np.load(Path(cache) / "derived/species_vision_seed.npz")
    D = torch.tensor(z["dino_mean"].astype(np.float32))                  # [N, 1024] mean DINOv3
    if fuse:
        B = torch.tensor(z["bio_mean"].astype(np.float32))              # [N, 768] mean BioCLIP-image
        seed = torch.cat([F.normalize(D, dim=-1), F.normalize(B, dim=-1)], dim=-1)
    else:
        seed = D
    return seed


# --- TRAIT axis (flag-gated; default probe path never calls these) -------------------------------
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


def nn_trait_acc(emb, y, obs, test):
    """held-out species -> nearest OBSERVED-TRAIN species (cosine) -> predict its categorical trait."""
    train = obs & (~test)
    tst = obs & test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[tst], dim=-1)
    nn = (ett @ et.t()).argmax(-1)
    return (y[train][nn] == y[tst]).float().mean().item()


def nn_trait_ap(emb, Y, obs, test, k=5):
    """held-out multi-label trait: mean over k nearest OBSERVED-TRAIN neighbours -> micro-AP vs prior.

    Positive micro-AP over the label-prior baseline = the embedding geometry carries this trait's
    structure. Reported gain = refined_AP - seed_AP with an identical NN protocol (fair control)."""
    from sklearn.metrics import average_precision_score
    train = obs & (~test)
    tst = obs & test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[tst], dim=-1)
    sim = ett @ et.t()
    topk = sim.topk(min(k, sim.shape[1]), dim=-1).indices
    pred = Y[train][topk].mean(1)                              # [ntest, L]
    yt = Y[tst].cpu().numpy().ravel()
    pr = pred.cpu().numpy().ravel()
    try:
        return float(average_precision_score(yt, pr))
    except Exception:
        return float("nan")


def nn_trait_num(emb, y, obs, test, k=5):
    """held-out continuous trait: predict mean of k nearest OBSERVED-TRAIN neighbours; score = Spearman
    rank-correlation between predicted and true value over held-out observed species. Rank-based so it is
    robust to the log-compression and directly comparable across seeds/sources (0=no signal, 1=perfect)."""
    from scipy.stats import spearmanr
    train = obs & (~test)
    tst = obs & test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[tst], dim=-1)
    sim = ett @ et.t()
    topk = sim.topk(min(k, sim.shape[1]), dim=-1).indices
    pred = y[train][topk].float().mean(1)
    yt = y[tst].detach().cpu().numpy()
    pr = pred.detach().cpu().numpy()
    try:
        r = spearmanr(yt, pr).correlation
        return float(r) if r == r else 0.0
    except Exception:
        return 0.0


# --- INTERACTION axis (rule 27; flag-gated; default probe path never calls these) ----------------
def load_interactions(cache: str, gidx, dev):
    """Plant->pollinator bipartite interactions + the pollinator seed, both aligned to their own vocab.

    Rule 27's data: a plant's observed pollinator set (GloBI marginals, top-K per plant). Returns
      Pmap   [Np]      plant vocab rows (subset of the 2141-plant graph) that HAVE >=1 pollinator,
      Ptgt   [Np, Nq]  0/1 target: which pollinators visit each plant (the bipartite adjacency),
      poll_text [Nq,1024] BioCLIP-2.5 text prior for the Nq pollinators actually observed (the 2nd seed).
    Only pollinators that appear in >=1 in-vocab plant's list are kept, so the pollinator graph is over
    the interacting pollinator set. This is the cross-tree object a single plant phylo-graph cannot hold."""
    cachep = Path(cache)
    z = np.load(cachep / "gbif_pollinator_dist.npz", allow_pickle=True)
    pi = z["plant_idx"]                                          # [P] global species_index rows
    ppi = z["marg_poll_idx"]                                     # [P, K] pollinator vocab ids (-1 pad)
    pfr = z["marg_poll_frq"] if "marg_poll_frq" in z else None   # [P, K] visitation frequency per (plant, pollinator)
    npo = z["marg_npoll"]                                        # [P] valid count per row
    g2l = -np.ones(int(gidx.max()) + 1, dtype=np.int64); g2l[gidx] = np.arange(len(gidx))
    plant_local = g2l[pi]                                        # plant -> graph row (-1 if not in E1 vocab)
    keep_p = (plant_local >= 0) & (npo > 0)                      # in-vocab plants that have interactions
    rows = np.where(keep_p)[0]
    poll_used = np.unique(ppi[rows][ppi[rows] >= 0])             # pollinators appearing for kept plants
    p2c = -np.ones(int(ppi.max()) + 1, dtype=np.int64); p2c[poll_used] = np.arange(len(poll_used))
    Np, Nq = len(rows), len(poll_used)
    Ptgt = np.zeros((Np, Nq), dtype=np.float32)
    Pfrq = np.zeros((Np, Nq), dtype=np.float32)                  # frequency-weighted adjacency (for a richer covis edge weight)
    for r, gr in enumerate(rows):
        k = npo[gr]
        cols = p2c[ppi[gr, :k]]
        valid = cols >= 0
        Ptgt[r, cols[valid]] = 1.0
        if pfr is not None:
            Pfrq[r, cols[valid]] = pfr[gr, :k][valid]
    Pmap = torch.tensor(plant_local[rows], dtype=torch.long).to(dev)
    Ptgt = torch.tensor(Ptgt).to(dev)
    Pfrq = torch.tensor(Pfrq).to(dev)
    poll_text_all = np.load(cachep / "pollinator_taxon_text_emb.npy").astype(np.float32)
    poll_text = torch.tensor(poll_text_all[poll_used]).to(dev)  # [Nq, 1024] 2nd-tree seed
    return Pmap, Ptgt, poll_text, Pfrq


def interaction_ap(P, Q, W, Pmap, Ptgt, test, k_train=None, qcol=None):
    """Held-out interaction recovery: for each held-out plant, score all pollinators via the bilinear form
    (P[plant] @ W @ Q.T) and rank; report micro-AP against the true pollinator set. `test` masks held-out
    PLANTS (their relatives stay in train). `qcol` (optional bool [Nq]) restricts scoring to held-out
    POLLINATORS -- for the bidirectional protocol the recovered object is held-out-plant x held-out-pollinator,
    so both trees must reconstruct their own held-out members. Fair, protocol-identical across all controls."""
    from sklearn.metrics import average_precision_score
    scores = (P[Pmap] @ W) @ Q.t()                              # [Np, Nq] bilinear plant->pollinator affinity
    tst = test[Pmap]
    Yt, Pr = Ptgt[tst], scores[tst]
    if qcol is not None:
        Yt, Pr = Yt[:, qcol], Pr[:, qcol]                       # score only held-out pollinator columns
    yt = Yt.detach().cpu().numpy().ravel()
    pr = Pr.detach().cpu().numpy().ravel()
    try:
        return float(average_precision_score(yt, pr))
    except Exception:
        return float("nan")


def run_interaction(a, dev):
    """Rule-27 two-tree bilinear probe. A plant graph and a SEPARATE pollinator graph each phylo-refine
    their own kingdom; a bilinear head decodes plant->pollinator interactions against the REFINED pollinator
    embeddings. Signal flows between two trees -- structurally impossible for one phylo-graph.

    Controls (all same head, same holdout, same protocol):
      two_tree  : refine BOTH plant & pollinator graphs   (the mechanism)
      one_tree  : refine ONLY plants, pollinators = raw seed   (isolates the 2nd tree)
      seed_only : neither graph refines (both raw seed)        (untouched-seed baseline)
    Reported cross_tree_gain = AP(two_tree) - AP(one_tree): the marginal value of the pollinator tree,
    the piece a single plant phylo-graph cannot express. bio_gain = AP(two_tree) - AP(seed_only)."""
    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    E1, tip_row = E1.to(dev), tip_row.to(dev)
    N = E1.shape[0]
    Pmap, Ptgt, poll_text, Pfrq = load_interactions(a.cache_dir, gidx, dev)
    Nq = poll_text.shape[0]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    test = (torch.rand(N, generator=g) < a.holdout).to(dev)     # held-out PLANTS (relatives remain in train)
    train = ~test
    # rule-27 bidirectional: also hold out a fraction of POLLINATORS -> recover their reps from pollinator relatives.
    # The held-out plants' interactions are then decoded against masked pollinators reconstructed by the 2nd tree,
    # so signal must flow BOTH ways (plant tree recovers plants, pollinator tree recovers pollinators).
    gq = torch.Generator(device="cpu").manual_seed(a.seed + 10007)
    qtest = (torch.rand(Nq, generator=gq) < a.holdout).to(dev) if a.bidir_mask else torch.zeros(Nq, dtype=torch.bool, device=dev)
    qtrain = ~qtest

    plant = SpeciesGraph(N, a.d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=E1).to(dev)                # tree1: plant phylogeny
    if a.poll_dist == "covis":                                  # tree2 from co-visitation over TRAIN plants (no held-out leak)
        base = Pfrq if a.covis_weighted else Ptgt              # weighted: shared-visitation MASS (seed-orthogonal, richer than binary)
        A = base.clone(); A[~train[Pmap]] = 0.0                # (frequency-weighted) bipartite adjacency, TRAIN plants only
        if a.bidir_mask:                                        # a masked pollinator's own covis edges must be inferred from relatives too
            A[:, qtest] = 0.0
        if a.covis_weighted:                                   # log1p compresses the heavy-tailed GloBI visit counts before co-occurrence
            A = torch.log1p(A)
        cov = A.t() @ A                                         # [Nq,Nq] shared-plant (co-visitation mass) between pollinators
        Fn = F.normalize(cov + 1e-6 * torch.eye(Nq, device=dev), dim=-1)
        poll_dist = SpeciesGraph.distance_from_embedding(Fn)   # pollinators sharing plants are neighbors (seed-orthogonal)
    elif a.poll_dist == "taxo":                                 # tree2 = TAXONOMIC ultrametric: agglomerative dendrogram over the
        # pollinator text prior, cophenetic (tree-depth-to-common-ancestor) distance. On-box the discrete GBIF rank labels
        # (order/family/genus) are NOT provisioned, so the strongest taxonomy available is the hierarchy the BioCLIP-2.5
        # taxonomy-string embedding encodes, quantised into a TREE by average-linkage. This yields a genuine tree topology
        # (clade depths) but note it is a monotone re-expression of the SAME seed axis, so it is expected to be
        # seed-correlated rather than seed-orthogonal (unlike covis). Additive; only fires under --poll_dist taxo.
        from scipy.cluster.hierarchy import linkage, cophenet
        from scipy.spatial.distance import squareform
        tn = F.normalize(poll_text, dim=-1)
        Dt = (1.0 - tn @ tn.t()).clamp(min=0.0).cpu().numpy()  # text-cosine distance among pollinators
        Dt = 0.5 * (Dt + Dt.T); np.fill_diagonal(Dt, 0.0)
        Z = linkage(squareform(Dt, checks=False), method="average")            # taxonomic dendrogram
        Dco = squareform(cophenet(Z)).astype(np.float32)       # cophenetic = tree-depth-to-common-ancestor (ultrametric)
        poll_dist = torch.tensor(Dco / (Dco.max() + 1e-9), device=dev)         # normalised patristic-style tree distance
    elif a.poll_dist == "realtree":                             # tree2 = REAL DATED pollinator phylogeny (pollitree pipeline):
        # OToL induced-subtree topology + taxonomic-rank Myr calibration over the recovered pollinator names,
        # precomputed and aligned to poll_used in derived/pollinator_distance_real.npy. This is a genuine
        # cross-tree structure NOT derivable from the plant graph nor a monotone re-expression of the text seed.
        Dreal = np.load(Path(a.cache_dir) / "derived/pollinator_distance_real.npy").astype(np.float32)
        assert Dreal.shape == (Nq, Nq), (Dreal.shape, Nq)       # must be aligned to the probe's poll_used order
        poll_dist = torch.tensor(Dreal / (Dreal.max() + 1e-9), device=dev)  # normalised patristic tree distance
    else:
        poll_dist = SpeciesGraph.distance_from_embedding(F.normalize(poll_text, dim=-1))  # tree2 distance from BioCLIP prior
    poll = SpeciesGraph(Nq, a.d_model, operator="ou-attention", phylo_distance=poll_dist,
                        n_heads=4, n_layers=2, species_text=poll_text).to(dev)        # tree2: pollinator phylogeny
    W = torch.nn.Parameter(torch.eye(a.d_model, device=dev) + 0.01 * torch.randn(a.d_model, a.d_model, device=dev))

    def embeds(refine_plant, refine_poll):
        Pm = test if refine_plant else None                     # hide held-out plants' seed -> reconstruct from relatives
        Qm = qtest if (refine_poll and a.bidir_mask) else None   # hide held-out pollinators' seed -> reconstruct from relatives
        P = plant(mask=Pm) if refine_plant else plant._seed()
        Q = poll(mask=Qm) if refine_poll else poll._seed()
        return P, Q

    params = list(plant.parameters()) + list(poll.parameters()) + [W]
    opt = torch.optim.Adam(params, lr=a.lr)

    qcol = qtest if a.bidir_mask else None                      # bidirectional: metric measures held-out plant x held-out pollinator
    with torch.no_grad():                                       # untouched-seed baseline (no refinement, head=identity-ish)
        Ps, Qs = plant._seed(), poll._seed()
        seed_ap = interaction_ap(Ps, Qs, torch.eye(a.d_model, device=dev), Pmap, Ptgt, test, qcol=qcol)

    t0 = time.time(); last = float("nan")
    for step in range(a.steps):                                # train the bilinear interaction head + both graphs
        # rule 27/25: also mask a fraction of TRAIN plants -> recover their interactions from relatives
        mask = ((torch.rand(N, device=dev) < a.mask_frac) & train) | test if not a.no_mask else test
        P = plant(mask=mask)
        if a.bidir_mask:                                        # rule-27 both-ways: also mask a fraction of TRAIN pollinators
            qmask = ((torch.rand(Nq, device=dev) < a.mask_frac) & qtrain) | qtest if not a.no_mask else qtest
            Q = poll(mask=qmask)
        else:
            Q = poll()
        logits = (P[Pmap] @ W) @ Q.t()                          # [Np, Nq]
        sel = train[Pmap]                                       # supervise on TRAIN plants only (held-out never seen)
        if a.bidir_mask:                                        # ...and TRAIN pollinators only (held-out pollinators never supervised)
            loss = F.binary_cross_entropy_with_logits(logits[sel][:, qtrain], Ptgt[sel][:, qtrain])
        else:
            loss = F.binary_cross_entropy_with_logits(logits[sel], Ptgt[sel])
        if not torch.isfinite(loss):
            print(f"  DIVERGED (nan) at step {step}"); break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); last = loss.item()
    dt = time.time() - t0

    with torch.no_grad():
        P2, Q2 = embeds(True, True);  two_tree = interaction_ap(P2, Q2, W, Pmap, Ptgt, test, qcol=qcol)
        P1, Q1 = embeds(True, False); one_tree = interaction_ap(P1, Q1, W, Pmap, Ptgt, test, qcol=qcol)
        P0, Q0 = embeds(False, False); no_tree = interaction_ap(P0, Q0, W, Pmap, Ptgt, test, qcol=qcol)
    cross = two_tree - one_tree
    print(f"=== BIOLOGICAL encoder (standalone, rule-27 TWO-TREE) | plants={N} interacting={Pmap.numel()} "
          f"pollinators={Nq} held-out={int(test.sum())} ===")
    print(f"  interaction micro-AP | seed-only {seed_ap:.4f} | no-tree(trained-head) {no_tree:.4f} | "
          f"one-tree(plant only) {one_tree:.4f} | TWO-TREE {two_tree:.4f}")
    print(f"  cross_tree_gain (two-tree - one-tree) {cross:+.4f}   bio_gain (two-tree - seed-only) {two_tree - seed_ap:+.4f}")
    print(f"  [profile] final_bce={last:.5f}  {a.steps} steps in {dt:.1f}s ({dt/max(a.steps,1)*1000:.0f} ms/step)")
    return {"bio_gain": two_tree - seed_ap, "cross_tree_gain": cross, "two_tree_ap": two_tree,
            "one_tree_ap": one_tree, "seed_ap": seed_ap, "no_tree_ap": no_tree, "steps": a.steps, "seconds": dt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--mask_frac", type=float, default=0.15)   # rule-25 withhold rate (the lever to sweep)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.2)      # fraction of species held out (same-family relatives remain)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supervised", action="store_true")       # co-train a family-classification head on the REFINED emb
    ap.add_argument("--sup_weight", type=float, default=1.0)    # weight of the family CE loss vs the recon loss
    ap.add_argument("--sup_graph_only", action="store_true")   # CE grads shape ONLY the operator (seed detached) -> isolate graph gain
    ap.add_argument("--sup_masked", action="store_true")        # classify the MASKED (reconstructed-from-relatives) TRAIN species -> aligns CE with the held-out imputation task
    ap.add_argument("--no_recon", action="store_true")          # drop the identity-recon loss; train ONLY the family CE -> seed stays untouched (clean bio_gain)
    ap.add_argument("--random_seed_text", action="store_true")  # replace E1 with random vectors -> can the pure phylo operator recover family from the TREE alone?
    ap.add_argument("--vision_seed", action="store_true")       # rule-26 root-cause fix: reseed species graph with mean-DINO VISION (orthogonal to tree topology) instead of the phylo-coherent E1 text prior
    ap.add_argument("--fuse_seed", action="store_true")         # seed = vision (DINO) concat text (BioCLIP image bio), each L2-normed -> does appearance ADD to text where text saturates family?
    ap.add_argument("--objective", choices=["family", "trait", "interaction"], default="family")  # family=byte-identical default; interaction=rule-27 two-tree
    ap.add_argument("--trait_key", default="cat_plant_type")     # trait target when --objective trait (cat_* single-label or multi_* multi-label)
    ap.add_argument("--no_mask", action="store_true")           # interaction: don't rule-25 mask TRAIN plants during training (ablate the recover-from-relatives signal)
    ap.add_argument("--poll_dist", choices=["text", "covis", "taxo", "realtree"], default="text")  # pollinator-tree topology: text=BioCLIP prior (redundant w/ seed); covis=co-visitation over TRAIN plants (seed-orthogonal); taxo=taxonomic ultrametric (cophenetic dendrogram over the text prior; strong but seed-correlated)
    ap.add_argument("--bidir_mask", action="store_true")        # rule-27 both-ways: also hold out+mask a fraction of pollinators, recover via pollinator relatives; metric = held-out plant x held-out pollinator
    ap.add_argument("--covis_weighted", action="store_true")    # covis edges from log1p visitation-FREQUENCY mass (marg_poll_frq) not binary adjacency -> richer seed-orthogonal 2nd tree
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    if a.objective == "interaction":                            # rule-27 plant<->pollinator bilinear across two trees (additive, isolated path)
        return run_interaction(a, dev)

    E1, fam_id, tree, tip_row, gidx = load_species(a.cache_dir)
    E1, fam_id, tip_row = E1.to(dev), fam_id.to(dev), tip_row.to(dev)
    N = E1.shape[0]
    if a.random_seed_text:                                     # kill the family-coherent prior: seed carries NO family info
        E1 = torch.randn_like(E1)                              # operator must recover family structure from tree topology alone
    if a.vision_seed or a.fuse_seed:                           # rule-26: reseed the species graph with VISION (orthogonal to tree topology)
        E1 = load_vision_seed(a.cache_dir, gidx, fuse=a.fuse_seed).to(dev)  # SpeciesGraph probes any seed dim -> d_model
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    test = (torch.rand(N, generator=g) < a.holdout).to(dev)   # held-out species (same-family relatives stay in train)

    graph = SpeciesGraph(N, a.d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=E1).to(dev)              # seed = probe(E1) + free ; refined by the tree

    # objective: family (default, single hard label) OR trait (functional-trait axis the seed may not saturate)
    trait_kind = None
    if a.objective == "trait":
        trait_kind, trait_Y, trait_obs, trait_out = load_trait(a.cache_dir, gidx, a.trait_key, dev)
        def score(emb):                                        # trait-NN metric (parallels nn_family_acc)
            return (nn_trait_acc if trait_kind == "cat" else nn_trait_ap)(emb, trait_Y, trait_obs, test)
        head_out = trait_out
    else:
        def score(emb):
            return nn_family_acc(emb, fam_id, test)
        head_out = int(fam_id.max().item()) + 1

    fam_head = torch.nn.Linear(a.d_model, head_out).to(dev) if a.supervised else None
    params = list(graph.parameters()) + (list(fam_head.parameters()) if fam_head is not None else [])
    opt = torch.optim.Adam(params, lr=a.lr)

    with torch.no_grad():
        seed = graph._seed().detach()
    seed_acc = score(seed)

    t0 = time.time()
    last_loss = float("nan")
    for step in range(a.steps):                                # rule 25: mask TRAIN species, reconstruct from relatives
        mask = (torch.rand(N, device=dev) < a.mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)
        target = graph._seed().detach()
        loss = (1.0 - F.cosine_similarity(refined[mask], target[mask], dim=-1)).mean()  # scale-invariant (family-NN uses direction)
        if a.no_recon:
            loss = loss * 0.0                                  # train ONLY the CE term -> seed untouched, clean bio_gain
        if fam_head is not None:                               # co-train: make the REFINED emb family-separable (TRAIN only)
            if a.sup_graph_only:                               # CE grads shape ONLY the operator: run it on a DETACHED seed
                h0 = graph._seed().detach()
                h0 = torch.where(mask.unsqueeze(-1), graph.mask_token, h0)
                sup_refined = graph.clade(h0)                  # operator on frozen seed -> only clade/tree params get CE grad
            else:
                sup_refined = refined
            if a.sup_masked:                                   # supervise the MASKED-and-reconstructed TRAIN species (== held-out imputation task)
                sel = mask                                     # these tips saw mask_token -> reconstructed purely from relatives
            else:
                sel = ~test                                    # all TRAIN species (keep their own seed)
            if a.objective == "trait":
                sel = sel & trait_obs                          # only species with an observed trait carry a target
                if sel.any():
                    logits = fam_head(sup_refined[sel])
                    if trait_kind == "cat":
                        sup_loss = F.cross_entropy(logits, trait_Y[sel])
                    else:                                      # multi-label -> BCE over the label vector
                        sup_loss = F.binary_cross_entropy_with_logits(logits, trait_Y[sel])
                    loss = loss + a.sup_weight * sup_loss
            else:
                logits = fam_head(sup_refined[sel])            # classify family from the refined embedding
                loss = loss + a.sup_weight * F.cross_entropy(logits, fam_id[sel])
        if not torch.isfinite(loss):
            print(f"  DIVERGED (nan) at step {step}"); break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
        opt.step(); last_loss = loss.item()

    with torch.no_grad():
        seed = graph._seed().detach()
        refined_impute = graph(mask=test)                      # IMPUTATION: held-out species from relatives (rule 25)
        refined_rep = graph(mask=None)                         # REPRESENTATION: all species refined (no mask)
        seed_acc = score(seed)
        impute_acc = score(refined_impute)
        rep_acc = score(refined_rep)
        move = (refined_rep - seed).norm(dim=-1).mean().item()
    dt = time.time() - t0
    metric = ("trait[%s]-NN %s" % (a.trait_key, "AP" if trait_kind == "multi" else "acc")) if a.objective == "trait" else "family-NN acc"
    print(f"=== BIOLOGICAL encoder (standalone) | N={N} in-tree={tip_row.numel()} held-out={int(test.sum())} obj={a.objective} ===")
    print(f"  {metric} | seed {seed_acc:.4f} | refined-representation {rep_acc:.4f} | imputed-from-relatives {impute_acc:.4f}")
    print(f"  {'trait_gain' if a.objective=='trait' else 'bio_gain'} (representation) {rep_acc - seed_acc:+.4f}   (imputation) {impute_acc - seed_acc:+.4f}")
    print(f"  [profile] refined_seed_norm={move:.4f}  final_recon_mse={last_loss:.5f}")
    print(f"  {a.steps} steps in {dt:.1f}s ({dt/max(a.steps,1)*1000:.0f} ms/step)")
    return {"bio_gain": rep_acc - seed_acc, "imputation_gain": impute_acc - seed_acc, "seed_acc": seed_acc,
            "trait_gain": rep_acc - seed_acc, "refined_seed_norm": move, "steps": a.steps, "seconds": dt}


if __name__ == "__main__":
    main()
