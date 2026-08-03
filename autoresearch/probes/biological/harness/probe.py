"""Standalone biological-encoder probe -- train + evaluate the species-graph IN ISOLATION.

No fusion model, no 621k observations, no full benchmark suite -- just the ~2141-species table + the
phylogeny. Measures the encoder's science (science.md rules 9, 25, 27): does refining a species through
its phylogenetic relatives let you impute its biology (family) for held-out species? Seconds, not minutes.

Objective (standalone `bio_gain`): family-NN accuracy of the graph-refined species embedding MINUS the raw
seed, on held-out species reconstructed from relatives (rule-25 mask). >0 ⟹ the phylogeny adds
family-discriminative structure the seed lacks. Reuses SpeciesGraph unchanged (no core edit).

  python -m deepearth.autoresearch.probes.biological.harness.probe --cache_dir autoresearch/data/deepcal --steps 400
"""
import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deepearth.autoresearch.probes.biological.editable_files.phylogenomic import SpeciesGraph, build_tree_buffers
from deepearth.autoresearch.probes.biological.editable_files.lib.training import (
    train_graph_family, train_interaction_graphs,
)
from deepearth.autoresearch.probes.biological.harness.board import PROTOCOL, _set_result_sink, declare
from deepearth.autoresearch.probes.biological.harness.nulltree import (
    FAIR_CONTROL_DRAWS, fair_gain as nulltree_fair_gain, null_family,
)


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
    qcol = qtest if a.bidir_mask else None                      # bidirectional: metric measures held-out plant x held-out pollinator

    def fit_two_tree(plant_tree, plant_tip_row):
        """Fit one clean arm through the editable mechanism and score it with the fixed evaluator."""
        (plant, poll, interaction, last_loss,
         initial_plant_seed, initial_pollinator_seed) = train_interaction_graphs(
            E1, poll_text, plant_tree, plant_tip_row, poll_dist, Pmap, Ptgt,
            train, test, qtrain, qtest, dev, a.d_model, a.steps, a.mask_frac, a.lr,
            run_seed=a.seed, bidirectional_mask=a.bidir_mask, no_mask=a.no_mask)

        def embeds(refine_plant, refine_poll):
            plant_mask = test if refine_plant else None
            poll_mask = qtest if (refine_poll and a.bidir_mask) else None
            plant_rep = plant(mask=plant_mask) if refine_plant else plant._seed()
            poll_rep = poll(mask=poll_mask) if refine_poll else poll._seed()
            return plant_rep, poll_rep

        with torch.no_grad():
            p2, q2 = embeds(True, True)
            p1, q1 = embeds(True, False)
            p0, q0 = embeds(False, False)
            return {
                "two_tree": interaction_ap(p2, q2, interaction, Pmap, Ptgt, test, qcol=qcol),
                "one_tree": interaction_ap(p1, q1, interaction, Pmap, Ptgt, test, qcol=qcol),
                "no_tree": interaction_ap(p0, q0, interaction, Pmap, Ptgt, test, qcol=qcol),
                "seed": interaction_ap(initial_plant_seed, initial_pollinator_seed,
                                       torch.eye(a.d_model, device=dev),
                                       Pmap, Ptgt, test, qcol=qcol),
                "loss": last_loss,
            }

    t0 = time.time()
    real = fit_two_tree(tree, tip_row)
    dt = time.time() - t0
    two_tree, one_tree = real["two_tree"], real["one_tree"]
    no_tree, seed_ap, last = real["no_tree"], real["seed"], real["loss"]
    cross = two_tree - one_tree
    print(f"=== BIOLOGICAL encoder (standalone, rule-27 TWO-TREE) | plants={N} interacting={Pmap.numel()} "
          f"pollinators={Nq} held-out={int(test.sum())} ===")
    print(f"  interaction micro-AP | seed-only {seed_ap:.4f} | no-tree(trained-head) {no_tree:.4f} | "
          f"one-tree(plant only) {one_tree:.4f} | TWO-TREE {two_tree:.4f}")
    print(f"  cross_tree_gain (two-tree - one-tree) {cross:+.4f}   bio_gain (two-tree - seed-only) {two_tree - seed_ap:+.4f}")
    print(f"  [profile] final_bce={last:.5f}  {a.steps} steps in {dt:.1f}s ({dt/max(a.steps,1)*1000:.0f} ms/step)")

    nulls = {}
    if not a.no_control:
        for label, ntree, ntip in null_family(E1, tree, tip_row, draws=a.control_draws):
            nulls[label] = fit_two_tree(ntree, ntip)["two_tree"]
            print(f"  [control] {label:32s} {nulls[label]:.4f}", flush=True)
    fair, best_label, best_null = nulltree_fair_gain(two_tree, nulls)
    if fair is not None:
        print(f"  vs null-tree (FAIR) {fair:+.4f}   strongest null: {best_label} {best_null:.4f}")
    declare(capability="pollinator_transfer",
            mode=f"TWO-TREE(poll_dist={a.poll_dist},bidir={a.bidir_mask})",
            metric="interaction_micro_ap", value=two_tree, split="species-random",
            gains={"vs null-tree": fair, "vs seed": two_tree - seed_ap},
            baselines={"seed": seed_ap, "one-tree": one_tree, "no-tree": no_tree, **nulls,
                       **({"null-tree/best": best_null} if best_null is not None else {})},
            diagnostic=(fair is None),
            diagnostic_reason=("--no_control: no null-tree baseline, so there is no fair gain"
                               if fair is None else ""),
            seed_score=seed_ap, cross_tree_gain=cross, n_plants=N, n_pollinators=Nq,
            interacting=int(Pmap.numel()), held_out=int(test.sum()), strongest_null=best_label,
            seconds=dt)
    return {"bio_gain": two_tree - seed_ap, "fair_gain": fair, "cross_tree_gain": cross,
            "two_tree_ap": two_tree, "null_scores": nulls,
            "one_tree_ap": one_tree, "seed_ap": seed_ap, "no_tree_ap": no_tree, "steps": a.steps, "seconds": dt}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="autoresearch/data/deepcal")
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
    # THE FAIR CONTROL (nulltree.py). On by default: a run without it measures the operator against its
    # own input, which is the ablation this loop mistook for a control for its whole history.
    ap.add_argument("--no_control", action="store_true",
                    help="skip the null-tree family — screens fast, but the result is DIAGNOSTIC and can never record")
    ap.add_argument("--control_draws", type=int, default=FAIR_CONTROL_DRAWS,
                    help="tip-label permutations in the null family (the seed-dendrogram is always added)")
    ap.add_argument("--result-json", dest="result_json", default="",
                    help="write the ProbeResult here for the board to gate")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"
    _set_result_sink(a.result_json, "", PROTOCOL, a, {
        "objective": a.objective, "d_model": a.d_model, "steps": a.steps, "mask_frac": a.mask_frac,
        "lr": a.lr, "holdout": a.holdout, "operator": "latent-clade", "supervised": a.supervised,
        "no_recon": a.no_recon, "vision_seed": a.vision_seed, "fuse_seed": a.fuse_seed,
        "random_seed_text": a.random_seed_text, "control_draws": a.control_draws,
        "train_encoder": True,
    })

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

    def fit(tree_buf, tip_row_):
        """Fit and score one arm; the editable trainer is identical for real and null trees."""
        target = trait_Y if a.objective == "trait" else fam_id
        observed = trait_obs if a.objective == "trait" else None
        seed_rep, refined_rep, refined_impute, movement, last_loss = train_graph_family(
            E1, tree_buf, tip_row_, test, dev, a.d_model, a.steps, a.mask_frac, a.lr,
            run_seed=a.seed, supervised=a.supervised, no_recon=a.no_recon,
            sup_graph_only=a.sup_graph_only, sup_masked=a.sup_masked, sup_weight=a.sup_weight,
            target_kind=(trait_kind or "cat"), target=target, target_obs=observed,
            target_out=head_out)
        return {"seed": score(seed_rep), "rep": score(refined_rep),
                "impute": score(refined_impute), "move": movement, "loss": last_loss}

    t0 = time.time()
    real = fit(tree, tip_row)
    seed_acc, rep_acc, impute_acc, move = real["seed"], real["rep"], real["impute"], real["move"]

    # THE FAIR CONTROL. Same operator, same budget, a tree that is not the phylogeny — see nulltree.py.
    nulls = {}
    if not a.no_control:
        for label, ntree, ntip in null_family(E1, tree, tip_row, draws=a.control_draws):
            nulls[label] = fit(ntree, ntip)["impute"]
            print(f"  [control] {label:32s} {nulls[label]:.4f}", flush=True)
    fair, best_label, best_null = nulltree_fair_gain(impute_acc, nulls)
    dt = time.time() - t0

    metric = ("trait[%s]-NN %s" % (a.trait_key, "AP" if trait_kind == "multi" else "acc")) if a.objective == "trait" else "family-NN acc"
    print(f"=== BIOLOGICAL encoder (standalone) | N={N} in-tree={tip_row.numel()} held-out={int(test.sum())} obj={a.objective} ===")
    print(f"  {metric} | seed {seed_acc:.4f} | refined-representation {rep_acc:.4f} | imputed-from-relatives {impute_acc:.4f}")
    print(f"  vs seed (NOT fair, the old bio_gain) {rep_acc - seed_acc:+.4f}   (imputation) {impute_acc - seed_acc:+.4f}")
    if fair is not None:
        print(f"  vs null-tree (FAIR) {fair:+.4f}   strongest null: {best_label} {best_null:.4f}")
    print(f"  [profile] refined_seed_norm={move:.4f}  final_recon_mse={real['loss']:.5f}")
    print(f"  {a.steps} steps in {dt:.1f}s")

    metric_name = (f"trait_{a.trait_key}_nn_" + ("ap" if trait_kind == "multi" else "acc")
                   if a.objective == "trait" else "family_nn_accuracy")
    baselines = {"seed": seed_acc, **nulls}
    if best_null is not None:
        baselines["null-tree/best"] = best_null
    declare(
        capability=("family_from_phylo" if a.objective == "family" else ""),
        mode=f"MASK-RECON(latent-clade,mask={a.mask_frac})",
        metric=metric_name, value=impute_acc, split="species-random",
        # The labels are written out LITERALLY, not assembled into a variable first, because the audit
        # reads this call statically to check that exactly one fair label is published. A computed dict
        # is invisible to it, and an invisible fair label is how a board silently gates on the wrong
        # baseline. `vs null-tree` is None when --no_control is set; fair_gain skips None entries, and
        # the run is declared diagnostic below so it cannot record anyway.
        gains={"vs null-tree": fair, "vs seed": impute_acc - seed_acc},
        baselines=baselines,
        # A run with the control switched off measures the encoder against its own input and nothing
        # else. It is a legitimate quick screen, but it can never set a record, and saying so here is
        # better than letting it onto the board against a baseline that is not one.
        diagnostic=(fair is None or a.objective != "family"),
        diagnostic_reason=("--no_control: no null-tree baseline, so there is no fair gain" if fair is None
                           else f"objective {a.objective!r} is not a board capability" if a.objective != "family"
                           else ""),
        seed_score=seed_acc, imputation=impute_acc, refined_seed_norm=move,
        n_species=N, in_tree=int(tip_row.numel()), held_out=int(test.sum()),
        control_members=sorted(nulls), seconds=dt)
    return {"bio_gain": impute_acc - seed_acc, "fair_gain": fair, "imputation_gain": impute_acc - seed_acc,
            "seed_acc": seed_acc, "null_scores": nulls,
            "refined_seed_norm": move, "steps": a.steps, "seconds": dt}


if __name__ == "__main__":
    main()
