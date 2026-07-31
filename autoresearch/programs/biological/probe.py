"""Standalone biological-encoder probe -- train + evaluate the species-graph IN ISOLATION.

No fusion model, no 621k observations, no full benchmark suite -- just the ~2141-species table + the
phylogeny. Measures the encoder's science (science.md rules 9, 25, 27): does refining a species through
its phylogenetic relatives let you impute its biology (family) for held-out species? Seconds, not minutes.

Objective (standalone `bio_gain`): family-NN accuracy of the graph-refined species embedding MINUS the raw
seed, on held-out species reconstructed from relatives (rule-25 mask). >0 ⟹ the phylogeny adds
family-discriminative structure the seed lacks. Reuses SpeciesGraph unchanged (no core edit).

  python -m deepearth.autoresearch.programs.biological.probe --cache_dir data/deepcal --steps 400
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
    return E1, fam_id, tree, tip_row


def nn_family_acc(emb: torch.Tensor, fam_id: torch.Tensor, test: torch.Tensor) -> float:
    """held-out species -> nearest TRAIN species (cosine) -> predict its family. accuracy over held-out."""
    train = ~test
    et = F.normalize(emb[train], dim=-1)
    ett = F.normalize(emb[test], dim=-1)
    nn = (ett @ et.t()).argmax(-1)
    return (fam_id[train][nn] == fam_id[test]).float().mean().item()


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
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    E1, fam_id, tree, tip_row = load_species(a.cache_dir)
    E1, fam_id, tip_row = E1.to(dev), fam_id.to(dev), tip_row.to(dev)
    N = E1.shape[0]
    if a.random_seed_text:                                     # kill the family-coherent prior: seed carries NO family info
        E1 = torch.randn_like(E1)                              # operator must recover family structure from tree topology alone
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    test = (torch.rand(N, generator=g) < a.holdout).to(dev)   # held-out species (same-family relatives stay in train)

    graph = SpeciesGraph(N, a.d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=E1).to(dev)              # seed = probe(E1) + free ; refined by the tree
    n_fam = int(fam_id.max().item()) + 1
    fam_head = torch.nn.Linear(a.d_model, n_fam).to(dev) if a.supervised else None
    params = list(graph.parameters()) + (list(fam_head.parameters()) if fam_head is not None else [])
    opt = torch.optim.Adam(params, lr=a.lr)

    with torch.no_grad():
        seed = graph._seed().detach()
    seed_acc = nn_family_acc(seed, fam_id, test)

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
            logits = fam_head(sup_refined[sel])                # classify from the refined embedding
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
        seed_acc = nn_family_acc(seed, fam_id, test)
        impute_acc = nn_family_acc(refined_impute, fam_id, test)
        rep_acc = nn_family_acc(refined_rep, fam_id, test)
        move = (refined_rep - seed).norm(dim=-1).mean().item()
    dt = time.time() - t0
    print(f"=== BIOLOGICAL encoder (standalone) | N={N} in-tree={tip_row.numel()} held-out={int(test.sum())} ===")
    print(f"  family-NN acc | seed {seed_acc:.4f} | refined-representation {rep_acc:.4f} | imputed-from-relatives {impute_acc:.4f}")
    print(f"  bio_gain (representation) {rep_acc - seed_acc:+.4f}   (imputation) {impute_acc - seed_acc:+.4f}")
    print(f"  [profile] refined_seed_norm={move:.4f}  final_recon_mse={last_loss:.5f}")
    print(f"  {a.steps} steps in {dt:.1f}s ({dt/max(a.steps,1)*1000:.0f} ms/step)")
    return {"bio_gain": rep_acc - seed_acc, "imputation_gain": impute_acc - seed_acc, "seed_acc": seed_acc,
            "refined_seed_norm": move, "steps": a.steps, "seconds": dt}


if __name__ == "__main__":
    main()
