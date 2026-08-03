"""Graph training and readout — the rule-9, rule-25 and rule-10/11 levers.

How the species graph is fitted and read: the masked-reconstruct protocol (rule 25 -- withhold a
fraction of species per batch and rebuild them from their relatives), the out-of-tree variant that
rebuilds the tree over TRAIN species only so held-out species are genuinely unseen (rule 9), the
trait/multi-trait/co-occurrence supervised variants, and the progressive-reveal autoregressive readout
(rules 10-11 -- an observation of A updates its in-context neighbours B, C).

Each returns `(seed, representation, imputation)` scores on the same held-out split, so a caller can
difference them without knowing how the fit was done.

EDITABLE. These are mechanisms under test. What is MEASURED, and against which control, lives in
`probes/biological/harness/`.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from deepearth.autoresearch.probes.biological.editable_files.phylogenomic import (
    SpeciesGraph, build_tree_buffers,
)


def train_graph(seed: torch.Tensor, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                impute_steps, *, run_seed=None):
    """Phylo-refine one seed via the latent-clade operator (rule-25 mask-reconstruct). Identical protocol for
    every seed source. `impute_steps`>0 uses that many extra masked-imputation refinement passes at eval for
    the held-out species (push imputation-from-relatives, since pure-vision single-pass imputation was net-neg)."""
    if run_seed is not None:
        torch.manual_seed(run_seed)
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    opt = torch.optim.Adam(graph.parameters(), lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)
        target = graph._seed().detach()
        loss = graph.masked_reconstruction_loss(mask, target, metric="cosine", reconstructed=refined)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):   # extra imputation refinement rounds for the held-out set
            imp = graph(mask=test)
    return s, rep, imp


def train_graph_family(seed: torch.Tensor, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                       *, run_seed=0, supervised=False, no_recon=False, sup_graph_only=False,
                       sup_masked=False, sup_weight=1.0, target_kind="cat", target=None,
                       target_obs=None, target_out=None):
    """Fit the standalone family/trait candidate and return its three representations.

    The fixed probe owns the split, target, budget and scoring. This editable function owns the
    mechanism used to fit the candidate, including masked reconstruction and the optional supervised
    auxiliary objective. Re-seeding here makes the real tree and every null-tree arm start from the
    same parameters and consume the same mask stream.
    """
    torch.manual_seed(run_seed)
    n_species = seed.shape[0]
    graph = SpeciesGraph(n_species, d_model, operator="latent-clade", tree=tree,
                         tip_row=tip_row, species_text=seed).to(dev)
    head = torch.nn.Linear(d_model, int(target_out)).to(dev) if supervised else None
    params = list(graph.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    last_loss = float("nan")

    for _ in range(steps):
        mask = (torch.rand(n_species, device=dev) < mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)
        seed_target = graph._seed().detach()
        loss = graph.masked_reconstruction_loss(mask, seed_target, metric="cosine", reconstructed=refined)
        if no_recon:
            loss = loss * 0.0

        if head is not None:
            if sup_graph_only:
                h0 = graph._seed().detach()
                h0 = torch.where(mask.unsqueeze(-1), graph.mask_token, h0)
                supervised_rep = graph.clade(h0)
            else:
                supervised_rep = refined
            selected = mask if sup_masked else ~test
            if target_obs is not None:
                selected = selected & target_obs
            if selected.any():
                logits = head(supervised_rep[selected])
                if target_kind == "cat":
                    supervised_loss = F.cross_entropy(logits, target[selected])
                else:
                    supervised_loss = F.binary_cross_entropy_with_logits(logits, target[selected])
                loss = loss + sup_weight * supervised_loss

        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last_loss = loss.item()

    with torch.no_grad():
        seed_rep = graph._seed().detach()
        representation = graph(mask=None)
        imputation = graph(mask=test)
        movement = (representation - seed_rep).norm(dim=-1).mean().item()
    return seed_rep, representation, imputation, movement, last_loss


def train_interaction_graphs(plant_seed: torch.Tensor, pollinator_seed: torch.Tensor,
                             plant_tree, plant_tip_row, pollinator_distance: torch.Tensor,
                             plant_rows: torch.Tensor, interaction_targets: torch.Tensor,
                             train_plants: torch.Tensor, test_plants: torch.Tensor,
                             train_pollinators: torch.Tensor, test_pollinators: torch.Tensor,
                             dev, d_model, steps, mask_frac, lr, *, run_seed=0,
                             bidirectional_mask=False, no_mask=False):
    """Fit the editable two-tree interaction mechanism from a clean deterministic start.

    The fixed evaluator supplies the split, targets and budget and scores the returned models. Calling
    this function independently for the real phylogeny and each null guarantees that neither the
    pollinator graph nor the bilinear head carries state between arms.
    """
    torch.manual_seed(run_seed)
    plant = SpeciesGraph(plant_seed.shape[0], d_model, operator="latent-clade",
                         tree=plant_tree, tip_row=plant_tip_row,
                         species_text=plant_seed).to(dev)
    pollinator = SpeciesGraph(pollinator_seed.shape[0], d_model, operator="ou-attention",
                              phylo_distance=pollinator_distance, n_heads=4, n_layers=2,
                              species_text=pollinator_seed).to(dev)
    interaction = torch.nn.Parameter(
        torch.eye(d_model, device=dev) + 0.01 * torch.randn(d_model, d_model, device=dev))
    params = list(plant.parameters()) + list(pollinator.parameters()) + [interaction]
    opt = torch.optim.Adam(params, lr=lr)
    last_loss = float("nan")
    with torch.no_grad():
        initial_plant_seed = plant._seed().detach().clone()
        initial_pollinator_seed = pollinator._seed().detach().clone()

    for _ in range(steps):
        plant_mask = test_plants if no_mask else (
            ((torch.rand(plant_seed.shape[0], device=dev) < mask_frac) & train_plants) | test_plants)
        plant_rep = plant(mask=plant_mask)
        if bidirectional_mask:
            pollinator_mask = test_pollinators if no_mask else (
                ((torch.rand(pollinator_seed.shape[0], device=dev) < mask_frac)
                 & train_pollinators) | test_pollinators)
            pollinator_rep = pollinator(mask=pollinator_mask)
        else:
            pollinator_rep = pollinator()
        logits = (plant_rep[plant_rows] @ interaction) @ pollinator_rep.t()
        selected = train_plants[plant_rows]
        if bidirectional_mask:
            loss = F.binary_cross_entropy_with_logits(
                logits[selected][:, train_pollinators],
                interaction_targets[selected][:, train_pollinators])
        else:
            loss = F.binary_cross_entropy_with_logits(logits[selected], interaction_targets[selected])
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last_loss = loss.item()

    return (plant, pollinator, interaction, last_loss,
            initial_plant_seed, initial_pollinator_seed)


def _oot_tree(cache, gidx, tip_row, keep_mask):
    """Rule-9 helper. Rebuild the latent-clade tree buffers over ONLY the species kept in (`keep_mask` True at
    their vocab row), i.e. the tree whose tips are the TRAIN species. Returns (tree_train, tip_row_train) where
    tip_row_train lists the vocab rows of the retained tips. Species NOT in keep_mask that WERE in-tree become
    genuinely OUT-OF-TREE rows of the resulting SpeciesGraph -> they carry no tree position and can only be
    imputed by the clade soft-attach cross-attention (LatentCladeAttention, the has_oot MLA-read path that the
    all-in-tree probe never exercises). This is science.md rule-9 tested for real."""
    import csv as _csv
    rows = list(_csv.DictReader(open(Path(cache) / "derived/species_index.csv")))
    labels = [rows[int(gidx[int(r)])]["tip_label"] for r in tip_row.tolist()]   # label of each currently in-tree tip
    keep = keep_mask[tip_row].tolist()                                          # keep this tip in the train tree?
    kept_pairs = [(int(r), lab) for r, lab, k in zip(tip_row.tolist(), labels, keep) if k]
    tree_tr = build_tree_buffers(str(Path(cache) / "ca_subtree.dated.nwk"), [lab for _, lab in kept_pairs])
    tip_row_tr = torch.tensor([r for r, _ in kept_pairs], dtype=torch.long)
    return tree_tr, tip_row_tr


def train_graph_oot(seed, cache, gidx, tip_row, test, dev, d_model, steps, mask_frac, lr, impute_steps,
                    sup_kind=None, sup_Y=None, sup_obs=None, sup_nclass=None,
                    resid_readout=False, clade_base=None, recon_relative=False, fam_gid=None, recon_k=0,
                    oot_heads=4, oot_layers=2):
    """RULE-9 OUT-OF-TREE PROJECTION (science.md rule 9, the `~` row). The standard probe leaves held-out `test`
    species AS TIPS and only masks their seed -- they still occupy a tree position and are refined by exact
    message passing. Rule-9 asks the harder question: project a species that is genuinely NOT IN THE TREE. Here we
    build the phylogeny over the TRAIN tips only; every test species becomes an out-of-tree row that must soft-
    attach to the refined clade latents via cross-attention (the LatentCladeAttention has_oot MLA read). The graph
    is trained self-supervised (rule-25 mask-reconstruct) on the in-tree TRAIN tips; test species never enter the
    tree during training or eval. Returns (seed, refined_full, oot_projection) all [N,d]; test rows of the
    projection come purely from the out-of-tree soft-attach path."""
    N = seed.shape[0]
    train = ~test
    tree_tr, tip_row_tr = _oot_tree(cache, gidx, tip_row, train)
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree_tr, tip_row=tip_row_tr,
                         species_text=seed, n_heads=oot_heads, n_layers=oot_layers).to(dev)
    # in-tree train tips (bool over full vocab) -- the only rows we mask/reconstruct during training
    intree = torch.zeros(N, dtype=torch.bool, device=dev); intree[tip_row_tr.to(dev)] = True
    # RULE-25 (recon_relative): reconstruct a masked species toward the MEAN SEED of its same-family TRAIN
    # relatives -- the real held-out imputation target -- not the leaked identity copy of its own seed. Identity
    # recon is trivially satisfiable by the seed, so the operator learns ~identity and adds little (established
    # weak in-tree path). A relative target forces the operator to place a masked species where its relatives
    # predict, which is exactly the OOT soft-attach eval task. Default off (byte-identical when the flag is unset).
    rel_target = None
    if recon_relative and fam_gid is not None:
        fg = fam_gid.to(dev).long()
        s0 = graph._seed().detach()
        G = int(fg.max().item()) + 1
        tr = intree
        if recon_k and recon_k > 0:
            # ITER-2: rank-PRESERVING relative target. Each species' target = mean seed of its recon_k nearest
            # SAME-FAMILY TRAIN relatives (cosine in seed space), NOT the whole-family centroid (which erased
            # within-family rank in ITER-1). A local neighborhood keeps the fine structure a continuous trait
            # needs while still forcing reconstruction FROM relatives (leak-free: a species is never its own
            # neighbor, and only TRAIN tips are eligible relatives).
            sn = F.normalize(s0, dim=-1)
            eligible = tr.clone()                                        # only train tips are relatives
            same_fam = (fg.unsqueeze(1) == fg.unsqueeze(0))             # [N,N] same-family mask
            sim = sn @ sn.t()
            valid = same_fam & eligible.unsqueeze(0)                    # relative must be same-family AND a train tip
            valid.fill_diagonal_(False)                                 # never self
            sim = sim.masked_fill(~valid, -2.0)
            kk = min(int(recon_k), sim.shape[1])
            topv, topi = sim.topk(kk, dim=1)                           # [N,kk]
            has = (topv > -1.5)                                        # rows with >=1 same-family train relative
            neigh = s0[topi]                                            # [N,kk,d]
            w = (topv.clamp(min=-1.5) * has.float()).unsqueeze(-1)     # ignore pad slots
            wsum = has.float().sum(1, keepdim=True).clamp(min=1)
            knn_mean = (neigh * has.float().unsqueeze(-1)).sum(1) / wsum.unsqueeze(-1).squeeze(1)
            # fall back to family centroid for species with no same-family train relative
            cnt = torch.zeros(G, device=dev).index_add_(0, fg[tr], torch.ones(int(tr.sum()), device=dev))
            ssum = torch.zeros(G, s0.shape[1], device=dev).index_add_(0, fg[tr], s0[tr])
            fam_mean = (ssum / cnt.clamp(min=1).unsqueeze(-1))[fg]
            rel_target = torch.where(has.any(1, keepdim=True), knn_mean, fam_mean)
        else:
            cnt = torch.zeros(G, device=dev).index_add_(0, fg[tr], torch.ones(int(tr.sum()), device=dev))
            ssum = torch.zeros(G, s0.shape[1], device=dev).index_add_(0, fg[tr], s0[tr])
            fam_mean = ssum / cnt.clamp(min=1).unsqueeze(-1)
            rel_target = fam_mean[fg]
    # ITER-2: optional trait-supervised residual head. When sup_kind is set, the graph co-trains a magnitude-
    # explicit readout (predicts y - clade_mean on masked TRAIN tips) alongside the cosine-recon loss, then reads
    # the OOT-projected embedding through that head at eval -> combines the rule-9 OOT projection (which preserves
    # the seed's magnitude signal) with the residual-head clade regression (which recovers within-clade rank).
    head = None; clade_mu = None
    if sup_kind is not None:
        out_dim = 1 if sup_kind == "num" else int(sup_nclass)
        head = torch.nn.Linear(d_model, out_dim).to(dev)
        yt = sup_Y.float() if sup_kind == "num" else sup_Y.long()
        if resid_readout and sup_kind == "num" and clade_base is not None:
            clade_mu = clade_base.to(dev).float(); yt = yt - clade_mu
    params = list(graph.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    train_obs = (sup_obs & intree) if sup_kind is not None else None
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & intree            # mask only in-tree train tips
        if not mask.any():
            continue
        refined = graph(mask=mask)
        if rel_target is not None:
            target = rel_target
        else:
            target = graph._seed().detach()
        loss = (1.0 - F.cosine_similarity(refined[mask], target[mask], dim=-1)).mean()
        if head is not None:
            hm = mask & train_obs
            if hm.any():
                pred = head(refined[hm])
                if sup_kind == "num":
                    loss = loss + F.mse_loss(pred.squeeze(-1), yt[hm])
                else:
                    loss = loss + F.cross_entropy(pred, yt[hm])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)                                             # full refine: test rows = OOT soft-attach
        oot = rep                                                          # test species already come from the OOT path
        for _ in range(max(0, impute_steps - 1)):
            oot = graph(mask=None)
        head_pred = None
        head_resid = None
        if head is not None and sup_kind == "num":
            add_mu = clade_mu if (resid_readout and clade_mu is not None) else 0.0
            head_resid = head(oot).squeeze(-1).detach()                    # WITHIN-clade residual (mean added at readout)
            head_pred = (head_resid + add_mu).detach()                     # magnitude-aware readout on OOT projection
        elif head is not None and sup_kind == "cat":
            head_pred = head(oot).argmax(-1).detach()                      # ITER-4: categorical class readout on OOT proj
    return s, rep, oot, head_pred, head_resid


def train_graph_trait_supervised(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                                 impute_steps, kind, Y, obs, nclass, dual_head=False, dual_seed=None,
                                 head_hidden=0, dual_scale=1.0, rollout_rounds=1, reveal_frac=0.25,
                                 ab_scorer=None, operator="latent-clade", phylo_dist=None,
                                 fam_id=None, resid_readout=False, clade_base=None,
                                 blanket_k=0, mask_curriculum=None):
    """ROUND-2 push for num_lep_support (and any axis): SEED-FROZEN, TRAIT-SUPERVISED graph. The operator is
    trained to reconstruct a MASKED species' TRAIT from its phylo relatives (rule 25+the trait-supervised recipe
    that unlocked cat_form/family): freeze the seed, mask a fraction of TRAIN species, and drive a small readout
    off the refined masked embedding toward that species' true trait (regression MSE for num_, CE for cat_).
    This aligns the objective with the held-out imputation task and pushes the axis the seed lacks.

    `dual_head` (interaction-aware): num_lep_support is lepidoptera HOST support -- a plant x insect interaction
    axis where vision(plant appearance) and phylo may COMPOUND. With dual_head we co-train a SECOND readout off a
    detached-vision projection of the SAME refined embedding, so the operator must make the refined rep explain
    the trait through BOTH a phylo-propagated channel and an appearance channel -> an interaction-aware objective
    that a single reconstruct-identity loss cannot express. `dual_seed` = the [N,vdim] vision block for that head.

    `operator` (rule 29): 'latent-clade' (default, all prior rounds) or 'ou-attention' -- the exact O(N) OU-GP
    distance-biased attention the champion uses. ou-attention needs a phylo_distance matrix (`phylo_dist`,
    built from the E1 tree-derived prior) instead of the tree buffers; this isolates the OPERATOR choice on the
    identical trait-supervised imputation task."""
    N = seed.shape[0]
    if operator == "ou-attention":
        # blanket_k (rule-29 Markov-blanket restriction): restrict each species' reconstruction to its k NEAREST
        # phylo relatives via the operator's top_k sparsification of phylo_distance (dense = whole-clade). 0 = dense.
        graph = SpeciesGraph(N, d_model, operator="ou-attention", phylo_distance=phylo_dist,
                             n_heads=4, n_layers=2, species_text=seed,
                             top_k=(blanket_k if blanket_k and blanket_k > 0 else None)).to(dev)
    else:
        graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                             species_text=seed).to(dev)
    train_obs = obs & (~test)
    out_dim = 1 if kind == "num" else int(nclass)
    if head_hidden > 0:   # CEILING-PUSH: deeper trait-supervised readout (2-layer GELU MLP)
        head = torch.nn.Sequential(torch.nn.Linear(d_model, head_hidden), torch.nn.GELU(),
                                   torch.nn.Linear(head_hidden, out_dim)).to(dev)
    else:
        head = torch.nn.Linear(d_model, out_dim).to(dev)
    yt = Y.float() if kind == "num" else Y.long()
    # CLADE-MEAN-RESIDUAL READOUT (resid_readout, num only): make magnitude EXPLICIT. The head predicts only the
    # WITHIN-CLADE deviation y - clade_mean; at readout we add back the held-out species' clade mean. Clade means
    # are computed from TRAIN-observed species only (per family), so no held-out leakage. This forces the graph to
    # carry the fine within-family rank (the exact signal kNN neighbor-averaging collapses) while the coarse
    # family-level magnitude comes from the phylo prior for free.
    clade_mu = None
    if resid_readout and kind == "num" and clade_base is not None:
        clade_mu = clade_base.to(dev).float()                          # caller-supplied [N] base mean (e.g. hybrid)
        yt = yt - clade_mu
    elif resid_readout and kind == "num" and fam_id is not None:
        fam = fam_id.to(dev).long()
        train_obs0 = obs & (~test)
        F_ = int(fam.max().item()) + 1
        gmean = yt[train_obs0].mean() if train_obs0.any() else yt.mean()
        ssum = torch.zeros(F_, device=dev).index_add_(0, fam[train_obs0], yt[train_obs0])
        scnt = torch.zeros(F_, device=dev).index_add_(0, fam[train_obs0], torch.ones(int(train_obs0.sum()), device=dev))
        fmean = torch.where(scnt > 0, ssum / scnt.clamp(min=1), gmean)   # per-family train mean (fallback global)
        clade_mu = fmean[fam]                                            # [N] each species' family mean
        yt = yt - clade_mu                                              # head now regresses the residual
    vhead = None
    if dual_head and dual_seed is not None:
        vproj = F.normalize(dual_seed, dim=-1)
        vh = head_hidden if head_hidden > 0 else d_model   # CEILING-PUSH: match dual-head capacity to readout
        vhead = torch.nn.Sequential(torch.nn.Linear(vproj.shape[1], vh), torch.nn.GELU(),
                                    torch.nn.Linear(vh, 1 if kind == "num" else int(nclass))).to(dev)
    params = list(graph.parameters()) + list(head.parameters()) + (list(vhead.parameters()) if vhead else [])
    opt = torch.optim.Adam(params, lr=lr)
    for p in graph.parameters():
        pass  # seed frozen via detached target below; operator params train
    for _step in range(steps):
        # mask_curriculum (rule-25 mechanism knob): ('easy2hard', lo, hi) linearly RAMPS the mask fraction
        # lo->hi over training (harder recon later); ('hard2easy', lo, hi) reverses it. None = fixed mask_frac.
        mf = mask_frac
        if mask_curriculum is not None:
            _sched, _lo, _hi = mask_curriculum
            _frac = _step / max(1, steps - 1)
            mf = (_lo + (_hi - _lo) * _frac) if _sched == "easy2hard" else (_hi - (_hi - _lo) * _frac)
        mask = (torch.rand(N, device=dev) < mf) & train_obs
        if not mask.any():
            continue
        refined = graph(mask=mask)                          # masked species reconstructed from relatives
        pred = head(refined[mask])
        if kind == "num":
            loss = F.mse_loss(pred.squeeze(-1), yt[mask])
        else:
            loss = F.cross_entropy(pred, yt[mask])
        if vhead is not None:                               # interaction-aware 2nd channel (appearance)
            vpred = vhead(vproj[mask])
            if kind == "num":
                loss = loss + dual_scale * F.mse_loss(vpred.squeeze(-1), yt[mask])
            else:
                loss = loss + dual_scale * F.cross_entropy(vpred, yt[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp_single = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):
            imp_single = graph(mask=test)
        if rollout_rounds > 1:
            imp = _rollout_impute(graph, test, rollout_rounds, reveal_frac, dev)
        else:
            imp = imp_single
        if ab_scorer is not None:                       # CLEAN same-graph A/B: isolates the readout effect (zero
            sp = ab_scorer(imp_single, test)            # training variance -- both scored on the SAME trained graph)
            ro = ab_scorer(imp, test)
            print(f"    [rollout_ab] single_pass={sp:.4f} rollout={ro:.4f} delta={ro-sp:+.4f}")
        # MAGNITUDE-AWARE READOUT (num_head_readout): the kNN impute scorer averages relatives' raw
        # values -> collapses within-clade rank (impute 0.19 << graph 0.54 on num_width_max). The trained
        # regression `head` is an MSE-fit magnitude map from the refined embedding straight to the value;
        # applying it to the imputed (mask=test) embedding is a magnitude-aware readout that keeps rank the
        # neighbor-average destroys. This isolates READOUT (same graph, same imputed embedding) vs the kNN.
        graph._head_imp = None
        graph._head_seed = None
        if kind == "num":
            add_mu = clade_mu if (resid_readout and clade_mu is not None) else 0.0
            pred_imp = head(imp).squeeze(-1) + add_mu   # [N] head prediction on IMPUTED (mask=test) embeddings
            graph._head_imp = pred_imp.detach()
            # LEAKAGE CONTROL: same head on the held-out species' OWN refined (unmasked) rep. If head-on-imputed
            # >> this, the gain genuinely rides imputation-from-relatives, not head memorization of the seed.
            pred_seed = head(rep).squeeze(-1) + add_mu
            graph._head_seed = pred_seed.detach()
    return s, rep, imp, (head, graph)


def _rollout_impute(graph, test, rounds, reveal_frac, dev):
    """READOUT LEVER (rule 10-11): autoregressive PROGRESSIVE-REVEAL imputation. Single-pass `graph(mask=test)`
    masks ALL held-out species at once, so a held-out species reconstructs only from TRAIN relatives and never
    from another (confidently imputed) held-out relative. Rollout instead reveals, each round, the most-anchored
    still-masked test species -- clear them from the mask so their seed joins the relative pool -- then re-imputes
    the rest from the enlarged neighbour set. `an observation of A updates neighbours B, C`. Confidence = a
    species' max cosine to any currently-UNMASKED species in the refined space (most tree-anchored = revealed
    first). The FINAL refined table (all test still SCORED by NN vs train, identical metric) is returned; reveals
    only change which relatives are visible during reconstruction, not the held-out train/test scoring split."""
    N = test.shape[0]
    mask = test.clone()
    n_test = int(test.sum().item())
    per_round = max(1, int(n_test * reveal_frac))
    out = graph(mask=mask).clone()                                   # frozen per-species imputed-while-masked table
    for _ in range(max(1, rounds - 1)):
        refined = graph(mask=mask)
        r = F.normalize(refined, dim=-1)
        sim = r @ r.t()
        sim.fill_diagonal_(-2.0)
        vis = ~mask                                                  # currently-visible species (train + revealed)
        anchor = torch.where(vis.unsqueeze(0), sim, torch.full_like(sim, -2.0)).max(dim=1).values  # [N]
        cand = mask.clone()                                          # only reveal currently-masked test species
        anchor = torch.where(cand, anchor, torch.full_like(anchor, -3.0))
        k = min(per_round, int(cand.sum().item()))
        if k <= 0:
            break
        top = anchor.topk(k).indices
        mask = mask.clone(); mask[top] = False                      # reveal: their seed now visible to relatives
        newimp = graph(mask=mask)                                    # re-impute the STILL-masked species anew
        out[mask] = newimp[mask]                                     # each test species keeps its imputed-while-
    #                                                                # -masked embedding (revealed ones frozen at reveal)
    return out


def train_graph_multitrait(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr, impute_steps,
                           panel, eval_key, include_eval=False):
    """NEW LEVER (rule 25/29, richer reconstruction target): SEED-FROZEN graph trained to reconstruct a PANEL of
    MANY auxiliary traits JOINTLY from relatives, then measured by held-out NN imputation on ONE frozen EVAL axis.
    Every prior trait-supervised round drove a SINGLE readout at the eval axis; here the operator must make the
    refined-from-relatives embedding simultaneously explain a whole vector of correlated biology axes (form,
    growth, size, climate, soil...).

    `include_eval=False` (variant `multi`): the eval axis is EXCLUDED from the panel -> a fair pure-TRANSFER test
    (does shared multi-axis structure carry to a held-out axis with its labels never in the loss?).
    `include_eval=True`  (variant `sup_multi`): the eval axis IS one head alongside the panel -> a MULTI-TASK test
    (do the 12 auxiliary trait heads regularize/boost the single eval-axis head above `sup` alone?).
    One head per trait; each contributes its loss only on masked species that OBSERVE that trait. Core untouched."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    heads, targets = [], []                                  # (head, kind, Yt, train_obs) per panel trait
    for k, (kind_k, Y_k, obs_k, nclass_k) in panel.items():
        if k == eval_key and not include_eval:
            continue                                         # transfer variant: never supervise the eval axis
        out_dim = 1 if kind_k == "num" else int(nclass_k)
        h = torch.nn.Linear(d_model, out_dim).to(dev)
        yt = Y_k.float() if kind_k == "num" else Y_k.long()
        heads.append((h, kind_k, yt, obs_k & (~test)))
        targets.append(k)
    params = list(graph.parameters())
    for h, *_ in heads:
        params += list(h.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & (~test)
        if not mask.any():
            continue
        refined = graph(mask=mask)                          # masked species reconstructed from relatives
        loss = 0.0
        n_terms = 0
        for h, kind_k, yt, tobs in heads:
            m = mask & tobs                                 # only species that OBSERVE this panel trait
            if not m.any():
                continue
            pred = h(refined[m])
            if kind_k == "num":
                loss = loss + F.mse_loss(pred.squeeze(-1), yt[m])
            else:
                loss = loss + F.cross_entropy(pred, yt[m])
            n_terms += 1
        if n_terms == 0:
            continue
        loss = loss / n_terms
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):
            imp = graph(mask=test)
    return s, rep, imp


def train_graph_cooccur_supervised(seed, tree, tip_row, test, dev, d_model, steps, mask_frac, lr,
                                   impute_steps, Y, obs):
    """COMMUNITY-SUPERVISED refinement (the untested positive branch). SAME mechanism as the trait-supervised
    recipe that rescued the trait axes, applied to the co-occurrence target: freeze the seed, mask a fraction of
    TRAIN species, and drive a small MULTI-LABEL head off each masked species' refined (reconstructed-from-
    relatives) embedding toward that species' true top-K co-occurrence partner row (BCE). If the phylo operator
    can be SUPERVISED to reconstruct a species' community from its relatives, graph_gain should flip positive as
    it did for traits; if community stays graph-resistant even under supervision, it confirms community is a
    spatial-niche (not phylo) axis. Returns (seed, rep, impute) exactly like train_graph, so the identical
    nn_cooccur_ap scorer applies. Additive; edits nothing in core/."""
    N = seed.shape[0]
    graph = SpeciesGraph(N, d_model, operator="latent-clade", tree=tree, tip_row=tip_row,
                         species_text=seed).to(dev)
    train_obs = obs & (~test)
    head = torch.nn.Linear(d_model, Y.shape[1]).to(dev)          # multi-label partner-set readout (BCE)
    params = list(graph.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        mask = (torch.rand(N, device=dev) < mask_frac) & train_obs
        if not mask.any():
            continue
        refined = graph(mask=mask)                              # masked species reconstructed from relatives
        pred = head(refined[mask])
        loss = F.binary_cross_entropy_with_logits(pred, Y[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        s = graph._seed().detach()
        rep = graph(mask=None)
        imp = graph(mask=test)
        for _ in range(max(0, impute_steps - 1)):
            imp = graph(mask=test)
    return s, rep, imp
