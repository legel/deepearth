"""Editable Earth4D readouts, objectives, and candidate training behavior.

The fixed probe supplies a budget iterator and records outcomes. This module
owns how candidate representations are fit and decoded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from deepearth.autoresearch.probes.spacetime.editable_files.lib.candidate_data import CONFIG
from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import build_probe_readout


def _candidate_steps(stepper, steps, tag):
    if stepper is None:
        raise ValueError("candidate training requires the fixed probe's budget iterator")
    return stepper(steps, tag)


GROUP_DRO_TEMPERATURE = 10.0


def _domain_groups(train_domains, n_train, dev):
    """Validate train-aligned chronological domain ids and return their row-index banks."""
    if train_domains is None:
        return None
    domains = torch.as_tensor(train_domains, dtype=torch.long, device=dev)
    if domains.ndim != 1 or domains.numel() != n_train:
        raise ValueError("chronological domains must align exactly with the training rows")
    values = torch.unique(domains, sorted=True)
    groups = [torch.nonzero(domains == value, as_tuple=False).squeeze(1) for value in values]
    if len(groups) < 2 or any(group.numel() == 0 for group in groups):
        raise ValueError("GroupDRO needs at least two non-empty chronological domains")
    return groups


def _group_dro_indices(groups, batch_size, generator):
    """Draw an equal subbatch from every domain with a sampler independent of model RNG use."""
    per_group = batch_size // len(groups)
    if per_group * len(groups) != batch_size:
        raise ValueError("GroupDRO batch size must divide evenly across domains")
    parts = [group[torch.randint(group.numel(), (per_group,), device=group.device,
                                 generator=generator)] for group in groups]
    return torch.cat(parts), per_group


def _smooth_worst_domain_ce(logits, targets, per_group, n_groups):
    losses = torch.stack([
        F.cross_entropy(logits[i * per_group:(i + 1) * per_group],
                        targets[i * per_group:(i + 1) * per_group])
        for i in range(n_groups)
    ])
    return torch.logsumexp(GROUP_DRO_TEMPERATURE * losses, dim=0) / GROUP_DRO_TEMPERATURE


def evaluate_candidate(feats, fam, test, n_fam, dev, steps, lr, tag, head_hidden=0, seed=0,
             train_domains=None, support_feats=None, support_fam=None, support_partner_indices=None,
             temporal_phase=None, stepper=None):
    """Train a linear head feats->family on TRAIN locations; report held-out-block accuracy.

    knn_readout > 0 swaps the parametric head for a NON-PARAMETRIC one: the k nearest TRAIN rows in this
    arm's own feature space vote (inverse-distance weighted). The hypothesis is that env->family is a LOCAL
    DENSITY question -- which families occur in this kind of place -- and a linear softmax over a 64-d
    AlphaEarth embedding cannot express a local frequency table however many columns are appended to it.
    Every arm (raw, RFF, Earth4D, env, fused) gets the identical readout, so the comparison stays fair."""
    torch.manual_seed(seed)
    if CONFIG["knn_readout"] > 0:
        k = CONFIG["knn_readout"]
        Xtr = feats[~test].to(dev); ytr = fam[~test].to(dev)
        Xte = feats[test].to(dev); yte = fam[test].to(dev)
        Xtr = Xtr / Xtr.norm(dim=1, keepdim=True).clamp_min(1e-6)
        Xteh = Xte / Xte.norm(dim=1, keepdim=True).clamp_min(1e-6)
        hits = torch.zeros(Xte.shape[0], dtype=torch.bool, device=dev)
        top5h = torch.zeros(Xte.shape[0], dtype=torch.bool, device=dev)
        for i in range(0, Xteh.shape[0], 2048):
            sim = Xteh[i:i + 2048] @ Xtr.T
            v, idx = sim.topk(k, dim=1)
            w = (v.clamp_min(0.0) + 1e-6)
            votes = torch.zeros(idx.shape[0], n_fam, device=dev)
            votes.scatter_add_(1, ytr[idx], w)
            hits[i:i + 2048] = votes.argmax(1) == yte[i:i + 2048]
            top5h[i:i + 2048] = (votes.topk(5, 1).indices == yte[i:i + 2048, None]).any(1)
        acc = hits.float().mean().item(); t5 = top5h.float().mean().item()
        print(f"  [{tag}] knn{k} acc {acc:.4f}", flush=True)
        return acc, t5
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train = ~test
    Xtr, ytr = feats[train].to(dev), fam[train].to(dev)
    Xte, yte = feats[test].to(dev), fam[test].to(dev)
    if support_feats is not None:
        Xs, ys = support_feats.to(dev), support_fam.to(dev)
        if Xs.shape[1] != Xtr.shape[1] or len(Xs) != len(ys):
            raise ValueError("historical range-support features and species labels must align")
        Xbank, ybank = torch.cat([Xtr, Xs]), torch.cat([ytr, ys])
    else:
        Xbank, ybank = Xtr, ytr
    partners = None
    if support_partner_indices is not None:
        if support_feats is None:
            raise ValueError("local cross-era partners require support features")
        partners = support_partner_indices.to(dev)
        if partners.shape != ytr.shape or torch.any((partners < -1) | (partners >= len(Xs))):
            raise ValueError("local cross-era partners must align with current training rows and support")
    Ptr = Pte = None
    if temporal_phase is not None:
        if len(temporal_phase) != len(feats) or partners is not None:
            raise ValueError("temporal transport must align with observations and cannot replace cross-era pairing")
        Ptr, Pte = temporal_phase[train].to(dev), temporal_phase[test].to(dev)
    head = build_probe_readout(
        feats.shape[1], head_hidden, n_fam,
        cross_era=partners is not None, temporal=Ptr is not None,
    ).to(dev)
    domain_groups = _domain_groups(train_domains, Xtr.shape[0], dev)
    domain_generator = (torch.Generator(device=dev).manual_seed(seed)
                        if domain_groups is not None else None)

    def _knn_logp(Q, K=int(CONFIG["knn_vote_k"]), tau=float(CONFIG["knn_vote_tau"]), chunk=2048):
        """Soft k-NN class log-posterior against dated train + historical support."""
        Xn = F.normalize(Xbank, dim=-1)
        out = []
        for i in range(0, Q.shape[0], chunk):
            sim = F.normalize(Q[i:i + chunk], dim=-1) @ Xn.t()
            v, j = sim.topk(min(K, sim.shape[1]), dim=-1)
            w = torch.softmax(v / tau, dim=-1)
            p = torch.zeros(j.shape[0], n_fam, device=dev)
            p.scatter_add_(1, ybank[j], w)
            out.append((p + 1e-6).log())
        return torch.cat(out)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in _candidate_steps(stepper, steps, tag):
        if domain_groups is None:
            idx = torch.randint(0, Xtr.shape[0], (4096,), device=dev)
            if partners is not None:
                paired = partners[idx]
                valid_rows = torch.nonzero(paired >= 0, as_tuple=False).squeeze(1)
                loss = head.loss(Xtr[idx], ytr[idx], Xs[paired[valid_rows]], valid_rows)
            else:
                logits = head(Xtr[idx], Ptr[idx]) if Ptr is not None else head(Xtr[idx])
                loss = F.cross_entropy(logits, ytr[idx])
        else:
            idx, per_group = _group_dro_indices(domain_groups, 4096, domain_generator)
            logits = head(Xtr[idx], Ptr[idx]) if Ptr is not None else head(Xtr[idx])
            loss = _smooth_worst_domain_ce(logits, ytr[idx], per_group, len(domain_groups))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = head(Xte, Pte) if Pte is not None else head(Xte)
        logits = logits + (CONFIG["knn_vote"] * _knn_logp(Xte) if CONFIG["knn_vote"] else 0.0)
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5


def train_candidate(enc, coords, fam, test, n_fam, dev, steps, lr, tag, head_hidden=0,
                       enc_lr_mult=0.05, warmup=0.15, c2f=0.5, clip=1.0, seed=0, side=None,
                       train_domains=None, support_coords=None, support_fam=None,
                       support_partner_indices=None, temporal_phase=None, stepper=None):
    """Train the ENCODER end-to-end with the head, instead of reading a frozen random hash table.

    Every other probe path calls enc(coords) under no_grad on a freshly-initialized Earth4D, so its hash table
    stays RANDOM: the reported fair-gains compare architectural priors as fixed random feature maps, not a
    trained encoder. For an architecture whose premise is a LEARNED table that is close to its worst case.

    A hash grid does not train stably by default here (a bare table on a Poisson objective returned +0.88 /
    +0.44 / +0.35 across seeds), so three standard stabilizers are on:
      * the encoder gets its OWN param group at lr*enc_lr_mult with no weight decay,
      * linear LR WARMUP over the first `warmup` fraction of steps (project memory: off-champion configs
        NaN/collapse without it),
      * COARSE-TO-FINE level unmasking -- fine hash levels are zeroed early and released over the first `c2f`
        fraction, the standard remedy for hash-grid overfitting/instability.
    Returns (acc, top5) with the SAME protocol as evaluate() so the numbers stay comparable."""
    torch.manual_seed(seed)
    train = ~test
    Ctr, ytr = coords[train].to(dev), fam[train].to(dev)
    Cte, yte = coords[test].to(dev), fam[test].to(dev)
    Cs = ys = None
    if support_coords is not None:
        Cs, ys = support_coords.to(dev), support_fam.to(dev)
        if Cs.shape[1] != Ctr.shape[1] or len(Cs) != len(ys):
            raise ValueError("historical range-support coordinates and species labels must align")
        if side is not None:
            raise ValueError("historical range support is defined for bare space-time only")
    partners = None
    if support_partner_indices is not None:
        if Cs is None:
            raise ValueError("local cross-era partners require support coordinates")
        partners = support_partner_indices.to(dev)
        if partners.shape != ytr.shape or torch.any((partners < -1) | (partners >= len(Cs))):
            raise ValueError("local cross-era partners must align with current training rows and support")
    # `side` = static per-observation features (the ENV channel) concatenated to the encoder output, so the
    # FUSED arm can be trained end-to-end. Without it the fused arm always reads a frozen random hash table:
    # train_encoder moved the Earth4D-alone arm 0.0938 -> 0.1105 and left the fused primary byte-identical at
    # 0.142318, because `fused` was built from the frozen features before any training happened.
    Str = Ste = None
    if side is not None:
        Str, Ste = side[train].to(dev), side[test].to(dev)
    with torch.no_grad():
        edim = enc(Ctr[:8]).shape[1]
        fdim = edim + (0 if side is None else Str.shape[1])
    Ptr = Pte = None
    if temporal_phase is not None:
        if len(temporal_phase) != len(coords) or partners is not None:
            raise ValueError("temporal transport must align with observations and cannot replace cross-era pairing")
        Ptr, Pte = temporal_phase[train].to(dev), temporal_phase[test].to(dev)
    head = build_probe_readout(
        fdim, head_hidden, n_fam,
        cross_era=partners is not None, temporal=Ptr is not None,
    ).to(dev)
    domain_groups = _domain_groups(train_domains, Ctr.shape[0], dev)
    domain_generator = (torch.Generator(device=dev).manual_seed(seed)
                        if domain_groups is not None else None)
    opt = torch.optim.Adam([{"params": head.parameters(), "lr": lr},
                            {"params": list(enc.parameters()), "lr": lr * enc_lr_mult, "weight_decay": 0.0}])
    # per-level feature mask over the SPATIAL block (levels are contiguous, features_per_level each)
    fpl = getattr(enc, "features_per_level", 2); sdim = getattr(enc, "spatial_dim", edim)
    n_lv = max(int(sdim // max(fpl, 1)), 1)
    lvl_of = (torch.arange(edim, device=dev) // max(fpl, 1)).clamp(max=n_lv - 1)   # ENCODER dims only
    warm_n, c2f_n = max(int(steps * warmup), 1), max(int(steps * c2f), 1)
    _p0 = {n: q.detach().clone() for n, q in enc.named_parameters()}   # sanity: did the encoder ACTUALLY move?
    for it in _candidate_steps(stepper, steps, tag):
        for gi, base in enumerate((lr, lr * enc_lr_mult)):
            opt.param_groups[gi]["lr"] = base * min(1.0, (it + 1) / warm_n)      # linear warmup
        keep = n_lv if it >= c2f_n else max(1, int(n_lv * (it + 1) / c2f_n))     # coarse-to-fine
        if domain_groups is None:
            idx = torch.randint(0, Ctr.shape[0], (4096,), device=dev)
            per_group = None
        else:
            idx, per_group = _group_dro_indices(domain_groups, 4096, domain_generator)
        if partners is not None:
            paired = partners[idx]
            valid_rows = torch.nonzero(paired >= 0, as_tuple=False).squeeze(1)
            encoded = enc(torch.cat([Ctr[idx], Cs[paired[valid_rows]]], dim=0))
            f, fp = encoded[:len(idx)], encoded[len(idx):]
        else:
            valid_rows = None
            f, fp = enc(Ctr[idx]), None
        if keep < n_lv:
            level_mask = (lvl_of < keep).to(f.dtype)
            f = f * level_mask
            if fp is not None:
                fp = fp * level_mask
        if Str is not None:
            f = torch.cat([f, Str[idx]], 1)
        if partners is not None:
            loss = head.loss(f, ytr[idx], fp, valid_rows)
        else:
            logits = head(f, Ptr[idx]) if Ptr is not None else head(f)
            loss = (F.cross_entropy(logits, ytr[idx]) if domain_groups is None else
                    _smooth_worst_domain_ce(logits, ytr[idx], per_group, len(domain_groups)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), clip)
        opt.step()
    with torch.no_grad():
        moved = {n: (q - _p0[n]).norm().item() / max(_p0[n].norm().item(), 1e-9) for n, q in enc.named_parameters()}
        tot = sum(1 for v in moved.values() if v > 1e-6)
        print(f"  [train_encoder] {tot}/{len(moved)} encoder tensors moved; "
              f"rel-delta " + ", ".join(f"{n.split('.')[-1]}={v:.3g}" for n, v in list(moved.items())[:6]), flush=True)
        def _feat(C, S):
            f = enc(C)
            return f if S is None else torch.cat([f, S], 1)
        Fte = torch.cat([_feat(Cte[i:i + 8192], None if Ste is None else Ste[i:i + 8192])
                         for i in range(0, Cte.shape[0], 8192)])
        logits = head(Fte, Pte) if Pte is not None else head(Fte)
        # ESTIMATOR PARITY with evaluate(): the frozen champion path adds a soft k-NN class log-vote
        # over the TRAIN rows (knn_vote=0.5 for family_from_spacetime). Omitting it here made A-vs-B a
        # comparison of two ESTIMATORS, not of frozen-vs-trained hash tables. Same K/tau/cosine, over
        # the TRAINED features. No-op when knn_vote == 0, so every other capability is byte-identical.
        if CONFIG["knn_vote"]:
            Ftr = torch.cat([_feat(Ctr[i:i + 8192], None if Str is None else Str[i:i + 8192])
                             for i in range(0, Ctr.shape[0], 8192)])
            if Cs is not None:
                Fs = torch.cat([_feat(Cs[i:i + 8192], None)
                                for i in range(0, Cs.shape[0], 8192)])
                Fbank, ybank = torch.cat([Ftr, Fs]), torch.cat([ytr, ys])
            else:
                Fs = None
                Fbank, ybank = Ftr, ytr
            _K, _tau = int(CONFIG["knn_vote_k"]), float(CONFIG["knn_vote_tau"])
            _Xn = F.normalize(Fbank, dim=-1)
            _out = []
            for i in range(0, Fte.shape[0], 2048):
                _sim = F.normalize(Fte[i:i + 2048], dim=-1) @ _Xn.t()
                _v, _j = _sim.topk(min(_K, _sim.shape[1]), dim=-1)
                _w = torch.softmax(_v / _tau, dim=-1)
                _pr = torch.zeros(_j.shape[0], n_fam, device=dev)
                _pr.scatter_add_(1, ybank[_j], _w)
                _out.append((_pr + 1e-6).log())
            logits = logits + CONFIG["knn_vote"] * torch.cat(_out)
            del Ftr, Fbank, Fs, _Xn
        acc = (logits.argmax(-1) == yte).float().mean().item()
        top5 = (logits.topk(5, -1).indices == yte[:, None]).any(-1).float().mean().item()
    return acc, top5
