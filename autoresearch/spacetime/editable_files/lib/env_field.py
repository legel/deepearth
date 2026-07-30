"""Environment-supervised dense-field decode for the spacetime encoder (science.md rule 24, done right).

Prior finding: training the 4D hash field end-to-end on the SPARSE family target (`--field_decode`) was
ACTIVELY HARMFUL (-0.10 vs a generic PE) -- the field memorized sparse training cells and generalized worst.
Rule 24 does not actually ask for a sparse-label field; it asks the encoder to "infer EVERY VARIABLE at every
space-time point, sampling between sparse observations". The physically-real, spatially-SMOOTH variables are
the ENVIRONMENT (worldclim/soil/elev), not the categorical family. This module supplies that target.

Each encoder (Earth4D / coord-MLP / fixed-RFF) is trained with a JOINT objective at the TRAIN obs:
    L = CE(family | feat)  +  aux_w * MSE(worldclim | feat)
so the positional field is shaped to reconstruct the smooth environment field it should represent, and biology
is read off the same features. We then measure biology accuracy at the strict held-out (future+new-place) set.
Fair control = the identical joint objective on the generic PEs. st_gain = env-supervised-Earth4D biology-acc
MINUS the best generic control -> isolates whether a physically-real smooth field target lets the 4D hash field
finally beat a generic learned/fixed PE, where the sparse family-supervised field failed.

Additive + flag-gated: imported only when probe.py is called with --env_decode; the default path never touches
this file.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.spacetime.editable_files.lib.recurrence import _CoordMLP


def run_env_decode(kind, coords4, rn_in, env_tgt, fam, test, n_fam, dev, enc=None, feat_dim=96,
                   steps=4000, lr=3e-3, head_hidden=256, aux_w=1.0, wd=0.0):
    """Joint env-recon + biology decode, then held-out biology accuracy.

    kind='earth4d' : trainable Earth4D encoder (enc) -> shared features
    kind='mlp'     : trainable coord-MLP on (lat/90,lon/180[,t]) -> shared features   (generic learned PE)
    kind='rff'     : FIXED random-Fourier features -> shared features                  (fixed positional control)

    env_tgt [N, E] : standardized environment field (worldclim) -- the smooth rule-24 target.
    Returns (bio_acc, bio_top5, n_held, env_val_r2) where env_val_r2 is 1 - MSE/Var of the env recon on the
    held-out set (aux fit quality, diagnostic only)."""
    coords4 = coords4.to(dev)
    rn = torch.tensor(rn_in).to(dev)
    y = torch.tensor(fam).long().to(dev)
    E = torch.tensor(env_tgt).to(dev)
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

    bio_head = nn.Sequential(nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, n_fam)).to(dev)
    env_head = nn.Sequential(nn.Linear(in_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, E.shape[1])).to(dev)
    params = list(bio_head.parameters()) + list(env_head.parameters()) + (list(encoder.parameters()) if encoder is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def feats_of(idx):
        if kind == "rff":
            return enc_in[idx]
        return encoder(enc_in[idx])

    Ntr = tr_i.shape[0]
    for _ in range(steps):
        sel = tr_i[torch.randint(0, Ntr, (4096,), device=dev)]
        f = feats_of(sel)
        bio_loss = F.cross_entropy(bio_head(f), y[sel])
        env_loss = F.mse_loss(env_head(f), E[sel])
        loss = bio_loss + aux_w * env_loss
        opt.zero_grad(); loss.backward(); opt.step()
    if encoder is not None: encoder.eval()
    bio_head.eval(); env_head.eval()
    with torch.no_grad():
        accs, t5s, tot = 0, 0, 0
        env_sse, env_n = 0.0, 0
        for s in range(0, te_i.shape[0], 8192):
            b = te_i[s:s + 8192]
            f = feats_of(b); yy = y[b]
            logits = bio_head(f)
            accs += (logits.argmax(-1) == yy).sum().item()
            t5s += (logits.topk(5, -1).indices == yy[:, None]).any(-1).sum().item()
            tot += b.shape[0]
            env_sse += F.mse_loss(env_head(f), E[b], reduction="sum").item()
            env_n += b.shape[0] * E.shape[1]
        # env_tgt is standardized (unit variance per col) so held-out Var ~ 1 -> R2ish = 1 - MSE
        env_r2 = 1.0 - (env_sse / max(env_n, 1))
    return accs / tot, t5s / tot, tot, env_r2
