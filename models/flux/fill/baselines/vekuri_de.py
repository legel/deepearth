"""Deep ensemble with a Laplace likelihood of Vekuri et al. 2025 as a FluxFill-Bench method.

Vekuri, H., Tuovinen, J.-P., Kulmala, L., et al. 2025. Neural networks provide a robust gap-filling method with
uncertainty estimates for eddy covariance CO2 fluxes (deep ensemble with aleatoric and epistemic uncertainty).
Code: github.com/hvekuri/nn_gapfilling_unc, commit 7edd07a6775aa89b3ace059bde60c770376f2c3f (local clone
/home/photon/ecological/flux_gapfill/baselines_src/nn_gapfilling_unc). The released code is TensorFlow/Keras; this
is a PyTorch port of the same network, loss, optimiser and training schedule.

Ported, with the released lines (gapfill_NN.py unless stated):
- Inputs: TA, SW_IN, VPD, DOY, Month_sin, Month_cos, Hour_sin, Hour_cos (gapfilling_demo.ipynb cell 4; time
  variables from `add_time_vars`, cell 2).
- Scaling: StandardScaler on all rows' inputs (line 146) and on the training rows' target (line 147).
- Network: Dense(50, relu, HeNormal, L2 1e-4) -> Dense(50, relu, HeNormal, L2 1e-4) -> Dense(2, linear,
  GlorotUniform), outputs location and log scale (lines 45-64).
- Loss: Laplace negative log likelihood, mean over the batch, plus the constant 5 (lines 16-28); Keras adds the L2
  penalties to both the training and the validation loss.
- Optimiser: Adam, learning rate 1e-4 (cell 4; line 66), Keras update rule (epsilon 1e-7 added to sqrt(v) without
  bias correction, bias correction folded into the step size).
- Training: batch 64, at most 500 epochs, reshuffled every epoch, early stopping on validation loss with patience
  20 and restore_best_weights=False (lines 86-103). Each member draws its own random 10 % validation split
  (train_test_split(test_size=0.1), line 120).
- Ensemble: N = 5 members (cell 4). Aleatoric = Laplace scale per member; epistemic = spread of member means
  (lines 172-191).

Changes, each stated:
1. Training rows: the site's other calendar years (runner `context`) plus the visible part of the window, where the
   flux is measured; inputs hidden by the mask, or missing, are replaced by ERA5 (common.py). The released code fits
   one site-year table.
2. Month_sin and Month_cos are computed from each half-hour's position in its calendar year,
   2 pi ((doy - 1) 48 + slot) / (365 48), which is what `add_time_vars` computes for a table that starts on 1 January
   and has every row. The released code uses the row number within each year, which shifts the phase when rows are
   missing or the table starts mid-year; its start-of-year test `month == 1 & day == 1` also parses as a chained
   comparison with `1 & day`.
3. The members of all masks of one window and all its fluxes are trained together as one batched network
   (independent weights, data, validation splits, batch orders, optimiser states and early stopping per member),
   for speed; on the GPU one training step is replayed as a CUDA graph.
   Each member's updates are those of its own Keras run up to floating-point order; a member that has stopped keeps
   the weights of its stopping epoch.
4. Keras initialisers: HeNormal = truncated normal (2 standard deviations) with stddev sqrt(2 / fan_in) / 0.8796;
   GlorotUniform limit sqrt(6 / (fan_in + fan_out)); zero biases.

Samples (joint over time within a member): sample s belongs to member s mod 5 and is that member's mean trajectory
plus Laplace noise with the member's scale, independent per half-hour. A member's mean is a coherent trajectory, so
the spread of sums across samples is the between-member spread of summed means (epistemic, coherent over the gap)
plus the summed independent aleatoric variance; this is the structure of the released sum uncertainty
(gapfilling_demo.ipynb cell 6: epistemic = 1.96 x std of member sums, aleatoric = sqrt(sum of squared aleatoric)).
The predictive distribution of one value is the equal mixture of the members' Laplace distributions.
"""

import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from common import Context, context_keys, fill_output, flux_rows, place, seed_of, window_driver

DRIVERS = ("TA", "SW_IN", "VPD")
HPARAMS = dict(lr=1e-4, batch=64, l2=1e-4, epochs=500, patience=20, hidden=50, members=5, val_frac=0.1)


def time_vars(t: pd.DatetimeIndex) -> np.ndarray:
    """DOY, Month_sin, Month_cos, Hour_sin, Hour_cos (add_time_vars, gapfilling_demo.ipynb cell 2; change 2)."""
    doy = t.dayofyear.to_numpy()
    slot = (t.hour * 2 + t.minute // 30).to_numpy()
    xnorm = 2 * np.pi * ((doy - 1) * 48 + slot) / (365 * 48)
    hour = (t.hour + t.minute / 60).to_numpy()
    return np.stack([doy, np.sin(xnorm), np.cos(xnorm), np.sin(hour * 2 * (2 * np.pi / 48)),
                     np.cos(hour * 2 * (2 * np.pi / 48))], 1).astype(np.float64)


class _Batched:
    """N independent 2-hidden-layer MLPs as batched tensors, trained with per-network Adam and early stopping."""

    def __init__(self, n_net: int, d_in: int, hp: Dict, gen, device):
        import torch
        h = hp["hidden"]

        def he(fan_in, shape):
            std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
            w = torch.empty(shape, dtype=torch.float32)
            torch.nn.init.trunc_normal_(w, 0.0, 1.0, -2.0, 2.0, generator=gen)
            return w * std

        lim = math.sqrt(6.0 / (h + 2))
        self.p = [he(d_in, (n_net, d_in, h)), torch.zeros(n_net, 1, h), he(h, (n_net, h, h)), torch.zeros(n_net, 1, h),
                  (torch.rand((n_net, h, 2), generator=gen) * 2 - 1) * lim, torch.zeros(n_net, 1, 2)]
        self.p = [x.to(device).requires_grad_(True) for x in self.p]
        for x in self.p:
            x.grad = torch.zeros_like(x)
        self.m = [torch.zeros_like(x) for x in self.p]
        self.v = [torch.zeros_like(x) for x in self.p]
        self.t = torch.zeros(n_net, device=device)
        self.hp = hp

    def forward(self, x, p=None):
        import torch
        w1, b1, w2, b2, w3, b3 = self.p if p is None else p
        h = torch.relu(torch.bmm(x, w1) + b1)
        h = torch.relu(torch.bmm(h, w2) + b2)
        return torch.bmm(h, w3) + b3

    def loss(self, x, y, wt, p=None):
        """Per-network Keras loss: weighted mean Laplace NLL + 5 + L2 on the hidden kernels. (N,)"""
        import torch
        o = self.forward(x, p)
        mu, logb = o[..., 0], o[..., 1]
        nll = logb + math.log(2.0) + (y - mu).abs() / logb.exp()
        data = (nll * wt).sum(1) / wt.sum(1).clamp_min(1.0) + 5.0
        P = self.p if p is None else p
        reg = self.hp["l2"] * ((P[0] ** 2).sum((1, 2)) + (P[2] ** 2).sum((1, 2)))
        return data + reg

    def step(self, active):
        """Keras Adam update on the networks flagged active (bool tensor (N,))."""
        import torch
        lr, b1, b2, eps = self.hp["lr"], 0.9, 0.999, 1e-7
        with torch.no_grad():
            self.t += active.float()
            t = self.t.clamp_min(1.0)
            alpha = lr * torch.sqrt(1 - b2 ** t) / (1 - b1 ** t)
            a = active.float().view(-1, 1, 1)
            for x, m, v in zip(self.p, self.m, self.v):
                g = x.grad
                m.add_(a * (1 - b1) * (g - m))
                v.add_(a * (1 - b2) * (g * g - v))
                x.sub_(a * alpha.view(-1, 1, 1) * m / (v.sqrt() + eps))
                g.zero_()


def train_ensemble(Xtr: List[np.ndarray], ytr: List[np.ndarray], hp: Dict, seed: int, device="cpu", log=None):
    """Train one network per (X, y) pair (already scaled). Returns the parameter list of the final weights."""
    import torch
    N = len(Xtr)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    d_in = Xtr[0].shape[1]
    tr_idx, va_idx = [], []
    for X in Xtr:
        perm = rng.permutation(len(X))
        nv = int(math.ceil(hp["val_frac"] * len(X)))           # sklearn test_size=0.1 rounds up
        va_idx.append(perm[:nv]); tr_idx.append(perm[nv:])
    ntr = np.array([len(i) for i in tr_idx])
    nmax, vmax = int(ntr.max()), int(max(len(i) for i in va_idx))
    Xt = torch.zeros(N, nmax, d_in); Yt = torch.zeros(N, nmax)
    Xv = torch.zeros(N, vmax, d_in); Yv = torch.zeros(N, vmax); Wv = torch.zeros(N, vmax)
    for k in range(N):
        a, b = tr_idx[k], va_idx[k]
        Xt[k, :len(a)] = torch.from_numpy(Xtr[k][a]).float(); Yt[k, :len(a)] = torch.from_numpy(ytr[k][a]).float()
        Xv[k, :len(b)] = torch.from_numpy(Xtr[k][b]).float(); Yv[k, :len(b)] = torch.from_numpy(ytr[k][b]).float()
        Wv[k, :len(b)] = 1.0
    Xt, Yt, Xv, Yv, Wv = (x.to(device) for x in (Xt, Yt, Xv, Yv, Wv))
    net = _Batched(N, d_in, hp, gen, device)
    B = hp["batch"]
    nb = np.ceil(ntr / B).astype(int)
    best = np.full(N, np.inf); wait = np.zeros(N, int); stopped = np.zeros(N, bool)
    final = [x.detach().clone() for x in net.p]
    it_s = torch.zeros((N, B), dtype=torch.int64, device=device)
    wt_s = torch.zeros((N, B), dtype=torch.float32, device=device)

    def one_step():
        xb = torch.gather(Xt, 1, it_s[..., None].expand(-1, -1, d_in)); yb = torch.gather(Yt, 1, it_s)
        net.loss(xb, yb, wt_s).sum().backward()
        net.step(wt_s[:, 0] > 0)

    graph = None
    if str(device).startswith("cuda"):
        # One CUDA graph replays the whole step (gather, forward, backward, Adam): the same kernels as the eager
        # step, launched at once. Warm-up and capture run with every batch weight 0, so no network is updated.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                one_step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            one_step()
    for ep in range(hp["epochs"]):
        J = int(nb[~stopped].max())
        P = np.zeros((N, J * B), np.int64); W = np.zeros((N, J * B), np.float32)
        for k in np.flatnonzero(~stopped):                  # each member's own reshuffle of its training rows
            P[k, :ntr[k]] = rng.permutation(ntr[k]); W[k, :ntr[k]] = 1.0
        P, W = torch.from_numpy(P).to(device), torch.from_numpy(W).to(device)
        for j in range(J):
            it_s.copy_(P[:, j * B:(j + 1) * B]); wt_s.copy_(W[:, j * B:(j + 1) * B])
            if graph is not None:
                graph.replay()
            else:
                one_step()
        with torch.no_grad():
            vl = net.loss(Xv, Yv, Wv).cpu().numpy()
        for k in np.flatnonzero(~stopped):
            if vl[k] < best[k]:
                best[k], wait[k] = vl[k], 0
            else:
                wait[k] += 1
                if wait[k] >= hp["patience"]:
                    stopped[k] = True
                    for f, x in zip(final, net.p):
                        f[k] = x.detach()[k]
        if log is not None:
            log.append(dict(epoch=ep, active=int((~stopped).sum()), val=float(np.mean(vl))))
        if os.environ.get("FLUXFILL_VERBOSE") and ep % 10 == 0:
            print(f"  vekuri epoch {ep}: {int((~stopped).sum())}/{N} networks training", flush=True)
        if stopped.all():
            break
    for f, x in zip(final, net.p):          # members still running at the epoch limit keep their last weights
        f[~torch.from_numpy(stopped)] = x.detach()[~torch.from_numpy(stopped)]
    return net, final


class VekuriDE:
    name = "vekuri_de"

    def __init__(self, n_samples: int = 80, hparams: Dict = None, device: str = "cpu", threads: int = 2, seed: int = 0):
        self.hp = dict(HPARAMS if hparams is None else hparams)
        self.n, self.device, self.threads, self.seed = n_samples, device, threads, seed
        self.log = []

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        return self.predict_many([(w, mask)], context)[0]

    def predict_many(self, items, context: Dict):
        """Fill every (visible window, mask) of one window; all their members train as one batch. Each mask's
        networks see only that mask's visible values."""
        import torch
        torch.set_num_threads(self.threads)
        M = self.hp["members"]
        w0 = items[0][0]
        start = str(w0.start.date())
        ctx = Context.from_runner(context, context_keys(DRIVERS))
        Xc = np.column_stack([ctx.driver(d) for d in DRIVERS] + [time_vars(ctx.t)])
        outs, Xs, ys, meta = [], [], [], []
        for n_item, (w, mask) in enumerate(items):
            outs.append(fill_output(w, mask, self.n))
            Xw = np.column_stack([window_driver(w, mask, d) for d in DRIVERS] + [time_vars(w.t)])
            Xall = np.vstack([Xc, Xw])
            keep = np.isfinite(Xall).all(1)                    # rows with every input (the released demo table has
            mu_x = Xall[keep].mean(0); sd_x = Xall[keep].std(0)  # gap-filled drivers, so it drops none)
            sd_x = np.where(sd_x > 0, sd_x, 1.0)                 # StandardScaler on all rows' inputs (line 146)
            for i in flux_rows(w, mask):
                k = w.keys[i]
                yc, yw = ctx.get(k), np.where(mask[i], np.nan, w.values[i])
                y = np.concatenate([yc, yw])
                ok = np.isfinite(y) & keep
                if ok.sum() < 48 * 7:
                    continue                                    # under a week of training data: left unfilled
                my, sy = y[ok].mean(), y[ok].std() or 1.0       # StandardScaler on the target (line 147)
                for m in range(M):
                    Xs.append((Xall[ok] - mu_x) / sd_x); ys.append((y[ok] - my) / sy)
                meta.append((n_item, i, my, sy, (np.where(np.isfinite(Xw), Xw, mu_x) - mu_x) / sd_x))
        if not meta:
            return outs
        log = []
        key = tuple(int(mk.any(0).sum()) for _, mk in items)
        net, final = train_ensemble(Xs, ys, self.hp, seed_of(self.seed, w0.site, start, key), self.device, log)
        self.log.append(dict(site=w0.site, window=start, n_masks=len(items), epochs=len(log), n_net=len(Xs)))
        rng = np.random.default_rng(seed_of(self.seed, w0.site, start, key, "s"))
        member = np.arange(self.n) % M
        for v, (n_item, i, my, sy, Xp) in enumerate(meta):
            w, mask = items[n_item]
            with torch.no_grad():
                xp = torch.from_numpy(Xp).float().to(self.device)[None].expand(M, -1, -1).contiguous()
                o = net.forward(xp, [f[v * M:(v + 1) * M] for f in final]).cpu().numpy().astype(np.float64)
            hid = mask[i]
            mu = o[:, :, 0] * sy + my                                  # (M, T) member means (line 174)
            b = np.exp(o[:, :, 1]) * sy                                # (M, T) member Laplace scales
            s = np.full((self.n, w.values.shape[1]), np.nan)
            s[:, hid] = mu[member][:, hid] + b[member][:, hid] * rng.laplace(size=(self.n, int(hid.sum())))
            place(outs[n_item], i, hid, s)
        return outs
