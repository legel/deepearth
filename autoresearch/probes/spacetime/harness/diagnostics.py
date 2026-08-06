"""Encoder diagnostics for the space-time probe.

Lifted out of `scoring/` -- these measure an encoder, they do not score a run, and nothing on the
fusion path calls them. Bodies moved verbatim.
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def enforce_determinism(seed: int = 0) -> dict:
    """Make a whole RUN reproducible, not just the hash backward.

    EARTH4D_DETERMINISTIC=1 makes the hash-grid gradient bit-identical -- verified on the box, all four
    encoders. It is not sufficient. Measured after that fix, four seed-0 runs of species_from_spacetime
    still gave 0.032906 / 0.033466 / 0.034178 / 0.036721: a spread of 0.0038, which is LARGER than the
    0.002 noise-barrier floor a record has to clear. The frozen RFF control was bit-identical across the
    same runs (0.041806530207395554), so the nondeterminism is in the TRAINED path, not the data or the
    split.

    What is left after the kernel: cuBLAS split-k reductions pick different orders per launch, TF32 lets
    matmuls use a lower-precision path chosen at runtime, cuDNN autotunes an algorithm per shape, and
    torch's scatter/index kernels have nondeterministic variants. Every one of those sits between the
    encoder output and the loss.

    This pins all of them. It costs some throughput, which is the correct trade: a number nobody can
    reproduce cannot set a record, and that is exactly why the trained protocol went unused for so long.
    Returns what it set, so a run can record the guarantee it was made under.
    """
    import os
    import random
    import torch
    # cuBLAS needs this set BEFORE the first CUDA context or the workspace is already allocated.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False        # no per-shape autotune
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 picks precision at runtime
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    return {
        "determinism_hash_kernel": os.environ.get("EARTH4D_DETERMINISTIC", "") in ("1", "true", "True"),
        "determinism_cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "determinism_tf32_off": True,
        "determinism_torch_algorithms": True,
    }


def _as_device(dev):
    """Accept "cuda:0" or torch.device. probe.py passes the raw --device STRING, and an axis that
    assumed the object died with AttributeError: 'str' has no attribute 'type'. Normalise once here so
    every axis takes either."""
    import torch
    return dev if hasattr(dev, "type") else torch.device(dev)


def science_axes(enc, coords, dev, warmup: int = 5, iters: int = 20) -> dict:
    """R5 (capacity) and R4/R21 (throughput). Both are cheap and both are currently unscored.

    R5: science.md says a small model is "no less than 100M parameters". Nothing has ever asserted it.
    Counted from the live module, so a config that quietly shrinks the encoder is visible immediately
    (the v4 champion ran ~37.7M -- one table, tri-planes dropped -- and nothing said so).

    R21: "speed is a first-class score lever ... a non-compromising speedup MUST score strictly
    higher". The probe budget is CONFIG["steps"], so throughput cannot move the primary metric at all.
    Reporting fwd+bwd wall-clock per 1k coords at least makes a speedup VISIBLE while the budget is
    still counted in steps.
    """
    dev = _as_device(dev)
    import time
    import torch
    n_params = sum(p.numel() for p in enc.parameters())
    hash_params = sum(p.numel() for n, p in enc.named_parameters() if n.endswith("embeddings"))

    x = coords[: min(len(coords), 65536)].to(dev)
    for _ in range(warmup):
        enc(x).sum().backward()
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        enc(x).sum().backward()
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()
    ms_per_1k = (time.time() - t0) / iters / (len(x) / 1000.0) * 1000.0
    enc.zero_grad(set_to_none=True)

    return {
        "axis_R5_params_M": round(n_params / 1e6, 2),
        "axis_R5_hash_params_M": round(hash_params / 1e6, 2),
        "axis_R5_meets_100M_floor": bool(n_params >= 100_000_000),
        "axis_R21_fwd_bwd_ms_per_1k_coords": round(ms_per_1k, 4),
        "axis_R21_deterministic": os.environ.get("EARTH4D_DETERMINISTIC", "") in ("1", "true", "True"),
    }

def signal_capture(lat, lon, days, fam, test, n_fam, encoder_acc: float, cells=(0.05, 0.1, 0.25, 0.5)) -> dict:
    """R-signal: what FRACTION of the signal present in the coordinates does the architecture capture?

    `fair_gain vs RFF` answers "did we beat this particular competitor". It cannot answer "is the
    architecture leaving signal on the table", because RFF is an arbitrary reference with no relation
    to how much structure the coordinates actually contain. So the board could not distinguish an
    encoder that had exhausted the available signal from one capturing a third of it, and there was no
    way to know when to stop pushing architecture and start adding data channels.

    This brackets the encoder between two non-parametric references, both fit on TRAIN and scored on
    TEST under the identical split:

      FLOOR    predict the train marginal argmax, ignoring position entirely.
               = the score with ZERO coordinate information.
      CEILING  the empirical conditional p(family | spatial cell), at the finest cell size that still
               has train support, backing off to coarser cells for test points whose cell is unseen.
               This is a direct estimate of the Bayes-optimal predictor given position -- a perfect
               memorizer of the training distribution -- so no function of the coordinates can
               reliably beat it on this split.

      captured = (encoder - floor) / (ceiling - floor)

    Read it as: 1.0 means the architecture has extracted everything the coordinates hold and further
    architecture work is wasted -- go get another data channel. A low value with a high ceiling means
    the signal IS there and the architecture is failing to represent it, which is the case that
    justifies more architecture. A LOW ceiling means the coordinates are simply uninformative for this
    target on this split, and no encoder will fix that.

    Cell sizes are in degrees; the backoff makes the ceiling honest rather than a memorization artifact
    (a cell containing exactly one train point would otherwise "predict" it perfectly and inflate the
    ceiling toward 1.0 while being pure overfit).
    """
    tr, te = ~test, test
    fam_np = fam.numpy() if hasattr(fam, "numpy") else np.asarray(fam)

    counts = np.bincount(fam_np[tr], minlength=n_fam)
    floor = float((fam_np[te] == int(counts.argmax())).mean())

    # finest-first backoff: each test point is predicted by the finest cell that had train support
    pred = np.full(te.sum(), -1, dtype=np.int64)
    unfilled = np.ones(te.sum(), dtype=bool)
    for deg in sorted(cells):
        key_tr = (np.floor(lat[tr] / deg).astype(np.int64) * 100003
                  + np.floor(lon[tr] / deg).astype(np.int64))
        key_te = (np.floor(lat[te] / deg).astype(np.int64) * 100003
                  + np.floor(lon[te] / deg).astype(np.int64))
        table = {}
        order = np.argsort(key_tr, kind="stable")
        ks, fs = key_tr[order], fam_np[tr][order]
        bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo >= 3:                       # >=3 train points, else it is memorization
                table[int(ks[lo])] = int(np.bincount(fs[lo:hi], minlength=n_fam).argmax())
        for j in np.flatnonzero(unfilled):
            hit = table.get(int(key_te[j]))
            if hit is not None:
                pred[j] = hit
                unfilled[j] = False
        if not unfilled.any():
            break
    pred[unfilled] = int(counts.argmax())          # never seen at any resolution -> the floor
    ceiling = float((pred == fam_np[te]).mean())

    # The span must clear SAMPLING NOISE before a fraction of it means anything. With span > 1e-9 a
    # target carrying NO signal read floor 0.180 / ceiling 0.210 -- a 0.03 gap that is pure binomial
    # noise on 800 test points -- and an encoder sitting at 0.21 scored "captured 1.0, headroom 0.0",
    # i.e. the loop would have been told the coordinates were exhausted when they were empty. That is
    # the exact misreading this measurement exists to prevent.
    #
    # Require the span to exceed 2 standard errors of the ceiling estimate. Below that, floor and
    # ceiling are indistinguishable and the honest answer is "no measurable signal here", not a ratio.
    # The standard error of the DIFFERENCE, not of one proportion: floor and ceiling are two estimates
    # on the same test set, and the quantity that has to clear noise is the gap between them. A single-
    # proportion SE was too permissive -- a signal-free target (floor 0.180, ceiling 0.210 on 800 test
    # points) passed a 2-SE test at 0.0288 with a span of 0.030 and reported "captured 1.0".
    #
    # The ceiling is also a MAXIMUM over four cell sizes with backoff, i.e. a selected statistic, so its
    # upward bias is real. Requiring 2 SE of the difference is the minimum honest bar; anything below it
    # means floor and ceiling are the same number and no fraction of the gap is meaningful.
    n_te = max(int(te.sum()), 1)
    se = float(np.sqrt((max(ceiling * (1 - ceiling), 0.0) + max(floor * (1 - floor), 0.0)) / n_te))
    span = ceiling - floor
    measurable = span > 2.0 * se
    captured = (encoder_acc - floor) / span if measurable else float("nan")
    return {
        "axis_signal_floor": round(floor, 6),
        "axis_signal_ceiling": round(ceiling, 6),
        "axis_signal_captured": None if captured != captured else round(captured, 4),
        "axis_signal_headroom": None if captured != captured else round(1.0 - captured, 4),
        # span <= 2*SE => floor and ceiling are the same number within noise; no fraction is reportable.
        "axis_signal_measurable": bool(measurable),
    }



def field_interpolation(enc, coords, env, dev, cell_deg: float = 0.25, steps: int = 400,
                        lr: float = 3e-3, seed: int = 0) -> dict:
    """R24 — does the encoder infer a variable at a coordinate where NOTHING was observed?

    science.md rule 24: model the dense 4D field, "sampling between sparse observations in space and
    time". Every other measurement on this board scores at held-out OBSERVATION points, so a code that
    memorises observed positions perfectly and interpolates not at all scores identically to a genuine
    field. That is exactly how CMAC tile coding -- a one-hot cell indicator that CANNOT interpolate by
    construction -- came to hold the Earth4D record, and why deleting the space-time tri-planes read as
    free.

    Whole spatial CELLS are held out, so a test coordinate has no training observation anywhere near it
    and the encoder must generalise across the gap rather than look the answer up. The target is a dense
    env channel (always available at any coordinate, unlike species), reconstructed from encoder
    features by a linear head fit on train cells only.

    Control is NEAREST-NEIGHBOUR from the train cells: the interpolation any method gets for free. A
    gain over it is evidence of a learned field; at or below it, the encoder is a lookup table.
    """
    dev = _as_device(dev)
    import torch
    torch.manual_seed(seed)
    lat, lon = np.asarray(coords[:, 0].cpu()), np.asarray(coords[:, 1].cpu())
    cell = (np.floor(lat / cell_deg).astype(np.int64) * 100003
            + np.floor(lon / cell_deg).astype(np.int64))
    uniq = np.unique(cell)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(uniq, max(1, len(uniq) // 5), replace=False).tolist())
    te = np.array([c in held for c in cell])
    tr = ~te
    if tr.sum() < 32 or te.sum() < 32:
        return {"axis_R24_measurable": False}

    Y = torch.as_tensor(np.asarray(env), dtype=torch.float32, device=dev)
    m, sd = Y[torch.as_tensor(tr)].mean(0), Y[torch.as_tensor(tr)].std(0).clamp_min(1e-6)
    Y = (Y - m) / sd

    with torch.no_grad():
        F = enc(coords.to(dev)).float()
    Ftr, Fte = F[torch.as_tensor(tr)], F[torch.as_tensor(te)]
    Ytr, Yte = Y[torch.as_tensor(tr)], Y[torch.as_tensor(te)]
    head = torch.nn.Linear(F.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(head(Ftr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        enc_mse = float(torch.nn.functional.mse_loss(head(Fte), Yte))

    # nearest train observation in raw space -- the free interpolation
    ll_tr = torch.as_tensor(np.stack([lat[tr], lon[tr]], 1), dtype=torch.float32, device=dev)
    ll_te = torch.as_tensor(np.stack([lat[te], lon[te]], 1), dtype=torch.float32, device=dev)
    nn_idx = torch.cdist(ll_te, ll_tr).argmin(1)
    with torch.no_grad():
        nn_mse = float(torch.nn.functional.mse_loss(Ytr[nn_idx], Yte))
    var = float(Yte.var())
    return {
        "axis_R24_measurable": True,
        "axis_R24_encoder_r2": round(1.0 - enc_mse / max(var, 1e-9), 4),
        "axis_R24_nearest_r2": round(1.0 - nn_mse / max(var, 1e-9), 4),
        "axis_R24_vs_nearest": round((nn_mse - enc_mse) / max(var, 1e-9), 4),
        "axis_R24_held_cells": len(held),
    }


def relative_transfer(enc, coords, fam, dev, steps: int = 400, lr: float = 3e-3, seed: int = 0) -> dict:
    """R2b — does the relative encoder carry a pattern ACROSS absolute position?

    science.md rule 2B: a relative encoder over "limited context windows, focused on a limited spatial
    region, going back in time". `earth4d.py` implements it (`encode_relative`) and `fusion.py:54` calls
    it; no probe mode ever has, so half of rule 2 has never been measured.

    The test is transfer. Train on one spatial half, evaluate on the other. An ABSOLUTE encoding of a
    coordinate in region B was never seen during training, so it should collapse. An encoding of the
    OFFSET between two nearby observations is the same vector in both regions, so if the relative
    channel works it should hold up. Returns the gap.

    Requires enable_relative=True; without it the axis reports unmeasurable rather than a wrong number.
    """
    dev = _as_device(dev)
    import torch
    if not getattr(enc, "enable_relative", False):
        return {"axis_R2b_measurable": False,
                "axis_R2b_reason": "encoder built with enable_relative=False"}
    torch.manual_seed(seed)
    lat = np.asarray(coords[:, 0].cpu())
    tr = lat < np.median(lat)
    te = ~tr
    y = torch.as_tensor(np.asarray(fam), dtype=torch.long, device=dev)
    n_cls = int(y.max()) + 1

    # pair each observation with its nearest neighbour INSIDE its own half, encode the offset
    def _pairs(mask):
        idx = np.flatnonzero(mask)
        ll = torch.as_tensor(np.stack([lat[idx], np.asarray(coords[:, 1].cpu())[idx]], 1),
                             dtype=torch.float32, device=dev)
        dist = torch.cdist(ll, ll)
        dist.fill_diagonal_(float("inf"))
        j = dist.argmin(1)
        a = coords.to(dev)[torch.as_tensor(idx, device=dev)]
        b = a[j]
        return a, b, y[torch.as_tensor(idx, device=dev)]

    a_tr, b_tr, y_tr = _pairs(tr)
    a_te, b_te, y_te = _pairs(te)
    with torch.no_grad():
        rel_tr = enc.encode_relative(a_tr - b_tr).float()
        rel_te = enc.encode_relative(a_te - b_te).float()
        abs_tr, abs_te = enc(a_tr).float(), enc(a_te).float()

    def _fit(Xtr, Xte):
        head = torch.nn.Linear(Xtr.shape[1], n_cls).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(Xtr), y_tr).backward()
            opt.step()
        with torch.no_grad():
            return float((head(Xte).argmax(1) == y_te).float().mean())

    rel, absol = _fit(rel_tr, rel_te), _fit(abs_tr, abs_te)
    return {
        "axis_R2b_measurable": True,
        "axis_R2b_relative_transfer": round(rel, 4),
        "axis_R2b_absolute_transfer": round(absol, 4),
        "axis_R2b_gain": round(rel - absol, 4),
    }



def autoregressive_rollout(enc, coords, fam, days, test, dev, K: int = 16, horizon: int = 2,
                           steps: int = 400, lr: float = 3e-3, seed: int = 0) -> dict:
    """R1 — does the model CONSUME observed past state and roll its own predictions forward?

    science.md rule 1: "a causal auto-regressive model trained to forecast future states from past
    states". program.md's evidence standard #3 sharpens it: "Consume observed past state; roll your own
    predictions forward. A positional lookup at t-lag is a delayed basis, not memory."

    Every capability on this board is a past->future SPLIT, which is not the same thing. A split asks
    "does a coordinate in the future decode?"; autoregression asks "does knowing what happened nearby
    BEFORE help, and does that survive being fed the model's own output?" Nothing has ever asked the
    second question, so rule 1 has been unmeasured while the loop reported a forecast metric.

    Three arms, identical head capacity and budget:

      POSITIONAL  encoder(query coordinate) alone.               the control -- no state consumed
      OBSERVED    + aggregated state of the K nearest STRICTLY-PAST train neighbours
      ROLLED      same, but the state is the model's OWN prediction from the previous step,
                  applied `horizon` times. This is the part that separates memory from a delayed basis:
                  a delayed basis degrades to the control as soon as its input is synthetic.

    axis_R1_gain_observed  = OBSERVED - POSITIONAL   (does history help at all?)
    axis_R1_gain_rolled    = ROLLED   - POSITIONAL   (does it survive self-feeding?)

    Neighbours are drawn from TRAIN rows only and are strictly earlier in time, so no test label and no
    future information can enter the state.
    """
    dev = _as_device(dev)
    import torch
    torch.manual_seed(seed)
    lat = np.asarray(coords[:, 0].cpu()); lon = np.asarray(coords[:, 1].cpu())
    dy = np.asarray(days); te = np.asarray(test); tr = ~te
    y = torch.as_tensor(np.asarray(fam), dtype=torch.long, device=dev)
    n_cls = int(y.max()) + 1
    if tr.sum() < 64 or te.sum() < 64:
        return {"axis_R1_measurable": False, "axis_R1_reason": "split too small"}

    with torch.no_grad():
        P = enc(coords.to(dev)).float()

    tr_idx = np.flatnonzero(tr)
    ll_tr = torch.as_tensor(np.stack([lat[tr_idx], lon[tr_idx]], 1), dtype=torch.float32, device=dev)
    d_tr = torch.as_tensor(dy[tr_idx], dtype=torch.float32, device=dev)

    def _past_state(idx, state_src):
        """Mean one-hot state of the K nearest TRAIN neighbours strictly earlier in time."""
        ll = torch.as_tensor(np.stack([lat[idx], lon[idx]], 1), dtype=torch.float32, device=dev)
        dq = torch.as_tensor(dy[idx], dtype=torch.float32, device=dev)
        dist = torch.cdist(ll, ll_tr)
        dist = dist.masked_fill(d_tr.unsqueeze(0) >= dq.unsqueeze(1), float("inf"))  # strictly past
        k = min(K, ll_tr.shape[0])
        nn = dist.topk(k, largest=False).indices
        valid = torch.isfinite(dist.gather(1, nn)).float().unsqueeze(-1)
        return (state_src[nn] * valid).sum(1) / valid.sum(1).clamp_min(1.0)

    onehot_tr = torch.nn.functional.one_hot(y[torch.as_tensor(tr_idx, device=dev)], n_cls).float()
    tr_state, te_state = _past_state(tr_idx, onehot_tr), _past_state(np.flatnonzero(te), onehot_tr)
    Ptr, Pte = P[torch.as_tensor(tr)], P[torch.as_tensor(te)]
    ytr, yte = y[torch.as_tensor(tr)], y[torch.as_tensor(te)]

    def _fit(Xtr, Xte):
        head = torch.nn.Linear(Xtr.shape[1], n_cls).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(Xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            return head, float((head(Xte).argmax(1) == yte).float().mean())

    _, positional = _fit(Ptr, Pte)
    head, observed = _fit(torch.cat([Ptr, tr_state], 1), torch.cat([Pte, te_state], 1))

    # ROLLED: replace the observed state with the model's own output, `horizon` times.
    rolled_state = te_state
    with torch.no_grad():
        for _ in range(horizon):
            pred = torch.softmax(head(torch.cat([Pte, rolled_state], 1)), dim=1)
            rolled_state = pred
        rolled = float((head(torch.cat([Pte, rolled_state], 1)).argmax(1) == yte).float().mean())

    return {
        "axis_R1_measurable": True,
        "axis_R1_positional": round(positional, 4),
        "axis_R1_observed": round(observed, 4),
        "axis_R1_rolled": round(rolled, 4),
        "axis_R1_gain_observed": round(observed - positional, 4),
        "axis_R1_gain_rolled": round(rolled - positional, 4),
        "axis_R1_horizon": horizon,
    }


