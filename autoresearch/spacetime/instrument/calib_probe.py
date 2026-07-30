"""Additive, flag-gated CALIBRATION probe for the encoder SDM head (science.md rule 17).

DOES NOT touch core/fusion.py, evaluate.py, encoders/*, earth4d.py, phenology.py, recurrence.py,
or any existing probe's default path. Default-off. Scratch only. Nothing committed by this file.

Question (the untested vertical, Bayesian-calibration gap ~0.143, B23 species_calibration /
B53 pollinator_calibration): the encoders may be ACCURATE but OVERCONFIDENT. On a nature-derived
held-out prediction task (env/coords -> family-presence SDM, exactly the head `probe.py:evaluate`
already trains), do the raw predicted class-probabilities match empirical frequencies (low ECE,
good reliability)? And do post-hoc calibration MECHANISMS cut ECE/Brier WITHOUT hurting accuracy
or ranking? And is the resulting confidence a HONEST predictor of the model's own error
(AUROC of confidence vs correctness)?

Every feature here is nature-derived (occurrence lat/lon, worldclim/soil/elev env), identical
provenance to probe.py -- no external data, fully science-respecting. Calibration is a pure
modeling/uncertainty choice layered on top of the same held-out predictions.

Usage on newbox (real Earth4D held-out logits, reuses probe.py loaders + head):
  CUDA_VISIBLE_DEVICES=0 python -m deepearth.autoresearch.spacetime.instrument.calib_probe \
      --cache_dir data/deepcal --feature env --steps 400 --seed 0

If the token cache is absent (e.g. code-only checkout), pass --surrogate to run the IDENTICAL
calibration machinery on a nature-structured surrogate SDM task (spatially-blocked families with
env-correlated logits) so the mechanisms themselves are measured end-to-end. The calibration
math (ECE/Brier/temperature/isotonic/conformal/ensemble/MC-dropout/AUROC) is byte-identical in
both modes; only the source of the held-out logits differs.
"""
_MODULE_NAME = "deepearth.autoresearch.spacetime.instrument.calib_probe"

if __name__ == "__main__":
    import sys as _entry_sys

    from deepearth.autoresearch.spacetime.instrument.recurrence import (
        require_recorded_entrypoint as _require_recorded,
    )

    _require_recorded("calib_probe.py", module=_MODULE_NAME, argv=_entry_sys.argv[1:])

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.spacetime.instrument.recurrence import (
    require_recorded_entrypoint,
)


# --------------------------------------------------------------------------------------------
# Calibration metrics -- all operate on held-out (probs [N,C], labels [N]); pure, no data source.
# --------------------------------------------------------------------------------------------
def _confidences(probs):
    pred = probs.argmax(1)
    conf = probs.max(1)
    return conf, pred


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (equal-width confidence bins) + max-calibration-error."""
    conf, pred = _confidences(probs)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    mce = 0.0
    N = len(labels)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        w = m.sum() / N
        e += w * abs(acc_b - conf_b)
        mce = max(mce, abs(acc_b - conf_b))
        rows.append((0.5 * (lo + hi), conf_b, acc_b, int(m.sum())))
    return e, mce, rows


def brier(probs, labels):
    """Multiclass Brier score = mean squared error between one-hot label and probability vector."""
    C = probs.shape[1]
    oh = np.eye(C, dtype=np.float64)[labels]
    return float(((probs - oh) ** 2).sum(1).mean())


def nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean())


def conf_error_auroc(probs, labels):
    """AUROC of confidence predicting correctness -- the HONEST test of uncertainty.

    A well-calibrated/meaningful uncertainty should rank correct predictions above wrong ones.
    Rank-statistic AUROC (Mann-Whitney), so temperature-scaling (a monotone conf transform)
    leaves it invariant by construction -- it measures whether the ORDERING of confidence is
    informative, which post-hoc scaling cannot create or destroy."""
    conf, pred = _confidences(probs)
    correct = (pred == labels).astype(np.int64)
    n_pos = correct.sum()
    n_neg = len(correct) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(len(conf), dtype=np.float64)
    ranks[order] = np.arange(1, len(conf) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(conf, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    auc = (ranks[correct == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------------------------------------------------
# Calibration MECHANISMS -- fit on a held-out CAL split, apply to TEST. Pure post-hoc.
# --------------------------------------------------------------------------------------------
def temperature_scale(logits_cal, labels_cal, logits_test, iters=200, lr=0.05):
    """Single-temperature scaling (Guo 2017): min NLL over T>0 on cal split, apply exp/T to test.

    Monotone -> accuracy and confidence-AUROC are exactly preserved; only sharpness changes."""
    lc = torch.as_tensor(logits_cal, dtype=torch.float64)
    yc = torch.as_tensor(labels_cal, dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lc / logT.exp(), yc)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(logT.exp().item())
    lt = torch.as_tensor(logits_test, dtype=torch.float64)
    return F.softmax(lt / T, dim=1).numpy(), T


def isotonic_1d(x, y):
    """Pool-adjacent-violators isotonic regression y ~ x, returns a step-function predictor.

    Fit on (confidence, correctness) of the cal split -> a nonparametric calibration map."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(np.float64)
    w = np.ones_like(ys)
    # PAVA
    vals = ys.copy()
    wts = w.copy()
    idx = list(range(len(vals)))
    i = 0
    lvl_val = [vals[0]]
    lvl_w = [wts[0]]
    lvl_x = [[xs[0]]]
    for k in range(1, len(vals)):
        lvl_val.append(vals[k]); lvl_w.append(wts[k]); lvl_x.append([xs[k]])
        while len(lvl_val) > 1 and lvl_val[-2] > lvl_val[-1]:
            v = (lvl_val[-2] * lvl_w[-2] + lvl_val[-1] * lvl_w[-1]) / (lvl_w[-2] + lvl_w[-1])
            w2 = lvl_w[-2] + lvl_w[-1]
            xs2 = lvl_x[-2] + lvl_x[-1]
            lvl_val = lvl_val[:-2] + [v]; lvl_w = lvl_w[:-2] + [w2]; lvl_x = lvl_x[:-2] + [xs2]
    # build breakpoints: for each level, the max x it covers -> its value
    bp_x, bp_y = [], []
    for v, xg in zip(lvl_val, lvl_x):
        bp_x.append(max(xg)); bp_y.append(v)
    bp_x = np.array(bp_x); bp_y = np.array(bp_y)

    def predict(q):
        j = np.searchsorted(bp_x, q, side="left")
        j = np.clip(j, 0, len(bp_y) - 1)
        return bp_y[j]

    return predict


def isotonic_calibrate(probs_cal, labels_cal, probs_test):
    """Apply isotonic regression to the top-class confidence (Zadrozny-Elkan style).

    Recalibrates the winning-class probability to empirical accuracy; renormalizes the vector."""
    conf_c, pred_c = _confidences(probs_cal)
    correct_c = (pred_c == labels_cal).astype(np.float64)
    f = isotonic_1d(conf_c, correct_c)
    conf_t, pred_t = _confidences(probs_test)
    new_conf = np.clip(f(conf_t), 1e-6, 1.0)
    out = probs_test.copy()
    # scale the winning class to new_conf, distribute the remainder proportionally
    C = probs_test.shape[1]
    for i in range(len(conf_t)):
        k = pred_t[i]
        rest = out[i].sum() - out[i, k]
        out[i, k] = new_conf[i]
        if rest > 1e-12:
            out[i] *= 1.0
            out[i, np.arange(C) != k] *= (1.0 - new_conf[i]) / rest
    out /= out.sum(1, keepdims=True)
    return out


def conformal_coverage(probs_cal, labels_cal, probs_test, labels_test, alpha=0.1):
    """Split-conformal (APS-style top-k by 1-p) coverage guarantee at level 1-alpha.

    Nonconformity = 1 - p(true class). Threshold = the ceil((n+1)(1-alpha))/n cal-quantile.
    Reports empirical test coverage (should be >= 1-alpha) and mean prediction-set size."""
    s_cal = 1.0 - probs_cal[np.arange(len(labels_cal)), labels_cal]
    n = len(s_cal)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = np.quantile(s_cal, q_level, method="higher")
    sets = (1.0 - probs_test) <= qhat            # [N,C] bool prediction sets
    covered = sets[np.arange(len(labels_test)), labels_test].mean()
    set_size = sets.sum(1).mean()
    return float(covered), float(set_size), float(qhat)


# --------------------------------------------------------------------------------------------
# Held-out logit sources
# --------------------------------------------------------------------------------------------
def _train_head_logits(feats, fam, n_fam, test_mask, dev, steps, lr, dropout=0.0, seed=0,
                       overconf=False):
    """Train a linear (or dropout-MLP) head feats->family on TRAIN, return TEST logits + labels.

    Byte-identical training recipe to probe.py:evaluate (Adam, CE, 4096 minibatch). With dropout>0
    the head is an MLP and we can do MC-dropout at test for epistemic variance. With overconf=True a
    higher-capacity 2-layer MLP is trained LONGER with no regularization -- this reproduces the real
    SDM/pollinator overconfidence pathology (a capacity-rich head driven to near-zero train loss
    grows over-sharp logits that do NOT hold at held-out spatial blocks). This is the regime the
    B23/B53 calibration gap lives in and the one temperature scaling is designed for."""
    torch.manual_seed(seed)
    train = ~test_mask
    Xtr = torch.as_tensor(feats[train], dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(fam[train], dtype=torch.long, device=dev)
    Xte = torch.as_tensor(feats[test_mask], dtype=torch.float32, device=dev)
    yte = fam[test_mask].astype(np.int64)
    if overconf:
        head = nn.Sequential(nn.Linear(feats.shape[1], 512), nn.ReLU(),
                             nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, n_fam)).to(dev)
    elif dropout > 0:
        head = nn.Sequential(nn.Linear(feats.shape[1], 256), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(256, n_fam)).to(dev)
    else:
        head = nn.Linear(feats.shape[1], n_fam).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (min(4096, Xtr.shape[0]),), device=dev)
        loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        logits = head(Xte).cpu().numpy().astype(np.float64)
    # MC-dropout variance (epistemic), only meaningful if dropout>0
    mc_var = None
    if dropout > 0:
        head.train()  # re-enable dropout
        stack = []
        with torch.no_grad():
            for _ in range(20):
                stack.append(F.softmax(head(Xte), dim=1).cpu().numpy())
        mc = np.stack(stack, 0)                 # [T,N,C]
        mc_var = mc.var(0).sum(1)               # predictive variance per test point [N]
    return logits, yte, mc_var


def _real_features(cache, feature, n_shards, dev, seed):
    """Reuse probe.py loaders for genuine encoder/env features. Returns (feats, fam, n_fam, lat, lon)."""
    from deepearth.autoresearch.spacetime.instrument import probe as P
    from deepearth.encoders.spacetime.earth4d import Earth4D
    lat, lon, fam, n_fam, _, gid, _sp = P.load_obs(cache, n_shards, with_gid=(feature == "env"))
    if feature == "raw":
        feats = np.stack([lat, lon], 1).astype(np.float32)
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    elif feature == "env":
        feats = P.load_env(cache, gid)
    elif feature == "earth4d":
        e4d = Earth4D().to(dev).eval()
        la = torch.as_tensor(lat, device=dev); lo = torch.as_tensor(lon, device=dev)
        t = torch.zeros_like(la)
        with torch.no_grad():
            feats = e4d(torch.stack([la, lo, t, t], 1)).cpu().numpy().astype(np.float32)
    else:
        raise ValueError(feature)
    return feats, fam, n_fam, lat, lon



def _real_pollinator(cache, feature, n_shards, dev, seed, topk=40):
    """REAL B53 pollinator-guild task: predict a plant-observation's DOMINANT pollinator taxon
    from the SAME env/coords/earth4d features used by the plant SDM. NOT a surrogate.

    Source: gbif_pollinator_dist.npz marginal pollinator distribution per plant species_local
    (marg_poll_idx/marg_poll_frq, GloBI-derived, derived/pollinator_globi_interactions.tsv). The
    label per observation = argmax-frequency pollinator of that obs's plant species. Classes are
    restricted to the topk most frequent pollinators (rest dropped) so the head is well-posed, the
    same way family SDM is a bounded-class problem. This is the low-data, phylogenetically-packed
    B53 regime -- fewer obs, pollinator (not plant-family) classes -- measured on real data."""
    from deepearth.autoresearch.spacetime.instrument import probe as P
    from deepearth.encoders.spacetime.earth4d import Earth4D
    lat, lon, fam, n_fam0, _, gid, _sp = P.load_obs(cache, n_shards, with_gid=True)
    sp = P.load_species(cache, n_shards)                       # plant species_local per obs
    d = np.load(f"{cache}/gbif_pollinator_dist.npz", allow_pickle=True)
    mpi, mpf, npoll = d["marg_poll_idx"], d["marg_poll_frq"], d["marg_npoll"]
    dom = np.full(len(npoll), -1, np.int64)
    has = npoll > 0
    dom[has] = mpi[np.arange(len(npoll)), mpf.argmax(1)][has]  # dominant pollinator per plant sp
    obs_dom = dom[sp]                                          # dominant pollinator per obs
    valid = obs_dom >= 0
    # restrict to topk most frequent pollinator classes among valid obs
    u, c = np.unique(obs_dom[valid], return_counts=True)
    keep = set(u[np.argsort(c)[::-1][:topk]].tolist())
    mask = valid & np.array([p in keep for p in obs_dom])
    remap = {p: i for i, p in enumerate(sorted(keep))}
    poll = np.array([remap.get(p, -1) for p in obs_dom], np.int64)
    lat, lon, gid = lat[mask], lon[mask], gid[mask]
    poll = poll[mask]
    n_poll = len(keep)
    if feature == "raw":
        feats = np.stack([lat, lon], 1).astype(np.float32)
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    elif feature == "env":
        feats = P.load_env(cache, gid)
    elif feature == "earth4d":
        e4d = Earth4D().to(dev).eval()
        la = torch.as_tensor(lat, device=dev); lo = torch.as_tensor(lon, device=dev)
        t = torch.zeros_like(la)
        with torch.no_grad():
            feats = e4d(torch.stack([la, lo, t, t], 1)).cpu().numpy().astype(np.float32)
    else:
        raise ValueError(feature)
    return feats, poll, n_poll, lat, lon


def _surrogate(n=20000, n_fam=40, dev="cpu", seed=0, overconf=True, niche=1.2, noise=0.4):
    """Nature-structured surrogate SDM: spatially-blocked families with env-correlated logits.

    Same shape as the real task (lat/lon in a bounded region, env covariate that correlates with
    family, spatial-block holdout). Deliberately induces the OVERCONFIDENCE pathology real SDM
    heads exhibit (a high-capacity head fit on limited spatial blocks): the train head sees a
    biased env->family mapping and extrapolates over-sharply to held-out blocks. This lets the
    calibration mechanisms be measured end-to-end when the token cache is absent. Provenance-
    equivalent to probe.py (occurrence coords + env), no external data."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(24.0, 49.0, n).astype(np.float32)       # CONUS-ish band, like GBIF plants
    lon = rng.uniform(-125.0, -66.0, n).astype(np.float32)
    # latent env field (smooth in space) -> each family occupies an env niche
    env = np.stack([
        np.sin(lat / 6.0) + 0.3 * rng.standard_normal(n),
        np.cos(lon / 8.0) + 0.3 * rng.standard_normal(n),
        (lat - 36) / 12 + 0.3 * rng.standard_normal(n),
    ], 1).astype(np.float32)
    # niche spread: SMALLER -> classes packed closer together (more phylo-confusable, B53 pollinator
    # regime where congeneric pollinators share niches); LARGER -> well-separated (B23 plant SDM regime).
    fam_centers = rng.standard_normal((n_fam, 3)).astype(np.float32) * niche
    d = ((env[:, None, :] - fam_centers[None]) ** 2).sum(2)     # [n, n_fam]
    logits_true = -d + 0.5 * rng.standard_normal((n, n_fam))
    fam = logits_true.argmax(1).astype(np.int64)
    # observed feature = env + noise (the head never sees fam_centers) -> genuine held-out difficulty
    feats = env + noise * rng.standard_normal((n, 3)).astype(np.float32)
    return feats.astype(np.float32), fam, n_fam, lat, lon


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/deepcal")
    ap.add_argument("--feature", default="env", choices=["env", "raw", "earth4d"])
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--cal_frac", type=float, default=0.5)     # fraction of held-out used as CAL (rest=TEST)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.1)        # conformal miscoverage level
    ap.add_argument("--dropout", type=float, default=0.2)      # MC-dropout / MLP head
    ap.add_argument("--ensemble", type=int, default=5)         # deep-ensemble members
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overconf", action="store_true")        # high-capacity MLP head (overconfidence regime)
    ap.add_argument("--surrogate", action="store_true")
    ap.add_argument("--topk", type=int, default=40)  # B53: number of pollinator classes (real regime)
    ap.add_argument("--regime", default="plant", choices=["plant", "pollinator"])  # B23 (many obs, separated niches) vs B53 (few obs, packed niches)
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cuda")   # accepted for harness compat; GPU selected via CUDA_VISIBLE_DEVICES
    a = ap.parse_args(argv)
    require_recorded_entrypoint("calib_probe.py", module=_MODULE_NAME,
                               argv=(argv if argv is not None else sys.argv[1:]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    # regime presets: plant/B23 = many obs, well-separated niches, low noise (SDM);
    #                 pollinator/B53 = few obs, packed/confusable niches, higher noise.
    if a.regime == "pollinator":
        sk = dict(n=4000, niche=0.7, noise=0.6)
    else:
        sk = dict(n=20000, niche=1.2, noise=0.4)

    if a.surrogate:
        feats, fam, n_fam, lat, lon = _surrogate(seed=a.seed, dev=dev, **sk)
        src = f"surrogate:{a.regime}"
    else:
        try:
            if a.regime == "pollinator":
                feats, fam, n_fam, lat, lon = _real_pollinator(a.cache_dir, a.feature, a.n_shards, dev, a.seed, topk=a.topk)
                src = f"real-pollinator:{a.feature}"
            else:
                feats, fam, n_fam, lat, lon = _real_features(a.cache_dir, a.feature, a.n_shards, dev, a.seed)
                src = f"real:{a.feature}"
        except Exception as e:                                  # cache absent -> honest fallback
            print(f"[calib] real features unavailable ({type(e).__name__}: {e}); using surrogate")
            feats, fam, n_fam, lat, lon = _surrogate(seed=a.seed, dev=dev, **sk)
            src = f"surrogate:{a.regime}(fallback)"

    # spatial-block holdout (probe.py logic, inlined to avoid importing when cache absent)
    block = 0.5
    blk = (np.floor(lat / block).astype(np.int64) * 100000 + np.floor(lon / block).astype(np.int64))
    ublk = np.unique(blk)
    rng = np.random.default_rng(a.seed); rng.shuffle(ublk)
    held = set(ublk[: int(len(ublk) * a.holdout)].tolist())
    test_all = np.array([b in held for b in blk])

    # train head on TRAIN, get held-out logits; split held-out into CAL/TEST (by block, no leakage)
    logits_all, y_all, mc_var = _train_head_logits(
        feats, fam, n_fam, test_all, dev, a.steps, a.lr, dropout=a.dropout, seed=a.seed,
        overconf=a.overconf)
    held_blocks = np.array(sorted(held)); rng.shuffle(held_blocks)
    n_cal_blk = int(len(held_blocks) * a.cal_frac)
    cal_blk = set(held_blocks[:n_cal_blk].tolist())
    held_idx = np.where(test_all)[0]
    is_cal = np.array([blk[i] in cal_blk for i in held_idx])
    Lc, yc = logits_all[is_cal], y_all[is_cal]
    Lt, yt = logits_all[~is_cal], y_all[~is_cal]
    Pc = _softmax(Lc); Pt = _softmax(Lt)

    print(f"\n=== CALIBRATION PROBE  src={src}  feature={a.feature}  seed={a.seed}  {a.label} ===")
    print(f"n_fam={n_fam}  n_train={(~test_all).sum()}  n_cal={len(yc)}  n_test={len(yt)}  dev={dev}")

    # ---- (1) BASELINE mis-calibration on TEST ----
    acc0 = accuracy(Pt, yt)
    ece0, mce0, rows0 = ece(Pt, yt, a.n_bins)
    br0 = brier(Pt, yt); nll0 = nll(Pt, yt); auc0 = conf_error_auroc(Pt, yt)
    mean_conf0 = Pt.max(1).mean()
    print(f"\n[BASELINE raw softmax]  acc={acc0:.4f}  meanconf={mean_conf0:.4f}  "
          f"gap(conf-acc)={mean_conf0 - acc0:+.4f}")
    print(f"  ECE={ece0:.4f}  MCE={mce0:.4f}  Brier={br0:.4f}  NLL={nll0:.4f}  "
          f"conf->correct AUROC={auc0:.4f}")
    print("  reliability (mid_conf, conf, acc, n):")
    for mid, cf, ac, nb in rows0:
        print(f"    {mid:.3f}  conf={cf:.3f}  acc={ac:.3f}  n={nb}")

    # ---- (2) TEMPERATURE scaling ----
    Pt_T, T = temperature_scale(Lc, yc, Lt)
    accT = accuracy(Pt_T, yt); eceT, mceT, _ = ece(Pt_T, yt, a.n_bins)
    brT = brier(Pt_T, yt); nllT = nll(Pt_T, yt); aucT = conf_error_auroc(Pt_T, yt)
    print(f"\n[TEMPERATURE T={T:.3f}]  acc={accT:.4f}(dAcc={accT - acc0:+.4f})  "
          f"ECE={eceT:.4f}(d{eceT - ece0:+.4f})  Brier={brT:.4f}(d{brT - br0:+.4f})  "
          f"NLL={nllT:.4f}  AUROC={aucT:.4f}")

    # ---- (3a) ISOTONIC regression ----
    Pt_I = isotonic_calibrate(Pc, yc, Pt)
    accI = accuracy(Pt_I, yt); eceI, _, _ = ece(Pt_I, yt, a.n_bins)
    brI = brier(Pt_I, yt); nllI = nll(Pt_I, yt); aucI = conf_error_auroc(Pt_I, yt)
    print(f"[ISOTONIC]          acc={accI:.4f}(dAcc={accI - acc0:+.4f})  "
          f"ECE={eceI:.4f}(d{eceI - ece0:+.4f})  Brier={brI:.4f}(d{brI - br0:+.4f})  "
          f"NLL={nllI:.4f}  AUROC={aucI:.4f}")

    # ---- (3b) CONFORMAL coverage ----
    cov, sz, qhat = conformal_coverage(Pc, yc, Pt, yt, a.alpha)
    print(f"[CONFORMAL a={a.alpha}]  target_cov={1 - a.alpha:.3f}  emp_cov={cov:.4f}  "
          f"mean_set_size={sz:.3f}  qhat={qhat:.3f}")

    # ---- (3c) DEEP ENSEMBLE (epistemic) ----
    ens_probs = []
    for m in range(a.ensemble):
        Lm, ym, _ = _train_head_logits(feats, fam, n_fam, test_all, dev, a.steps, a.lr,
                                       dropout=0.0, seed=a.seed + 100 + m, overconf=a.overconf)
        ens_probs.append(_softmax(Lm[~is_cal]))
    Pe = np.stack(ens_probs, 0).mean(0)
    ens_var = np.stack(ens_probs, 0).var(0).sum(1)
    accE = accuracy(Pe, yt); eceE, _, _ = ece(Pe, yt, a.n_bins)
    brE = brier(Pe, yt); nllE = nll(Pe, yt)
    # ensemble AFTER temperature (ensemble mean often still overconfident)
    Le = np.log(np.clip(Pe, 1e-12, 1)); Le_cal = np.log(np.clip(np.stack(
        [_softmax(_train_head_logits(feats, fam, n_fam, test_all, dev, a.steps, a.lr, 0.0,
                                     a.seed + 100 + m, a.overconf)[0][is_cal]) for m in range(a.ensemble)],
        0).mean(0), 1e-12, 1))
    Pe_T, Te = temperature_scale(Le_cal, yc, Le)
    eceET, _, _ = ece(Pe_T, yt, a.n_bins)
    # honest uncertainty: does ensemble variance rank correctness? (use -var as confidence)
    pred_e = Pe.argmax(1); correct_e = (pred_e == yt).astype(np.int64)
    auc_var = conf_error_auroc(np.stack([1 - ens_var, ens_var], 1)
                               if False else _neg_as_conf(ens_var), correct_e) \
        if ens_var.std() > 0 else float("nan")
    print(f"[DEEP-ENS k={a.ensemble}]  acc={accE:.4f}(dAcc={accE - acc0:+.4f})  "
          f"ECE={eceE:.4f}(d{eceE - ece0:+.4f})  Brier={brE:.4f}(d{brE - br0:+.4f})  "
          f"NLL={nllE:.4f}  ECE+temp={eceET:.4f}(T={Te:.2f})")
    print(f"           ens-variance -> correctness AUROC={_var_auroc(ens_var, correct_e):.4f}")
    # MECHANISM LEVER (confidence SIGNAL, not post-hoc scaling): conf_auroc is a RANK statistic, so temperature
    # and isotonic scaling leave it invariant BY CONSTRUCTION -- the only way to move it is a better uncertainty
    # signal. Single-model max-softmax (auc0) was the only one ever scored. These are the ensemble's epistemic
    # signals: the ensemble-mean confidence, predictive ENTROPY (total uncertainty), and MUTUAL INFORMATION /
    # BALD (epistemic only = total minus mean member entropy, the part that reflects model disagreement).
    _P = np.stack(ens_probs, 0)
    auc_ens_conf = conf_error_auroc(Pe, yt)
    _ent = -(Pe * np.log(np.clip(Pe, 1e-12, 1))).sum(1)                       # H[E[p]] = total
    _mean_ent = -(_P * np.log(np.clip(_P, 1e-12, 1))).sum(2).mean(0)          # E[H[p]] = aleatoric
    auc_ent = _var_auroc(_ent, correct_e)
    auc_bald = _var_auroc(_ent - _mean_ent, correct_e)                        # mutual information = epistemic
    auc_ensvar = _var_auroc(ens_var, correct_e)
    print(f"           [SIGNAL SWAP] conf_auroc: softmax={auc0:.4f} | ens-mean-conf={auc_ens_conf:.4f} | "
          f"pred-entropy={auc_ent:.4f} | BALD/MI={auc_bald:.4f} | ens-var={auc_ensvar:.4f}")

    # ---- (3d) MC-DROPOUT (epistemic) ----
    if mc_var is not None:
        mc_var_t = mc_var[~is_cal]
        pred0 = Pt.argmax(1); correct0 = (pred0 == yt).astype(np.int64)
        print(f"[MC-DROPOUT p={a.dropout}]  predvar->correctness AUROC="
              f"{_var_auroc(mc_var_t, correct0):.4f}  (softmax-conf AUROC={auc0:.4f})")

    # ---- SUMMARY line (grep-able) ----
    print(f"\nSUMMARY {src} feat={a.feature} seed={a.seed} "
          f"acc={acc0:.4f} ECE_raw={ece0:.4f} ECE_temp={eceT:.4f} ECE_iso={eceI:.4f} "
          f"ECE_ens={eceE:.4f} Brier_raw={br0:.4f} Brier_temp={brT:.4f} "
          f"conf_auroc={auc0:.4f} conf_auroc_ens={auc_ens_conf:.4f} conf_auroc_ent={auc_ent:.4f} "
          f"conf_auroc_bald={auc_bald:.4f} conf_auroc_ensvar={auc_ensvar:.4f} "
          f"conformal_cov={cov:.4f} setsize={sz:.3f} T={T:.3f} "
          f"t={time.time() - t0:.1f}s")


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _neg_as_conf(var):
    c = 1.0 - (var - var.min()) / (var.max() - var.min() + 1e-12)
    return np.stack([1 - c, c], 1)


def _var_auroc(var, correct):
    """AUROC of (-variance) predicting correctness: low epistemic variance -> correct."""
    if var.std() == 0 or correct.sum() == 0 or correct.sum() == len(correct):
        return float("nan")
    conf = -var
    n_pos = correct.sum(); n_neg = len(correct) - n_pos
    _, inv, counts = np.unique(conf, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[correct == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


if __name__ == "__main__":
    main()
