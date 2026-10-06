"""A GPU reimplementation of maxent.jar's model fitting (MaxEnt 3.4.x), batched over many runs at once.

maxent.jar fits one presence-background MaxEnt model per run on a single CPU thread: a sequential, coordinate-wise
L1-regularized optimizer that adds one feature at a time from a very large pool of candidate threshold and hinge
features. A species in our pipeline needs 30 such runs (5 regularization multipliers x 5 cross-validation folds,
then 5 subsampled replicates), about 100 s of maxent.jar per species. This module performs the same computation on
a GPU, with every run of a species (or of several species) advanced in lock-step as one batch.

The algorithm is maxent.jar's, step for step (see docs/maxent_fitting_spec.md for the specification and the Java
source lines each step follows):

* data: duplicate presences (same coordinates) removed; training presences whose environment vector is not
  already a background row are appended to the background (``addsamplestobackground``);
* features: linear (scaled to [0, 1] over background plus presences), threshold and forward/reverse hinge
  features on a knot grid at the midpoints between consecutive distinct values (with maxent.jar's
  significant-digit grouping), regularization constants from maxent.jar's sample-size tables;
* objective: L1-regularized negative log-likelihood of the presences under a Gibbs density on the background;
* optimizer: per iteration the feature (or candidate threshold/hinge) with the best bound on the loss decrease is
  chosen and updated by a sign-preserving Newton step (bound-based step for threshold features, line search when
  Newton disappoints), with the early-iteration step damping, a joint Newton step along the recent update
  direction every 30 iterations, maxent.jar's stale-expectation bookkeeping, and its convergence test
  (loss decrease < 1e-5 over 20 iterations, at most 500 iterations);
* outputs: ``.lambdas`` files, ``*_samplePredictions.csv`` and ``maxentResults.csv`` (gains, AUCs, the standard
  threshold rules) in maxent.jar's formats, so downstream code reads them unchanged.

Everything is float64. GPU reductions sum in a different order from Java, so results agree with maxent.jar to
rounding until a near-tie in feature selection resolves differently; docs/maxent_torch.md quantifies the
resulting differences.

Randomness: cross-validation folds are drawn exactly as maxent.jar draws them (java.util.Random(0)), so they are
the same folds. Subsampled replicates cannot match any particular maxent.jar run, because maxent.jar seeds them
from the wall clock (it forces ``randomseed=true`` for replicated runs); here they are drawn with maxent.jar's
procedure from a fixed seed, which makes them reproducible.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

EPS = 1e-6                    # Sequential.eps
NODATA = -9999.0              # SampleSet.NODATA_value (parameter ``nodata``)
MAX_ITER = 500                # parameter ``maximumiterations``
CONVERGENCE = 1e-5            # parameter ``convergencethreshold``
PARALLEL_EVERY = 30           # parameter ``parallelupdatefrequency``
TEST_EVERY = 20               # Sequential.convergenceTestFrequency
UPDATE_CYCLE, RECENT, TOP_SELECT = 20, 10, 5   # Sequential.featuresToUpdate


# ---------------------------------------------------------------------------------------------------- Java helpers
class JavaRandom:
    """java.util.Random (48-bit LCG), for maxent.jar's fold and test-point draws."""
    _MULT, _ADD, _MASK = 0x5DEECE66D, 0xB, (1 << 48) - 1

    def __init__(self, seed: int):
        self.seed = (seed ^ self._MULT) & self._MASK

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * self._MULT + self._ADD) & self._MASK
        return self.seed >> (48 - bits)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))


def jstr(x: float) -> str:
    """Java's Double.toString (shortest round-trip digits; decimal for 1e-3 <= |x| < 1e7, else d.dddE<n>)."""
    x = float(x)
    if 1e-3 <= abs(x) < 1e7:                    # Python's repr is then decimal and identical to Java's
        return repr(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    r = repr(x)
    if "e" in r:
        m, e = r.split("e")
        return f"{m if '.' in m else m + '.0'}E{int(e)}"
    sign, r = ("-", r[1:]) if r[0] == "-" else ("", r)
    ip, fp = r.split(".")
    if ip != "0":                               # |x| >= 1e7
        digits, e = (ip + fp).rstrip("0"), len(ip) - 1
    else:                                       # 1e-4 <= |x| < 1e-3
        st = fp.lstrip("0")
        digits, e = st.rstrip("0"), -(len(fp) - len(st) + 1)
    digits = digits or "0"
    return f"{sign}{digits[0]}.{digits[1:] or '0'}E{e}"


def fmt4(x) -> str:
    """maxentResults.csv number format (java.text.NumberFormat, 4 fraction digits, HALF_EVEN)."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, str):
        return x
    x = float(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{Decimal(x).quantize(Decimal('0.0001'), ROUND_HALF_EVEN):f}"


def fmt_sci3(x: float) -> str:
    """java.text.DecimalFormat("#.###E0")."""
    if x == 0 or x != x:
        return "0E0" if x == 0 else "NaN"
    e = math.floor(math.log10(abs(x)))
    m = Decimal(x) / (Decimal(10) ** e)
    m = m.quantize(Decimal("0.001"), ROUND_HALF_EVEN)
    if abs(m) >= 10:
        e += 1
        m = (Decimal(x) / (Decimal(10) ** e)).quantize(Decimal("0.001"), ROUND_HALF_EVEN)
    ms = f"{m:f}".rstrip("0").rstrip(".")
    return f"{ms}E{e}"


def _java_int(a: np.ndarray) -> np.ndarray:
    """(int) cast of doubles: NaN -> 0, truncation toward zero, saturating."""
    a = np.nan_to_num(a, nan=0.0, posinf=2.0 ** 31 - 1, neginf=-(2.0 ** 31))
    return np.trunc(np.clip(a, -(2.0 ** 31), 2.0 ** 31 - 1)).astype(np.int64)


def precision(values: np.ndarray) -> np.ndarray:
    """SortedFeatureGenerator.precision: about half a unit of the last significant digit (at most 6 digits),
    treating trailing 00/01/99/98 digit pairs as floating-point noise. precision(0) = 0."""
    f = np.abs(np.asarray(values, np.float64))
    with np.errstate(all="ignore"):
        cp = np.power(10.0, np.floor(np.log(f) / np.log(10.0)))
        res = np.full(f.shape, np.nan)
        done = np.zeros(f.shape, bool)
        last = np.zeros(f.shape, np.int64)
        for i in range(1, 7):
            cd = _java_int(f / cp)
            last = np.fmod(last * 10 + cd, 100)
            if i >= 2:
                hit = ~done & np.isin(last, (0, 1, 99, 98))
                res[hit] = cp[hit] * 50.0
                done |= hit
            f = f - cd * cp
            cp = cp / 10.0
        res[~done] = cp[~done] * 5.0
    return res


def group_starts(u: np.ndarray, prec: float) -> np.ndarray:
    """Start values of maxent.jar's value groups over sorted distinct values ``u``: a new group starts at the first
    value v with v - start(previous group) > prec (SortedFeatureGenerator constructor). Vectorized: next-start
    pointers, then the chain from the first value by pointer doubling."""
    n = len(u)
    if n < 2 or not prec > 0 or bool(((u[1:] - u[:-1]) > prec).all()):
        return u
    i = np.arange(n)
    j = np.searchsorted(u, u + prec, side="right")
    for _ in range(64):                                 # exact Java predicate u[j] - u[i] > prec (rounding of u + prec)
        jm = np.clip(j - 1, 0, n - 1)
        down = (j - 1 > i) & (u[jm] - u - prec > 0)
        jc = np.clip(j, 0, n - 1)
        up = (j < n) & ~(u[jc] - u > prec)
        if not (down.any() or up.any()):
            break
        j = j - down + up
    nxt = np.append(j, n)                               # node n = past the end
    on = np.zeros(n + 1, bool)
    on[0] = True
    jump = nxt
    for _ in range(int(np.ceil(np.log2(n + 1))) + 1):
        on[jump[on]] = True
        jump = jump[jump]
    return u[np.flatnonzero(on[:n])]


def _interp(xs: Sequence[int], ys: Sequence[float], n: int) -> float:
    """Runner.interpolate."""
    for i, x in enumerate(xs):
        if n <= x:
            break
    else:
        return ys[-1]
    if i == 0:
        return ys[0]
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (n - xs[i - 1]) / (xs[i] - xs[i - 1])


def betas_for(m: int, beta_multiplier: float) -> tuple[float, float, float]:
    """(linear, threshold, hinge) regularization parameters (Runner.autoSetBeta, linear/threshold/hinge only)."""
    return (_interp([10, 30, 100], [1.0, 0.2, 0.05], m) * beta_multiplier,
            _interp([0, 100], [2.0, 1.0], m) * beta_multiplier,
            0.5 * beta_multiplier)


def _seq_sum(a: np.ndarray) -> float:
    return float(np.cumsum(a)[-1]) if len(a) else 0.0


def _interval(avg, std, beta, m):
    """FeaturedSpace.Interval of a feature divided by the (constant 1) bias interval: (mid, dev)."""
    lo = avg - beta / math.sqrt(m) * std
    hi = avg + beta / math.sqrt(m) * std
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


# ------------------------------------------------------------------------------------------------- one MaxEnt run
@dataclass
class Run:
    """One maxent.jar run: training presences, optional test presences, background, regularization multiplier."""
    label: str
    beta_multiplier: float
    background: np.ndarray            # (Nbg, V) float64
    train: np.ndarray                 # (m, V)
    train_xy: np.ndarray              # (m, 2) lon, lat
    test: Optional[np.ndarray] = None
    test_xy: Optional[np.ndarray] = None


@dataclass
class _Gen:
    """Candidate knots of one variable in one direction (x for thresholds and forward hinges, -x for reverse)."""
    thr: np.ndarray                   # (T,) knots
    maxval: float
    perm: np.ndarray                  # (N,) points sorted by value
    q: np.ndarray                     # (T,) number of points in groups <= t (start of "above t" in perm order)
    above: np.ndarray                 # (N,) number of knots strictly below each point (exported-feature eval)
    n1: dict                          # kind -> (T,) sample expectation
    dev: dict                         # kind -> (T,) sample deviation
    valid: dict                       # kind -> (T,) bool


@dataclass
class _Prepared:
    run: Run
    points: np.ndarray                # (N, V): background, then added training presences
    lin_min: np.ndarray
    lin_max: np.ndarray
    scale: np.ndarray
    scaled: np.ndarray                # (N, V)
    n1_lin: np.ndarray
    dev_lin: np.ndarray
    gens: list                        # [v] -> (fwd _Gen, rev _Gen)
    m: int


def _generator(pts: np.ndarray, smp: np.ndarray, prec: float, beta_thr: float, beta_hge: float, min_dev: float,
               with_thresholds: bool) -> _Gen:
    perm = np.argsort(pts, kind="stable")
    ps = pts[perm]
    ss = np.sort(smp)
    u = np.unique(np.concatenate([ps, ss]))
    starts = group_starts(u, prec)
    thr = (starts[1:] + starts[:-1]) / 2.0
    T = len(thr)
    maxval = float(max(ps[-1], ss[-1]))
    m = len(smp)
    gs = np.searchsorted(starts, ss, side="right") - 1          # group of each presence (ascending)
    gp = np.searchsorted(starts, ps, side="right") - 1          # group of each point (in sorted order)
    q = np.searchsorted(gp, np.arange(T), side="right")         # points in groups <= t
    above = np.empty(len(pts), np.int64)
    above[perm] = np.searchsorted(thr, ps, side="left")         # knots strictly below each point
    # presences above knot t are those in groups >= t+1; Java accumulates them from the largest value down
    sd = ss[::-1]
    cnt = m - np.searchsorted(gs, np.arange(1, T + 1), side="left")
    c1 = np.concatenate([[0.0], np.cumsum(sd)])
    c2 = np.concatenate([[0.0], np.cumsum(sd * sd)])
    sum1, sum2, wsum = c1[cnt], c2[cnt], cnt.astype(np.float64)
    n1, dev, valid = {}, {}, {}
    sm = math.sqrt(m)
    with np.errstate(all="ignore"):
        # threshold features (ThrFeatureGenerator.setSampleExpectations)
        if m > 1:
            avg = wsum / m
            std = np.where(wsum < m * avg * avg, 0.0, np.sqrt((wsum - m * avg * avg) / (m - 1)))
            std = np.minimum(std, 0.5)
        else:
            avg, std = wsum.copy(), np.full(T, 0.5)
        lo, hi = avg - beta_thr / sm * std, avg + beta_thr / sm * std
        n1["thr"], dev["thr"] = 0.5 * (lo + hi), np.maximum(0.5 * (hi - lo), min_dev)
        first = np.searchsorted(thr, ss[0], side="left")
        last = np.searchsorted(thr, ss[-1], side="left")
        valid["thr"] = (np.arange(T) >= first) & (np.arange(T) < last) & with_thresholds
        # hinge features (HingeFeatureGenerator.setSampleExpectations)
        dd = maxval - thr
        csum1 = (sum1 - thr * wsum) / dd
        csum2 = (sum2 - 2 * sum1 * thr + thr * thr * wsum) / dd / dd
        avg = csum1 / m
        if m > 1:
            std = np.where(csum2 < m * avg * avg, 0.0, np.sqrt((csum2 - m * avg * avg) / (m - 1)))
            std = np.minimum(std, 0.5)
            std = np.maximum(std, 1.0 / math.sqrt(m))
        else:
            std = np.full(T, 0.5)
        lo, hi = avg - beta_hge / sm * std, avg + beta_hge / sm * std
        n1["hinge"], dev["hinge"] = 0.5 * (lo + hi), np.maximum(0.5 * (hi - lo), min_dev)
        valid["hinge"] = np.ones(T, bool)
    return _Gen(thr, maxval, perm, q, above, n1, dev, valid)


_BG_CACHE: dict = {}


def _background_info(bg: np.ndarray):
    """Per-background constants shared by all runs of a species: row keys (for adding presences) and the
    minimum value precision of each column."""
    c = _BG_CACHE.get(id(bg))
    if c is None or c[0] is not bg:
        if len(_BG_CACHE) > 16:
            _BG_CACHE.clear()
        c = (bg, {r.tobytes() for r in np.ascontiguousarray(bg)}, precision(bg).min(0))
        _BG_CACHE[id(bg)] = c
    return c[1], c[2]


def prepare(run: Run) -> _Prepared:
    bg, tr = run.background, run.train
    m, V = tr.shape
    # presences added to the background unless their environment vector is already a background row
    bgkeys, bgprec = _background_info(bg)
    added = np.array([r.tobytes() not in bgkeys for r in np.ascontiguousarray(tr)], bool)
    pts = np.vstack([bg, tr[added]]) if added.any() else bg.copy()
    bl, bt, bh = betas_for(m, run.beta_multiplier)
    min_dev = 0.001 * run.beta_multiplier
    lo, hi = pts.min(0), pts.max(0)
    scale = hi - lo
    scale = np.where(scale == 0, 1.0, scale)
    scaled = (pts - lo) / scale
    n1_lin, dev_lin = np.zeros(V), np.zeros(V)
    for v in range(V):
        sv = (tr[:, v] - lo[v]) / scale[v]
        smin, smax = scaled[:, v].min(), scaled[:, v].max()
        avg, sq = _seq_sum(sv), _seq_sum(sv * sv)
        if m == 1:
            std = 0.5 * (smax - smin)
        else:
            avg /= m
            std = 0.0 if sq < m * avg * avg else math.sqrt((sq - m * avg * avg) / (m - 1))
            std = min(std, 0.5 * (smax - smin))
        n1_lin[v], dev_lin[v] = _interval(avg, std, bl, m)
        dev_lin[v] = max(dev_lin[v], min_dev)
    prec = np.minimum(bgprec, precision(tr).min(0))       # over points and presences (points = bg + some presences)
    gens = [(_generator(pts[:, v], tr[:, v], prec[v], bt, bh, min_dev, True),
             _generator(-pts[:, v], -tr[:, v], prec[v], bt, bh, min_dev, False)) for v in range(V)]
    return _Prepared(run, pts, lo, hi, scale, scaled, n1_lin, dev_lin, gens, m)


# --------------------------------------------------------------------------------------------- the batched solver
@dataclass
class Fitted:
    """Result of one run, enough to write maxent.jar's outputs."""
    prep: _Prepared
    variables: list
    lam_lin: np.ndarray
    exported: list                    # [(v, kind, t, lambda)] in export order; kind 0 thr, 1 fwd hinge, 2 rev hinge
    lpn: float
    dnorm: float
    entropy: float
    density: np.ndarray               # (N,) unnormalized training density (maxent.jar's density[])
    loss: float
    l1: float
    iterations: int
    seconds: float = 0.0

    def lambdas_text(self) -> str:
        p, out = self.prep, []
        for v, name in enumerate(self.variables):
            out.append(f"{name}, {jstr(self.lam_lin[v])}, {jstr(p.lin_min[v])}, {jstr(p.lin_max[v])}")
        for v, kind, t, lam in self.exported:
            if lam == 0.0:
                continue
            name = self.variables[v]
            g = p.gens[v][1 if kind == 2 else 0]
            if kind == 0:
                out.append(f"({jstr(g.thr[t])}<{name}), {jstr(lam)}, 0.0, 1.0")
            elif kind == 1:
                out.append(f"'{name}, {jstr(lam)}, {jstr(g.thr[t])}, {jstr(g.maxval)}")
            else:
                out.append(f"`{name}, {jstr(lam)}, {jstr(-g.maxval)}, {jstr(-g.thr[t])}")
        out += [f"linearPredictorNormalizer, {jstr(self.lpn)}", f"densityNormalizer, {jstr(self.dnorm)}",
                f"numBackgroundPoints, {len(p.points)}", f"entropy, {jstr(self.entropy)}"]
        return "\n".join(out) + "\n"


def _deriv(E, N1, beta, lam):
    g = E - N1
    at0 = torch.where(g + beta > 0, g + beta, torch.where(g - beta < 0, g - beta, torch.zeros_like(g)))
    return torch.where(lam > 0, g + beta, torch.where(lam < 0, g - beta, at0))


def _good_alpha(W1, N1, beta, lam):
    W0, N0 = 1 - W1, 1 - N1
    a1 = torch.log((N1 - beta) * W0 / ((N0 + beta) * W1))
    a2 = torch.log((N1 + beta) * W0 / ((N0 - beta) * W1))
    c1 = (N1 - beta > EPS) & (a1 + lam > 0)
    c2 = (N0 - beta > EPS) & (a2 + lam < 0)
    a = torch.where(c1, a1, torch.where(c2, a2, -lam))
    return torch.where((W0 < EPS) | (W1 < EPS), torch.zeros_like(a), a)


def _bound(W1, N1, beta, lam):
    a = _good_alpha(W1, N1, beta, lam)
    b = -N1 * a + torch.log((1 - W1) + W1 * torch.exp(a)) + beta * (torch.abs(lam + a) - torch.abs(lam))
    b = torch.where(torch.isnan(b) | torch.isinf(a), torch.zeros_like(b), b)
    return b


def _reduce(a, it):
    if it < 10:
        return a / 50
    if it < 20:
        return a / 10
    if it < 50:
        return a / 3
    return a


def _cand_bound_impl(C, Cx, invZ, src, thr, imt, isthr, n1, dev, R1, R2, L1, L2, ok1, ok2, lam, elam, valid):
    """Expectation W and loss-reduction bound (Sequential.deltaLossBound with goodAlpha) of every candidate
    threshold / hinge feature, fused into one GPU kernel. Static parts of goodAlpha are precomputed:
    log((N1-b)W0/((N0+b)W1)) = L1 + log(W0/W1), with R1 = exp(L1)."""
    D = C.gather(1, src)
    XD = Cx.gather(1, src)
    iz = invZ.unsqueeze(1)
    W = torch.where(isthr, D * iz, (XD - thr * D) * iz * imt)
    W0 = 1 - W
    q = W0 / W
    lq = torch.log(q)
    a1, a2 = L1 + lq, L2 + lq
    c1 = ok1 & (a1 + lam > 0)
    c2 = ok2 & (a2 + lam < 0)
    small = (W0 < EPS) | (W < EPS)
    a = torch.where(small, torch.zeros_like(W), torch.where(c1, a1, torch.where(c2, a2, -lam)))
    ea = torch.where(small, torch.ones_like(W), torch.where(c1, R1 * q, torch.where(c2, R2 * q, elam)))
    b = -n1 * a + torch.log(W0 + W * ea) + dev * (torch.abs(lam + a) - torch.abs(lam))
    b = torch.where(torch.isnan(b) | torch.isinf(a), torch.zeros_like(b), b)
    return torch.where(valid, b, torch.full_like(b, math.inf)), W


_cand_bound = None


def _get_cand_bound():
    """The candidate kernel compiled with torch.compile (one fused GPU kernel), falling back to eager PyTorch if
    compilation is unavailable (errors surface at the first call, so the fallback is decided there)."""
    global _cand_bound
    if _cand_bound is None:
        compiled = torch.compile(_cand_bound_impl, dynamic=True)
        state = {"fn": compiled}

        def call(*args):
            try:
                return state["fn"](*args)
            except Exception:
                if state["fn"] is _cand_bound_impl:
                    raise
                state["fn"] = _cand_bound_impl
                return _cand_bound_impl(*args)
        _cand_bound = call
    return _cand_bound


class Batch:
    """Many MaxEnt runs advanced together in lock-step (runs may differ in variables, points and presences).

    Candidate threshold / hinge features of each run are held as one flat list in maxent.jar's selection order
    (threshold generators of each variable, then forward / reverse hinge generators alternating per variable,
    knots ascending within a generator), restricted to the candidates maxent.jar considers. Column ``C`` (one
    past the last candidate) is a dummy slot used by padded entries."""

    def __init__(self, preps: list[_Prepared], device="cuda"):
        self.preps, self.dev = preps, torch.device(device)
        B = len(preps)
        V = max(p.points.shape[1] for p in preps)
        N = max(len(p.points) for p in preps)
        T = max(1, max(max(len(g.thr) for pair in p.gens for g in pair) for p in preps))
        G = 3 * V
        X = np.zeros((B, N, V)); L = np.zeros((B, N, V)); ptm = np.zeros((B, N), bool)
        vm = np.zeros((B, V), bool); n1l = np.zeros((B, V)); devl = np.ones((B, V))
        thr2 = np.zeros((B, 2 * V, T)); M2 = np.ones((B, 2 * V))
        permd = np.full((B, 2 * V, N), N, np.int64); xsd = np.zeros((B, 2 * V, N))
        qd = np.zeros((B, 2 * V, T), np.int64); above = np.zeros((B, 2 * V, N), np.int64)
        cands = []
        for b, p in enumerate(preps):
            n, nv = p.points.shape
            X[b, :n, :nv] = p.points; L[b, :n, :nv] = p.scaled; ptm[b, :n] = True
            vm[b, :nv] = True; n1l[b, :nv] = p.n1_lin; devl[b, :nv] = p.dev_lin
            for v, (gf, gr) in enumerate(p.gens):
                for k, g in enumerate((gf, gr)):
                    s, t = 2 * v + k, len(g.thr)
                    thr2[b, s, :t] = g.thr; M2[b, s] = g.maxval
                    desc = g.perm[::-1]
                    permd[b, s, :n] = desc
                    xsd[b, s, :n] = (p.points[:, v] if k == 0 else -p.points[:, v])[desc]
                    qd[b, s, :t] = n - g.q
                    above[b, s, :n] = g.above
            # flat candidate list in maxent.jar's order
            parts = []
            groups = [(v, 0, "thr") for v in range(nv)] + [(v, k, "hinge") for v in range(nv) for k in (0, 1)]
            for gi, (v, k, kind) in enumerate(groups):
                gen = p.gens[v][k]
                ts = np.flatnonzero(gen.valid[kind])
                if not len(ts):
                    continue
                gidx = v if kind == "thr" else V + 2 * v + k
                n1, dv = gen.n1[kind][ts], gen.dev[kind][ts]
                parts.append(dict(g=np.full(len(ts), gidx), t=ts, src=(2 * v + k) * T + ts, thr=gen.thr[ts],
                                  M=np.full(len(ts), gen.maxval), isthr=np.full(len(ts), kind == "thr"), n1=n1, dev=dv))
            cands.append({key: np.concatenate([q_[key] for q_ in parts]) for key in parts[0]})
        Cn = max(len(c["g"]) for c in cands)
        self.B, self.V, self.N, self.T, self.G, self.C = B, V, N, T, G, Cn
        W1 = Cn + 1
        cg = np.zeros((B, W1), np.int64); ct = np.zeros((B, W1), np.int64); src = np.zeros((B, W1), np.int64)
        cthr = np.zeros((B, W1)); cM = np.full((B, W1), 2.0); cisthr = np.zeros((B, W1), bool)
        cn1 = np.full((B, W1), 0.5); cdev = np.ones((B, W1)); valid = np.zeros((B, W1), bool)
        for b, c in enumerate(cands):
            m = len(c["g"])
            cg[b, :m], ct[b, :m], src[b, :m] = c["g"], c["t"], c["src"]
            cthr[b, :m], cM[b, :m], cisthr[b, :m] = c["thr"], c["M"], c["isthr"]
            cn1[b, :m], cdev[b, :m], valid[b, :m] = c["n1"], c["dev"], True
        with np.errstate(all="ignore"):
            n1_, dv_ = cn1, cdev
            R1 = (n1_ - dv_) / ((1 - n1_) + dv_)
            R2 = (n1_ + dv_) / ((1 - n1_) - dv_)
            ok1, ok2 = (n1_ - dv_ > EPS) & valid, ((1 - n1_) - dv_ > EPS) & valid
            L1 = np.where(ok1, np.log(np.where(ok1, R1, 1.0)), 0.0)
            L2 = np.where(ok2, np.log(np.where(ok2, R2, 1.0)), 0.0)
            imt = np.where(cisthr | ~valid, 0.0, 1.0 / (cM - cthr))
        f = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=self.dev)
        self.X, self.L, self.ptm = f(X), f(L), f(ptm, torch.bool)
        self.vm, self.n1l, self.devl = f(vm, torch.bool), f(n1l), f(devl)
        self.thr2, self.M2 = f(thr2), f(M2)
        self.permd, self.xsd, self.above = f(permd, torch.long), f(xsd), f(above, torch.long)
        self.qd_m1, self.qd_pos = f(np.maximum(qd - 1, 0), torch.long), f(qd > 0)       # sum of the first qd values
        self.cg, self.ct, self.src = f(cg, torch.long), f(ct, torch.long), f(src, torch.long)
        self.cthr, self.cM, self.cisthr = f(cthr), f(cM), f(cisthr, torch.bool)
        self.cn1, self.cdev, self.cvalid = f(cn1), f(cdev), f(valid, torch.bool)
        self.R1, self.R2, self.L1, self.L2 = f(np.nan_to_num(R1)), f(np.nan_to_num(R2)), f(L1), f(L2)
        self.ok1, self.ok2, self.imt = f(ok1, torch.bool), f(ok2, torch.bool), f(imt)
        self.cflat = self.cg * T + self.ct                   # (B, C+1) index into the (G, T) layout
        gvar = [g if g < V else (g - V) // 2 for g in range(G)]
        gdir = [0 if g < V else (g - V) % 2 for g in range(G)]
        self.gvar, self.gdir = f(gvar, torch.long), f(gdir, torch.long)
        self.npts = f([len(p.points) for p in preps])
        self.trace = None                  # set to [] to record per-iteration diagnostics
        self.cand_bound = _get_cand_bound()

    # ---- expectations
    def sums(self, d):
        """For every knot: sum over points above it of the density, and of value * density (the generators'
        sums, accumulated from the largest value down as maxent.jar does). Returned flat (B, 2V*T)."""
        B, V, N = self.B, self.V, self.N
        dext = torch.cat([d, d.new_zeros(B, 1)], 1)
        ds = dext.gather(1, self.permd.view(B, -1)).view(B, 2 * V, N)
        C = ds.cumsum(-1).gather(-1, self.qd_m1) * self.qd_pos
        Cx = (ds * self.xsd).cumsum(-1).gather(-1, self.qd_m1) * self.qd_pos
        return C.view(B, -1), Cx.view(B, -1)

    def candidates(self, d, Z, lam, elam):
        """(B, C+1) loss bounds and expectations of all candidates (column C: the dummy slot, bound +inf)."""
        C, Cx = self.sums(d)
        return self.cand_bound(C, Cx, 1.0 / Z, self.src, self.cthr, self.imt, self.cisthr, self.cn1, self.cdev,
                               self.R1, self.R2, self.L1, self.L2, self.ok1, self.ok2, lam, elam, self.cvalid)

    def lin_expect(self, d, Z):
        return torch.einsum("bn,bnv->bv", d, self.L) / Z.unsqueeze(1)

    def contrib(self, c):
        """Sum over threshold / hinge features of c_f * f(x) at every point; c: (B, C+1) flat candidate coefficients."""
        B, V, T, G = self.B, self.V, self.T, self.G
        cc = c.new_zeros(B, G * T).scatter_add_(1, self.cflat, c).view(B, G, T)
        z = c.new_zeros(B, V, 1)
        kf, kr = self.above[:, 0::2], self.above[:, 1::2]
        x = self.X.permute(0, 2, 1)
        thf, thr = self.thr2[:, 0::2], self.thr2[:, 1::2]
        a1 = cc[:, V::2] / (self.M2[:, 0::2].unsqueeze(-1) - thf)
        a2 = cc[:, V + 1::2] / (self.M2[:, 1::2].unsqueeze(-1) - thr)
        P0 = torch.cat([z, cc[:, :V].cumsum(-1)], -1).gather(-1, kf)
        A1 = torch.cat([z, a1.cumsum(-1)], -1).gather(-1, kf)
        C1 = torch.cat([z, (a1 * thf).cumsum(-1)], -1).gather(-1, kf)
        A2 = torch.cat([z, a2.cumsum(-1)], -1).gather(-1, kr)
        C2 = torch.cat([z, (a2 * thr).cumsum(-1)], -1).gather(-1, kr)
        out = P0 + (x * A1 - C1) + (-x * A2 - C2)
        return (out * self.vm.unsqueeze(-1)).sum(1) * self.ptm

    def density(self, lp):
        lpn = lp.max(1).values
        d = torch.exp(lp - lpn.unsqueeze(1))
        return lpn, d, d.sum(1)

    def features_to_update(self, it, dlb_l, dlb_e, last_chg):
        """Sequential.featuresToUpdate: linear features changed in the last 10 iterations or due in the 20-iteration
        refresh cycle, plus those among the 5 features (linear or exported) with the best loss bounds."""
        B, V = self.B, self.V
        j = torch.arange(V, device=self.dev)
        m1 = self.vm & ((it < last_chg + RECENT) | ((j % UPDATE_CYCLE) == (it % UPDATE_CYCLE)).unsqueeze(0))
        allv = torch.cat([dlb_l, dlb_e], 1)
        top = allv.topk(min(TOP_SELECT, allv.shape[1]), largest=False).indices
        m2 = torch.zeros(B, V + 1, dtype=torch.bool, device=self.dev)
        m2.scatter_(1, torch.where(top < V, top, torch.full_like(top, V)), True)
        return m1 | (m2[:, :V] & self.vm)

    # ---- the optimizer
    def run(self, max_iter=MAX_ITER, conv=CONVERGENCE, par_every=PARALLEL_EVERY) -> list[Fitted]:
        B, V, N, Cn, dev = self.B, self.V, self.N, self.C, self.dev
        t0 = time.time()
        bi = torch.arange(B, device=dev)
        f64 = dict(dtype=torch.float64, device=dev)
        lp = torch.where(self.ptm, torch.zeros(B, N, **f64), torch.full((B, N), -math.inf, **f64))
        lpn, d, Z = self.density(lp)
        lam_l = torch.zeros(B, V, **f64)
        lam_c = torch.zeros(B, Cn + 1, **f64)                # flat candidates + dummy slot
        elam = torch.ones(B, Cn + 1, **f64)                  # exp(-lambda), maintained where lambda changes
        prev_l = lam_l.clone()
        K = max_iter
        exp_idx = torch.full((B, K), Cn, dtype=torch.long, device=dev)   # exported features, export order
        prev_e = torch.zeros(B, K, **f64)
        n_exp = torch.zeros(B, dtype=torch.long, device=dev)
        exported = torch.zeros(B, Cn + 1, dtype=torch.bool, device=dev)
        kar = torch.arange(K, device=dev)
        S = torch.zeros(B, **f64)                            # sum of lambda * sample expectation
        reg = torch.zeros(B, **f64)
        E_l = self.lin_expect(d, Z)
        last_upd = torch.full((B, V), -1, dtype=torch.long, device=dev)
        last_chg = torch.full((B, V), -1, dtype=torch.long, device=dev)
        loss = (-S + lpn) + torch.log(Z) + reg
        act = torch.ones(B, dtype=torch.bool, device=dev)
        iters = torch.full((B,), max_iter, dtype=torch.long, device=dev)
        prev_loss = loss.clone()
        inf_l = torch.full((B, V), math.inf, **f64)
        n1c, devc, n1l, devl = self.cn1, self.cdev, self.n1l, self.devl

        def loss_of(S_, lpn_, Z_, reg_):
            return (-S_ + lpn_) + torch.log(Z_) + reg_

        ev = lambda a: a.gather(1, exp_idx)                  # (B, C+1) -> (B, K) in export order
        emask = lambda: kar.unsqueeze(0) < n_exp.unsqueeze(1)

        for it in range(max_iter):
            if it % 10 == 0 and not bool(act.any()):
                break
            old_loss = loss
            dlb_c, W = self.candidates(d, Z, lam_c, elam)        # (B, C+1); column C is the dummy slot
            dlb_cx, Wx = dlb_c, W
            if it > 0 and it % par_every == 0:
                # ------------------------------------------------ joint Newton step along the recent change
                em = emask()
                lam_e = ev(lam_c)
                nb_e = em & ~ev(self.cisthr)
                u_l = torch.where(self.vm & (lam_l != 0), lam_l - prev_l, torch.zeros_like(lam_l)) * act.unsqueeze(1)
                u_e = torch.where(nb_e & (lam_e != 0), lam_e - prev_e, torch.zeros_like(lam_e)) * act.unsqueeze(1)
                prev_l, prev_e = lam_l.clone(), torch.where(em, lam_e, prev_e)
                u_c = torch.zeros(B, Cn + 1, **f64).scatter_add_(1, exp_idx, u_e)
                Fu = torch.einsum("bnv,bv->bn", self.L, u_l) + self.contrib(u_c)
                ch = u_l != 0
                E_l = torch.where(ch, self.lin_expect(d, Z), E_l)
                last_upd = torch.where(ch, torch.full_like(last_upd, it), last_upd)
                uty = (d * Fu).sum(1) / Z
                uhu = (d * Fu * Fu).sum(1) / Z - uty * uty
                W_e, n1_e, dv_e = ev(Wx), ev(n1c), ev(devc)
                xty = (_deriv(E_l, n1l, devl, lam_l) * u_l).sum(1) + (_deriv(W_e, n1_e, dv_e, lam_e) * u_e).sum(1)
                step = torch.where(uhu < EPS * EPS, torch.zeros_like(uhu), -xty / uhu)
                lam_all, u_all = torch.cat([lam_l, lam_e], 1), torch.cat([u_l, u_e], 1)
                cross = (step.unsqueeze(1) * u_all + lam_all) * lam_all < 0
                r = torch.where(cross, -lam_all / torch.where(u_all == 0, torch.ones_like(u_all), u_all),
                                torch.full_like(u_all, math.inf))
                step = torch.where(cross.any(1), r.gather(1, r.abs().argmin(1, keepdim=True)).squeeze(1), step)
                a_l, a_e = step.unsqueeze(1) * u_l, step.unsqueeze(1) * u_e
                fl, fe = self.vm & act.unsqueeze(1), em & act.unsqueeze(1)
                a_l = torch.where(fl & (a_l != -lam_l) & ((a_l + lam_l).abs() < EPS), -lam_l, a_l)
                a_e = torch.where(fe & (a_e != -lam_e) & ((a_e + lam_e).abs() < EPS), -lam_e, a_e) * em
                loss_was = loss
                dlb_l = torch.where(self.vm, _bound(E_l, n1l, devl, lam_l), inf_l)
                upd = self.features_to_update(it, dlb_l, torch.where(em, ev(dlb_cx), math.inf), last_chg)
                a_c = torch.zeros(B, Cn + 1, **f64).scatter_add_(1, exp_idx, a_e)
                delta = torch.einsum("bnv,bv->bn", self.L, a_l) + self.contrib(a_c)

                def state(ll, le, lp_):
                    lpn_, d_, Z_ = self.density(lp_)
                    reg_ = (ll.abs() * devl).sum(1) + (le.abs() * dv_e * em).sum(1)
                    S_ = (ll * n1l).sum(1) + (le * n1_e * em).sum(1)
                    return ll, le, lp_, lpn_, d_, Z_, reg_, S_, loss_of(S_, lpn_, Z_, reg_)
                st = state(lam_l + a_l, lam_e + a_e, lp + delta)
                lam_c = lam_c.scatter_add(1, exp_idx, a_e)
                worse = act & (st[-1] > loss_was)
                if bool(worse.any()):
                    # undo: maxent.jar adds -alpha to the updated values (not an exact restore)
                    un = state(st[0] - a_l, st[1] - a_e, st[2] - delta)
                    st = tuple(torch.where(worse.view(-1, *([1] * (x.dim() - 1))), y, x) for x, y in zip(st, un))
                    lam_c = lam_c.scatter_add(1, exp_idx, torch.where(worse.unsqueeze(1), -a_e, torch.zeros_like(a_e)))
                lam_l, _, lp, lpn, d, Z, reg, S, loss = st
                lam_c[:, Cn] = 0.0
                elam = elam.scatter(1, exp_idx, torch.exp(-ev(lam_c)))
                upd = upd & act.unsqueeze(1)
                E_l = torch.where(upd, self.lin_expect(d, Z), E_l)
                last_upd = torch.where(upd, torch.full_like(last_upd, it), last_upd)
            else:
                # ------------------------------------------------ sequential step on the best feature
                dlb_l = torch.where(self.vm, _bound(E_l, n1l, devl, lam_l), inf_l)
                ml, il = dlb_l.min(1)
                mc, ic = dlb_c.min(1)
                is_lin = ml <= mc                              # candidates must be strictly better (Java order)
                g, t = self.cg[bi, ic], self.ct[bi, ic]
                v = torch.where(is_lin, il, self.gvar[g])
                is_bin = (~is_lin) & self.cisthr[bi, ic]
                dirh = self.gdir[g]
                thr_h, M_h = self.cthr[bi, ic], self.cM[bi, ic]
                xv = self.X[bi, :, v]
                xd = torch.where((dirh == 1).unsqueeze(1), -xv, xv)
                h_hinge = torch.where(xd > thr_h.unsqueeze(1), (xd - thr_h.unsqueeze(1)) / (M_h - thr_h).unsqueeze(1),
                                      torch.zeros_like(xd))
                h = torch.where(is_lin.unsqueeze(1), self.L[bi, :, v],
                                torch.where(is_bin.unsqueeze(1), (xv > thr_h.unsqueeze(1)).double(), h_hinge)) * self.ptm
                N1_h = torch.where(is_lin, n1l[bi, v], n1c[bi, ic])
                dv_h = torch.where(is_lin, devl[bi, v], devc[bi, ic])
                lam_h = torch.where(is_lin, lam_l[bi, v], lam_c[bi, ic])
                need = is_lin & act & (last_upd[bi, v] != it - 1)
                E_l[bi, v] = torch.where(need, (d * h).sum(1) / Z, E_l[bi, v])
                last_upd[bi, v] = torch.where(need, torch.full_like(v, it), last_upd[bi, v])
                E_h = torch.where(is_lin, E_l[bi, v], W[bi, ic])
                last_chg[bi, v] = torch.where(is_lin & act, torch.full_like(v, it), last_chg[bi, v])
                new = (~is_lin) & act & ~exported[bi, ic]
                exported[bi, ic] = exported[bi, ic] | new
                slot = n_exp.clamp(max=K - 1)
                exp_idx[bi, slot] = torch.where(new, ic, exp_idx[bi, slot])
                prev_e[bi, slot] = torch.where(new, torch.zeros_like(lam_h), prev_e[bi, slot])
                n_exp = n_exp + new.long()
                dlb_l = torch.where(self.vm, _bound(E_l, n1l, devl, lam_l), inf_l)
                upd = self.features_to_update(it, dlb_l, torch.where(emask(), ev(dlb_cx), math.inf), last_chg)
                dlb_h = _bound(E_h, N1_h, dv_h, lam_h)
                ga = _good_alpha(E_h, N1_h, dv_h, lam_h)
                var = (d * h * h).sum(1) / Z - E_h * E_h
                nt = torch.where(var < EPS * EPS, torch.zeros_like(var), -_deriv(E_h, N1_h, dv_h, lam_h) / var)
                nt = torch.where((nt + lam_h) * lam_h < 0, -lam_h, nt)
                a1 = _reduce(torch.where(is_bin, ga, nt), it) * act
                lam1 = lam_h + a1
                reg1 = reg + (torch.abs(lam1) - torch.abs(lam_h)) * dv_h
                S1 = S + a1 * N1_h
                lp1 = lp + a1.unsqueeze(1) * h
                lpn1, d1, Z1 = self.density(lp1)
                loss1 = loss_of(S1, lpn1, Z1, reg1)
                undo = act & (~is_bin) & (loss1 - old_loss > dlb_h)
                if bool(undo.any()):
                    ix = undo.nonzero().squeeze(1)
                    hh, aa, l1_, dvx, n1x = h[ix], a1[ix], lam1[ix], dv_h[ix], N1_h[ix]
                    lamu = l1_ + (-aa)
                    regu = reg1[ix] + (torch.abs(lamu) - torch.abs(l1_)) * dvx
                    Su = S1[ix] - aa * n1x
                    lpu = lp1[ix] + (-aa).unsqueeze(1) * hh
                    lpnu, du, Zu = self.density(lpu)
                    Eu = (du * hh).sum(1) / Zu
                    vi, lin_ix = v[ix], is_lin[ix]
                    E_l[ix, vi] = torch.where(lin_ix, Eu, E_l[ix, vi])
                    last_upd[ix, vi] = torch.where(lin_ix, torch.full_like(vi, it), last_upd[ix, vi])
                    a2 = _reduce(self.search_alpha(hh, _good_alpha(Eu, n1x, dvx, lamu), du, n1x, dvx, lamu), it)
                    lam2 = lamu + a2
                    reg2 = regu + (torch.abs(lam2) - torch.abs(lamu)) * dvx
                    S2 = Su + a2 * n1x
                    lp2 = lpu + a2.unsqueeze(1) * hh
                    lpn2, d2, Z2 = self.density(lp2)
                    lam1 = lam1.index_put((ix,), lam2); reg1 = reg1.index_put((ix,), reg2)
                    S1 = S1.index_put((ix,), S2); lp1 = lp1.index_put((ix,), lp2)
                    lpn1 = lpn1.index_put((ix,), lpn2); d1 = d1.index_put((ix,), d2); Z1 = Z1.index_put((ix,), Z2)
                    loss1 = loss1.index_put((ix,), loss_of(S2, lpn2, Z2, reg2))
                if self.trace is not None:
                    self.trace.append((it, loss1.clone(), is_lin.clone(), v.clone(), g.clone(), t.clone(), upd.clone()))
                lam_l[bi, v] = torch.where(is_lin, lam1, lam_l[bi, v])
                lam_c[bi, ic] = torch.where(is_lin, lam_c[bi, ic], lam1)
                elam[bi, ic] = torch.where(is_lin, elam[bi, ic], torch.exp(-lam1))
                lp, lpn, d, Z, reg, S, loss = lp1, lpn1, d1, Z1, reg1, S1, loss1
                upd = upd & act.unsqueeze(1)
                E_l = torch.where(upd, self.lin_expect(d, Z), E_l)
                last_upd = torch.where(upd, torch.full_like(last_upd, it), last_upd)
            # ---- termination test (Sequential.terminationTest)
            if it == 0:
                prev_loss = loss.clone()
            elif it % TEST_EVERY == 0:
                gain = torch.log(self.npts) - loss
                stop = act & (((prev_loss - loss) < conv) | (gain > 10000))
                iters = torch.where(stop, torch.full_like(iters, it), iters)
                act = act & ~stop
                prev_loss = torch.where(act, loss, prev_loss)
        p = d / Z.unsqueeze(1)
        ent = -(torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))).sum(1)
        lam_e = ev(lam_c)
        l1 = (lam_l.abs() * devl).sum(1) + (lam_e.abs() * ev(devc) * emask()).sum(1)
        secs = time.time() - t0
        out = []
        cpu = lambda x: x.cpu().numpy()
        lam_l_, lam_e_, idx_, n_, lpn_, Z_, ent_, d_, loss_, l1_, it_, cg_, ct_ = map(
            cpu, (lam_l, lam_e, exp_idx, n_exp, lpn, Z, ent, d, loss, l1, iters, self.cg, self.ct))
        for b, prep in enumerate(self.preps):
            ex = []
            for k in range(int(n_[b])):
                gg, tt = int(cg_[b, idx_[b, k]]), int(ct_[b, idx_[b, k]])
                var, kind = (gg, 0) if gg < V else ((gg - V) // 2, 1 + (gg - V) % 2)
                ex.append((var, kind, tt, float(lam_e_[b, k])))
            out.append(Fitted(prep, [], lam_l_[b, :prep.points.shape[1]].copy(), ex, float(lpn_[b]), float(Z_[b]),
                              float(ent_[b]), d_[b, :len(prep.points)].copy(), float(loss_[b]), float(l1_[b]),
                              int(it_[b]), secs))
        return out

    @staticmethod
    def search_alpha(h, a0, d, N1, beta, lam, K: int = 48):
        """Sequential.searchAlpha: multiply the step by 4 while the loss keeps dropping, then try doubling."""
        def L(a):
            Za = (d.unsqueeze(1) * torch.exp(a.unsqueeze(-1) * h.unsqueeze(1))).sum(-1)
            return (-a * N1.unsqueeze(1) + torch.log(Za)
                    + (torch.abs(lam.unsqueeze(1) + a) - torch.abs(lam.unsqueeze(1))) * beta.unsqueeze(1))
        pw = torch.pow(torch.tensor(4.0, dtype=h.dtype, device=h.device),
                       torch.arange(K + 1, device=h.device, dtype=h.dtype))
        al = a0.unsqueeze(1) * pw
        Ls = L(al)
        cont = (Ls[:, 1:] < Ls[:, :-1]) & torch.isfinite(Ls[:, 1:])
        stopk = torch.where(cont.all(1), torch.full_like(a0, K, dtype=torch.long), (~cont).long().argmax(1))
        a = al.gather(1, stopk.unsqueeze(1)).squeeze(1)
        cur = Ls.gather(1, stopk.unsqueeze(1)).squeeze(1)
        L2 = L((2 * a).unsqueeze(1)).squeeze(1)
        return torch.where((L2 < cur) & torch.isfinite(L2), 2 * a, a)


# ------------------------------------------------------------------------------------------- evaluation & outputs
def java_auc(pres: np.ndarray, bg: np.ndarray) -> tuple[float, float]:
    """FeaturedSpace.getAUC: AUC of presences vs background values (ties count 1/2) and its standard deviation."""
    d, y = np.sort(pres), np.sort(bg)
    dn, yn = len(d), len(y)
    if dn == 0:
        return 0.0, -1.0
    u = np.unique(np.concatenate([d, y]))
    de = (np.searchsorted(d, u, "right") - np.searchsorted(d, u, "left")).astype(np.int64)
    ye = (np.searchsorted(y, u, "right") - np.searchsorted(y, u, "left")).astype(np.int64)
    l = np.cumsum(ye) - ye
    g = dn - np.cumsum(de)
    auc2 = int(np.sum(de * l * 2 + de * ye))
    e104 = int(np.sum(de * (l * (l - 1) * 2 + ye * (ye - 1) // 2 + l * ye * 2)))
    e014 = int(np.sum(ye * (g * (g - 1) * 2 + de * (de - 1) // 2 + g * de * 2)))
    e114 = int(np.sum(de * l * 4 + de * ye))
    auc = auc2 / (dn * yn * 2.0)
    e10 = e104 / (2.0 * dn * yn * (yn - 1)) - auc * auc
    e01 = e014 / (2.0 * dn * (dn - 1) * yn) - auc * auc if dn > 1 else 0.0
    e11 = e114 / (4.0 * dn * yn) - auc * auc
    var = ((yn - 1) * e10 + (dn - 1) * e01 + e11) / (dn * yn)
    return auc, (-1.0 if dn == 1 else math.sqrt(max(var, 0.0)))


def cloglog(raw, entropy):
    return 1 - np.exp(-np.asarray(raw) * math.exp(entropy))


def _c_phi(x: float) -> float:
    R = [1.25331413731550025, .421369229288054473, .236652382913560671, .162377660896867462, .123131963257932296,
         .0990285964717319214, .0827662865013691773, .0710695805388521071, .0622586659950261958]
    j = int(.5 * (abs(x) + 1))
    if j >= len(R):
        return 0.0
    a, z = R[j], 2 * j
    b = a * z - 1
    h = abs(x) - z
    s, t, q, pwr, i = a + h * b, a, h * h, 1.0, 2
    while s != t:
        a = (a + z * b) / i
        b = (b + z * a) / (i + 1)
        pwr *= q
        t = s
        s = t + pwr * (a + h * b)
        i += 2
    s = s * math.exp(-.5 * x * x - .91893853320467274178)
    return s if x >= 0 else 1. - s


def _binomial(n, p, successrate):
    mean, sd = n * p, math.sqrt(n * p * (1 - p))
    if sd == 0:
        return 0.0 if (n > 0 and p != successrate) else 1.0
    return _c_phi((n * successrate - mean) / sd)


def _exact_binomial(success, n, p):
    prob = 0.0
    for i in range(success, n + 1):
        fac = 1
        ii = i if i >= n // 2 else n - i
        for j in range(ii + 1, n + 1):
            fac *= j
        for j in range(1, n - ii + 1):
            fac //= j
        prob += fac * math.pow(p, i) * math.pow(1 - p, n - i)
    return prob


RULES = ["Fixed cumulative value 1", "Fixed cumulative value 5", "Fixed cumulative value 10", "Minimum training presence",
         "10 percentile training presence", "Equal training sensitivity and specificity",
         "Maximum training sensitivity plus specificity", "Equal test sensitivity and specificity",
         "Maximum test sensitivity plus specificity", "Balance training omission, predicted area and threshold value",
         "Equate entropy of thresholded and original distributions"]


def thresholds(weights: np.ndarray, trainvals: np.ndarray, testvals: np.ndarray, entropy: float):
    """Runner.writeCumulativeIndex threshold rules. Returns ({rule: attrs}, cumulative-interpolation function)."""
    w = np.sort(weights)
    tr = np.sort(trainvals)
    te = np.asarray(testvals)
    hastest = len(te) > 0
    cw = 100.0 * np.cumsum(w) / np.cumsum(w)[-1]
    u = np.unique(np.concatenate([w, tr, te]))
    c0 = np.searchsorted(w, u, "left")
    area = 1.0 - c0 / len(w)
    trom = np.searchsorted(tr, u, "left") / len(tr)
    teom = np.searchsorted(np.sort(te), u, "left") / len(te) if hastest else np.zeros(len(u))

    def cum_at(x, i):
        x = np.asarray(x, np.float64)
        i = np.asarray(i)
        out = np.empty(x.shape)
        ic = np.clip(i, 0, len(w) - 1)
        eq = (i < len(w)) & (w[ic] == x)
        lo = np.clip(i - 1, 0, len(w) - 1)
        with np.errstate(all="ignore"):
            mid = cw[lo] + (x - w[lo]) / (w[ic] - w[lo]) * (cw[ic] - cw[lo])
        out = np.where(eq, cw[ic], np.where(i == 0, 0.0, np.where(i >= len(w), cw[-1], mid)))
        return out

    cum = cum_at(u, c0)
    occ = cloglog(u, entropy)

    def first(cond):
        k = np.flatnonzero(cond)
        return int(k[0]) if len(k) else None

    def argmin(v):
        return int(np.argmin(v))

    pick = {RULES[0]: first(cum >= 1), RULES[1]: first(cum >= 5), RULES[2]: first(cum >= 10),
            RULES[3]: first(u == tr[0]), RULES[4]: first(u == tr[len(tr) // 10]),
            RULES[5]: argmin(np.abs(trom - area)), RULES[6]: argmin(trom + area),
            RULES[9]: argmin(6 * trom + cum / 25.0 + area * 1.6),
            RULES[10]: first(area < math.exp(entropy) / len(w))}
    if hastest:
        pick[RULES[7]] = argmin(np.abs(teom - area))
        pick[RULES[8]] = argmin(teom + area)
    res = {}
    for rule in RULES:
        if rule not in pick:
            continue
        k = pick[rule]
        if k is None:
            res[rule] = None
            continue
        res[rule] = {"cumulative": float(cum[k]), "occ": float(occ[k]), "area": float(area[k]),
                     "trainomission": float(trom[k]), "testomission": float(teom[k]), "threshold": float(u[k])}
    for j, f in enumerate((1, 5, 10)):
        if res.get(RULES[j]) is not None:
            res[RULES[j]]["cumulative"] = float(f)

    def cumulative(raw):
        raw = np.asarray(raw, np.float64)
        i = np.searchsorted(w, raw, "right")          # interpolate(x, -1, ...): past all equal raw values
        i = np.where(np.isin(raw, w), np.searchsorted(w, raw, "left"), i)
        return cum_at(raw, i)
    return res, cumulative


def score_and_write(fit: Fitted, variables: list[str], outdir: Path, device="cpu") -> dict:
    """Write {label}.lambdas and {label}_samplePredictions.csv; return the maxentResults.csv row."""
    from .maxent import MaxentModel
    fit.variables = list(variables)
    run, prep = fit.prep.run, fit.prep
    text = fit.lambdas_text()
    (outdir / f"{run.label}.lambdas").write_text(text)
    model = MaxentModel.from_text(text, variables=variables, dtype=torch.float64, device=device)
    has_test = run.test is not None and len(run.test) > 0
    rows = [prep.points, run.train] + ([run.test] if has_test else [])
    # one evaluation of points and presences (per-row arithmetic is batch-invariant, so a presence and its copy
    # among the points get identical values, as in maxent.jar's AUC)
    lp_all = model.linear_predictor(torch.as_tensor(np.vstack(rows), dtype=torch.float64, device=device)).cpu().numpy()
    n_p, n_t = len(prep.points), len(run.train)
    lp_pts, lp_tr, lp_te = lp_all[:n_p], lp_all[n_p:n_p + n_t], lp_all[n_p + n_t:]
    pts_fresh = np.exp(lp_pts - fit.lpn)
    tr_d = np.exp(lp_tr - fit.lpn)
    te_d = np.exp(lp_te - fit.lpn) if has_test else np.zeros(0)
    train_auc, _ = java_auc(tr_d, pts_fresh)
    test_auc, auc_sd = java_auc(te_d, pts_fresh) if has_test else (0.0, -1.0)
    weights = fit.density / fit.dnorm
    trainvals = np.minimum(np.nan_to_num(tr_d / fit.dnorm, nan=1.0, posinf=1.0), 1.0)
    testvals = np.minimum(np.nan_to_num(te_d / fit.dnorm, nan=1.0, posinf=1.0), 1.0)
    thr, cumulative = thresholds(weights, trainvals, testvals, fit.entropy)
    N = len(prep.points)
    gain = math.log(N) - fit.loss
    row = {"Species": run.label, "#Training samples": len(run.train), "Regularized training gain": gain,
           "Unregularized training gain": gain + fit.l1, "Iterations": fit.iterations, "Training AUC": train_auc}
    if has_test:
        test_gain = math.log(N) - ((-float(np.mean(lp_te)) + fit.lpn) + math.log(fit.dnorm))
        row.update({"#Test samples": len(run.test), "Test gain": test_gain, "Test AUC": test_auc,
                    "AUC Standard Deviation": auc_sd})
    row["#Background points"] = N
    row["Entropy"] = fit.entropy
    row["Prevalence (average probability of presence over background sites)"] = float(np.mean(cloglog(weights, fit.entropy)))
    for rule in RULES:
        if not has_test and rule in (RULES[7], RULES[8]):
            continue
        t = thr.get(rule)
        cols = ["cumulative threshold", "Cloglog threshold", "area", "training omission"] + \
               (["test omission", "binomial probability"] if has_test else [])
        if t is None:
            for c in cols:
                row[f"{rule} {c}"] = "na"
            continue
        row[f"{rule} cumulative threshold"] = t["cumulative"]
        row[f"{rule} Cloglog threshold"] = t["occ"]
        row[f"{rule} area"] = t["area"]
        row[f"{rule} training omission"] = t["trainomission"]
        if has_test:
            n = len(run.test)
            p = (_binomial(n, t["area"], 1 - t["testomission"]) if n > 25 else
                 _exact_binomial(int(round((1 - t["testomission"]) * n)), n, t["area"]))
            row[f"{rule} test omission"] = t["testomission"]
            row[f"{rule} binomial probability"] = fmt_sci3(p)
    # sample predictions (train rows in training order, then test rows)
    lines = ["X,Y,Test or train,Raw prediction,Cumulative prediction,Cloglog prediction"]
    raw_tr = tr_d / fit.dnorm
    for xy_, raw, kind in ((run.train_xy, raw_tr, "train"), (run.test_xy if has_test else np.zeros((0, 2)), testvals, "test")):
        for (x, y), r_, c_, o_ in zip(xy_.tolist(), raw.tolist(), cumulative(raw).tolist(), cloglog(raw, fit.entropy).tolist()):
            lines.append(f"{jstr(x)},{jstr(y)},{kind},{jstr(r_)},{jstr(c_)},{jstr(o_)}")
    (outdir / f"{run.label}_samplePredictions.csv").write_text("\n".join(lines) + "\n")
    return row


def write_results(rows: list[dict], label: str, path: Path) -> None:
    """maxentResults.csv: one row per replicate and maxent.jar's '(average)' row (mean of the printed values)."""
    cols = list(dict.fromkeys(c for r in rows for c in r))
    protect = lambda x: f'"{x}"' if "," in x else x       # CsvWriter.protect
    out = [",".join(map(protect, cols))]
    printed = [{c: fmt4(r[c]) if c in r else "" for c in cols} for r in rows]
    for p in printed:
        out.append(",".join(protect(p[c]) for c in cols))
    if len(rows) > 1:
        avg = {"Species": f"{label} (average)"}
        for c in cols[1:]:
            try:
                avg[c] = fmt4(float(np.mean([float(p[c]) for p in printed])))
            except ValueError:
                avg[c] = ""
        out.append(",".join(protect(avg.get(c, "")) for c in cols))
    path.write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------------------------ maxent.jar's run protocols
def read_swd(samples: Path, background: Path):
    """Read SWD files as maxent.jar does: variables are the background's columns from the 4th on, in sorted order
    (ParamsPre.getSelected sorts layer names; the order fixes feature order and tie-breaking); rows with the NODATA
    value are dropped; duplicate presence coordinates are removed (first kept)."""
    bg = pd.read_csv(background, float_precision="round_trip")
    variables = sorted(bg.columns[3:])
    s = pd.read_csv(samples, float_precision="round_trip")
    B = bg[variables].to_numpy(np.float64)
    B = B[~(B == NODATA).any(1)]
    S = s[variables].to_numpy(np.float64)
    xy = s.iloc[:, 1:3].to_numpy(np.float64)
    ok = ~(S == NODATA).any(1) & ~np.isnan(S).any(1)
    S, xy = S[ok], xy[ok]
    seen, keep = set(), []
    for i, (x, y) in enumerate(xy):
        k = (y, x)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    keep = np.asarray(keep, np.int64)
    return variables, B, S[keep], xy[keep]


def cv_folds(n: int, k: int = 5) -> np.ndarray:
    """SampleSet.splitForCV with java.util.Random(0): fold index of each presence (in file order)."""
    rnd = JavaRandom(0)
    r = np.array([rnd.next_double() for _ in range(n)])
    order = np.argsort(r, kind="stable")
    return order % min(n, k)


def subsample_splits(n: int, replicates: int = 5, percent: int = 25, seed: int = 0) -> list[np.ndarray]:
    """maxent.jar's replicatetype=subsample test draws (SampleSet.replicate + randomSample) with a fixed seed: the
    base species list is drawn first, then each replicate's. Returns the test indices of each replicate, in draw
    order."""
    rnd = JavaRandom(seed)
    out = []
    for j in range(-1, replicates):
        pool = list(range(n))
        test = []
        for _ in range(int((percent * n) / 100.0)):
            if pool:
                sel = int(rnd.next_double() * len(pool))
                test.append(pool.pop(sel))
        if j >= 0:
            out.append(np.asarray(test, np.int64))
    return out


def cv_runs(label, beta, B, S, xy, k=5):
    fold = cv_folds(len(S), k)
    runs = []
    for j in range(min(len(S), k)):
        tr, te = fold != j, fold == j
        runs.append(Run(f"{label}_{j}", beta, B, S[tr], xy[tr], S[te], xy[te]))
    return runs


def subsample_runs(label, beta, B, S, xy, replicates=5, percent=25, seed=0):
    runs = []
    for j, te in enumerate(subsample_splits(len(S), replicates, percent, seed)):
        tr = np.setdiff1d(np.arange(len(S)), te)
        runs.append(Run(f"{label}_{j}", beta, B, S[tr], xy[tr], S[te], xy[te]))
    return runs


def runs_from_sample_predictions(run_dir: Path, label: str, beta: float, S: np.ndarray, xy: np.ndarray) -> list[Run]:
    """Recreate the train/test splits of an existing run directory (maxent.jar's or ours) from its
    ``*_samplePredictions.csv`` files (training presences in training order, test presences in draw order), so the
    same replicates can be refitted by either engine."""
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(xy)}
    runs = []
    for f in sorted(Path(run_dir).glob(f"{label}_[0-9]*_samplePredictions.csv"),
                    key=lambda p: int(p.name[len(label) + 1:].split("_")[0])):
        sp = pd.read_csv(f, float_precision="round_trip")
        idx = [index[(x, y)] for x, y in zip(sp.X, sp.Y)]
        is_tr = (sp["Test or train"] == "train").values
        tr, te = np.asarray(idx)[is_tr], np.asarray(idx)[~is_tr]
        name = f.name[: -len("_samplePredictions.csv")]
        runs.append(Run(name, beta, None, S[tr], xy[tr], S[te] if len(te) else None, xy[te] if len(te) else None))
    return runs


def fit_runs(runs: list[Run], device="cuda", max_batch_points: Optional[int] = None) -> list[Fitted]:
    """Prepare and fit runs on the GPU in batches bounded by total padded size (runs x points x variables;
    about 0.7 GB of GPU memory per million, default 8 million, env MAXENT_TORCH_BATCH_POINTS)."""
    import os
    max_batch_points = max_batch_points or int(os.environ.get("MAXENT_TORCH_BATCH_POINTS", 8_000_000))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(8, len(runs))) as ex:
        preps = list(ex.map(prepare, runs))
    out: list[Optional[Fitted]] = [None] * len(preps)
    order = sorted(range(len(preps)), key=lambda i: len(preps[i].points))
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order):
            grp = [preps[k] for k in order[i:j + 1]]
            size = len(grp) * max(len(p.points) for p in grp) * max(p.points.shape[1] for p in grp)
            if size > max_batch_points:
                break
            j += 1
        fits = Batch([preps[k] for k in order[i:j]], device).run()
        for k, f in zip(order[i:j], fits):
            out[k] = f
        i = j
    return out


def write_run_dir(outdir: Path, label: str, fits: list[Fitted], variables: list[str], device="cpu") -> None:
    from concurrent.futures import ThreadPoolExecutor
    outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(min(5, len(fits))) as ex:
        rows = list(ex.map(lambda f: score_and_write(f, variables, outdir, device), fits))
    write_results(rows, label, outdir / "maxentResults.csv")
    (outdir / "maxent.log").write_text(
        f"Fitted by ranges.maxent_torch (GPU reimplementation of maxent.jar 3.4.x fitting)\n"
        f"runs: {len(fits)}; beta multiplier: {fits[0].prep.run.beta_multiplier}; "
        f"iterations: {[f.iterations for f in fits]}; GPU seconds: {fits[0].seconds:.2f}\n")


def cv_test_auc(results_csv: Path) -> float:
    res = pd.read_csv(results_csv)
    return float(res.loc[res.Species.str.contains("average"), "Test AUC"].iloc[0])


def fit_species_dir(workdir: Path, label: str, betas: Sequence[float] = (2, 5, 10, 15, 20), replicates: int = 5,
                    seed: int = 0, device: str = "cuda", speculative_final: bool = True) -> tuple[float, dict]:
    """maxent.jar's protocol in ``modelling.fit_species`` from the SWD files in ``workdir``: 5-fold CV test AUC per
    beta (cv_beta<beta>/), best beta (first among equal printed AUCs), then ``replicates`` subsampled replicates
    with 25 % test points at that beta (final/). With ``speculative_final`` the replicates of every beta are fitted
    in the same GPU batch as the cross-validation runs and only the chosen beta's are written (one batch instead
    of two)."""
    return fit_species_dirs([(workdir, label)], betas, replicates, seed, device, speculative_final)[0]


def fit_species_dirs(species: Sequence[tuple[Path, str]], betas: Sequence[float] = (2, 5, 10, 15, 20),
                     replicates: int = 5, seed: int = 0, device: str = "cuda",
                     speculative_final: bool = True) -> list[tuple[float, dict]]:
    """``fit_species_dir`` for several species at once: all their runs share GPU batches (higher throughput when
    refitting many species). Returns (best beta, CV AUC per beta) for each species."""
    plan, allruns = [], []
    for workdir, label in species:
        workdir = Path(workdir)
        variables, B, S, xy = read_swd(workdir / "samples.csv", workdir / "background.csv")
        cv = {beta: cv_runs(label, beta, B, S, xy) for beta in betas}
        fin = {beta: subsample_runs(label, beta, B, S, xy, replicates, 25, seed) for beta in betas} if speculative_final else {}
        k0 = len(allruns)
        allruns += [r for b in betas for r in cv[b]] + [r for b in fin for r in fin[b]]
        plan.append((workdir, label, variables, B, S, xy, cv, fin, k0))
    fits = fit_runs(allruns, device)
    out = []
    for workdir, label, variables, B, S, xy, cv, fin, k in plan:
        cvauc = {}
        for beta in betas:
            n = len(cv[beta])
            d = workdir / f"cv_beta{beta}"
            write_run_dir(d, label, fits[k:k + n], variables, device)
            cvauc[beta] = cv_test_auc(d / "maxentResults.csv")
            k += n
        best = max(cvauc, key=cvauc.get)
        if speculative_final:
            for beta in betas:
                n = len(fin[beta])
                if beta == best:
                    write_run_dir(workdir / "final", label, fits[k:k + n], variables, device)
                k += n
        else:
            write_run_dir(workdir / "final", label,
                          fit_runs(subsample_runs(label, best, B, S, xy, replicates, 25, seed), device), variables, device)
        out.append((best, cvauc))
    return out
