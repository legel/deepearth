"""FluxFill-Bench scores: CRPS (from samples and in closed form), interval coverage, PIT, sums and gap classes.

Every score is computed on the masked, measured values only; the caller passes those values and their predictions.
CRPS is the primary score (Gneiting and Raftery 2007, J. Am. Stat. Assoc. 102:359). The sample estimator is the
fair (unbiased in the ensemble size) form of Ferro 2014 (Q. J. R. Meteorol. Soc. 140:1917); the closed forms for the
Gaussian, Laplace and Student-t follow Jordan, Krueger and Lerch 2019 (J. Stat. Softw. 90(12), scoringRules).
"""

from typing import Dict, Tuple

import numpy as np
from scipy import special, stats

GAP_CLASSES = ((1, 1), (2, 8), (9, 64), (65, 576), (577, np.inf))
"""Gap-length classes in half-hours (Irvin et al. 2021): 1; 2 to 8; 9 to 64 (1.5 d); 65 to 576 (12 d); longer."""


def crps_samples(samples: np.ndarray, y: np.ndarray, fair: bool = True) -> np.ndarray:
    """CRPS of each truth `y[j]` against the predictive samples `samples[:, j]` (m samples x n values).

    E|X - y| - 0.5 E|X - X'|, with E|X - X'| from the sorted samples: 2 / (m (m - 1)) sum_i (2i - m - 1) x_(i)
    when `fair`, else with m^2 in place of m (m - 1)."""
    x = np.sort(np.asarray(samples, np.float64), axis=0)
    y = np.asarray(y, np.float64)
    m = x.shape[0]
    if m < 2:
        raise ValueError("CRPS from samples needs at least 2 samples")
    term1 = np.abs(x - y[None]).mean(axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    spread = 2.0 * (w[:, None] * x).sum(axis=0) / (m * (m - 1) if fair else m * m)
    return term1 - 0.5 * spread


def crps_gaussian(mu, sigma, y) -> np.ndarray:
    """CRPS of N(mu, sigma^2): sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)], z = (y - mu) / sigma."""
    sigma = np.asarray(sigma, np.float64)
    z = (np.asarray(y, np.float64) - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_laplace(mu, b, y) -> np.ndarray:
    """CRPS of Laplace(mu, b): b [|z| + exp(-|z|) - 3/4], z = (y - mu) / b."""
    b = np.asarray(b, np.float64)
    z = np.abs((np.asarray(y, np.float64) - mu) / b)
    return b * (z + np.exp(-z) - 0.75)


def crps_student_t(mu, sigma, nu, y) -> np.ndarray:
    """CRPS of a location-scale Student-t with nu > 1 degrees of freedom (Jordan et al. 2019, eq. for crps_t)."""
    sigma, nu = np.asarray(sigma, np.float64), np.asarray(nu, np.float64)
    if np.any(nu <= 1):
        raise ValueError("the Student-t CRPS is finite only for nu > 1")
    z = (np.asarray(y, np.float64) - mu) / sigma
    f, F = stats.t.pdf(z, nu), stats.t.cdf(z, nu)
    beta = special.beta(0.5, nu - 0.5) / special.beta(0.5, nu / 2) ** 2
    return sigma * (z * (2 * F - 1) + 2 * f * (nu + z ** 2) / (nu - 1) - 2 * np.sqrt(nu) * beta / (nu - 1))


def coverage_samples(samples: np.ndarray, y: np.ndarray, levels=(0.5, 0.8, 0.95)) -> Dict[float, float]:
    """Fraction of `y` inside the central interval of each level, from the empirical sample quantiles."""
    out = {}
    for lv in levels:
        lo, hi = np.quantile(samples, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        out[lv] = float(np.mean((y >= lo) & (y <= hi)))
    return out


def pit_samples(samples: np.ndarray, y: np.ndarray, rng=None) -> np.ndarray:
    """Randomised probability integral transform: P(X < y) + U P(X = y), U uniform (ties among samples)."""
    rng = np.random.default_rng(0) if rng is None else rng
    below = (samples < y[None]).mean(axis=0)
    equal = (samples == y[None]).mean(axis=0)
    return below + rng.uniform(size=y.shape) * equal


def gap_runs(mask: np.ndarray) -> np.ndarray:
    """Length of the contiguous masked run that each element belongs to (0 where not masked), along the last axis."""
    mask = np.asarray(mask, bool)
    out = np.zeros(mask.shape, np.int64)
    flat_m, flat_o = mask.reshape(-1, mask.shape[-1]), out.reshape(-1, mask.shape[-1])
    for r in range(flat_m.shape[0]):
        row = flat_m[r]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        for start, stop in zip(edges[::2], edges[1::2]):
            flat_o[r, start:stop] = stop - start
    return out


def gap_class(lengths: np.ndarray) -> np.ndarray:
    """Index into GAP_CLASSES for each gap length (-1 for 0, i.e. not masked)."""
    lengths = np.asarray(lengths)
    cls = np.full(lengths.shape, -1, np.int64)
    for k, (lo, hi) in enumerate(GAP_CLASSES):
        cls[(lengths >= lo) & (lengths <= hi)] = k
    return cls


def block_sums(samples: np.ndarray, y: np.ndarray, scored: np.ndarray, block: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sums of the predictive samples and of the truth over consecutive blocks of `block` steps, counting only the
    steps in `scored` (masked and measured). Blocks with no scored step are dropped.

    samples: (m, T) joint samples over one window; y, scored: (T,). Returns (sample sums (m, nb), true sums (nb,))."""
    T = y.shape[0]
    nb = T // block
    s = np.where(scored[None, : nb * block], samples[:, : nb * block], 0.0).reshape(samples.shape[0], nb, block).sum(-1)
    t = np.where(scored[: nb * block], np.nan_to_num(y[: nb * block]), 0.0).reshape(nb, block).sum(-1)
    keep = scored[: nb * block].reshape(nb, block).any(-1)
    return s[:, keep], t[keep]


def point_scores(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, nMAE (MAE / std of truth), sklearn-style R2, squared Pearson r and bias (pred - truth)."""
    e = pred - y
    sd = np.std(y)
    ss_res, ss_tot = np.sum(e ** 2), np.sum((y - y.mean()) ** 2)
    r = np.corrcoef(pred, y)[0, 1] if len(y) > 1 and np.std(pred) > 0 else np.nan
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "nmae": float(np.mean(np.abs(e)) / sd) if sd > 0 else np.nan,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "pearson_r2": float(r ** 2),
            "bias": float(np.mean(e)), "n": int(len(y))}
