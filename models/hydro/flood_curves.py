"""FLOOD as a few numbers per wet cell: the 5-year storm's depth over time, and the peak water surface.

The storm is solved once (`solver.py`, from the soil water balance's state at its first hour, through its recession).
Each cell ever deeper than CUT_M keeps a piecewise-linear depth curve through KNOTS of the solver's own frames, chosen
greedily where the curve so far errs most (Douglas and Peucker 1973 with a fixed count), and its peak water surface.
A LiDAR return is flooded where it lies below its cell's peak surface.
"""

from typing import Dict, Tuple

import numpy as np
import torch

KNOTS = 8
CUT_M = 0.003                  # shallower is drawn dry
T_UNIT_S = 10.0                # the stored knots' time step
H_UNIT_M = 1e-4                # and depth step


def fit_knots(depth, times_s, k: int = KNOTS, cut: float = CUT_M, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Per cell, `k` of its frames through which a piecewise-linear curve best holds its depth, added one at a time
    where the curve so far errs most; always the frame before it first wets and the last.

    Args:
        depth: [cells, frames] m. times_s: [frames] s.
    Returns:
        (t [cells, k] s, h [cells, k] m), knots in time order.
    """
    D = torch.as_tensor(np.asarray(depth, np.float32), device=device)
    T = torch.as_tensor(np.asarray(times_s, np.float64), device=device)
    n, f = D.shape
    wet = D >= cut
    first = torch.where(wet.any(1), wet.float().argmax(1), torch.zeros(n, dtype=torch.long, device=device))
    start = torch.clamp(first - 1, min=0)
    idx = torch.stack([start, torch.full_like(start, f - 1)], 1)
    fr = torch.arange(f, device=device)
    for _ in range(k - 2):
        err = (_interp(D, idx, fr) - D).abs()
        err[fr[None, :] < start[:, None]] = -1.0
        err[torch.zeros_like(err, dtype=torch.bool).scatter_(1, idx, True)] = -1.0
        idx, _ = torch.sort(torch.cat([idx, err.argmax(1)[:, None]], 1), 1)
    return T[idx].cpu().numpy(), torch.gather(D, 1, idx).double().cpu().numpy()


def _interp(D, idx, fr):
    """Each row of D linear through its frames `idx` (sorted), at every frame `fr`; flat before the first."""
    n, m = idx.shape
    seg = torch.searchsorted(idx.contiguous(), fr[None, :].expand(n, -1).contiguous(), right=True) - 1
    seg = torch.clamp(seg, 0, m - 2)
    a, b = torch.gather(idx, 1, seg), torch.gather(idx, 1, seg + 1)
    ha, hb = torch.gather(D, 1, a), torch.gather(D, 1, b)
    w = torch.where(b > a, (fr[None, :] - a).float() / (b - a).clamp(min=1).float(), torch.ones_like(ha))
    out = ha + (hb - ha) * w.clamp(0.0, 1.0)
    return torch.where(fr[None, :] < idx[:, :1], ha, out)


def depth_at(t_knots: np.ndarray, h_knots: np.ndarray, t: float) -> np.ndarray:
    """Every cell's depth at time t: linear between its knots, flat outside them."""
    tk, hk = np.asarray(t_knots, np.float64), np.asarray(h_knots, np.float64)
    out = np.empty(len(tk))
    before, after = t <= tk[:, 0], t >= tk[:, -1]
    out[before], out[after] = hk[before, 0], hk[after, -1]
    mid = ~(before | after)
    if mid.any():
        tm, hm = tk[mid], hk[mid]
        j = np.clip((tm <= t).sum(1) - 1, 0, tk.shape[1] - 2)
        r = np.arange(len(tm))
        t0, t1, h0, h1 = tm[r, j], tm[r, j + 1], hm[r, j], hm[r, j + 1]
        span = np.where(t1 > t0, t1 - t0, 1.0)
        out[mid] = h0 + (h1 - h0) * np.where(t1 > t0, (t - t0) / span, 1.0)
    return out


def quantize(t_s: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    """uint16 [2][k][cells]: times in T_UNIT_S, then depths in H_UNIT_M."""
    tq = np.clip(np.round(np.asarray(t_s) / T_UNIT_S), 0, 65535).astype("<u2")
    hq = np.clip(np.round(np.asarray(h_m) / H_UNIT_M), 0, 65535).astype("<u2")
    return np.ascontiguousarray(np.stack([tq.T, hq.T]))


def dequantize(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return raw[0].T.astype(np.float64) * T_UNIT_S, raw[1].T.astype(np.float64) * H_UNIT_M


def peak_surface(ground_m: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Each wet cell's peak water surface, m: its ground plus its deepest frame."""
    return np.asarray(ground_m, np.float64) + np.asarray(depth, np.float64).max(1)


def flooded(z_point: np.ndarray, surface_of_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per LiDAR return, whether the water stood above it and by how much (m): surface - z where positive. NaN surface
    (a dry cell) is never flooded."""
    d = np.asarray(surface_of_point, np.float64) - np.asarray(z_point, np.float64)
    wet = np.isfinite(d) & (d > 0)
    return wet, np.where(wet, d, 0.0)


def error(depth: np.ndarray, times_s: np.ndarray, t_knots: np.ndarray, h_knots: np.ndarray, cut: float = CUT_M) -> Dict:
    """The curves against the solver's own frames: RMS and 99th-percentile depth error over cell-frames wet on either
    side; the wet footprint's intersection over union, pooled over every frame, at the cut, 1 cm and 3 cm; the stored
    volume's error at the fullest frame."""
    se, n, errs, vol = 0.0, 0, [], []
    pool = {c: [0, 0] for c in (cut, 0.01, 0.03)}
    for i, t in enumerate(times_s):
        truth = np.asarray(depth[:, i], np.float64)
        rec = depth_at(t_knots, h_knots, float(t))
        m = (truth >= cut) | (rec >= cut)
        e = np.abs(rec - truth)[m]
        se += float((e ** 2).sum())
        n += int(m.sum())
        errs.append(e)
        for c in pool:
            pool[c][0] += int(((truth >= c) & (rec >= c)).sum())
            pool[c][1] += int(((truth >= c) | (rec >= c)).sum())
        vol.append((float(truth.sum()), float(rec.sum())))
    allerr = np.concatenate(errs) if n else np.zeros(1)
    tv = np.array(vol)
    ipk = int(np.argmax(tv[:, 0]))
    return {"rmse_m": (se / max(n, 1)) ** 0.5, "p99_abs_m": float(np.percentile(allerr, 99)),
            "iou": {f"{c:g}": v[0] / max(v[1], 1) for c, v in pool.items()},
            "volume_rel_err_at_fullest": float(abs(tv[ipk, 1] - tv[ipk, 0]) / max(tv[ipk, 0], 1e-9))}
