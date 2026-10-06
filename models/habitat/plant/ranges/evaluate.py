"""Daru's (2024) per-species model metrics (his Fig. S2), computed the way phyloregion 1.0.9 computes them.

* AUC: test presences vs background (``predicts::pa_evaluate``).
* TSS: phyloregion's ``myMAXENT`` takes the **median** of (TPR + TNR − 1) over all thresholds of the
  evaluation's confusion table — not the usual maximum — which is why Daru's median TSS is only 0.42. Both the
  median (Daru) and the maximum (conventional) are reported.
* Boyce index: phyloregion ``boyce()`` (continuous Boyce index, moving window of 1/10 of the prediction range,
  101 windows, Spearman correlation of predicted-to-expected ratio vs window position).
Medians over the five replicates are returned. Background predictions are recomputed exactly from each
replicate's lambdas and the saved background SWD files.

Which background the metrics are scored against matters (D10). phyloregion's ``sdm`` scores test presences
against the bias-weighted points it was given, but Daru's reported medians (AUC 0.91, Boyce 0.87) are only
reached against the background the model was fitted with (uniform, D9). Both are returned: unsuffixed keys use
the fitting background; ``*_eval_bg`` keys use ``evaluation_background.csv`` (bias-weighted) when it exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .maxent import MaxentModel


def boyce(fit: np.ndarray, obs: np.ndarray, res: int = 100) -> float:
    """phyloregion::boyce with nclass = 0 (moving window), window.w = range(fit)/10, rm.duplicate = TRUE."""
    mini, maxi = min(fit.min(), obs.min()), max(fit.max(), obs.max())
    w = (fit.max() - fit.min()) / 10
    vec = np.linspace(mini, maxi - w, res + 1)
    vec[res] = vec[res] + 1
    f = []
    for lo in vec:
        hi = lo + w
        pi = np.mean((obs >= lo) & (obs <= hi))
        ei = np.mean((fit >= lo) & (fit <= hi))
        f.append(round(pi / ei, 10) if ei > 0 else np.nan)
    f = np.asarray(f)
    keep = np.isfinite(f)
    f, v = f[keep], vec[keep]
    if len(f) < 2:
        return float("nan")
    r = np.nonzero(f != np.append(f[1:], True))[0]          # drop runs of duplicated ratios
    return float(spearmanr(f[r], v[r]).statistic)


def _confusion_tss(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    """TPR + TNR − 1 at every threshold of predicts::pa_evaluate (each unique predicted value)."""
    th = np.unique(np.concatenate([p, a]))
    ps, as_ = np.sort(p), np.sort(a)
    tp = len(p) - np.searchsorted(ps, th, side="left")       # presences >= threshold
    tn = np.searchsorted(as_, th, side="left")                # absences < threshold
    return tp / len(p) + tn / len(a) - 1


def _metrics(models: list, test: list, raw_test: list, xb: torch.Tensor) -> dict:
    rows = []
    for m, t, rt in zip(models, test, raw_test):
        a = np.sort(m.cloglog(xb).numpy())
        lo, hi = np.searchsorted(a, t, side="left"), np.searchsorted(a, t, side="right")
        auc = (lo.sum() + 0.5 * (hi - lo).sum()) / (len(t) * len(a))
        tss = _confusion_tss(t, a)
        rows.append({"AUC": auc, "TSS_median_daru": float(np.median(tss)), "TSS_max": float(tss.max()),
                     "Boyce": boyce(m.raw(xb).numpy(), rt)})
    d = pd.DataFrame(rows)
    return {k: float(d[k].median()) for k in d.columns}


def daru_metrics(final_dir: Path, label: str, variables: list[str], background_csv: Path) -> dict:
    """``background_csv``: the fitting background (MaxEnt SWD). Metrics against the bias-weighted
    ``evaluation_background.csv`` next to it are added with the suffix ``_eval_bg`` when that file exists."""
    models, test, raw_test = [], [], []
    for lam in sorted(final_dir.glob(f"{label}_[0-9]*.lambdas")):
        i = lam.stem.rsplit("_", 1)[1]
        sp = pd.read_csv(final_dir / f"{label}_{i}_samplePredictions.csv")
        is_test = sp["Test or train"] == "test"
        if not is_test.any():
            continue
        models.append(MaxentModel.from_lambdas(lam, variables=variables))
        test.append(sp.loc[is_test, "Cloglog prediction"].values)
        raw_test.append(sp.loc[is_test, "Raw prediction"].values)
    load = lambda f: torch.tensor(pd.read_csv(f)[variables].values, dtype=torch.float64)
    out = _metrics(models, test, raw_test, load(background_csv)) | {"replicates": len(models)}
    ev = Path(background_csv).with_name("evaluation_background.csv")
    if ev.exists():
        out |= {f"{k}_eval_bg": v for k, v in _metrics(models, test, raw_test, load(ev)).items()}
    return out
