"""Chronos-2 zero-shot forecasting with covariates (Ansari et al. 2025) as a FluxFill-Bench method.

Ansari, A. F., et al. 2025. Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.
Package chronos-forecasting 2.3.2 (`chronos.Chronos2Pipeline`), weights amazon/chronos-2 from the Hugging Face hub
(snapshot 29ec3766d36d6f73f0696f85560a422f50e8498c). No fine-tuning: the model is used zero-shot.

Gap filling as forecasting. Each contiguous hidden block [a, b) of the window is forecast from the series before
it. The target is every hidden tower variable of the block, forecast jointly as one multivariate group per kind
(fluxes; meteorology), so the model shares information across them. Covariates: the 7 ERA5 drivers and the
hour-of-day and day-of-year as sine and cosine, given over the history and as known future values over the block
(the runner never hides ERA5 in M2). History: the `CONTEXT_STEPS` half-hours before a, from the window's visible
values and, before the window start, from the site's other calendar years (`context`; rows of the window's own
calendar years are absent there and enter as missing). Chronos-2 masks missing values in the history.

Forecast direction: forward when the history before the block holds at least one day of observed target values;
otherwise the series is reversed in time and the block is forecast from the visible values after it (the same model,
applied to the time-reversed series and covariates). When neither side holds a day of observations the block is
left unfilled. Blocks longer than the model's 1,024-step output use the package's own long-horizon rollout
(predicted quantiles fed back as context).

Memory and precision. During long-horizon rollout the pipeline expands each series by its unrolled quantiles, and
the peak GPU memory grows by about 0.4 GB per series at an 8,640-step horizon (measured: 15 series 3.5 GB, 19 series
4.9 GB, 23 series 6.5 GB in bfloat16). A joint group of 12 meteorological variables with 11 covariates does not fit
the 4 GB GPU share this baseline runs under, so meteorology is forecast per variable. The weights are stored in
float32 and run in bfloat16 on a GPU and in float32 on a CPU (`dtype`); float32 needs 3.9 GB for the flux group alone. The bfloat16 scores were
checked against a float32 CPU run on the same windows (FLUXFILL_STATE.md, 2026-10-27).

Uncertainty: the model's 21 quantiles (0.01 to 0.99) per step become samples by inverse-CDF draws
(common.quantile_samples), independent per half-hour. Sums from these samples assume independence.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common import seed_of, quantile_samples

os.environ.setdefault("HF_HOME", "/home/photon/4tb/hf")
MODEL_ID = "amazon/chronos-2"
CONTEXT_STEPS = 2048
ERA5 = ("ERA5_TA", "ERA5_SW_IN", "ERA5_LW_IN", "ERA5_VPD", "ERA5_PA", "ERA5_P", "ERA5_WS")
FLUX = ("NEE", "LE", "H", "FCH4")


def _time_cov(t: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    h = 2 * np.pi * (t.hour * 2 + t.minute // 30).to_numpy() / 48.0
    a = 2 * np.pi * (t.dayofyear.to_numpy() - 1) / 365.0
    return {"hour_sin": np.sin(h), "hour_cos": np.cos(h), "doy_sin": np.sin(a), "doy_cos": np.cos(a)}


def runs(x: np.ndarray):
    e = np.flatnonzero(np.diff(np.concatenate(([0], x.astype(np.int8), [0]))))
    return list(zip(e[::2], e[1::2]))


class Chronos2:
    name = "chronos2"

    def __init__(self, n_samples: int = 64, device: str = "cuda", context_steps: int = CONTEXT_STEPS,
                 batch_size: int = 64, seed: int = 0, dtype: Optional[str] = None):
        import torch
        from chronos import Chronos2Pipeline
        dev = device if torch.cuda.is_available() else "cpu"
        dtype = dtype or ("bfloat16" if dev.startswith("cuda") else "float32")   # CPUs lack fast bfloat16 kernels
        self.pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=dev, dtype=getattr(torch, dtype))
        self.q = np.asarray(self.pipe.quantiles, np.float64)
        self.n, self.cl, self.bs, self.seed = n_samples, context_steps, batch_size, seed
        self.log: List[Dict] = []

    def _series(self, w, mask, context, keys: List[str], a: int, b: int):
        """History (context_steps before a) of the target keys and covariates, and covariates over [a, b)."""
        t_hist = pd.date_range(w.t[a] - pd.Timedelta(minutes=30) * self.cl, periods=self.cl, freq="30min")
        tab, idx = context["table"], pd.DatetimeIndex(context["index"])
        ctab = tab.set_index(idx)
        ctab = ctab[~ctab.index.duplicated()].reindex(t_hist)
        in_win = t_hist >= w.t[0]
        pos = ((t_hist[in_win] - w.t[0]) // pd.Timedelta(minutes=30)).to_numpy()

        def hist(k: str) -> np.ndarray:
            x = np.full(self.cl, np.nan)
            if k in ctab:
                v = ctab[k].to_numpy(np.float64)
                f = ctab.get(f"{k}_flag")
                ok = np.isfinite(v) & ((f.to_numpy() == 0) if f is not None else True)
                x = np.where(ok, v, np.nan)
            if k in w.keys:
                i = w.keys.index(k)
                x[in_win] = np.where(mask[i, pos], np.nan, w.values[i, pos])
            return x

        target = np.stack([hist(k) for k in keys])
        cov_keys = [k for k in ERA5 if k in w.keys]
        past = {k: hist(k) for k in cov_keys}
        past.update(_time_cov(t_hist))
        fut = {}
        for k in cov_keys:
            i = w.keys.index(k)
            fut[k] = np.where(mask[i, a:b], np.nan, w.values[i, a:b])
        fut.update(_time_cov(w.t[a:b]))
        return target, past, fut

    def _forecast(self, w, mask, context, keys, a, b) -> Optional[np.ndarray]:
        """(len(keys), Q, b - a) quantiles, forward or time-reversed (see module docstring)."""
        T = w.values.shape[1]
        tgt, past, fut = self._series(w, mask, context, keys, a, b)
        if np.isfinite(tgt).sum(1).max() >= 48:
            inp = {"target": tgt, "past_covariates": past, "future_covariates": fut}
            return self.pipe.predict([inp], prediction_length=b - a, batch_size=self.bs)[0].cpu().numpy(), "forward"
        # reversed: history = visible window values after the block (reversed), covariates reversed
        n_after = min(self.cl, T - b)
        if n_after < 48:
            return None, "none"
        rows = [w.keys.index(k) for k in keys]
        sl = slice(b, b + n_after)
        tg = np.stack([np.where(mask[i, sl], np.nan, w.values[i, sl]) for i in rows])[:, ::-1]
        if np.isfinite(tg).sum(1).max() < 48:
            return None, "none"
        cov_keys = [k for k in ERA5 if k in w.keys]
        pc = {k: np.where(mask[w.keys.index(k), sl], np.nan, w.values[w.keys.index(k), sl])[::-1].copy() for k in cov_keys}
        pc.update({k: v[::-1].copy() for k, v in _time_cov(w.t[sl]).items()})
        fc = {k: np.where(mask[w.keys.index(k), a:b], np.nan, w.values[w.keys.index(k), a:b])[::-1].copy() for k in cov_keys}
        fc.update({k: v[::-1].copy() for k, v in _time_cov(w.t[a:b]).items()})
        inp = {"target": np.ascontiguousarray(tg), "past_covariates": pc, "future_covariates": fc}
        q = self.pipe.predict([inp], prediction_length=b - a, batch_size=self.bs)[0].cpu().numpy()
        return q[:, :, ::-1], "reversed"

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        out = np.full((self.n,) + w.values.shape, np.nan)
        tower = [i for i, k in enumerate(w.keys) if w.provenance[i] == "tower" and mask[i].any()]
        if not tower:
            return out
        rng = np.random.default_rng(seed_of(self.seed, w.site, str(w.start.date()), int(mask.any(0).sum())))
        # blocks: runs of steps where any tower row is hidden; within a block each row is filled where hidden
        for a, b in runs(mask[tower].any(0)):
            groups = [[i for i in tower if w.keys[i] in FLUX]] + [[i] for i in tower if w.keys[i] not in FLUX]
            for group in groups:
                if not group:
                    continue
                keys = [w.keys[i] for i in group]
                q, how = self._forecast(w, mask, context, keys, a, b)
                self.log.append(dict(site=w.site, window=str(w.start.date()), a=int(a), b=int(b), keys=",".join(keys),
                                     direction=how))
                if q is None:
                    continue
                for r, i in enumerate(group):
                    hid = mask[i, a:b]
                    s = quantile_samples(self.q, q[r], self.n, rng)
                    cols = a + np.flatnonzero(hid)
                    out[:, i, cols] = s[:, hid]
        return out
