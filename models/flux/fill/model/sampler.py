"""Training and validation batches: 180-day windows as day-tokens with a training mask.

One example holds, for the 22 tabular rows (V) and 180 days (D) of 48 half-hours:
- `z` standardised visible values (0 where hidden or not measured), `obs` the visible-and-measured flags;
- `tgt`, `tmask`: the standardised truth and where it is a target (measured and hidden); loss uses only these;
- `present` (V,): the variable is measured somewhere in the window (the benchmark's window keys);
- imagery: per-day Clay CLS, SCL meta, `img_obs` (visible acquisition), `img_tgt` (hidden acquisition, a target),
  `img_present` (the site has imagery and the modality is not dropped);
- `doy` (D, 4) and `site` (4,) features.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

import masking
from data import (STEPS_PER_DAY, V_TAB, WINDOW_DAYS, SiteRecord, Stats, doy_features, site_features,
                  valid_starts, window_tokens, window_values)
from imagery import N_META, SiteImagery, window_imagery


def make_example(rec: SiteRecord, start_day: int, stats: Stats, hide: np.ndarray, img_hide: np.ndarray,
                 img: Optional[SiteImagery], mem=None) -> Dict[str, np.ndarray]:
    """`mem`: (mean, sd, has) site-memory channels (V, D, 48) from memory.SiteMemory, or None (no memory)."""
    vals = window_values(rec, start_day)
    z, meas = window_tokens(vals, stats)
    h = hide.reshape(V_TAB, WINDOW_DAYS, STEPS_PER_DAY)
    obs = meas & ~h
    start = rec.t0 + pd.Timedelta(days=start_day)
    cls, meta, has = window_imagery(img, start, WINDOW_DAYS)
    img_obs = has & ~img_hide
    img_drop = bool(img_hide.all())
    return {
        "z": np.where(obs, z, 0.0).astype(np.float32), "obs": obs,
        "tgt": z, "tmask": meas & h, "present": meas.any((1, 2)),
        "img_cls": np.where(img_obs[:, None], cls, 0.0).astype(np.float32),
        "img_meta": np.where(img_obs[:, None], meta, 0.0).astype(np.float32),
        "img_obs": img_obs, "img_tgt_cls": cls, "img_tgt": has & img_hide,
        "img_present": np.array(img is not None and has.any() and not img_drop),
        "doy": doy_features(start), "site": site_features(rec.lat, rec.lon, rec.elev),
        **mem_arrays(mem),
    }


def mem_arrays(mem) -> Dict[str, np.ndarray]:
    if mem is None:
        z = np.zeros((V_TAB, WINDOW_DAYS, STEPS_PER_DAY), np.float32)
        return {"mem_m": z, "mem_s": z, "mem_k": z, "mem_h": np.zeros(z.shape, bool)}
    if len(mem) == 4:                 # with the skewness channel (E23)
        m, s_, k, h = mem
    else:
        (m, s_, h), k = mem, np.zeros(np.shape(mem[0]), np.float32)
    return {"mem_m": np.asarray(m, np.float32), "mem_s": np.asarray(s_, np.float32), "mem_k": np.asarray(k, np.float32),
            "mem_h": np.asarray(h, bool)}


class WindowSampler:
    """Draws (site, window, mask) triples. Sites are drawn by weight (uniform until weights.parquet gives site
    weights), then a window start uniformly among the site's usable starts."""

    def __init__(self, records: Dict[str, SiteRecord], sites, stats: Stats, imagery: Dict[str, SiteImagery],
                 train: bool, cfg: masking.MaskConfig = masking.MaskConfig(), site_weights: Optional[Dict] = None,
                 memory: bool = False, p_drop_mem: float = 0.2, memory_skew: bool = False,
                 fast_memory: bool = False):
        self.records, self.stats, self.imagery, self.cfg = records, stats, imagery, cfg
        self.fast_memory = fast_memory          # memory.window_memory_direct: no per-(site, years) table cache
        self.memory, self.p_drop_mem, self.memory_skew = memory, p_drop_mem, memory_skew
        self._mem, self._cache = {}, {}
        self.starts = {s: valid_starts(records[s], exclude_heldout=train) for s in sites if s in records}
        self.sites = [s for s, v in self.starts.items() if v.size]
        w = np.array([(site_weights or {}).get(s, 1.0) for s in self.sites], np.float64)
        self.p = w / w.sum()

    def draw(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        while True:
            s = self.sites[int(rng.choice(len(self.sites), p=self.p))]
            d = int(rng.choice(self.starts[s]))
            rec = self.records[s]
            present = np.isfinite(window_values(rec, d)).any(1)
            hide, img_hide, fam = masking.draw(rng, present, self.cfg)
            mem = self.window_memory(s, d, drop=rng.random() < self.p_drop_mem) if self.memory else None
            ex = make_example(rec, d, self.stats, hide, img_hide, self.imagery.get(s), mem)
            if ex["tmask"].any():
                ex["family"] = fam
                return ex


    def window_memory(self, s: str, d: int, drop: bool = False):
        """Site memory of window (s, d) without its own calendar years (cached per site and year span)."""
        if drop:
            return None
        if self.memory_skew:
            return self._window_memory_skew(s, d)
        from memory import SiteMemory, memory_from_sums, window_memory, year_sums
        rec = self.records[s]
        if s not in self._mem:
            self._mem[s] = SiteMemory(rec, [rec.heldout_year] if rec.heldout_year is not None else [])
        sm = self._mem[s]
        start = rec.t0 + pd.Timedelta(days=d)
        if self.fast_memory:
            from memory import window_memory_direct
            return window_memory_direct(sm, start, self.stats)
        years = tuple(range(start.year, (start + pd.Timedelta(days=WINDOW_DAYS - 1)).year + 1))
        key = (s, years)
        if key not in self._cache:
            S1, S2, N = (x.astype(np.float64) for x in (sm.S1, sm.S2, sm.N))
            for y in years:
                if y not in sm.never:
                    a, b, c = year_sums(rec, y)
                    S1 -= a; S2 -= b; N -= c
            mz, sz, has = memory_from_sums(S1, S2, np.maximum(N, 0), self.stats)
            if len(self._cache) >= 400:                  # bound RAM (about 1 GB): drop the oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (mz.astype(np.float16), sz.astype(np.float16), has)
        mz, sz, has = self._cache[key]
        m, sd, h = window_memory(mz, sz, has, start)
        return m.astype(np.float32), sd.astype(np.float32), h


    def _window_memory_skew(self, s: str, d: int):
        from memory import SiteMemorySkew, window_memory_skew
        rec = self.records[s]
        key = ("skew", s)
        if key not in self._mem:
            self._mem[key] = SiteMemorySkew(rec, [rec.heldout_year] if rec.heldout_year is not None else [])
        sm = self._mem[key]
        start = rec.t0 + pd.Timedelta(days=d)
        years = tuple(range(start.year, (start + pd.Timedelta(days=WINDOW_DAYS - 1)).year + 1))
        ck = ("skew", s, years)
        if ck not in self._cache:
            if len(self._cache) >= 400:
                self._cache.pop(next(iter(self._cache)))
            self._cache[ck] = tuple(x.astype(np.float16) if x.dtype != bool else x for x in sm.tables(years, self.stats))
        mz, sz, kz, has = self._cache[ck]
        return tuple(x.astype(np.float32) if x.dtype != bool else x for x in window_memory_skew(mz, sz, kz, has, start))


def collate(examples, device=None) -> Dict[str, torch.Tensor]:
    out = {}
    for k in examples[0]:
        if k == "family":
            out[k] = [e[k] for e in examples]
            continue
        x = torch.from_numpy(np.stack([e[k] for e in examples]))
        out[k] = x.to(device, non_blocking=True) if device is not None else x
    return out


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, sampler: WindowSampler, batch: int, seed: int):
        self.sampler, self.batch, self.seed = sampler, batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield collate([self.sampler.draw(rng) for _ in range(self.batch)])
