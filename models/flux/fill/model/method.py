"""FluxFill as a FluxFill-Bench method (bench/run.py protocol): predict(window, mask, context) -> (S, V, T) samples.

Two sampling modes from one checkpoint:
- `marginal` (v0): independent draws from the per-half-hour Student-t head. Window sums from these samples assume
  independence across half-hours and are labelled marginal.
- `joint` (v1): masked generation with continuous tokens (Li et al. 2024, arXiv:2406.11838). The hidden day-tokens are
  put in a random order (one order per sample); in each of `steps` rounds the encoder reads the window with every
  token generated so far shown as observed, and the flow head generates the next tokens, so later tokens are
  conditioned on earlier ones. The number of tokens still hidden after round k follows the cosine schedule of MaskGIT
  (Chang et al. 2022, CVPR) used by Li et al. Repeating with independent orders and noise gives joint samples over
  the window.

The method reads only what the runner passes: the window with hidden values set to NaN and `measured` False on
hidden values, the mask, and the context. Site location comes from `sites.parquet`; imagery from the Clay store.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (IS_LAND, STEPS_PER_DAY, V_TAB, VARS, WINDOW_DAYS, Stats, doy_features, land_matrix,  # noqa: E402
                  record_from_table, site_features, sites_table)
from memory import memory_from_table, memory_from_table_skew  # noqa: E402
from sampler import mem_arrays  # noqa: E402
from imagery import load_imagery, window_imagery  # noqa: E402
from net import FluxFillNet, mixture_cdf, sample_marginal_params  # noqa: E402

K = STEPS_PER_DAY


class FluxFill:
    def __init__(self, ckpt: str, mode: str = "marginal", n_samples: int = 64, steps: int = 32, flow_steps: int = 24,
                 name: Optional[str] = None, device: str = "cuda", chunk: int = 32, seed: int = 0,
                 imagery: bool = True, imagery_sites=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        c = ck["config"]
        self.stats = Stats.from_dict(c["stats"])
        self.net = FluxFillNet(d=c["d"], depth=c["depth"], heads=c["heads"], use_imagery=not c.get("no_imagery", False),
                               flow_width=c.get("flow_width", 256), flow_depth=c.get("flow_depth", 3),
                               memory=c.get("memory", False), mem_residual=c.get("mem_residual", False),
                               img_nll=c.get("img_nll", False), components=c.get("components", 1),
                               local_base=c.get("local_base", False), gated_base=c.get("gated_base", False),
                               gate_scale=c.get("gate_scale", False), edge_continue=c.get("edge_continue", False),
                               memory_skew=c.get("memory_skew", False), driver_anomaly=c.get("driver_anomaly", False),
                               memory_skew_fluxes_only=c.get("memory_skew_fluxes", False),
                               scale_anchor=c.get("scale_anchor", False), year_anomaly=c.get("year_anomaly", False),
                               sinr_place=c.get("sinr_place", -1), sinr_replace=c.get("sinr_replace", False))
        self.net.load_state_dict(ck["model"])
        self.net.use_imagery = self.net.use_imagery and imagery      # False: imagery modality removed (M4)
        # Imagery only at these sites (None: every site with a Clay store). Defaults to the list the checkpoint was
        # trained with, when its config records one, so a model is scored with the imagery coverage it was built for.
        self.imagery_sites = set(imagery_sites) if imagery_sites is not None else (
            set(c["imagery_sites_used"]) if "imagery_sites_used" in c and c.get("imagery_sites", "all") != "all" else None)
        self.net.to(device).eval()
        self.device, self.mode, self.n, self.steps, self.flow_steps, self.chunk = device, mode, n_samples, steps, flow_steps, chunk
        self.name = name or ("fluxfill-v0" if mode == "marginal" else "fluxfill-v1")
        self.sites = sites_table()
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self._img = {}
        self._memc = {}

    # ------------------------------------------------------------------------------------------------ inputs
    @staticmethod
    def window_rows(w):
        """Model rows of the window's rows that this model has, and their positions in the window. A window can
        carry rows the model lacks (--extra-tower rows with a base-channel model); those are not filled."""
        pos = [i for i, k in enumerate(w.keys) if k in VARS]
        return [VARS.index(w.keys[i]) for i in pos], pos

    def example(self, w, context: Dict) -> Dict[str, np.ndarray]:
        rows, pos = self.window_rows(w)
        off = self.utc_offset(context, w.site)
        vals = np.full((V_TAB, w.values.shape[1]), np.nan)
        vals[rows] = w.values[pos]
        vals[IS_LAND] = land_matrix(w.site, w.t, off)       # the reanalysis-land drivers of the window
        meas = np.isfinite(vals)
        meas[rows] &= w.measured[pos]
        z = np.where(meas, self.stats.norm(np.nan_to_num(vals)), 0.0).astype(np.float32).reshape(V_TAB, WINDOW_DAYS, K)
        present = meas.any(1)
        present[rows] = True
        lat = lon = elev = np.nan
        if w.site in self.sites.index:
            r = self.sites.loc[w.site]
            lat, lon, elev = float(r["lat"]), float(r["lon"]), float(r.get("elevation_m", np.nan))
        if w.site not in self._img:
            ok = self.imagery_sites is None or w.site in self.imagery_sites
            self._img[w.site] = load_imagery(w.site, off) if ok else None
        img = self._img[w.site]
        cls, meta, has = window_imagery(img, w.start, WINDOW_DAYS)
        mem = self.memory(w, context) if self.net.memory else None
        return {**mem_arrays(mem), "z": z, "obs": meas.reshape(V_TAB, WINDOW_DAYS, K), "present": present,
                "img_cls": cls, "img_meta": meta, "img_obs": has, "img_present": np.array(img is not None and has.any()),
                "doy": doy_features(w.start), "site": site_features(lat, lon, elev)}

    def utc_offset(self, context: Dict, site: str = "") -> float:
        """Local standard time minus UTC (hours) from the context table, else from the site table (the R3 regime
        has an empty context), else 0."""
        tab = context.get("table") if context else None
        if tab is not None and "t_lst" in tab and "t_utc" in tab and len(tab):
            return float(np.median((tab["t_lst"].to_numpy() - tab["t_utc"].to_numpy())[:1000]) / 60.0)
        if tab is not None and "time_utc" in tab and len(tab):
            u = pd.Timestamp(tab["time_utc"].iloc[0])
            return float((pd.Timestamp(tab["time_lst"].iloc[0]) - (u.tz_convert(None) if u.tzinfo else u)) / pd.Timedelta(hours=1))
        if site in self.sites.index and pd.notna(self.sites.loc[site].get("utc_offset_h", np.nan)):
            return float(self.sites.loc[site, "utc_offset_h"])
        return 0.0

    def memory(self, w, context: Dict):
        """Site memory from the runner's context table (the site's other years only)."""
        tab = context.get("table") if context else None
        if tab is None or len(tab) == 0:
            return None
        key = (w.site, str(w.start.date()), len(tab))
        if key not in self._memc:
            rec = record_from_table(tab, w.site)
            f = memory_from_table_skew if self.net.memory_skew else memory_from_table
            self._memc = {key: f(rec.values, rec.t0, self.stats, w.start)}
        return self._memc[key]

    def _batch(self, ex, n):
        return {k: torch.as_tensor(np.array(v), device=self.device)[None].expand(n, *v.shape).clone()
                for k, v in ex.items()}

    # ------------------------------------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        ex = self.example(w, context)
        rows, pos = self.window_rows(w)
        hide = np.zeros((V_TAB, mask.shape[1]), bool)
        hide[rows] = mask[pos]
        hide &= ex["present"][:, None]
        z = self.sample_joint(ex, hide) if self.mode == "joint" else self.sample_marginal(ex)   # (S, V_TAB, T) std.
        self._last = (np.array(rows), np.array(pos), self._params_last) if self.mode == "marginal" else None
        x = self.stats.denorm(z.reshape(self.n * V_TAB, -1).astype(np.float64),
                              np.tile(np.arange(V_TAB), self.n)).reshape(self.n, V_TAB, -1)
        out = np.full((self.n,) + w.values.shape, np.nan)
        out[:, pos] = x[:, rows]
        return np.where(mask[None], out, w.values[None])

    def pit(self, values: np.ndarray) -> np.ndarray:
        """Exact PIT F(y) of `values` (the window's rows, raw units) under the predictive distribution of the last
        `predict` call (marginal mode only; NaN in joint mode, whose flow samples have no closed-form CDF). The runner
        calls it after `predict` has returned, with the truth, to score coverage exactly."""
        if self.mode != "marginal" or getattr(self, "_last", None) is None:
            return np.full(values.shape, np.nan)
        rows, pos, params = self._last
        out = np.full(values.shape, np.nan)
        z = (values[pos] - self.stats.mean[rows, None]) / self.stats.std[rows, None]
        out[pos] = mixture_cdf([u[rows] for u in params], z)
        return out

    def sample_marginal(self, ex) -> np.ndarray:
        b = self._batch(ex, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, params = self.net(b)
        self._params_last = tuple(u[0].float().cpu().numpy().reshape(V_TAB, -1, *u.shape[4:]) for u in params)
        s = sample_marginal_params(tuple(u[0] for u in params), self.n)
        return s.reshape(self.n, V_TAB, -1).float().cpu().numpy()

    def sample_joint(self, ex, hide: np.ndarray) -> np.ndarray:
        out = []
        for a in range(0, self.n, self.chunk):
            out.append(self._generate(ex, hide, min(self.chunk, self.n - a)))
        return np.concatenate(out, 0)

    def _generate(self, ex, hide, S) -> np.ndarray:
        b = self._batch(ex, S)
        hid = torch.from_numpy(hide.reshape(V_TAB, WINDOW_DAYS, K)).to(self.device)
        tok = hid.any(-1).nonzero()                                    # (N, 2): (v, d) of hidden tokens
        N = tok.shape[0]
        z, obs = b["z"], b["obs"]
        if N == 0:
            return z.reshape(S, V_TAB, -1).float().cpu().numpy()
        order = torch.argsort(torch.rand(S, N, device=self.device, generator=self.gen), 1)   # per-sample order
        steps = min(self.steps, N)
        left = [int(math.floor(N * math.cos(math.pi / 2 * (k + 1) / steps))) for k in range(steps)]
        done = 0
        si = torch.arange(S, device=self.device)[:, None]
        for k in range(steps):
            n_new = (N - left[k]) - done
            if n_new <= 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.net.encode(b)
            sel = tok[order[:, done:done + n_new]]                     # (S, n_new, 2)
            vi, di = sel[..., 0], sel[..., 1]
            cond = h[si, vi, di].reshape(-1, h.shape[-1]).float()
            zv = z[si, vi, di].reshape(-1, K)
            ov = obs[si, vi, di].reshape(-1, K).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = self.net.head_flow.sample(cond, zv, ov, steps=self.flow_steps, generator=self.gen)
            hv = hid[vi, di].reshape(-1, K)
            x = x.float() + self.net.offset(b)[si, vi, di].reshape(-1, K)      # back from the memory residual
            newz = torch.where(hv, x, zv).reshape(S, n_new, K)
            z[si, vi, di] = newz
            obs[si, vi, di] = obs[si, vi, di] | hv.reshape(S, n_new, K)
            done += n_new
        return z.reshape(S, V_TAB, -1).float().cpu().numpy()
