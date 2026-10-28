"""SAITS and CSDI (PyPOTS) trained across training sites, as FluxFill-Bench methods.

Du, W., Cote, D., Liu, Y. 2023. SAITS: Self-attention-based imputation for time series. Expert Systems with
Applications 219:119619. https://doi.org/10.1016/j.eswa.2023.119619
Tashiro, Y., Song, J., Song, Y., Ermon, S. 2021. CSDI: Conditional score-based diffusion models for probabilistic
time series imputation. NeurIPS 34. https://arxiv.org/abs/2107.03502
Du, W. 2023. PyPOTS: a Python toolbox for data mining on partially-observed time series. arXiv:2305.18811.
Implementation: the PyPOTS 1.5 model cores `pypots.imputation.saits.core._SAITS` and
`pypots.imputation.csdi.core._CSDI` (network, losses and the CSDI sampler unchanged), driven by the training loop
below, so that the masks seen in training are the benchmark's kind of gaps.

Data. Every variable of the benchmark windows (15 tower variables, 7 ERA5 drivers; `model/data.VARS`), standardised
with the training sites' mean and standard deviation (`model/data.fit_stats`, held-out years excluded). Training uses
only training sites (`model/data.load_split`: the frozen towers/split.csv once it exists, else the provisional 20 %
hold-out by SHA-256 of the site id). Early stopping and the SAITS spread use the calibration sites of
`val_windows.py` (validation sites that are not scored). Scoring sites are never seen in training or calibration.

Representations (memory sets them; GPU budget 2 GB):
- SAITS: the 180-day window, daily-tokenised. Each day is one time step whose features are the 48 half-hours of each
  of the 22 variables (1,056 values) plus the day of year as sine and cosine; attention runs over the 180 days. The
  output is reshaped back to half-hours, so nothing is averaged.
- CSDI: half-hourly resolution in chunks of 6 days (288 steps), 22 variables plus hour-of-day and day-of-year as
  sine and cosine (always observed). The window is the 30 chunks that tile it; each chunk with a hidden value is
  sampled on its own, conditioned on that chunk's visible values. CSDI's attention over 8,640 half-hours of 26
  features does not fit the budget.

Training masks (masked imputation task of SAITS; the conditional/target split of CSDI): per example one of
(a) all tower rows hidden over a contiguous block of log-uniform length from 1 h to the example's length (p 0.5);
(b) one tower variable hidden over such a block (p 0.2); (c) 20 % of the measured values hidden at random, the PyPOTS
default (p 0.3). Reanalysis rows and time features are never hidden. The released PyPOTS training uses (c) only;
(a) and (b) are added because the benchmark's gaps are blocks.

Uncertainty. CSDI: 64 diffusion samples per chunk; joint within a chunk (6 days), independent between chunks.
Sampling runs the network under float16 autocast on the GPU (matrix products in float16, the sampler's arithmetic
in float32), which halves its cost; training is in float32.
SAITS gives points; its spread here is a residual-based Gaussian: root mean square of SAITS residuals per variable
and gap-length class on the calibration sites' windows under the benchmark's M2 masks, samples independent per
half-hour. This spread is ours, not part of SAITS.

Usage:
  python pypots_imputers.py train saits --out DIR [--steps 6000]
  python pypots_imputers.py train csdi  --out DIR [--steps 8000]
  python pypots_imputers.py spread --ckpt DIR/saits.pt      (SAITS residual spread on calibration sites)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, str(HERE))

from common import seed_of  # noqa: E402

STEPS_PER_DAY, WINDOW_DAYS = 48, 180
T_WIN = STEPS_PER_DAY * WINDOW_DAYS
CHUNK = 6 * STEPS_PER_DAY
VAL_JSON = Path("/home/photon/4tb/fluxfill/bench/val/windows_val.json")

SAITS_CFG = dict(n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=512, dropout=0.1, attn_dropout=0.0,
                 lr=1e-3, batch=16)
CSDI_CFG = dict(n_layers=4, n_heads=8, n_channels=64, d_time_embedding=128, d_feature_embedding=16,
                d_diffusion_embedding=128, n_diffusion_steps=50, schedule="quad", beta_start=1e-4, beta_end=0.5,
                lr=1e-3, batch=4)
"""CSDI values are the CSDI paper's and PyPOTS's defaults (Tashiro et al. 2021, appendix; 4 residual layers, 64
channels, 8 heads, 50 steps, quadratic schedule 1e-4 to 0.5). SAITS: the PyPOTS example scale (2 layers,
d_model 256, 4 heads)."""


def vars_():
    from data import VARS, IS_TOWER
    return list(VARS), np.asarray(IS_TOWER)


def gpu_cap(device: str) -> None:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        gb = float(os.environ.get("FLUXFILL_GPU_GB", "1.8"))
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total), 0)


# ------------------------------------------------------------------------------------------------- representations

def doy2(t0: pd.Timestamp, n_days: int) -> np.ndarray:
    d = pd.date_range(t0 + pd.Timedelta(hours=12), periods=n_days, freq="D")
    a = 2 * np.pi * (d.dayofyear.to_numpy() - 1) / 365.0
    return np.stack([np.sin(a), np.cos(a)], 1).astype(np.float32)


def time4(t: pd.DatetimeIndex) -> np.ndarray:
    h = 2 * np.pi * (t.hour * 2 + t.minute // 30).to_numpy() / 48.0
    a = 2 * np.pi * (t.dayofyear.to_numpy() - 1) / 365.0
    return np.stack([np.sin(h), np.cos(h), np.sin(a), np.cos(a)], 0).astype(np.float32)


def saits_tokens(z: np.ndarray, obs: np.ndarray, t0: pd.Timestamp):
    """(V, T) standardised values and observed flags -> X (D, V*48 + 2), mask (D, V*48 + 2)."""
    V, T = z.shape
    D = T // STEPS_PER_DAY
    x = np.where(obs, z, 0.0).reshape(V, D, STEPS_PER_DAY).transpose(1, 0, 2).reshape(D, V * STEPS_PER_DAY)
    m = obs.reshape(V, D, STEPS_PER_DAY).transpose(1, 0, 2).reshape(D, V * STEPS_PER_DAY).astype(np.float32)
    return (np.concatenate([x, doy2(t0, D)], 1).astype(np.float32),
            np.concatenate([m, np.ones((D, 2), np.float32)], 1))


def saits_untoken(x: np.ndarray, V: int) -> np.ndarray:
    D = x.shape[0]
    return x[:, : V * STEPS_PER_DAY].reshape(D, V, STEPS_PER_DAY).transpose(1, 0, 2).reshape(V, D * STEPS_PER_DAY)


def train_mask(obs: np.ndarray, tower: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Artificial hidden set (subset of `obs`) of one training example; see the module docstring."""
    V, T = obs.shape
    A = np.zeros_like(obs)
    u = rng.uniform()
    if u < 0.7:
        n = int(np.clip(np.exp(rng.uniform(np.log(2), np.log(T))), 2, T))
        a = int(rng.integers(0, T - n + 1))
        rows = np.flatnonzero(tower)
        if u >= 0.5:
            have = [r for r in rows if obs[r].any()]
            rows = np.array([rng.choice(have)]) if have else rows
        A[rows, a:a + n] = True
    else:
        A = rng.uniform(size=obs.shape) < 0.2
    return A & obs


# ------------------------------------------------------------------------------------------------- data

class Data:
    """Training-site records, statistics, and samplers of 180-day windows (SAITS) and 6-day chunks (CSDI)."""

    def __init__(self, verbose: bool = True):
        from data import heldout_year, fit_stats, load_records, load_split, table_dir, valid_starts
        sites = sorted(p.stem for p in table_dir().glob("*.parquet"))
        self.split, self.split_name = load_split(sites)
        calib = json.loads(VAL_JSON.read_text())["calibration_sites"] if VAL_JSON.exists() else []
        self.train = [s for s, v in self.split.items() if v == "train"]
        self.calib = [s for s in calib if self.split.get(s) == "val"]
        self.records = load_records(self.train + self.calib, verbose=verbose)
        for s in self.train:
            if s in self.records:
                self.records[s].heldout_year = heldout_year(self.records[s])
        self.train = [s for s in self.train if s in self.records]
        self.stats = fit_stats(self.records, self.train)
        self.starts = {s: valid_starts(self.records[s], exclude_heldout=s in self.train) for s in self.records}
        self.train = [s for s in self.train if self.starts[s].size]
        self.calib = [s for s in self.calib if s in self.records and self.starts[s].size]
        w = np.array([self.starts[s].size for s in self.train], np.float64)
        self.p = w / w.sum()
        self.V, self.tower = len(vars_()[0]), vars_()[1]

    def z(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.stats.mean[:, None]) / self.stats.std[:, None]).astype(np.float32)

    def window(self, site: str, day: int):
        r = self.records[site]
        a = day * STEPS_PER_DAY
        x = r.values[:, a:a + T_WIN].astype(np.float64)
        return x, r.t0 + pd.Timedelta(days=int(day))

    def draw(self, rng, sites=None, p=None):
        sites = self.train if sites is None else sites
        s = sites[int(rng.choice(len(sites), p=self.p if p is None and sites is self.train else p))]
        return s, int(rng.choice(self.starts[s]))

    def saits_batch(self, rng, B: int, sites=None):
        X, M, XO, IM = [], [], [], []
        for _ in range(B):
            s, d = self.draw(rng, sites)
            x, t0 = self.window(s, d)
            obs = np.isfinite(x)
            A = train_mask(obs, self.tower, rng)
            z = np.nan_to_num(self.z(x))
            xi, mi = saits_tokens(z, obs & ~A, t0)
            xo, mo = saits_tokens(z, obs, t0)
            X.append(xi); M.append(mi); XO.append(xo); IM.append(mo - mi)
        return [np.stack(a) for a in (X, M, XO, IM)]

    def csdi_batch(self, rng, B: int, sites=None):
        X, C, I = [], [], []
        while len(X) < B:
            s, d = self.draw(rng, sites)
            x, t0 = self.window(s, d)
            j = int(rng.integers(0, WINDOW_DAYS // 6))
            x = x[:, j * CHUNK:(j + 1) * CHUNK]
            obs = np.isfinite(x)
            if not obs[self.tower].any():
                continue
            A = train_mask(obs, self.tower, rng)
            t = pd.date_range(t0 + pd.Timedelta(days=6 * j), periods=CHUNK, freq="30min")
            z = np.concatenate([np.nan_to_num(self.z(x)), time4(t)], 0)
            o = np.concatenate([obs, np.ones((4, CHUNK), bool)], 0)
            A = np.concatenate([A, np.zeros((4, CHUNK), bool)], 0)
            X.append(z); C.append((o & ~A).astype(np.float32)); I.append(A.astype(np.float32))
        return [np.stack(a) for a in (X, C, I)]


# ------------------------------------------------------------------------------------------------- models

def build(kind: str, V: int):
    from pypots.nn.modules.loss import MAE
    if kind == "saits":
        from pypots.imputation.saits.core import _SAITS
        c = SAITS_CFG
        return _SAITS(c["n_layers"], WINDOW_DAYS, V * STEPS_PER_DAY + 2, c["d_model"], c["n_heads"], c["d_k"], c["d_v"],
                      c["d_ffn"], c["dropout"], c["attn_dropout"], True, 1, 1, MAE(), MAE())
    from pypots.imputation.csdi.core import _CSDI
    c = CSDI_CFG
    return _CSDI(V + 4, c["n_layers"], c["n_heads"], c["n_channels"], c["d_time_embedding"], c["d_feature_embedding"],
                 c["d_diffusion_embedding"], False, c["n_diffusion_steps"], c["schedule"], c["beta_start"], c["beta_end"])


def _inputs(kind, arrs, device):
    import torch
    t = [torch.from_numpy(a).to(device) for a in arrs]
    if kind == "saits":
        return {"X": t[0], "missing_mask": t[1], "X_ori": t[2], "indicating_mask": t[3]}
    B, K, L = t[0].shape
    return {"X_ori": t[0], "cond_mask": t[1], "indicating_mask": t[2],
            "observed_tp": torch.arange(L, device=device, dtype=torch.float32)[None].expand(B, -1)}


def train(kind: str, out: Path, steps: int, device: str, seed: int = 0, val_every: int = 250, n_val: int = 64):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    gpu_cap(device)
    data = Data()
    V = data.V
    net = build(kind, V).to(device)
    cfg = SAITS_CFG if kind == "saits" else CSDI_CFG
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    vrng = np.random.default_rng(seed + 1)
    batcher = data.saits_batch if kind == "saits" else data.csdi_batch
    val = [batcher(vrng, cfg["batch"], data.calib) for _ in range(max(1, n_val // cfg["batch"]))]
    out.mkdir(parents=True, exist_ok=True)
    best, hist, t0 = np.inf, [], time.time()
    for step in range(1, steps + 1):
        net.train()
        r = net(_inputs(kind, batcher(rng, cfg["batch"]), device), calc_criterion=True)
        opt.zero_grad(); r["loss"].backward(); opt.step()
        if step % val_every == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                vl = []
                for k, vb in enumerate(val):
                    torch.manual_seed(1000 + k)                 # fixed diffusion times and noise for CSDI
                    inp = _inputs(kind, vb, device)
                    if kind == "saits":
                        vl.append(float(net(inp, calc_criterion=True)["metric"]))
                    else:                                       # eval mode, five fixed diffusion times
                        side = net.get_side_info(inp["observed_tp"], inp["cond_mask"])
                        for tt in (2, 12, 24, 36, 48):
                            vl.append(float(net.backbone.calc_loss(inp["X_ori"], inp["cond_mask"],
                                                                   inp["indicating_mask"], side, set_t=tt)))
            v = float(np.mean(vl))
            hist.append(dict(step=step, train=float(r["loss"]), val=v, s=round(time.time() - t0, 1),
                             gpu_mb=round(torch.cuda.max_memory_allocated() / 1e6) if device.startswith("cuda") else 0))
            print(json.dumps(hist[-1]), flush=True)
            if v < best:
                best = v
                torch.save({"kind": kind, "state": net.state_dict(), "cfg": cfg, "V": V, "step": step,
                            "stats": {"mean": data.stats.mean.tolist(), "std": data.stats.std.tolist()},
                            "split": data.split_name, "train_sites": data.train, "calib_sites": data.calib},
                           out / f"{kind}.pt")
    (out / f"{kind}_history.json").write_text(json.dumps(hist, indent=1))


# ------------------------------------------------------------------------------------------------- methods

class _Loaded:
    def __init__(self, ckpt: str, device: str):
        import torch
        gpu_cap(device)
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ck, self.V, self.device = c, c["V"], device
        self.net = build(c["kind"], c["V"]).to(device)
        self.net.load_state_dict(c["state"]); self.net.eval()
        self.mean, self.std = np.array(c["stats"]["mean"]), np.array(c["stats"]["std"])
        self.vars = vars_()[0]

    def field(self, w, mask: np.ndarray):
        """(V_TAB, T) standardised visible values and observed flags of a benchmark window, on VARS rows."""
        T = w.values.shape[1]
        z = np.zeros((self.V, T), np.float32); obs = np.zeros((self.V, T), bool)
        for i, k in enumerate(w.keys):
            j = self.vars.index(k)
            x = np.where(mask[i], np.nan, w.values[i])
            o = np.isfinite(x)
            z[j] = np.where(o, (x - self.mean[j]) / self.std[j], 0.0); obs[j] = o
        return z, obs

    def rows(self, w, mask):
        """(window row, VARS row) of the tower rows with hidden values."""
        return [(i, self.vars.index(k)) for i, k in enumerate(w.keys) if w.provenance[i] == "tower" and mask[i].any()]


class SAITSMethod:
    """SAITS points with a residual-based Gaussian spread (labelled 'saits+gauss' in results)."""
    name = "saits"

    def __init__(self, ckpt: str = "/home/photon/4tb/fluxfill/baselines/pypots/saits.pt", device: str = "cuda",
                 n_samples: int = 64, spread: Optional[str] = None, seed: int = 0):
        import torch
        self.m = _Loaded(ckpt, device if torch.cuda.is_available() else "cpu")
        self.n, self.seed = n_samples, seed
        sp = Path(spread) if spread else Path(ckpt).with_name("saits_spread.json")
        self.spread = json.loads(sp.read_text()) if sp.exists() else None

    def point(self, w, mask) -> np.ndarray:
        import torch
        z, obs = self.m.field(w, mask)
        x, mm = saits_tokens(z, obs, w.start)
        with torch.no_grad():
            r = self.m.net({"X": torch.from_numpy(x[None]).to(self.m.device),
                            "missing_mask": torch.from_numpy(mm[None]).to(self.m.device)}, calc_criterion=False)
        zz = saits_untoken(r["imputation"][0].cpu().numpy(), self.m.V)
        return zz * self.m.std[:, None] + self.m.mean[:, None]

    def sigma(self, var: str, cls: np.ndarray) -> np.ndarray:
        from metrics import GAP_CLASSES
        s = self.spread[var] if self.spread and var in self.spread else None
        if s is None:
            return np.full(cls.shape, np.nan)
        return np.array([s["by_class"][c] if c >= 0 and s["by_class"][c] is not None else s["all"] for c in cls])

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        from metrics import gap_class, gap_runs
        out = np.full((self.n,) + w.values.shape, np.nan)
        rows = self.m.rows(w, mask)
        if not rows:
            return out
        p = self.point(w, mask)
        cls = gap_class(gap_runs(mask))
        rng = np.random.default_rng(seed_of(self.seed, w.site, str(w.start.date()), int(mask.any(0).sum())))
        for i, j in rows:
            hid = mask[i]
            sd = self.sigma(w.keys[i], cls[i, hid])
            out[:, i, hid] = p[j, hid][None] + sd[None] * rng.standard_normal((self.n, int(hid.sum())))
        return out


class CSDIMethod:
    name = "csdi"

    def __init__(self, ckpt: str = "/home/photon/4tb/fluxfill/baselines/pypots/csdi.pt", device: str = "cuda",
                 n_samples: int = 64, batch: int = 32, fp16: bool = True, seed: int = 0):
        import torch
        self.m = _Loaded(ckpt, device if torch.cuda.is_available() else "cpu")
        self.n, self.batch, self.seed = n_samples, batch, seed
        self.fp16 = fp16 and self.m.device.startswith("cuda")

    def predict(self, w, mask: np.ndarray, context: Dict) -> np.ndarray:
        import torch
        out = np.full((self.n,) + w.values.shape, np.nan)
        rows = self.m.rows(w, mask)
        if not rows:
            return out
        z, obs = self.m.field(w, mask)
        T = z.shape[1]
        tf = time4(w.t)
        hidden_cols = mask.any(0)
        chunks = [c for c in range(T // CHUNK) if hidden_cols[c * CHUNK:(c + 1) * CHUNK].any()]
        torch.manual_seed(seed_of(self.seed, w.site, str(w.start.date()), int(mask.any(0).sum())) % (2**31))
        net = self.m.net
        for c in chunks:
            sl = slice(c * CHUNK, (c + 1) * CHUNK)
            x = np.concatenate([z[:, sl], tf[:, sl]], 0)
            cm = np.concatenate([obs[:, sl], np.ones((4, CHUNK), bool)], 0).astype(np.float32)
            res = []
            for a in range(0, self.n, self.batch):
                b = min(self.batch, self.n - a)
                X = torch.from_numpy(x[None]).to(self.m.device).expand(b, -1, -1).contiguous()
                C = torch.from_numpy(cm[None]).to(self.m.device).expand(b, -1, -1).contiguous()
                tp = torch.arange(CHUNK, device=self.m.device, dtype=torch.float32)[None].expand(b, -1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    side = net.get_side_info(tp, C)
                    s = net.backbone(X, C, side, 1)[:, 0]               # (b, K, L)
                    s = X + s * (1 - C)
                res.append(s.cpu().numpy())
            s = np.concatenate(res, 0)[:, : self.m.V]                    # (n, V_TAB, L)
            s = s * self.m.std[None, :, None] + self.m.mean[None, :, None]
            for i, j in rows:
                hid = mask[i, sl]
                if hid.any():
                    out[:, i, c * CHUNK + np.flatnonzero(hid)] = s[:, j, hid]
        return out


# ------------------------------------------------------------------------------------------------- SAITS spread

def fit_spread(ckpt: str, device: str, per_site: int = 2) -> Dict:
    """Residual standard deviation of SAITS per variable and gap class on the calibration sites' M2 masks."""
    from metrics import GAP_CLASSES, gap_class, gap_runs
    from run import masks_for
    from windows import Window, load_window
    from evaluate import val_windows  # noqa: F401  (same window rule as the scored sites)
    from data import load_records, table_dir, valid_starts
    meth = SAITSMethod(ckpt, device)
    calib = json.loads(VAL_JSON.read_text())["calibration_sites"]
    recs = load_records(calib, verbose=False)
    res: Dict[str, List[List[float]]] = {}
    n_win = 0
    for s in calib:
        if s not in recs:
            continue
        st = valid_starts(recs[s], exclude_heldout=False)
        if st.size == 0:
            continue
        tab = pd.read_parquet(table_dir() / f"{s}.parquet")
        for j in np.unique(np.linspace(0, st.size - 1, per_site + 2).astype(int)[1:-1]):
            start = str((recs[s].t0 + pd.Timedelta(days=int(st[j]))).date())
            w = load_window(tab, s, start)
            n_win += 1
            for name, mask in masks_for(w, ["M2"]).items():
                vis = Window(w.site, w.start, w.t, w.keys, np.where(mask, np.nan, w.values), w.measured & ~mask,
                             w.provenance, w.meta)
                p = meth.point(vis, mask)
                cls = gap_class(gap_runs(mask))
                for i, k in enumerate(w.keys):
                    if w.provenance[i] != "tower":
                        continue
                    tgt = mask[i] & w.measured[i]
                    if not tgt.any():
                        continue
                    r = p[meth.m.vars.index(k), tgt] - w.values[i, tgt]
                    d = res.setdefault(k, [[] for _ in GAP_CLASSES])
                    for c in range(len(GAP_CLASSES)):
                        d[c].extend(r[cls[i, tgt] == c].tolist())
    out = {"_meta": {"calibration_sites": calib, "n_windows": n_win, "rule": "std of residuals per class; class "
                     "with < 100 residuals falls back to the variable's pooled std"}}
    for k, d in res.items():
        allr = np.concatenate([np.asarray(x) for x in d])
        out[k] = {"all": float(np.sqrt(np.mean(allr ** 2))) if allr.size else None,
                  "by_class": [float(np.sqrt(np.mean(np.square(x)))) if len(x) >= 100 else None for x in d],
                  "n": [len(x) for x in d]}
    Path(ckpt).with_name("saits_spread.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "spread"])
    ap.add_argument("kind", nargs="?", default="saits")
    ap.add_argument("--out", default="/home/photon/4tb/fluxfill/baselines/pypots")
    ap.add_argument("--ckpt", default="/home/photon/4tb/fluxfill/baselines/pypots/saits.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.kind, Path(a.out), a.steps, a.device)
    else:
        out = fit_spread(a.ckpt, a.device)
        print(json.dumps({k: v for k, v in out.items() if k != "_meta"})[:2000])


if __name__ == "__main__":
    main()
