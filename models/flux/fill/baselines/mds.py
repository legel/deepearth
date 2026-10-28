"""MDS (Reichstein et al. 2005; REddyProc, Wutzler et al. 2018) as a FluxFill-Bench method.

It fills each tower flux of the visible 180-day window from that window alone, as MDS runs on a site-year. Drivers
are Rg = SW_IN, Tair = TA and VPD. Where the tower driver is hidden or missing, the ONEFlux-downscaled ERA5 value
stands in: this is how FLUXNET products fill drivers (Vuichard and Papale 2015). The predictive distribution is
Gaussian with MDS's own spread (`_fsd`), independent per half-hour: MDS reports no joint uncertainty.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RSCRIPT = "/home/photon/4tb/envs/fluxfill/bin/Rscript"
FILL = ("NEE", "LE", "H", "FCH4")


class MDS:
    name = "mds"

    def __init__(self, n_samples: int = 64, lat: float = 0.0, lon: float = 0.0, utc_offset: float = 0.0):
        self.n, self.lat, self.lon, self.tz = n_samples, lat, lon, utc_offset

    def driver(self, w, key: str) -> np.ndarray:
        k = w.keys.index(key) if key in w.keys else None
        e = w.keys.index("ERA5_" + key) if "ERA5_" + key in w.keys else None
        x = w.values[k].copy() if k is not None else np.full(w.values.shape[1], np.nan)
        if e is not None:
            x = np.where(np.isfinite(x), x, w.values[e])
        return x

    def predict(self, w, mask: np.ndarray, context) -> np.ndarray:
        V, T = w.values.shape
        out = np.full((self.n, V, T), np.nan)
        rows = [i for i, k in enumerate(w.keys) if k in FILL and mask[i].any()]
        if not rows:
            return out
        # REddyProc stamps the end of each half-hour; the window grid stamps the start.
        df = pd.DataFrame({"DateTime": (w.t + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),
                           "Rg": np.clip(self.driver(w, "SW_IN"), 0, None), "Tair": self.driver(w, "TA"),
                           "VPD": self.driver(w, "VPD")})
        rng = np.random.default_rng(0)
        for i in rows:
            k = w.keys[i]
            if np.isfinite(w.values[i]).sum() < 48 * 3:
                continue                     # fewer than 3 days visible: MDS has nothing to sample; left unfilled
            d1 = df.copy()
            d1[k] = w.values[i]
            with tempfile.TemporaryDirectory() as td:
                fi, fo = Path(td) / "in.csv", Path(td) / "out.csv"
                d1.to_csv(fi, index=False, na_rep="NA")
                p = subprocess.run([RSCRIPT, str(HERE / "mds.R"), str(fi), str(fo), str(self.lat), str(self.lon),
                                    str(self.tz), k], capture_output=True, text=True)
                if p.returncode != 0:
                    self.failures = getattr(self, "failures", []) + [(w.site, str(w.start.date()), k, p.stderr[-300:])]
                    continue
                r = pd.read_csv(fo)
            mu, sd = r[f"{k}_f"].to_numpy(np.float64), r[f"{k}_fsd"].to_numpy(np.float64)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
            s = mu[None] + sd[None] * rng.standard_normal((self.n, T))
            out[:, i] = np.where(mask[i][None], s, w.values[i][None])
        return out
