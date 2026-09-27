"""Score a simulated hydrograph against an observed USGS gauge record.

Peak discharge, Nash-Sutcliffe efficiency and Kling-Gupta efficiency, all on storm excess above
baseflow, over the observed samples that fall inside the simulated window.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sites import SiteConfig, Storm

CFS_TO_CMS = 0.0283168466
"""Exact cubic feet to cubic metres."""

GAUGE_DT_H = 0.25
"""The 15-minute instantaneous-values sampling interval."""


@dataclass
class Score:
    """One simulated-vs-observed comparison. Discharges are storm excess [m^3/s]."""

    peak_sim_cms: float
    peak_obs_cms: float
    peak_time_sim_h: float
    peak_time_obs_h: float
    nse: float
    kge: float
    kge_r: float
    kge_alpha: float
    kge_beta: float
    n_samples: int
    window_h: Tuple[float, float]

    @property
    def peak_ratio(self) -> float:
        """Simulated over observed peak."""
        return self.peak_sim_cms / self.peak_obs_cms

    @property
    def peak_lag_h(self) -> float:
        """Simulated minus observed time of peak [h]; below `GAUGE_DT_H` it is unresolved."""
        return self.peak_time_sim_h - self.peak_time_obs_h

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d.update(peak_ratio=self.peak_ratio, peak_lag_h=self.peak_lag_h, gauge_dt_h=GAUGE_DT_H)
        return d

    def report(self) -> str:
        lines = [
            f"window {self.window_h[0]:.0f}-{self.window_h[1]:.0f} h, {self.n_samples} gauge samples",
            f"  peak       sim {self.peak_sim_cms:8.2f} m3/s at {self.peak_time_sim_h:6.2f} h   "
            f"obs {self.peak_obs_cms:8.2f} m3/s at {self.peak_time_obs_h:6.2f} h   "
            f"ratio {self.peak_ratio:.2f}   lag {self.peak_lag_h:+.2f} h",
            f"  NSE        {self.nse:8.3f}",
            f"  KGE        {self.kge:8.3f}   (r {self.kge_r:.3f}, alpha {self.kge_alpha:.3f}, "
            f"beta {self.kge_beta:.3f})",
        ]
        return "\n".join(lines)


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency; 1 is exact, 0 is the observed mean."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return float(1.0 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def kge(obs: np.ndarray, sim: np.ndarray) -> Tuple[float, float, float, float]:
    """Kling-Gupta efficiency and its components (r, alpha, beta)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean())
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), r, alpha, beta


def peak(t_h: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
    """(peak discharge, its time [h])."""
    i = int(np.argmax(q))
    return float(q[i]), float(t_h[i])


def align(t_sim_h: np.ndarray, q_sim: np.ndarray, t_obs_h: np.ndarray,
          q_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed samples inside the simulated window, with the simulation interpolated onto them.

    Returns:
        (t_h, sim, obs).
    """
    m = (t_obs_h >= t_sim_h.min()) & (t_obs_h <= t_sim_h.max())
    assert m.sum() >= 2, "fewer than two gauge samples inside the simulated window"
    return t_obs_h[m], np.interp(t_obs_h[m], t_sim_h, q_sim), q_obs[m]


def load_observed(site: SiteConfig, storm: Storm) -> Tuple[np.ndarray, np.ndarray]:
    """Observed discharge as (hours since the storm's t0, cfs)."""
    path = site.discharge(storm.name)
    assert path.exists(), f"{path} missing; run `python3 cli.py fetch --site {site.name}`"
    df = pd.read_csv(path, parse_dates=["dateTime"])
    t0 = pd.Timestamp(storm.start, tz="UTC")
    t = (df["dateTime"] - t0).dt.total_seconds().to_numpy() / 3600.0
    return t, df["discharge_cfs"].to_numpy(float)


def score(site: SiteConfig, storm: Storm, t_sim_h: np.ndarray, q_sim_cms: np.ndarray) -> Score:
    """Compare a simulated outflow series against the gauge record.

    Args:
        site: Site carrying the gauge.
        storm: Storm defining t0 and the observed record.
        t_sim_h: Simulated time axis [h since storm start].
        q_sim_cms: Simulated discharge [m^3/s].

    Returns:
        A `Score` on storm excess: observed minus baseflow, simulated minus its initial value.
    """
    assert site.gauge is not None, f"site {site.name} has no gauge to validate against"
    t_obs, q_obs = load_observed(site, storm)
    q_obs_excess = (q_obs - site.gauge.baseflow_cfs).clip(0) * CFS_TO_CMS
    q_sim_excess = (np.asarray(q_sim_cms, float) - float(q_sim_cms[0])).clip(0)
    t, sim, obs = align(np.asarray(t_sim_h, float), q_sim_excess, t_obs, q_obs_excess)
    peak_sim, t_peak_sim = peak(t, sim)
    peak_obs, t_peak_obs = peak(t, obs)
    k, r, alpha, beta = kge(obs, sim)
    return Score(
        peak_sim_cms=peak_sim, peak_obs_cms=peak_obs,
        peak_time_sim_h=t_peak_sim, peak_time_obs_h=t_peak_obs,
        nse=nse(obs, sim), kge=k, kge_r=r, kge_alpha=alpha, kge_beta=beta,
        n_samples=int(len(t)), window_h=(float(t.min()), float(t.max())),
    )
