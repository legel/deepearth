"""Score a simulated hydrograph against an observed USGS gauge record.

Two metrics, reported very differently on purpose.

Timing is claimable, but only above the gauge's own resolution: the record is instantaneous
values at 15 minutes, so no difference below 0.25 h is a measurement of anything. An earlier
"0.09 h rising limb" sat under that floor and was never an accuracy figure.

Magnitude is reported as a RANGE over baseflow-separation and integration-window choices, never
as one number. A single unreproducible 19.6 % runoff coefficient propagated through this
project's write-ups for months; like-for-like over the simulated window the observed figure is
28.9-31.4 %. The comparison is also structurally approximate -- simulated volume is integrated
over all four domain edges against the gauge's own catchment area -- and saying so is part of
the result.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sites import SiteConfig, Storm

CFS_TO_CMS = 0.0283168466
"""Exact cubic feet to cubic metres."""

GAUGE_DT_H = 0.25
"""The 15-minute instantaneous-values sampling interval. The resolution floor on any timing
claim, and what disqualified the peak-argmax metric entirely: the 99 % plateau is wider than
the difference being measured."""


@dataclass
class Score:
    """One simulated-vs-observed comparison."""

    rising_limb_sim_h: float
    rising_limb_obs_h: float
    runoff_sim: float
    runoff_obs: Dict[str, float]
    rain_mm: float
    domain_km2: float
    window_h: Tuple[float, float]

    @property
    def rising_limb_error_h(self) -> float:
        """Absolute timing difference [h]. Only meaningful above the gauge's own interval."""
        return abs(self.rising_limb_sim_h - self.rising_limb_obs_h)

    @property
    def rising_limb_resolved(self) -> bool:
        """False means the difference sits under the gauge's own sampling interval."""
        return self.rising_limb_error_h >= GAUGE_DT_H

    @property
    def runoff_obs_range(self) -> Tuple[float, float]:
        """Observed runoff coefficient over the simulated window, spanning baseflow choices."""
        like = [v for k, v in self.runoff_obs.items() if k.startswith("sim window")]
        return min(like), max(like)

    def report(self) -> str:
        """A human-readable summary, ranges intact."""
        lo, hi = self.runoff_obs_range
        if not np.isfinite(self.rising_limb_error_h):
            verdict = "NOT MEASURABLE -- one of the two series never rises"
        elif self.rising_limb_resolved:
            verdict = f"{self.rising_limb_error_h / GAUGE_DT_H:.1f}x the gauge sampling interval"
        else:
            verdict = (f"BELOW the gauge's own {GAUGE_DT_H:.2f} h interval -- agreement to "
                       f"within one sample; report as preserved, not as an accuracy figure")
        lines = [
            f"storm {self.rain_mm:.1f} mm over {self.domain_km2:.2f} km2, "
            f"window {self.window_h[0]:.0f}-{self.window_h[1]:.0f} h",
            "",
            "[1] RISING LIMB (50 % of peak)",
            f"  simulated  {self.rising_limb_sim_h:6.2f} h",
            f"  observed   {self.rising_limb_obs_h:6.2f} h",
            f"  difference {self.rising_limb_error_h:6.2f} h   {verdict}",
            "",
            "[2] RUNOFF COEFFICIENT",
            f"  simulated  {self.runoff_sim * 100:6.2f} %   (all four domain edges)",
            "  observed, by baseflow separation and integration window:",
        ]
        lines += [f"    {label:>34} {rc * 100:7.1f} %" for label, rc in self.runoff_obs.items()]
        lines += [
            "",
            f"  like-for-like over the simulated window: observed {lo * 100:.1f}-{hi * 100:.1f} %"
            f" against simulated {self.runoff_sim * 100:.2f} %",
            f"  ratio: {self.runoff_sim / hi:.1f}x-{self.runoff_sim / lo:.1f}x observed",
        ]
        return "\n".join(lines)


def rising_limb_50(t_h: np.ndarray, q: np.ndarray) -> float:
    """Time at which the rising limb first reaches half the peak, linearly interpolated.

    NaN when there is no peak to be half of. A series that never rises satisfies `q >= 0.5 * q`
    at its own argmax, so the crossing search would otherwise return t0 and a run that produced
    no flow at all would score as an instantaneous response -- a number where there is no
    measurement, which is the failure this module exists to avoid.
    """
    t, q = np.asarray(t_h, float), np.asarray(q, float)
    peak = int(np.argmax(q))
    half = 0.5 * q[peak]
    if not (q[peak] > 0.0):
        return float("nan")
    seg_q, seg_t = q[: peak + 1], t[: peak + 1]
    hit = np.where(seg_q >= half)[0]
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(seg_t[0])
    q0, q1, t0, t1 = seg_q[i - 1], seg_q[i], seg_t[i - 1], seg_t[i]
    return float(t0 + (half - q0) / (q1 - q0) * (t1 - t0)) if q1 > q0 else float(t1)


def load_observed(site: SiteConfig, storm: Storm) -> Tuple[np.ndarray, np.ndarray]:
    """Observed discharge as (hours since the storm's t0, cfs)."""
    path = site.discharge(storm.name)
    assert path.exists(), f"{path} missing; run `python3 cli.py fetch --site {site.name}`"
    df = pd.read_csv(path, parse_dates=["dateTime"])
    t0 = pd.Timestamp(storm.start, tz="UTC")
    t = (df["dateTime"] - t0).dt.total_seconds().to_numpy() / 3600.0
    return t, df["discharge_cfs"].to_numpy(float)


def score(site: SiteConfig, storm: Storm, t_sim_h: np.ndarray, q_sim_cms: np.ndarray,
          rain_mm: float, domain_m2: float) -> Score:
    """Compare a simulated outflow series against the gauge record.

    Args:
        site: Site carrying the gauge.
        storm: Storm defining t0 and the observed record.
        t_sim_h: Simulated time axis [h since storm start].
        q_sim_cms: Simulated discharge [m^3/s].
        rain_mm: Total storm depth [mm].
        domain_m2: Simulated domain area, the full grid rather than valid cells only.

    Returns:
        A `Score`; use `.report()` for the formatted version.
    """
    assert site.gauge is not None, f"site {site.name} has no gauge to validate against"
    t_obs, q_obs = load_observed(site, storm)

    # Baseflow separation must be symmetric. With a baseflow initial condition the simulated
    # series also starts non-zero and the pre-filled channel dumps at the edge on step one,
    # which put the "rising limb" at t = 0 -- an initial-condition transient, not a response.
    q_obs_excess = (q_obs - site.gauge.baseflow_cfs).clip(0) * CFS_TO_CMS
    q_sim_excess = (q_sim_cms - float(q_sim_cms[0])).clip(0)

    window = (float(t_sim_h.min()), float(t_sim_h.max()))
    gauge_area_m2 = site.gauge.documented_area_km2 * 1e6
    windows = {
        f"sim window {window[0]:.0f}-{window[1]:.0f} h": window,
        "full gauge record": (float(t_obs.min()), float(t_obs.max())),
        "0-48 h": (0.0, 48.0),
    }
    runoff_obs = {}
    for bf_cfs in (0.0, site.gauge.baseflow_cfs):
        for label, (lo, hi) in windows.items():
            m = (t_obs >= lo) & (t_obs <= hi)
            vol = np.trapz((q_obs[m] - bf_cfs).clip(0) * CFS_TO_CMS, t_obs[m] * 3600.0)
            key = f"{label}, baseflow {bf_cfs:.0f} cfs"
            runoff_obs[key] = vol / (rain_mm / 1000.0 * gauge_area_m2)

    return Score(
        rising_limb_sim_h=rising_limb_50(t_sim_h, q_sim_excess),
        rising_limb_obs_h=rising_limb_50(t_obs, q_obs_excess),
        runoff_sim=float(np.trapz(q_sim_cms, t_sim_h * 3600.0) / (rain_mm / 1000.0 * domain_m2)),
        runoff_obs=runoff_obs,
        rain_mm=rain_mm,
        domain_km2=domain_m2 / 1e6,
        window_h=window,
    )
