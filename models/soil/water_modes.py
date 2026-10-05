"""Water as three per-cell maps, each a sample of the flux tower's record over its last five complete years.

    SOIL MOISTURE  the typical year's mean root-zone water, the year whose rain is the median of the five
    DROUGHT        Stephenson's climatic water deficit, sum(ET0 - AET), over the 91 days of largest sum(ET0 - P)
    FLOOD          the largest rain event an independent gauge corroborates, solved in two dimensions
                   (../hydro/flood_curves.py)

The balance (`balance.py`) runs only the years these samples fall in, each spun up on the year before it
(`sampled_lanes`). Every LiDAR return takes its cell's value.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import balance

YEARS_BACK = 5                 # the sampled record: the last five complete calendar years
WINDOW_D = 91                  # the drought window, days: the WMO's scale for soil-moisture drought (WMO-No. 1090)
DAY_MIN_HOURS = 20             # hours of ET0 a day needs to count in the window (its ET0: their mean times 24)
ALBEDO_REF = 0.23              # FAO-56's grass reference surface, for the site's ET0 in choosing the window
U2_OF_U10 = 4.87 / np.log(67.8 * 10.0 - 5.42)
"""FAO-56 eq. 47: the 2 m wind from the 10 m reference speed."""
DRY_H = 6                      # dry hours that end one storm (the NWS and NRCS event separation)
MIN_TOTAL_MM = 5.0             # a wet spell below this is not a storm
CORROBORATE_FACTOR = 2.0       # an independent gauge reads at least 1/2 of the forcing gauge's peak hour and total
CORROBORATE_COVER = 0.8        # over at least 80 % of the storm's hours
RECESSION_H = 24               # hours FLOOD solves after the rain stops
MAX_H = 72                     # FLOOD's longest run, rain and recession
SIGMA = 5.670374e-8

CODE_TOP = 250                 # codes 0..250 over a map's [lo, hi]
CODE_SEALED = 251              # sealed ground (paving, vehicles) and ground under a crown that takes no water in
CODE_BUILDING = 252            # a building, a planted roof included
CODE_NONE = 255                # outside the site, or no state
MIN_SPAN = 0.25
"""SOIL MOISTURE's narrowest scale, as a share of the site's wilting point to saturation: differences below a soil
sensor's precision (0.01 to 0.03 m3/m3) are not stretched across the whole ramp."""


# ------------------------------------------------------------------------------------------------ the record

def complete_years(years: Sequence[int], this_year: int, n: int = YEARS_BACK) -> List[int]:
    """The last `n` complete calendar years of the record, before `this_year`."""
    return sorted(y for y in years if y < this_year)[-n:]


def et0_hourly_np(rn_w, rs_w, t, ea, pa, u2) -> np.ndarray:
    """`balance.et0_hourly` in numpy: ASCE-EWRI (2005) hourly short reference ET, mm/h."""
    rn = np.asarray(rn_w, np.float64) * 0.0036
    g = np.where(np.asarray(rs_w) > 0, 0.1 * rn, 0.5 * rn)
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    delta = 4098.0 * es / (t + 237.3) ** 2
    gamma = 0.000665 * pa
    cd = np.where(rn > 0, 0.24, 0.96)
    num = 0.408 * delta * (rn - g) + gamma * (37.0 / (t + 273.0)) * u2 * (es - ea)
    return np.clip(num / (delta + gamma * (1.0 + cd * u2)), 0.0, None)


def ffill(a) -> np.ndarray:
    """Each NaN takes the last value before it (a leading run, the first value after): the balance's air in a gap."""
    a = np.asarray(a, np.float64).copy()
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    first = np.flatnonzero(np.isfinite(a))
    if len(first):
        a[: first[0]] = a[first[0]]
    return a


def site_pet(ghi, ta, ea, pa, lw_in, u10) -> np.ndarray:
    """The site's hourly reference ET0 (mm) over open grass: net radiation from GHI at the reference albedo and the
    measured longwave, the 2 m wind from the 10 m reference speed. Air gaps are forward-filled as the balance fills
    them."""
    ta, ea, pa, lw_in, u10 = (ffill(v) for v in (ta, ea, pa, lw_in, u10))
    ghi = np.nan_to_num(np.asarray(ghi, np.float64))
    rn = (1.0 - ALBEDO_REF) * ghi + 0.98 * (lw_in - SIGMA * (ta + 273.15) ** 4)
    return et0_hourly_np(rn, ghi, ta, ea, pa, U2_OF_U10 * u10)


# ------------------------------------------------------------------------------------------------ the three samples

def storm_events(p_mm: np.ndarray, hours_s: np.ndarray, dry_h: int = DRY_H, min_total: float = MIN_TOTAL_MM) -> List[Dict]:
    """Runs of wet hours separated by fewer than `dry_h` dry ones, with at least `min_total` mm: start and end (UTC s,
    end exclusive), total mm, peak mm/h, wet hours."""
    p = np.nan_to_num(np.asarray(p_mm, np.float64))
    wet = np.nonzero(p > 0)[0]
    out = []
    if not len(wet):
        return out
    for g in np.split(wet, np.nonzero(np.diff(wet) > dry_h)[0] + 1):
        tot = float(p[g[0]:g[-1] + 1].sum())
        if tot >= min_total:
            out.append({"start_s": int(hours_s[g[0]]), "end_s": int(hours_s[g[-1]]) + 3600, "total_mm": round(tot, 1),
                        "peak_mm_h": round(float(p[g].max()), 1), "wet_hours": int(len(g))})
    return out


def corroborate(storm: Dict, hours_s: np.ndarray, p: np.ndarray, sources: Dict[str, np.ndarray],
                factor: float = CORROBORATE_FACTOR) -> Optional[bool]:
    """Whether an independent gauge saw the storm the forcing gauge recorded: a source holding CORROBORATE_COVER of
    the storm's hours supports it when its peak hour and its total both reach the forcing's over `factor`. `sources`:
    {name: hourly mm on `hours_s`, NaN where none}: other towers' gauges, a shielded ground gauge, gridded rain at the
    site; never the forcing gauge itself. None when no source covers the storm."""
    i0 = int(np.searchsorted(hours_s, storm["start_s"]))
    i1 = int(np.searchsorted(hours_s, storm["end_s"])) + 1
    f = np.nan_to_num(np.asarray(p[i0:i1], np.float64))
    peak, total = (float(f.max()) if len(f) else 0.0), float(f.sum())
    ok = []
    for v in sources.values():
        w = np.asarray(v[i0:i1], np.float64)
        if not len(w) or np.isfinite(w).mean() < CORROBORATE_COVER:
            continue
        ok.append(float(np.nanmax(w)) >= peak / factor and float(np.nansum(w)) >= total / factor)
    return any(ok) if ok else None


def five_year_storm(events: List[Dict], check: Callable[[Dict], Optional[bool]], years: Sequence[int]) -> Tuple[Optional[Dict], List[Dict]]:
    """FLOOD's storm: the largest event by depth starting in `years` that `check` (`corroborate`) does not refute.
    Returns (storm with its recession hours, the larger events refuted)."""
    yr = lambda e: int(str(np.datetime64(int(e["start_s"]), "s"))[:4])  # noqa: E731
    refuted = []
    for e in sorted((e for e in events if yr(e) in set(years)), key=lambda e: -e["total_mm"]):
        if check(e) is False:
            refuted.append(e)
            continue
        return dict(e, tail_h=recession_h(e)), refuted
    return None, refuted


def recession_h(storm: Dict) -> int:
    """Hours solved after the rain: RECESSION_H, within MAX_H of the storm's start."""
    return int(max(0, min(RECESSION_H, MAX_H - round((storm["end_s"] - storm["start_s"]) / 3600))))


def drought_window(hours_s: np.ndarray, rain: np.ndarray, pet: np.ndarray, std_offset_h: float,
                   days: int = WINDOW_D) -> Dict:
    """The `days` local standard-time days with the largest sum of daily (ET0 - P), the climatic water balance deficit
    SPEI accumulates (Vicente-Serrano et al. 2010). A day counts with DAY_MIN_HOURS hours of ET0; a window counts when
    every day does. Starts and ends at local midnight."""
    local = np.asarray(hours_s, np.int64) + int(round(std_offset_h * 3600))
    day = local // 86400
    d0 = int(day.min())
    k = day - d0
    nd = int(k.max()) + 1
    good = np.isfinite(pet)
    nh = np.bincount(k, minlength=nd)
    ng = np.bincount(k, weights=good.astype(np.float64), minlength=nd)
    full = (nh == 24) & (ng >= DAY_MIN_HOURS)
    dp = np.bincount(k, weights=np.where(good, pet, 0.0), minlength=nd) * 24.0 / np.maximum(ng, 1.0)
    dr = np.bincount(k, weights=np.nan_to_num(rain), minlength=nd)
    cs = np.r_[0.0, np.cumsum(dp - dr)]
    ok = np.convolve(full.astype(np.int64), np.ones(days, np.int64), "valid") == days
    run = np.where(ok, cs[days:] - cs[:-days], -np.inf)
    if not np.isfinite(run).any():
        raise ValueError(f"no {days}-day window has {DAY_MIN_HOURS} hours of ET0 every day")
    i = int(np.argmax(run))
    start = (d0 + i) * 86400 - int(round(std_offset_h * 3600))
    return {"start_s": int(start), "end_s": int(start + days * 86400), "days": days,
            "rain_mm": round(float(dr[i:i + days].sum()), 1), "pet_mm": round(float(dp[i:i + days].sum()), 1),
            "deficit_mm": round(float(run[i]), 1)}


def median_year(hours_s: np.ndarray, rain: np.ndarray, years: Sequence[int]) -> Dict:
    """SOIL MOISTURE's year: the one whose rain is the median of `years`."""
    yr = np.asarray(hours_s, np.int64).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    tot = {int(y): float(np.nansum(np.asarray(rain)[yr == y])) for y in years}
    order = sorted(tot, key=lambda y: tot[y])
    y = order[(len(order) - 1) // 2]
    return {"year": y, "rain_mm": round(tot[y], 1)}


def sampled_lanes(samples: Dict, record: Sequence[int]) -> List[Tuple[int, int]]:
    """The balance's (year, spin-up year) lanes the three samples need: each sample's year on the year before it (on
    itself where the record starts). A window or storm crossing into a new year takes both years."""
    want = {int(samples["typical"]["year"])}
    for key in ("drought", "storm"):
        a, b = samples[key]["start_s"], samples[key]["end_s"] - 1
        want |= {int(str(np.datetime64(int(a), "s"))[:4]), int(str(np.datetime64(int(b), "s"))[:4])}
    rec = set(int(y) for y in record)
    return [(y, y - 1 if y - 1 in rec else y) for y in sorted(want) if y in rec]


# ------------------------------------------------------------------------------------------------ the maps

def typical_theta(theta_months: np.ndarray, hours: Sequence[int]) -> np.ndarray:
    """SOIL MOISTURE per cell, m3/m3: the typical year's monthly mean root-zone water (`balance.root_zone`) weighted
    by each month's hours, sum_m theta_m h_m / sum_m h_m. theta_months: [12, cells]."""
    h = np.asarray(hours, np.float64)[:, None]
    return (np.asarray(theta_months, np.float64) * h).sum(0) / max(float(h.sum()), 1.0)


def drought_cwd(s: balance.State, k: balance.Cells, net, rain, rs_of, u2_of, air, hours: int,
                lateral: Optional[balance.LateralGraph] = None, roots: Optional[balance.RootShare] = None,
                kcb_of: Optional[Callable] = None) -> np.ndarray:
    """DROUGHT per cell, mm: the balance over the window's hours from the state at its first local midnight,
    CWD = sum_h max(0, ET0_h - AET_h) (Stephenson 1990), ET0 on the cell's own sunlight and 2 m wind; with `roots`, a
    crown's deficit charged to the soil its roots reach (`balance.Year.add`). NaN without soil. Arguments as
    `balance.run`."""
    return balance.run(s, k, net, hours, rain, rs_of, u2_of, air, lateral, roots=roots,
                       kcb_of=kcb_of).metrics(k)["cwd_mm"].cpu().numpy()


def plantable(no_soil: np.ndarray, roof: np.ndarray, perv: np.ndarray) -> np.ndarray:
    """Plantable ground: soil whose surface takes water in, not a roof. A crown over paving has roots and no plantable
    ground."""
    return ~np.asarray(no_soil, bool) & ~np.asarray(roof, bool) & (np.asarray(perv) > 0)


WALL_NZ = 0.5
WALL_M = 0.5
"""A return on a surface steeper than 60 degrees (its normal's vertical component under WALL_NZ) and more than WALL_M
over the bare earth is on a wall, whatever the classifier called it. No soil holds on a face that steep (the angle of
repose of soils is 30 to 45 degrees, Al-Hashemi and Al-Amoudi 2018), so a ground class there is a misclassified facade.
On UC Berkeley's campus 2.4 % of the pervious-ground returns stood on walls, up to over 10 m, and SOIL MOISTURE painted
them soil; at Harvard Forest 0.02 %. Within WALL_M of the ground a steep return stays ground: a wall's foot, a curb."""


def ground_on_walls(ground: np.ndarray, nz: np.ndarray, height: np.ndarray) -> np.ndarray:
    """The ground returns (`ground`: classed sealed or pervious ground) that stand on a wall: their normal's vertical
    component `nz` under WALL_NZ (negative under an overhang) and `height` over the bare earth over WALL_M. These are no
    ground, and a map of the ground never paints them."""
    return (np.asarray(ground, bool) & (np.nan_to_num(np.asarray(nz, float), nan=1.0) < WALL_NZ)
            & (np.nan_to_num(np.asarray(height, float), nan=0.0) > WALL_M))


def nice_bounds(v: np.ndarray, lo_pct: float, hi_pct: float, step: Optional[float] = None) -> Tuple[float, float]:
    """A fixed scale's ends: the values' percentiles, snapped outward to a round step."""
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    a, b = np.percentile(v, [lo_pct, hi_pct])
    if step is None:
        step = 10.0 ** np.floor(np.log10(max(b - a, 1e-9) / 4.0))
    lo, hi = np.floor(a / step) * step, np.ceil(b / step) * step
    return float(round(lo, 6)), float(round(hi if hi > lo else lo + step, 6))


def encode(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """uint8 codes 0..CODE_TOP over [lo, hi], clipped at the ends; NaN reads CODE_NONE."""
    v = np.asarray(values, np.float64)
    q = np.clip(np.round((np.nan_to_num(v) - lo) / max(hi - lo, 1e-12) * CODE_TOP), 0, CODE_TOP)
    return np.where(np.isfinite(v), q, CODE_NONE).astype(np.uint8)


def decode(codes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    c = np.asarray(codes)
    return np.where(c <= CODE_TOP, lo + c.astype(np.float64) / CODE_TOP * (hi - lo), np.nan)


def soil_moisture_map(theta: np.ndarray, no_soil: np.ndarray, roof: np.ndarray, perv: np.ndarray, site: np.ndarray,
                      wp: Optional[float] = None, theta_s: Optional[float] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    """SOIL MOISTURE's codes and scale. The scale: plantable cells' p10 to p90, snapped outward, and never narrower than
    MIN_SPAN of the site's wp to theta_s (medians), widened about its middle inside them."""
    plant = site & plantable(no_soil, roof, perv)
    v = theta[plant & np.isfinite(theta)]
    lo, hi = nice_bounds(v, 10.0, 90.0)
    if wp is not None and theta_s is not None and theta_s > wp and hi - lo < MIN_SPAN * (theta_s - wp):
        span = MIN_SPAN * (theta_s - wp)
        a = min(max(wp, 0.5 * (lo + hi) - span / 2), theta_s - span)
        step = 10.0 ** np.floor(np.log10(span / 4.0))
        lo, hi = float(round(np.floor(a / step) * step, 6)), float(round(np.ceil((a + span) / step) * step, 6))
    return _codes(theta, (lo, hi), no_soil, roof, perv, site), (lo, hi)


def drought_map(cwd: np.ndarray, no_soil: np.ndarray, roof: np.ndarray, perv: np.ndarray,
                site: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """DROUGHT's codes and scale: 0 to the plantable cells' p98, snapped up to 10 mm (1 mm below 50 mm). Every cell
    that is not plantable reads CODE_SEALED."""
    v = cwd[site & plantable(no_soil, roof, perv) & np.isfinite(cwd)]
    _, hi = nice_bounds(v, 2.0, 98.0, step=10.0 if (v.size and np.percentile(v, 98) > 50) else 1.0)
    return _codes(cwd, (0.0, hi), no_soil, roof, perv, site, CODE_SEALED), (0.0, hi)


def _codes(v, scale, no_soil, roof, perv, site, building=CODE_BUILDING):
    codes = encode(v, *scale)
    sealed = np.asarray(no_soil, bool) | ~(np.asarray(perv) > 0)
    codes[site & sealed] = CODE_SEALED
    codes[site & np.asarray(roof, bool)] = building
    codes[~site] = CODE_NONE
    return codes
