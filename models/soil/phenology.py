"""Each cover its own season: the trees' and the grass's basal crop coefficients through the year, and the grass that
grows under a savanna's crowns.

A site's MODIS LAI pixel (500 m) is mostly one cover. At Tonzi Ranch (blue oak savanna) it is the grass: its LAI peaks in
April with the grass while the oaks are bare until then, so trees whose Kcb followed it spent their water in February and
March (Kc 1.15 against the tower's 0.76). So each cover takes its own development stage:

    grass     stage g(d) from its own MCD12Q2 cycle (green-up, maturity, senescence, dormancy; FAO-56's curve, 0 to 1)
              Kcb = KC_MIN + (Kcb_full - KC_MIN) g(d)
    trees     where grass dominates the pixel, the record's own leaf area over the grass's dormant days (the trees alone
              in leaf then), shaped through the year by the trees' own MCD12Q2 cycle where a tree pixel gives one, else
              by the Growing Season Index of the site's air (Jolly, Nemani and Running 2005); elsewhere the site's LAI
              Kcb = KC_MIN + (Kcb_full - KC_MIN) (1 - exp(-0.7 LAI_cell))          Allen and Pereira 2009

The understory grass under a crown, where grass dominates the pixel (MOD44B), carries the open grass's Kcb scaled by the
pixel's own grass cover, nontree / (100 - tree): in blue oak savanna under 50 cm of rain the herbaceous biomass under the
canopy is about the open grassland's (Jackson, Strauss, Firestone and Bartolome 1990, Agric. Ecosyst. Environ. 32:89-105).
Nothing here is fitted to a tower.
"""

from typing import Optional, Tuple

import numpy as np
import torch

KC_MIN = 0.15                  # FAO-56: the basal coefficient of bare soil and a dormant cover
KD_K = 0.7                     # FAO-56 / Allen and Pereira 2009: the cover's extinction of the reference ET with leaf area
KCB_GRASS = 0.85               # the grass's mid-season Kcb (FAO-56 Table 17, turf; `balance.SURFACES` pervious ground)
GSI_TMIN_C = (-2.0, 5.0)
GSI_VPD_PA = (900.0, 4100.0)
GSI_PHOTO_H = (10.0, 11.0)
GSI_DAYS = 21
"""The Growing Season Index (Jolly, Nemani and Running 2005, Glob. Change Biol. 11:619-632): the product of three ramps,
daily minimum temperature -2 to 5 degC, daily mean vapor pressure deficit 900 to 4100 Pa (falling) and day length 10 to
11 h, averaged over 21 days. Its published limits; nothing fitted to a site."""


def kcb(kcb_full: torch.Tensor, lai_cell: torch.Tensor) -> torch.Tensor:
    """A canopy cell's basal coefficient from its green leaf area (Allen and Pereira 2009)."""
    return KC_MIN + (kcb_full - KC_MIN) * (1.0 - torch.exp(-KD_K * lai_cell))


def grass_kcb(kcb_full: torch.Tensor, stage) -> torch.Tensor:
    """An open cover's basal coefficient at its development stage (0 to 1): FAO-56's curve between KC_MIN and its Kcb.
    Covers whose Kcb is at or below KC_MIN (paving, roofs) keep it."""
    return torch.where(kcb_full > KC_MIN, KC_MIN + (kcb_full - KC_MIN) * stage, kcb_full)


def _leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def stage_daily(cycles: Optional[dict], year: int) -> np.ndarray:
    """A cover's development stage on each day of `year`, 0 to 1 (FAO-56's crop-coefficient curve): 0 before green-up,
    rising to 1 at maturity, 1 to senescence, falling to 0 at dormancy. `cycles` maps a year to its MCD12Q2 dates
    (ISO strings, keys Greenup, Maturity, Senescence, Dormancy; a cycle's green-up may fall in the year before); a year
    without its own cycle takes the median dates of the others, by day of year. None or empty: 1 every day."""
    n = 366 if _leap(year) else 365
    days = np.arange(n, dtype=np.float64)
    y0 = np.datetime64(f"{year}-01-01")

    def off(iso, y):
        return float((np.datetime64(iso) - np.datetime64(f"{y}-01-01")).astype(int))
    have = {int(k): v for k, v in (cycles or {}).items()}
    if not have:
        return np.ones(n)
    keys = ("Greenup", "Maturity", "Senescence", "Dormancy")
    med = {s: float(np.median([off(c[s], y) for y, c in have.items()])) for s in keys}
    g = np.zeros(n)
    for yc in (year - 1, year, year + 1):
        if yc in have:
            c = {s: float((np.datetime64(have[yc][s]) - y0).astype(int)) for s in keys}
        else:
            c = {s: med[s] + float((np.datetime64(f"{yc}-01-01") - y0).astype(int)) for s in keys}
        up = np.clip((days - c["Greenup"]) / max(c["Maturity"] - c["Greenup"], 1.0), 0.0, 1.0)
        down = np.clip((c["Dormancy"] - days) / max(c["Dormancy"] - c["Senescence"], 1.0), 0.0, 1.0)
        g = np.maximum(g, np.minimum(up, down))
    return g


def gsi_daily(ta_h: np.ndarray, ea_h: np.ndarray, lat: float) -> np.ndarray:
    """The Growing Season Index of each day of an hourly air record (ta degC, ea kPa, whole days), 21-day trailing mean."""
    n = len(ta_h) // 24
    ta = np.asarray(ta_h[:n * 24], np.float64).reshape(n, 24)
    ea = np.asarray(ea_h[:n * 24], np.float64).reshape(n, 24)
    es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    vpd = 1000.0 * np.nanmean(np.maximum(es - ea, 0.0), 1)
    tmin = np.nanmin(ta, 1)
    doy = np.arange(1, n + 1)
    dec = np.radians(23.44) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
    photo = 24.0 / np.pi * np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(dec), -1.0, 1.0))
    ramp = lambda x, lo, hi: np.clip((x - lo) / (hi - lo), 0.0, 1.0)  # noqa: E731
    gsi = np.nan_to_num(ramp(tmin, *GSI_TMIN_C) * (1.0 - ramp(vpd, *GSI_VPD_PA)) * ramp(photo, *GSI_PHOTO_H))
    c = np.cumsum(np.r_[np.zeros(GSI_DAYS), gsi])
    k = np.minimum(np.arange(1, n + 1), GSI_DAYS)
    return np.clip((c[GSI_DAYS:] - c[GSI_DAYS - k + np.arange(n)]) / k, 0.0, 1.0)


def class_daily(pheno: dict, leaf_years: dict, floor: float, year: int, ta_h: np.ndarray, ea_h: np.ndarray):
    """(the trees' leaf area on each day of `year`, the grass's stage on each day).

    pheno: {"dominant": "herbaceous" | "woody" | ..., "cycles": {"herbaceous": {year: dates}, "woody": {...}}, "lat"}
    (MCD12Q2 cycles of the site's pixel and its nearest tree-dominated pixel); leaf_years: {year: daily site LAI};
    floor: the record's winter LAI. Where grass does not dominate the pixel the trees keep the site's LAI."""
    site = np.asarray(leaf_years.get(year) or leaf_years.get(str(year)), np.float64)
    n = len(site)
    herb = stage_daily((pheno.get("cycles") or {}).get("herbaceous"), year)[:n]
    if len(herb) < n:
        herb = np.r_[herb, np.full(n - len(herb), herb[-1] if len(herb) else 1.0)]
    if pheno.get("dominant") != "herbaceous":
        return site, herb
    dormant = herb <= 0.05
    woody_cycles = (pheno.get("cycles") or {}).get("woody")
    if woody_cycles:
        g = stage_daily(woody_cycles, year)[:n]
        plateau = float(np.mean(site[dormant])) if dormant.any() else float(np.max(site))
        return floor + max(plateau - floor, 0.0) * g, herb
    gsi = gsi_daily(ta_h, ea_h, float(pheno["lat"]))
    gsi = np.r_[gsi, np.full(max(n - len(gsi), 0), gsi[-1] if len(gsi) else 0.0)][:n]
    if not dormant.any() or gsi[dormant].mean() <= 1e-6:
        return site, herb
    years = [np.asarray(v, np.float64) for v in leaf_years.values()]
    stages = [stage_daily((pheno.get("cycles") or {}).get("herbaceous"), int(k))[:len(v)] for k, v in leaf_years.items()]
    plateau = float(np.mean(np.concatenate([v[s <= 0.05] for v, s in zip(years, stages) if (s <= 0.05).any()])))
    return floor + max(plateau - floor, 0.0) * gsi / float(gsi[dormant].mean()), herb


def understory_share(pheno: Optional[dict], canopy: np.ndarray, over_pervious: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Each cell's understory grass under a crown, as a share of the open grass's cover, and its record.

    Where grass dominates the site's MODIS pixel the herb layer continues under the crowns at the pixel's own cover,
    nontree / (100 - tree) (MOD44B). A site the trees dominate (Harvard Forest, 72 % tree cover) carries none: its floor's
    light and water are the forest's. Only a crown over pervious ground has one."""
    pheno = pheno or {}
    if pheno.get("dominant") != "herbaceous" or pheno.get("tree_cover") is None or pheno.get("nontree_cover") is None:
        return np.zeros(len(canopy), np.float64), {"share": 0.0, "why": f"site dominant {pheno.get('dominant')}"}
    tree, non = float(pheno["tree_cover"]), float(pheno["nontree_cover"])
    h = float(np.clip(non / max(100.0 - tree, 1e-6), 0.0, 1.0))
    u = np.where(np.asarray(canopy, bool) & np.asarray(over_pervious, bool), h, 0.0)
    return u, {"share": round(h, 3), "tree_cover": tree, "nontree_cover": non,
               "source": "MOD44B nontree / (100 - tree); Jackson et al. 1990"}


def understory_kcb(u: torch.Tensor, stage=None) -> torch.Tensor:
    """The understory's basal coefficient (`balance.Cells.kcb_u`): its share `u` of the grass's own Kcb at its stage
    (FAO-56's curve between KC_MIN and KCB_GRASS; None: full). No light factor: Jackson et al.'s like biomass under the
    canopy already integrates the shade the grass grew in; its energy is the crown cell's own ET0."""
    g = KCB_GRASS if stage is None else KC_MIN + (KCB_GRASS - KC_MIN) * stage
    return u * g
