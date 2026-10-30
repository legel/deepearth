"""SITE CONDITIONS beside soil moisture: root-zone salinity, flooding and ponding, the water table's months near the
surface, and each cell's dry-season low and wet-season high moisture, on a site's water modes grid.

The plant model reads soil moisture as one typical-year value per cell (modes.json drainage). A tidal marsh, a sodic flat
and a seasonally ponded swale differ in what that value leaves out: the salt in the root zone, how often and how long
water stands on the ground, and how far the wettest and driest seasons sit from the year's mean. These layers state that
per cell from the USDA soil survey, the National Wetlands Inventory and SMAP-HydroBlocks. Nothing here changes the
moisture product.

SALINITY. SSURGO chorizon ec_r (electrical conductivity of the saturation extract, dS/m) and sar_r (sodium adsorption
ratio) over 0 to 30 cm. A component's horizons are weighted by their thickness inside that layer; a map unit's components
by comppct_r over the components that have a value (Urban land and Water have none, and the share that has one is
recorded). Classes (Soil Survey Manual, USDA Handbook 18, 2017, ch. 3): nonsaline under 2, very slightly saline 2 to 4,
slightly saline 4 to 8, moderately saline 8 to 16, strongly saline 16 dS/m or more; sodic at SAR 13 or more (Richards
1954, USDA Handbook 60). Both rasters are feathered over FEATHER_M as the moisture floors are (soil.moisture_truth), over
the surveyed cells only. A map unit whose name says "salin" or "tidal" is recorded beside its numbers as a check. NWI
estuarine polygons (Cowardin et al. 1979: E2 intertidal, the salt marsh and flats; E1 subtidal) are a flag raster of
their own, the tidal evidence where the survey has no horizon data.

FLOODING AND PONDING. The component month table (comonth), month by month: flooding frequency and duration class
(flodfreqcl, floddurcl) and ponding frequency, duration and depth (pondfreqcl, ponddurcl, ponddep_r), in the classes of
the National Soil Survey Handbook (NSSH) Part 618 (flooding frequency class, flooding duration class, ponding frequency
class, ponding duration class, ponding depth). A component's class is its most frequent month's, and its duration the
longest of its flooded (ponded) months. A map unit's class is the dominant condition over its components (the class
holding the most comppct_r, the wetter on a tie), the aggregation muaggatt's flodfreqdcd uses. Soil moisture by month
(cosoilmoist): a month counts as saturated near the surface when a layer with soimoiststat 'Wet' begins at 30 cm or
shallower (the hydric soil technical criteria put the water table at 1.0 ft or less; NTCHS 1995, Federal Register 60 FR
10349). months_saturated_30cm is that count, 0 to 12, weighted by comppct_r over the components that have soil moisture
records; a component with none is unknown, never dry.

SEASONALITY. On the relative scale the product already uses (0 wilting point, 1 field capacity, 2 saturation;
exposure.water_modes.relative_wetness), from three sources:
  annual   the published soil moisture (modes drainage codes), the typical year's mean per cell;
  surface  SMAP-HydroBlocks 30 m seasonal means of the top 5 cm (soil.hydroblocks.site_window wet_mean and dry_mean,
           Vergopolan et al. 2021), sampled bilinearly at each cell centre and put on the relative scale with the
           cell's own wilting point, field capacity and saturation (SSURGO wfifteenbar_r, wthirdbar_r, wsatiated_r over
           0 to 30 cm; where the survey has none, modes.json drainage.scale.absolute_m3m3.soil_wp_theta_s and the site's
           median field capacity). The lower of the two seasonal means is the dry season and the higher the wet one, so
           a summer-wet climate (Florida) reads the right way round;
  survey   the water table and the flooding and ponding classes above.
  The surface's seasonal swing is laid on the root zone's annual value, never its level: the top 5 cm dries to the
  wilting point in a California summer while the root zone below it holds water (a California site, 2026-10-28: the surface
  dry mean put every cell at 0). The swing is d_dry = (theta_dry - theta_mid) / band and d_wet = (theta_wet - theta_mid)
  / band, theta_mid the two seasons' mean and band the width of the half of the scale the annual value sits in (fc - wp
  below field capacity, theta_s - fc above):
  dry-season low  = clip(annual + d_dry, 0, 2), raised to SAT_FLOOR where the water table is within 30 cm in
                    every month (months_saturated_30cm >= ALL_YEAR_MONTHS) and to VERY_FREQUENT_FLOOR under very
                    frequent flooding;
  wet-season high = clip(annual + d_wet, 0, 2), raised to SAT_FLOOR where the water table is within 30 cm
                    in any month (>= ANY_MONTH) and to saturation (2) under frequent or very frequent flooding or
                    frequent ponding (standing water in more than half of all years).
  The root zone swings less than the top 5 cm, so the band is an outer bound. low <= high always; low <= annual
  <= high except where a survey floor lifts the low above the annual value. Both are NaN where the cell is not
  plantable ground (codes 251 to 255). SAT_FLOOR and VERY_FREQUENT_FLOOR are soil.moisture_truth's 30 cm water table
  floor and NWI wetland floor, so the two modules agree on what a water table at the surface means. Without
  HydroBlocks the swing is 0 and the survey alone widens the band.

PUBLISHED (2026-10-28). Seven uint8 layers in sims/v1/exposure/soil_conditions/ with a gzipped copy of each,
coded as exposure/water/modes drainage_u8.bin is: 0..250 on a fixed scale given in the sidecar (a class layer: its class
code), 251 sealed, 252 building, 253 open water copied from the moisture product, 255 no value; the same grid and row
order. They make three SOIL modes beside pH, Type and Organic (BUTTONS): Salinity (salinity, tidal), Flooding (flooding,
flood_freq, pond_freq) and Seasonal water (seasonal_dry, seasonal_wet). site_conditions.json carries each layer's
scale or classes, legend and the DATA panel's row (data_sources), and the per map unit table the plant model can join.
One Soil Data Access request per site (site_sql), one NWI query (paged past 500 polygons), one HydroBlocks window.

Standard library and numpy only: this runs beside the product, never inside the solver.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import moisture_truth as MT
from . import net, ssurgo
from . import truth_sources as TS

TOP_CM = 30.0
"""The surface layer [cm] salinity and retention are averaged over: the root zone the moisture product reports."""
WET_DEPTH_CM = 30.0
FEATHER_M = MT.FEATHER_M
EC_CLASSES = ((2.0, "nonsaline"), (4.0, "very slightly saline"), (8.0, "slightly saline"), (16.0, "moderately saline"),
              (math.inf, "strongly saline"))
SODIC_SAR = 13.0
SALINE_NAME = re.compile(r"salin|tidal", re.I)

FLOOD_FREQ = {"none": 0, "very rare": 1, "rare": 2, "occasional": 3, "common": 3, "frequent": 4, "very frequent": 5}
"""NSSH 618 flooding frequency: none (near 0 % chance in any year), very rare (under 1 %), rare (1 to 5 %), occasional
(5 to 50 %), frequent (over 50 % in any year), very frequent (over 50 % in all months). 'Common' is the legacy class for
occasional and frequent together; it is read as occasional, the lower bound."""
FLOOD_DUR = {"extremely brief": 1, "very brief": 2, "brief": 3, "long": 4, "very long": 5}
"""NSSH 618 flooding duration: extremely brief (0.1 to 4 h), very brief (4 to 48 h), brief (2 to 7 d), long (7 to 30 d),
very long (over 30 d); 0 not flooded."""
POND_FREQ = {"none": 0, "rare": 1, "occasional": 2, "frequent": 3}
"""NSSH 618 ponding frequency: none, rare (1 to 5 times in 100 years), occasional (5 to 50), frequent (over 50)."""
POND_DUR = {"very brief": 1, "brief": 2, "long": 3, "very long": 4}
"""NSSH 618 ponding duration: very brief (under 2 d), brief (2 to 7 d), long (7 to 30 d), very long (over 30 d); 0 not
ponded."""
NONE_U8 = 255

SAT_FLOOR = MT.WATER_TABLE_FLOORS[0][1]
VERY_FREQUENT_FLOOR = MT.WETLAND_FLOOR
SATURATION, FIELD_CAPACITY = MT.SATURATION, MT.FIELD_CAPACITY
ALL_YEAR_MONTHS = 11.5
"""months_saturated_30cm at or above which the water table is near the surface all year (12, less a feathered edge)."""
ANY_MONTH = 0.5
HB_MARGIN_M = 60.0
"""Metres read past the grid for SMAP-HydroBlocks: two 30 m cells, so every cell centre has its four neighbours."""
HB_MIN_WEIGHT = 0.25
"""A cell takes a bilinear value when at least this much of its weight falls on cells with a value (a coast's edge)."""

SOURCES = {"ssurgo": "USDA-NRCS SSURGO via Soil Data Access (mupolygon, component, chorizon: ec_r, sar_r, wthirdbar_r, "
                     "wfifteenbar_r, wsatiated_r; comonth: flodfreqcl, floddurcl, pondfreqcl, ponddurcl, ponddep_r; "
                     "cosoilmoist: soimoiststat, soimoistdept_r)",
           "nwi": TS.SOURCES["nwi"] + ", estuarine (E) polygons kept",
           "hydroblocks": "SMAP-HydroBlocks seasonal means, top 5 cm (soil.hydroblocks)"}

RULES = {
    "surface_layer_cm": [0, TOP_CM],
    "salinity": "ec_r [dS/m] and sar_r over 0 to 30 cm, horizon thickness weighted, then comppct_r weighted over the "
                "components with a value; feathered over the surveyed cells",
    "ec_classes_dS_m": {name: hi if math.isfinite(hi) else None for hi, name in EC_CLASSES},
    "sodic_sar_at_least": SODIC_SAR,
    "salinity_basis": "Soil Survey Manual (USDA Handbook 18, 2017) ch. 3 salinity classes; Richards 1954 (USDA "
                      "Handbook 60) SAR 13",
    "name_check": "map unit name matches /salin|tidal/i",
    "nwi_estuarine_codes": {"0": "no estuarine polygon", "1": "E2 intertidal (salt marsh, flats)", "2": "E1 subtidal"},
    "flooding_ponding_basis": "NSSH Part 618 flooding and ponding frequency, duration and depth classes",
    "flood_freq_codes": {"0": "none", "1": "very rare", "2": "rare", "3": "occasional (or legacy common)",
                         "4": "frequent", "5": "very frequent", "255": "no record"},
    "flood_dur_codes": {"0": "not flooded", "1": "extremely brief", "2": "very brief", "3": "brief", "4": "long",
                        "5": "very long", "255": "no record"},
    "pond_freq_codes": {"0": "none", "1": "rare", "2": "occasional", "3": "frequent", "255": "no record"},
    "pond_dur_codes": {"0": "not ponded", "1": "very brief", "2": "brief", "3": "long", "4": "very long",
                       "255": "no record"},
    "component_class": "most frequent month; duration the longest flooded (ponded) month",
    "map_unit_class": "dominant condition over components (most comppct_r, the wetter on a tie)",
    "months_saturated_30cm": "months with a cosoilmoist 'Wet' layer whose top is at 30 cm or shallower, comppct_r "
                             "weighted over components with soil moisture records (NTCHS 1995 hydric criterion: water "
                             "table at 1.0 ft or less); feathered over the surveyed cells",
    "seasonal": {"scale": "0 wilting point, 1 field capacity, 2 saturation (each soil's own)",
                 "dry_low": f"clip(annual + d_dry, 0, 2), d_dry = (theta_dry - theta_mid) / band (fc - wp below field "
                            f"capacity, theta_s - fc above); >= {SAT_FLOOR} where months_saturated_30cm >= {ALL_YEAR_MONTHS}; "
                            f">= {VERY_FREQUENT_FLOOR} under very frequent flooding",
                 "wet_high": f"clip(annual + d_wet, 0, 2); >= {SAT_FLOOR} where months_saturated_30cm >= "
                             f"{ANY_MONTH}; = {SATURATION} under frequent or very frequent flooding or frequent ponding",
                 "band": "the top 5 cm's seasonal swing laid on the root zone's annual value: an outer bound, since the "
                         "root zone swings less",
                 "seasons": "the lower of HydroBlocks' two seasonal means is the dry season, the higher the wet one",
                 "retention": "SSURGO wfifteenbar_r, wthirdbar_r, wsatiated_r [% vol] over 0 to 30 cm; else modes.json "
                              "absolute_m3m3.soil_wp_theta_s and the site's median field capacity, else fc "
                              "midway between wp and theta_s",
                 "annual": "the published soil moisture codes (modes.json drainage), decoded on their scale"},
    "feather_m": FEATHER_M,
}


def _num(v) -> Optional[float]:
    return MT._num(v)


def class_code(table: Dict[str, int], value) -> Optional[int]:
    """A class string as SDA writes it ('Very long (more than 30 days)') as its code in `table`; None when blank or
    unknown."""
    if value is None:
        return None
    k = str(value).split("(")[0].strip().lower()
    return table.get(k)


def ec_class(ec: Optional[float]) -> Optional[str]:
    if ec is None or not math.isfinite(ec):
        return None
    for hi, name in EC_CLASSES:
        if ec < hi:
            return name
    return EC_CLASSES[-1][1]


# ------------------------------------------------------------------------------------------------ queries

def _in(units) -> str:
    """The IN list of a query: a subquery string as given, or mukeys checked to be integers (anything else is SQL)."""
    if isinstance(units, str):
        return units
    keys = sorted({str(k) for k in units})
    bad = [k for k in keys if not k.isdigit()]
    if bad:
        raise ValueError(f"not mukeys: {bad[:5]}")
    return ",".join(f"'{k}'" for k in keys)


def under(bbox: TS.Bbox) -> str:
    """The map units under a box, as a subquery: SDA refuses DECLARE, so each statement writes the box again."""
    return (f"SELECT DISTINCT mukey FROM mupolygon WHERE mupolygongeo.STIntersects("
            f"geometry::STGeomFromText('{TS._box_wkt(bbox)}', 4326)) = 1")


def horizon_sql(units) -> str:
    """Every component of the map units with its horizons that begin above 30 cm (none for Urban land and Water)."""
    return ("SELECT c.mukey, c.cokey, c.compname, c.comppct_r, h.hzdept_r, h.hzdepb_r, h.ec_r, h.sar_r, h.wthirdbar_r, "
            "h.wfifteenbar_r, h.wsatiated_r FROM component c LEFT JOIN chorizon h ON h.cokey = c.cokey AND "
            f"h.hzdept_r < {TOP_CM:g} WHERE c.mukey IN ({_in(units)})")


def month_sql(units) -> str:
    """Every component month with its flooding and ponding classes and each of its soil moisture layers (a month without
    one comes back once with NULLs)."""
    return ("SELECT c.mukey, c.cokey, m.monthseq, m.flodfreqcl, m.floddurcl, m.pondfreqcl, m.ponddurcl, m.ponddep_r, "
            "s.soimoiststat, s.soimoistdept_r FROM component c JOIN comonth m ON m.cokey = c.cokey LEFT JOIN "
            f"cosoilmoist s ON s.comonthkey = m.comonthkey WHERE c.mukey IN ({_in(units)})")


def site_sql(bbox: TS.Bbox) -> str:
    """ONE Soil Data Access request per site: the polygons under the box (truth_sources.ssurgo_sql), their components'
    surface horizons and their component months, three statements SDA answers as Table, Table1 and Table2."""
    return "; ".join((TS.ssurgo_sql(bbox), horizon_sql(under(bbox)), month_sql(under(bbox))))


TABLE_KEYS = {"polygons": "wkt", "horizons": "hzdept_r", "months": "monthseq"}
"""Each result set told by a column only it has, never by its position: SDA may leave an empty one out."""


def split_tables(doc: Dict) -> Dict[str, List[Dict]]:
    """A multi-statement SDA body ({"Table": [cols, rows...], "Table1": ...}) as {polygons, horizons, months} rows."""
    out = {k: [] for k in TABLE_KEYS}
    for key, t in doc.items():
        if not (key.startswith("Table") and isinstance(t, list) and t):
            continue
        cols = t[0]
        for name, col in TABLE_KEYS.items():
            if col in cols:
                out[name] = [dict(zip(cols, r)) for r in t[1:]]
    return out


def sda_tables(sql: str, tries: int = 2, sleep=time.sleep) -> Dict[str, List[Dict]]:
    """POST one multi-statement query (soil.ssurgo.query's transport and its refusal and maintenance handling) and
    return every result set, split by `split_tables`."""
    body = json.dumps({"query": sql, "format": "JSON+COLUMNNAME"}).encode()
    raw, attempt, waited = "", 0, 0
    while attempt < tries:
        req = urllib.request.Request(ssurgo.SDA_URL, data=body, headers={"Content-Type": "application/json"})
        raw = net.urlopen(req, "USDA Soil Data Access", timeout=ssurgo._TIMEOUT).decode()
        if raw.lstrip().startswith("{"):
            return split_tables(json.loads(raw))
        if ssurgo.MAINTENANCE in raw and waited < ssurgo.MAINTENANCE_WAIT_S:
            sleep(ssurgo.MAINTENANCE_POLL_S)                  # the daily window is an outage, never the query's fault
            waited += ssurgo.MAINTENANCE_POLL_S
            continue
        attempt += 1
        if attempt < tries:
            sleep(2 * attempt)
    raise RuntimeError("SDA rejected the query: " + raw[:400])


def estuarine_code(attribute: Optional[str]) -> int:
    """1 for an NWI estuarine intertidal polygon (E2: salt marsh, flats), 2 for estuarine subtidal (E1), 0 otherwise."""
    a = (attribute or "").strip().upper()
    if not a.startswith("E"):
        return 0
    return 2 if a.startswith("E1") else 1


def fetch(bbox: TS.Bbox, post: Callable = sda_tables, get: Callable = TS._get_json) -> Dict:
    """The survey's polygons, horizons and months (one SDA request) and the NWI's estuarine polygons for one box."""
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    t = post(site_sql(bbox))
    t_ssurgo = stamp()
    wetlands = TS.nwi_wetlands(bbox, get)
    return {"bbox": list(bbox), "units": TS.parse_sda(t["polygons"]), "horizons": t["horizons"], "months": t["months"],
            "estuarine": [w for w in wetlands if estuarine_code(w["attribute"])], "nwi_polygons_all": len(wetlands),
            "fetched": {"ssurgo": t_ssurgo, "nwi": stamp()}, "requests": {"sda": 1}, "sources": dict(SOURCES)}


def place(sources: Dict, frame: Dict) -> Dict:
    """`fetch`'s polygons in the frame's metres."""
    return dict(sources, units=[dict(u, polygons=TS.to_local(frame, u["polygons"])) for u in sources["units"]],
                estuarine=[dict(w, polygons=TS.to_local(frame, w["polygons"])) for w in sources["estuarine"]])


# ------------------------------------------------------------------------------------------------ the survey's units

def layer_mean(horizons: Sequence[Dict], col: str, top: float = 0.0, bottom: float = TOP_CM) -> Optional[float]:
    """The thickness-weighted mean of `col` over [top, bottom] cm of a component's horizons that have a value."""
    num = den = 0.0
    for h in horizons:
        v, a, b = _num(h.get(col)), _num(h.get("hzdept_r")), _num(h.get("hzdepb_r"))
        if v is None or a is None or b is None:
            continue
        t = min(b, bottom) - max(a, top)
        if t > 0:
            num += v * t
            den += t
    return num / den if den > 0 else None


def _weighted(pairs) -> Tuple[Optional[float], float]:
    """(comppct-weighted mean of the values that are known, the comppct they hold)."""
    num = den = 0.0
    for p, v in pairs:
        if v is None or p is None or p <= 0:
            continue
        num += p * v
        den += p
    return (num / den if den > 0 else None), den


def _dominant(pairs) -> Optional[int]:
    """The code holding the most comppct, the higher (wetter) on a tie; None when no component has one."""
    acc: Dict[int, float] = {}
    for p, c in pairs:
        if c is not None:
            acc[c] = acc.get(c, 0.0) + max(p or 0.0, 0.0)
    return max(acc, key=lambda c: (acc[c], c)) if acc else None


def components(horizons: Sequence[Dict], months: Sequence[Dict]) -> Dict[str, Dict]:
    """Components by cokey: {mukey, cokey, compname, comppct_r, horizons, months {1..12: record}}."""
    comps: Dict[str, Dict] = {}
    for r in horizons:
        k = str(r["cokey"])
        c = comps.setdefault(k, {"mukey": str(r["mukey"]), "cokey": k, "compname": r.get("compname"),
                                 "comppct_r": _num(r.get("comppct_r")) or 0.0, "horizons": [], "months": {}})
        if _num(r.get("hzdept_r")) is not None:
            c["horizons"].append(r)
    for r in months:
        c = comps.get(str(r["cokey"]))
        m = _num(r.get("monthseq"))
        if c is None or m is None:
            continue
        e = c["months"].setdefault(int(m), {"flood": None, "fdur": None, "pond": None, "pdur": None, "pdep": None,
                                            "wet": [], "records": 0})
        # one row per soil moisture layer: the month's classes repeat on each, and a month may be listed twice
        for key, col, table in (("flood", "flodfreqcl", FLOOD_FREQ), ("fdur", "floddurcl", FLOOD_DUR),
                                ("pond", "pondfreqcl", POND_FREQ), ("pdur", "ponddurcl", POND_DUR)):
            v = class_code(table, r.get(col))
            if v is not None:
                e[key] = v if e[key] is None else max(e[key], v)
        d = _num(r.get("ponddep_r"))
        if d is not None:
            e["pdep"] = d if e["pdep"] is None else max(e["pdep"], d)
        st = (r.get("soimoiststat") or "").strip().lower()
        if st:
            e["records"] += 1
        top = _num(r.get("soimoistdept_r"))
        if st == "wet" and top is not None:
            e["wet"].append(top)
    return comps


def component_summary(c: Dict) -> Dict:
    """One component's surface salinity and retention, its months with water within 30 cm, and its classes."""
    hz = c["horizons"]
    out = {k: layer_mean(hz, col) for k, col in (("ec", "ec_r"), ("sar", "sar_r"), ("fc", "wthirdbar_r"),
                                                 ("wp", "wfifteenbar_r"), ("theta_s", "wsatiated_r"))}
    for k in ("fc", "wp", "theta_s"):                    # SSURGO writes these in % by volume
        out[k] = None if out[k] is None else out[k] / 100.0
    ms = c["months"]
    if any(e["records"] for e in ms.values()):
        wet = [m for m in range(1, 13) if m in ms and ms[m]["wet"] and min(ms[m]["wet"]) <= WET_DEPTH_CM]
        out["months_saturated"], out["wet_months"] = len(wet), wet
    else:
        out["months_saturated"], out["wet_months"] = None, None
    for f, d, dep in (("flood", "fdur", None), ("pond", "pdur", "pdep")):
        fr = [e[f] for e in ms.values() if e[f] is not None]
        out[f + "_freq"] = max(fr) if fr else None
        durs = [e[d] for e in ms.values() if e[f] and e[d] is not None]
        out[f + "_dur"] = (max(durs) if durs else 0) if fr else None
        if dep:
            deps = [e[dep] for e in ms.values() if e[f] and e[dep] is not None]
            out["pond_depth_cm"] = (max(deps) if deps else 0.0) if fr else None
    return out


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def unit_profiles(horizons: Sequence[Dict], months: Sequence[Dict]) -> Dict[str, Dict]:
    """Per map unit: ec, sar, wp, fc, theta_s (m3/m3), months_saturated_30cm, wet_share_by_month [12], flood and pond
    classes, pond depth, the comppct each figure rests on and the components themselves."""
    by: Dict[str, List[Tuple[Dict, Dict]]] = {}
    for c in components(horizons, months).values():
        by.setdefault(c["mukey"], []).append((c, component_summary(c)))
    out = {}
    for mk, cs in by.items():
        p: Dict = {}
        for k in ("ec", "sar", "wp", "fc", "theta_s"):
            p[k], share = _weighted([(c["comppct_r"], s[k]) for c, s in cs])
            if k == "ec":
                p["salinity_share_pct"] = round(share, 1)
        rec = [(c, s) for c, s in cs if s["months_saturated"] is not None]
        p["months_saturated_30cm"], share = _weighted([(c["comppct_r"], s["months_saturated"]) for c, s in rec])
        p["water_table_share_pct"] = round(share, 1)
        p["wet_share_by_month"] = ([round(_weighted([(c["comppct_r"], float(m in s["wet_months"])) for c, s in rec])[0], 3)
                                    for m in range(1, 13)] if share > 0 else None)
        for k in ("flood_freq", "flood_dur", "pond_freq", "pond_dur"):
            p[k] = _dominant([(c["comppct_r"], s[k]) for c, s in cs])
        p["flood_freq_max"] = max((s["flood_freq"] for _, s in cs if s["flood_freq"] is not None), default=None)
        p["pond_depth_cm"], _ = _weighted([(c["comppct_r"], s["pond_depth_cm"]) for c, s in cs])
        p["ec_class"], p["sodic"] = ec_class(p["ec"]), None if p["sar"] is None else bool(p["sar"] >= SODIC_SAR)
        p["components"] = [{"compname": c["compname"], "comppct_r": c["comppct_r"], "ec": _r(s["ec"], 2),
                            "sar": _r(s["sar"], 2), "months_saturated": s["months_saturated"],
                            "flood_freq": s["flood_freq"], "flood_dur": s["flood_dur"], "pond_freq": s["pond_freq"],
                            "pond_dur": s["pond_dur"]} for c, s in sorted(cs, key=lambda t: -t[0]["comppct_r"])]
        out[mk] = p
    return out


# ------------------------------------------------------------------------------------------------ onto the grid

def unit_index(units: Sequence[Dict], shape: Sequence[int], transform: Dict) -> Tuple[np.ndarray, List[str]]:
    """Each cell's map unit (index into the returned keys, -1 none) from placed SSURGO polygons; the first polygon to
    claim a cell keeps it (polygons do not overlap; a shared edge's cell centre falls in one)."""
    idx = np.full(tuple(shape), -1, np.int32)
    keys: List[str] = []
    for u in units:
        if u["mukey"] not in keys:
            keys.append(u["mukey"])
        k = keys.index(u["mukey"])
        for rings in u["polygons"]:
            (r0, r1, c0, c1), m = MT.polygon_mask(rings, shape, transform)
            sl = (slice(r0, r1), slice(c0, c1))
            idx[sl] = np.where(m & (idx[sl] < 0), k, idx[sl])
    return idx, keys


def gather(idx: np.ndarray, keys: Sequence[str], profiles: Dict[str, Dict], field: str, dtype=np.float32,
           none=np.nan) -> np.ndarray:
    """A per-unit figure on every cell; `none` off the survey and where the unit has no value."""
    lut = np.array([none if (profiles.get(k) or {}).get(field) is None else profiles[k][field] for k in keys] + [none],
                   dtype)
    return lut[np.where(idx >= 0, idx, len(keys))]


def feather_known(field: np.ndarray, transform: Dict, width_m: float = FEATHER_M) -> np.ndarray:
    """The box mean `width_m` wide over the cells that have a value (a survey line is only that accurate); a cell
    without one stays NaN, so a Water or Urban land unit borrows nothing from its neighbour."""
    known = np.isfinite(field)
    if not known.any():
        return field.astype(np.float32)
    num = MT.feather(np.where(known, field, 0.0).astype(np.float32), transform, width_m)
    den = MT.feather(known.astype(np.float32), transform, width_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(known & (den > 0), out, np.nan).astype(np.float32)


def estuarine_grid(wetlands: Sequence[Dict], shape: Sequence[int], transform: Dict) -> np.ndarray:
    """uint8 per cell: `estuarine_code` of the placed NWI polygons over it, the higher where two meet."""
    out = np.zeros(tuple(shape), np.uint8)
    for w in wetlands:
        c = estuarine_code(w.get("attribute"))
        if not c:
            continue
        for rings in w["polygons"]:
            (r0, r1, c0, c1), m = MT.polygon_mask(rings, shape, transform)
            sl = (slice(r0, r1), slice(c0, c1))
            out[sl] = np.where(m, np.maximum(out[sl], c), out[sl])
    return out


# ------------------------------------------------------------------------------------------------ HydroBlocks

def cell_lonlat(frame: Dict, shape: Sequence[int], transform: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Lon/lat [deg] of every cell centre: the four corner centres taken back through the frame's basis
    (sims.plan.ecef_to_geodetic) and interpolated bilinearly between them: 0.19 m from exact at the centre of a 3.5 km
    grid (measured, the Ames test grid), far inside a 30 m HydroBlocks cell."""
    from sims.plan import ecef_to_geodetic
    ny, nx = shape
    t = transform
    xs = (t["x0"] + 0.5 * t["dx"], t["x0"] + (nx - 0.5) * t["dx"])
    ys = (t["y0"] + 0.5 * t["dy"], t["y0"] + (ny - 0.5) * t["dy"])
    o, e, n = frame["anchor_ecef"], frame["basis_east"], frame["basis_north"]
    c = {(a, b): ecef_to_geodetic(*(o[i] + xs[a] * e[i] + ys[b] * n[i] for i in range(3)))[:2]
         for a in (0, 1) for b in (0, 1)}
    u = np.linspace(0.0, 1.0, nx)[None, :] if nx > 1 else np.zeros((1, 1))
    v = np.linspace(0.0, 1.0, ny)[:, None] if ny > 1 else np.zeros((1, 1))
    out = []
    for q in (0, 1):
        out.append(((1 - u) * (1 - v) * c[0, 0][q] + u * (1 - v) * c[1, 0][q] + (1 - u) * v * c[0, 1][q]
                    + u * v * c[1, 1][q]) * np.ones((ny, nx)))
    return out[0], out[1]


def sample_bilinear(a: np.ndarray, gt: Sequence[float], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A lon/lat raster `a` (GDAL geotransform `gt`, rows from the north) at each point, bilinear between the four cell
    centres around it over the ones that have a value; NaN where under HB_MIN_WEIGHT of the weight has one."""
    x0, dx, _, y0, _, dy = gt
    h, w = a.shape
    fc = (lon - x0) / dx - 0.5
    fr = (lat - y0) / dy - 0.5
    c0, r0 = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
    tx, ty = fc - c0, fr - r0
    acc = np.zeros(lon.shape, np.float64)
    wsum = np.zeros(lon.shape, np.float64)
    for di, dj, wt in ((0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx), (1, 0, ty * (1 - tx)), (1, 1, ty * tx)):
        r, c = r0 + di, c0 + dj
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        v = np.full(lon.shape, np.nan, np.float64)
        v[ok] = a[r[ok], c[ok]]
        good = ok & np.isfinite(v)
        acc += np.where(good, wt * np.nan_to_num(v), 0.0)
        wsum += np.where(good, wt, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum >= HB_MIN_WEIGHT, acc / wsum, np.nan).astype(np.float32)


def hydroblocks_on_grid(frame: Dict, shape: Sequence[int], transform: Dict, window: Optional[Callable] = None
                        ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict]:
    """({"wet", "dry"} m3/m3 per cell, info), or (None, info with the reason) when the reader or its layers are absent.

    Args:
        window: soil.hydroblocks.site_window or a stand-in with its signature (lon_min, lat_min, lon_max, lat_max,
            layers=...) returning {layer: float32 [rows, cols], 'geotransform', ...}.
    """
    if window is None:
        try:
            from .hydroblocks import site_window as window
        except ImportError:
            return None, {"applied": False, "reason": "soil.hydroblocks is not in this checkout"}
    bbox = TS.grid_bbox(frame, shape, transform, HB_MARGIN_M)
    try:
        w = window(*bbox, layers=("wet_mean", "dry_mean"))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
        # the layers are built nationally by another job; until they land the survey alone sets the band
        return None, {"applied": False, "reason": f"SMAP-HydroBlocks unreadable: {type(e).__name__}: {str(e)[:300]}"}
    lon, lat = cell_lonlat(frame, shape, transform)
    got = {k: sample_bilinear(np.asarray(w[layer], np.float32), w["geotransform"], lon, lat)
           for k, layer in (("wet", "wet_mean"), ("dry", "dry_mean"))}
    del lon, lat
    n = {k: int(np.isfinite(v).sum()) for k, v in got.items()}
    info = {"source": w.get("source"), "window": list(w.get("window") or []), "geotransform": list(w["geotransform"]),
            "cells_with_value": n, "layers": "wet_mean and dry_mean, bilinear at each cell centre",
            "depth": "0 to 5 cm (the source's top layer; the moisture product's root zone is deeper)"}
    if not any(n.values()):
        return None, dict(info, applied=False, reason="no SMAP-HydroBlocks value under the site")
    return got, dict(info, applied=True)


# ------------------------------------------------------------------------------------------------ seasonality

def relative(theta, wp, fc, theta_s) -> np.ndarray:
    """exposure.water_modes.relative_wetness's piecewise scale at one depth: (theta - wp) / (fc - wp) up to field
    capacity (1), then 1 + (theta - fc) / (theta_s - fc) up to saturation (2); NaN where any input is."""
    th, w, f, s = (np.asarray(x, np.float64) for x in (theta, wp, fc, theta_s))
    with np.errstate(invalid="ignore", divide="ignore"):
        low = (th - w) / np.maximum(f - w, 1e-3)
        high = 1.0 + (th - f) / np.maximum(s - f, 1e-3)
        out = np.clip(np.where(th <= f, low, high), 0.0, SATURATION)
    return np.where(np.isfinite(th) & np.isfinite(w) & np.isfinite(f) & np.isfinite(s), out, np.nan).astype(np.float32)


def retention(idx: np.ndarray, keys: Sequence[str], profiles: Dict[str, Dict], modes: Dict
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """(wp, fc, theta_s) m3/m3 per cell from the survey; the gaps from modes.json's site medians of wp and theta_s and
    the survey's site median fc, else fc midway between wp and theta_s (a site wholly Urban land, UC 2026-10-28, has
    no survey retention at all; fc only sets the swing's band width there); all three NaN where they do not order
    wp < fc < theta_s."""
    wp, fc, ts = (gather(idx, keys, profiles, k) for k in ("wp", "fc", "theta_s"))
    surveyed = np.isfinite(wp) & np.isfinite(fc) & np.isfinite(ts)
    pair = ((modes.get("drainage") or {}).get("scale") or {}).get("absolute_m3m3", {}) or {}
    pair = pair.get("soil_wp_theta_s")
    fc_site = float(np.nanmedian(fc)) if np.isfinite(fc).any() else None
    if pair and len(pair) == 2 and all(_num(v) is not None for v in pair):
        wp = np.where(np.isfinite(wp), wp, np.float32(pair[0]))
        ts = np.where(np.isfinite(ts), ts, np.float32(pair[1]))
    if fc_site is not None:
        fc = np.where(np.isfinite(fc), fc, np.float32(fc_site))
    mid = ~np.isfinite(fc) & np.isfinite(wp) & np.isfinite(ts)
    fc = np.where(mid, 0.5 * (wp + ts), fc)
    ok = (wp < fc) & (fc < ts)
    wp, fc, ts = (np.where(ok, a, np.nan).astype(np.float32) for a in (wp, fc, ts))
    return wp, fc, ts, {"surveyed_cells": int(surveyed.sum()), "usable_cells": int(ok.sum()),
                        "fallback_wp_theta_s": pair, "fallback_fc_site_median": _r(fc_site, 4),
                        "fc_midway_cells": int((mid & ok).sum())}


def swing(wet, dry, annual, wp, fc, theta_s) -> Dict[str, np.ndarray]:
    """The surface's seasonal swing about its own mean on the relative scale: {"down" <= 0, "up" >= 0}. A change in
    water content moves the relative value by itself over the width of the band the cell's annual value sits in
    (fc - wp below field capacity, theta_s - fc above), so a surface drier than the wilting point in both seasons
    still swings (a California site, 2026-10-28: both seasons' means under its clay's wilting point read no swing when clipped
    to the scale first)."""
    w, d = (np.asarray(x, np.float64) for x in (wet, dry))
    mid = 0.5 * (w + d)
    band = np.where(np.asarray(annual) < FIELD_CAPACITY, np.asarray(fc) - np.asarray(wp),
                    np.asarray(theta_s) - np.asarray(fc))
    band = np.maximum(band, 1e-3)
    with np.errstate(invalid="ignore"):
        return {"down": ((np.fmin(w, d) - mid) / band).astype(np.float32),
                "up": ((np.fmax(w, d) - mid) / band).astype(np.float32)}


def seasonal(annual: np.ndarray, months_sat: np.ndarray, flood_freq: np.ndarray, pond_freq: np.ndarray,
             surface: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(dry-season low, wet-season high) on the 0..2 scale per cell: the module docstring's SEASONALITY rule.

    Args:
        annual: The published typical-year moisture, NaN off plantable ground.
        months_sat: months_saturated_30cm, NaN where unknown.
        flood_freq, pond_freq: uint8 class codes, NONE_U8 where unknown.
        surface: {"down", "up"} the seasonal swing on the relative scale (`swing`), or None.
    """
    lo = np.asarray(annual, np.float32).copy()
    hi = lo.copy()
    if surface is not None:                            # a cell without a surface value keeps the annual
        lo = np.clip(lo + np.nan_to_num(surface["down"]), 0.0, SATURATION)
        hi = np.clip(hi + np.nan_to_num(surface["up"]), 0.0, SATURATION)
    m = np.nan_to_num(np.asarray(months_sat, np.float32), nan=-1.0)
    ff, pf = np.asarray(flood_freq), np.asarray(pond_freq)
    ff = np.where(ff == NONE_U8, 0, ff)
    pf = np.where(pf == NONE_U8, 0, pf)
    lo = np.where(m >= ALL_YEAR_MONTHS, np.fmax(lo, SAT_FLOOR), lo)
    lo = np.where(ff >= FLOOD_FREQ["very frequent"], np.fmax(lo, VERY_FREQUENT_FLOOR), lo)
    hi = np.where(m >= ANY_MONTH, np.fmax(hi, SAT_FLOOR), hi)
    hi = np.where((ff >= FLOOD_FREQ["frequent"]) | (pf >= POND_FREQ["frequent"]), SATURATION, hi)
    hi = np.fmax(hi, lo)
    plant = np.isfinite(annual)
    return (np.where(plant, lo, np.nan).astype(np.float32), np.where(plant, hi, np.nan).astype(np.float32))


# ------------------------------------------------------------------------------------------------ the whole site


FOLDER = "sims/v1/exposure/soil_conditions"
"""Where a site's layers are published, beside exposure/water/modes; the sidecar is site_conditions.json."""
CODES = {"0-250": "value = lo + code / 250 * (hi - lo) (the layer's scale); a class layer: its class code (classes)",
         "251": "sealed ground (paving, vehicles): not plantable, grey",
         "252": "a building (its roof, a planted roof included): not plantable, grey",
         "253": "open water (a pond, a lake, a creek, a pool): water, drawn in the water color",
         "255": "no value"}
"""exposure/water/modes drainage_u8.bin's coding, word for word on 251 to 255, so one reader serves both."""

SALINITY_HI = 32.0
"""dS/m at code 250: twice the strongly saline threshold; tidal marsh soils above it (one at 62) clip, as the scale
says. No tolerance class reaches past 16, so clipping changes no plant's gate."""

PRODUCT = {
    "salinity": {"file": "salinity_u8.bin", "kind": "value", "field": "ec_dS_m", "lo": 0.0, "hi": SALINITY_HI,
                 "unit": "dS/m", "ticks": [0, 2, 4, 8, 16, 32],
                 "legend": {"title": "Soil salinity", "lo_label": "Fresh", "hi_label": "Strongly saline",
                            "note": "Electrical conductivity of the top 30 cm of soil, in dS/m. Under 2 is nonsaline. "
                                    "16 or more is strongly saline. Values above 32 show as 32."}},
    "tidal": {"file": "tidal_u8.bin", "kind": "class", "field": "nwi_estuarine",
              "classes": [{"code": 0, "label": "Not tidal"}, {"code": 1, "label": "Tidal marsh or flat"},
                          {"code": 2, "label": "Tidal water"}],
              "legend": {"title": "Tidal wetlands", "note": "Estuarine wetlands mapped by the National Wetlands "
                                                            "Inventory. Tidal ground is usually saline."}},
    "flooding": {"file": "flooding_u8.bin", "kind": "value", "field": "months_saturated_30cm", "lo": 0.0, "hi": 12.0,
                 "unit": "months a year", "ticks": [0, 3, 6, 9, 12],
                 "legend": {"title": "Water table near the surface", "lo_label": "Never", "hi_label": "All year",
                            "note": "Months each year the water table is within 30 cm of the surface, from the soil "
                                    "survey's monthly records. Flood and pond frequency are given for each point."}},
    "flood_freq": {"file": "flood_freq_u8.bin", "kind": "class", "field": "flood_freq",
                   "classes": [{"code": 0, "label": "None"}, {"code": 1, "label": "Very rare", "chance": "under 1 % a year"},
                               {"code": 2, "label": "Rare", "chance": "1 to 5 % a year"},
                               {"code": 3, "label": "Occasional", "chance": "5 to 50 % a year"},
                               {"code": 4, "label": "Frequent", "chance": "over 50 % a year"},
                               {"code": 5, "label": "Very frequent", "chance": "over 50 % in every month"}],
                   "legend": {"title": "Flooding frequency", "note": "How often streams, lakes or tides flood the "
                                                                     "ground, by the soil survey's class."}},
    "pond_freq": {"file": "pond_freq_u8.bin", "kind": "class", "field": "pond_freq",
                  "classes": [{"code": 0, "label": "None"}, {"code": 1, "label": "Rare", "chance": "1 to 5 times in 100 years"},
                              {"code": 2, "label": "Occasional", "chance": "5 to 50 times in 100 years"},
                              {"code": 3, "label": "Frequent", "chance": "over 50 times in 100 years"}],
                  "legend": {"title": "Ponding frequency", "note": "How often water stands in a closed low spot, "
                                                                   "by the soil survey's class."}},
    "seasonal_dry": {"file": "seasonal_dry_u8.bin", "kind": "value", "field": "moisture_dry_low", "lo": 0.0,
                     "hi": 2.0, "unit": "relative to each soil", "ticks": [0.0, 0.5, 1.0, 1.5, 2.0],
                     "legend": {"title": "Soil moisture, dry season", "lo_label": "Dry", "fc_label": "Field capacity",
                                "hi_label": "Saturated",
                                "note": "The root zone in the driest season. 0 is the wilting point, 1 is field "
                                        "capacity and 2 is saturation, each for this soil."}},
    "seasonal_wet": {"file": "seasonal_wet_u8.bin", "kind": "value", "field": "moisture_wet_high", "lo": 0.0,
                     "hi": 2.0, "unit": "relative to each soil", "ticks": [0.0, 0.5, 1.0, 1.5, 2.0],
                     "legend": {"title": "Soil moisture, wet season", "lo_label": "Dry", "fc_label": "Field capacity",
                                "hi_label": "Saturated",
                                "note": "The root zone in the wettest season. Ground that floods or holds a water "
                                        "table near the surface reads saturated."}},
}

BUTTONS = [
    {"id": "salinity", "label": "Salinity", "layer": "salinity", "marks": ["tidal"]},
    {"id": "flooding", "label": "Flooding", "layer": "flooding", "at_point": ["flood_freq", "pond_freq"]},
    {"id": "seasonal", "label": "Seasonal water",
     "views": [{"id": "dry", "label": "Dry season", "layer": "seasonal_dry"},
               {"id": "wet", "label": "Wet season", "layer": "seasonal_wet"}]},
]
"""Three SOIL modes beside pH, Type and Organic; Seasonal water has two views, one raster each."""


def data_sources(sources: Dict, hb_info: Dict) -> List[Dict]:
    """The DATA panel's rows for the three modes, in the viewer's own shape (simulation.js dataSources: name, by,
    detail, license), each also carrying its resolution, date and method for a fuller note."""
    day = lambda k: (sources.get("fetched") or {}).get(k, "")[:10]  # noqa: E731
    out = [
        {"group": "Soil", "mode": "salinity", "name": "Soil salinity",
         "by": "USDA Natural Resources Conservation Service, Soil Survey (SSURGO)", "license": "Public domain",
         "detail": "electrical conductivity of the top 30 cm, by soil map unit",
         "resolution": "soil map units mapped at 1:12,000 to 1:24,000, drawn on the site's water grid",
         "date": f"read {day('ssurgo')}",
         "method": "Each soil's horizons are averaged over 0 to 30 cm by thickness. A map unit's soils are averaged "
                   "by their share of the unit."},
        {"group": "Soil", "mode": "salinity", "name": "Tidal wetlands",
         "by": "U.S. Fish and Wildlife Service, National Wetlands Inventory", "license": "Public domain",
         "detail": "estuarine marsh, flats and water",
         "resolution": "wetland polygons mapped from aerial imagery", "date": f"read {day('nwi')}",
         "method": "Estuarine polygons (Cowardin class E) mark tidal ground."},
        {"group": "Soil", "mode": "flooding", "name": "Flooding and water table",
         "by": "USDA Natural Resources Conservation Service, Soil Survey (SSURGO)", "license": "Public domain",
         "detail": "months a year with water within 30 cm, flood and pond frequency",
         "resolution": "soil map units mapped at 1:12,000 to 1:24,000", "date": f"read {day('ssurgo')}",
         "method": "The survey records each soil's water table, flooding and ponding by month. Each map unit takes "
                   "the condition that covers most of it."},
    ]
    sw = {"group": "Soil", "mode": "seasonal", "name": "Seasonal soil moisture", "license": "Public domain",
          "detail": "the driest and the wettest season's root zone",
          "resolution": "the site's water grid", "date": f"read {day('ssurgo')}",
          "method": "The year's soil moisture is widened by the soil survey's water table, flooding and ponding."}
    if hb_info.get("applied"):
        sw.update(by="NASA SMAP soil moisture at 30 m (SMAP-HydroBlocks, Vergopolan et al. 2021) and the USDA soil "
                     "survey", license="CC BY 4.0 and public domain", date="2015 to 2019",
                  resolution="30 m, sampled between pixel centres",
                  method="Seasonal means of the top 5 cm are put on each soil's scale from wilting point to "
                         "saturation. The year's soil moisture is widened by them and by the soil survey's water "
                         "table, flooding and ponding.")
    else:
        sw.update(by="The Simulator's soil moisture and the USDA soil survey (SSURGO)")
    out.append(sw)
    return out


def _stats(v: np.ndarray, plant: np.ndarray) -> Optional[Dict]:
    x = v[plant & np.isfinite(v)]
    if not x.size:
        return None
    q = np.percentile(x, [0, 10, 50, 90, 100])
    return {"cells": int(x.size), "min_p10_p50_p90_max": [round(float(a), 3) for a in q]}


def _hist(v: np.ndarray, plant: np.ndarray) -> Dict[str, int]:
    u, n = np.unique(v[plant], return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(u, n)}


def build(modes: Dict, codes: np.ndarray, frame: Dict, sources: Dict, hb_window: Optional[Callable] = None,
          use_hydroblocks: bool = True) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Every field on the modes grid, in its own units (float32 values, NaN unknown; uint8 classes, NONE_U8 unknown),
    and the record of how they were made; `encode` turns them into the published layers.

    Args:
        modes: modes.json (grid, drainage scale on the relative quantity).
        codes: The published moisture codes [ny * nx] (drainage_u8.bin).
        frame: The scene's frame (truth_sources.frame_of).
        sources: `fetch`'s record, lon/lat.
        hb_window: A stand-in for soil.hydroblocks.site_window (tests); None reads the real one.
    """
    ny, nx = modes["grid"]["shape"]
    tf = modes["grid"]["transform"]
    shape = (ny, nx)
    scale = modes["drainage"]["scale"]
    if scale.get("quantity", MT.RELATIVE) != MT.RELATIVE:
        raise ValueError(f"the seasonal band is on the relative scale; this grid is {scale.get('quantity')}")
    c = np.asarray(codes, np.uint8).reshape(shape)
    plant = c <= MT.CODE_TOP
    annual = np.where(plant, MT.decode(c, scale), np.nan).astype(np.float32)

    placed = place(sources, frame)
    prof = unit_profiles(sources["horizons"], sources["months"])
    for u in sources["units"]:                         # the name check, beside the numbers
        p = prof.setdefault(u["mukey"], {})
        p["muname"] = u.get("muname")
        p["name_saline_or_tidal"] = bool(SALINE_NAME.search(u.get("muname") or ""))
    idx, keys = unit_index(placed["units"], shape, tf)
    F: Dict[str, np.ndarray] = {}
    F["ec_dS_m"] = feather_known(gather(idx, keys, prof, "ec"), tf)
    F["sar"] = feather_known(gather(idx, keys, prof, "sar"), tf)
    F["nwi_estuarine"] = estuarine_grid(placed["estuarine"], shape, tf)
    F["months_saturated_30cm"] = feather_known(gather(idx, keys, prof, "months_saturated_30cm"), tf)
    for k in ("flood_freq", "flood_dur", "pond_freq", "pond_dur"):
        F[k] = gather(idx, keys, prof, k, np.uint8, NONE_U8)
    F["map_unit"] = idx

    hb, hb_info = (None, {"applied": False, "reason": "not asked (--no-hydroblocks)"})
    if use_hydroblocks:
        hb, hb_info = hydroblocks_on_grid(frame, shape, tf, hb_window)
    wp, fc, ts, ret_info = retention(idx, keys, prof, modes)
    surface = None
    if hb is not None:
        F["hb_wet_m3m3"], F["hb_dry_m3m3"] = hb["wet"], hb["dry"]
        surface = swing(hb["wet"], hb["dry"], annual, wp, fc, ts)
        d = hb["wet"] - hb["dry"]
        wetter = float(np.nanmean(d)) if np.isfinite(d).any() else None
        hb_info["wet_minus_dry_mean_m3m3"] = _r(wetter, 4)
        hb_info["wetter_season"] = None if wetter is None else ("wet_mean" if wetter >= 0 else "dry_mean")
        for k in ("wet", "dry"):                       # the seasons' own level, for the record only
            hb_info[f"{k}_relative"] = _stats(relative(hb[k], wp, fc, ts), plant)
        for k, v in surface.items():
            hb_info[f"swing_{k}"] = _stats(v, plant)
    del wp, fc, ts
    F["moisture_dry_low"], F["moisture_wet_high"] = seasonal(annual, F["months_saturated_30cm"], F["flood_freq"],
                                                             F["pond_freq"], surface)

    stats = {k: _stats(v, plant) for k, v in F.items() if v.dtype.kind == "f"}
    stats["annual"] = _stats(annual, plant)
    for k in ("nwi_estuarine", "flood_freq", "flood_dur", "pond_freq", "pond_dur"):
        stats[k] = _hist(F[k], plant)
    surveyed = idx >= 0
    tidal_name = np.array([bool((prof.get(k) or {}).get("name_saline_or_tidal")) for k in keys] + [False])
    units = []
    for i, k in enumerate(keys):
        p = prof.get(k) or {}
        u = {k_: v for k_, v in p.items() if k_ != "components"}
        for k_ in ("ec", "sar", "wp", "fc", "theta_s", "months_saturated_30cm", "pond_depth_cm"):
            if k_ in u:
                u[k_] = _r(u[k_], 4 if k_ in ("wp", "fc", "theta_s") else 2)
        u.update(mukey=k, cells=int((idx == i).sum()), plantable_cells=int(((idx == i) & plant).sum()),
                 components=p.get("components"))
        units.append(u)
    side = {
        "v": 1, "site": modes.get("site"), "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": modes["grid"], "rows": "the modes grid's own order (row 0 at y0, the north edge)",
        "sources": sources["sources"], "fetched": sources["fetched"], "bbox": sources["bbox"],
        "requests": sources.get("requests"), "rules": RULES,
        "cells": {"grid": ny * nx, "plantable": int(plant.sum()), "surveyed": int(surveyed.sum()),
                  "plantable_surveyed": int((plant & surveyed).sum()),
                  "plantable_name_saline_or_tidal": int((plant & tidal_name[np.where(surveyed, idx, len(keys))]).sum()),
                  "plantable_nwi_estuarine": int((plant & (F["nwi_estuarine"] > 0)).sum())},
        "stats_plantable": stats, "hydroblocks": hb_info, "retention": ret_info,
        "nwi": {"polygons_all": sources.get("nwi_polygons_all"), "estuarine": [w["attribute"] for w in sources["estuarine"]]},
        "map_units": units,
    }
    return F, side


def encode(F: Dict[str, np.ndarray], codes: np.ndarray, side: Dict) -> Tuple[Dict[str, bytes], Dict]:
    """The published layers: {file: bytes} and the sidecar (site_conditions.json) with each layer's coding, legend and
    DATA row. Every layer is coded as drainage_u8.bin: 251, 252 and 253 where the moisture product has them, 255
    where it has no value or the layer has none; a value layer 0..250 on its fixed scale (MT.encode, its clip), a
    class layer its class code."""
    c = np.asarray(codes, np.uint8).ravel()
    keep = c > MT.CODE_TOP                                 # 251, 252, 253 and 255 travel unchanged
    files, layers = {}, {}
    for name, p in PRODUCT.items():
        v = F[p["field"]].ravel()
        if p["kind"] == "value":
            q = MT.encode(v, (p["lo"], p["hi"]))
            meta = {"scale": {"lo": p["lo"], "hi": p["hi"], "unit": p["unit"], "ticks": p["ticks"],
                              "transform": "linear",
                              "clip": "fixed on every site; values past an end take its code (0 or 250)"}}
        else:
            q = v.astype(np.uint8)                         # NONE_U8 is 255: no record reads no value
            meta = {"classes": p["classes"]}
        q = np.where(keep, c, q).astype(np.uint8)
        files[p["file"]] = q.tobytes()
        layers[name] = dict(meta, file=p["file"], gz=p["file"] + ".gz", dtype="uint8", kind=p["kind"], codes=CODES,
                            legend=p["legend"], field=p["field"])
    out = dict(side, folder=FOLDER, layers=layers, buttons=BUTTONS,
               data_sources=data_sources({"fetched": side["fetched"]}, side["hydroblocks"]))
    return files, out
