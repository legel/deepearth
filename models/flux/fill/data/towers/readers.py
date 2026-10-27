"""Readers for the four source formats. Each returns a `Record`: TIMESTAMP_START in minutes, the step, the clock
convention, and for every dataset variable the measured values with the source's own fill flag.

Clock conventions:
- ONEFlux, FLUXNET-CH4 and AmeriFlux BASE: local standard time (no daylight saving), TIMESTAMP_START and
  TIMESTAMP_END as YYYYMMDDHHMM; the UTC offset is the site's BIF / BADM `UTC_OFFSET`.
- ICOS ETC L2: local standard time too. The offset is the difference between the file's first TIMESTAMP_START and the
  acquisition start in UTC that the Carbon Portal publishes for the object (UTC+2 in Finland and Greece, UTC+1 in
  Sweden and Switzerland).
"""

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import qc as Q
from variables import ERA5_COLUMNS, MAX_PROFILE, UNCERTAINTY, VARIABLES, columns

EPOCH = np.datetime64("1970-01-01T00:00")


def minutes(stamp) -> np.ndarray:
    """YYYYMMDDHHMM integers to minutes since 1970-01-01 (clock-naive)."""
    s = pd.to_datetime(pd.Series(np.asarray(stamp, np.int64)).astype(str), format="%Y%m%d%H%M")
    return ((s.values - EPOCH) // np.timedelta64(1, "m")).astype(np.int64)


@dataclass
class Series:
    value: np.ndarray
    """Measured values (NaN where not measured)."""
    filled: np.ndarray
    """True where the source holds a gap-filled value instead of a measurement."""
    column: str
    """The source column read."""
    quality: Optional[np.ndarray] = None
    """True where the source's own quality test failed (SSITC flag 2; Foken et al. 2004)."""


@dataclass
class Record:
    source: str
    site_id: str
    t: np.ndarray
    """TIMESTAMP_START, minutes since 1970-01-01 in the source's clock."""
    step: int
    utc_offset_h: Optional[float]
    """Hours to add to UTC to get the record's clock (the site's standard-time offset; +1 for ICOS L2)."""
    series: Dict[str, Series] = field(default_factory=dict)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)
    """Non-target columns: ERA5 drivers, SW_IN_POT, random uncertainty."""
    meta: Dict = field(default_factory=dict)


def _read_csv(z: zipfile.ZipFile, name: str, skip_comments: bool = False) -> pd.DataFrame:
    with z.open(name) as f:
        raw = f.read()
    if skip_comments:
        lines = raw.split(b"\n")
        k = 0
        while k < len(lines) and lines[k].startswith(b"#"):
            k += 1
        raw = b"\n".join(lines[k:])
    # The pyarrow parser reads these files 4 times faster than the C parser, with identical values (checked on
    # every column of an ONEFlux FLUXMET HH file).
    return pd.read_csv(io.BytesIO(raw), na_values=["-9999", "-9999.0"], engine="pyarrow")


def _member(z: zipfile.ZipFile, pattern: str) -> Optional[str]:
    hits = [n for n in z.namelist() if re.search(pattern, n)]
    return sorted(hits)[0] if hits else None


# ---- ONEFlux / FLUXNET Shuttle --------------------------------------------------------------------------------

def parse_bif(df: pd.DataFrame) -> Dict:
    """Site metadata from a BIF table (SITE_ID, GROUP_ID, VARIABLE_GROUP, VARIABLE, DATAVALUE)."""
    g = {gid: dict(zip(x.VARIABLE, x.DATAVALUE)) | {"_group": x.VARIABLE_GROUP.iloc[0]}
         for gid, x in df.groupby("GROUP_ID")}
    out: Dict = {"locations": [], "team": [], "ustar_vut": {}, "ustar_cut": None, "era_fit": {}}
    for d in g.values():
        grp = str(d["_group"]).removeprefix("GRP_")  # networks write LOCATION or GRP_LOCATION, TEAM or TEAM_MEMBER
        if grp == "LOCATION":
            out["locations"].append({"date_start": str(d.get("LOCATION_DATE_START", "")),
                                     "lat": float(d["LOCATION_LAT"]), "lon": float(d["LOCATION_LONG"]),
                                     "elev": float(d["LOCATION_ELEV"]) if pd.notna(d.get("LOCATION_ELEV")) else None,
                                     "comment": d.get("LOCATION_COMMENT")})
        elif grp in ("TEAM_MEMBER", "TEAM"):
            out["team"].append({"name": d.get("TEAM_MEMBER_NAME"), "role": d.get("TEAM_MEMBER_ROLE"),
                                "institution": d.get("TEAM_MEMBER_INSTITUTION")})
        elif grp == "FLUX_REF" and d.get("REF_VARIABLE") == "NEE" and d.get("REF_TIME_RESOLUTION") in ("HH", "HR"):
            thr = float(d["REF_USTAR_THRESHOLD"])
            if d.get("REF_VERSION") == "VUT" and pd.notna(d.get("REF_YEAR")):
                out["ustar_vut"][int(float(d["REF_YEAR"]))] = thr
            elif d.get("REF_VERSION") == "CUT":
                out["ustar_cut"] = thr
        elif grp == "ERA_DOWN":
            out["era_fit"][d["ERA_VARIABLE"]] = {k: float(v) for k, v in d.items()
                                                  if k.startswith("ERA_") and k != "ERA_VARIABLE"}
        else:
            for k, v in d.items():
                if k != "_group":
                    out.setdefault(k, v)
    out["locations"].sort(key=lambda r: r["date_start"])
    return out


def parse_varinfo(df: pd.DataFrame) -> pd.DataFrame:
    """Instrument records from BIFVARINFO: one row per (variable, date) with model and height."""
    rows = []
    for gid, x in df.groupby("GROUP_ID"):
        d = dict(zip(x.VARIABLE, x.DATAVALUE))
        if "VAR_INFO_VARNAME" in d:
            rows.append({"varname": d["VAR_INFO_VARNAME"], "date": str(d.get("VAR_INFO_DATE", "")),
                         "model": d.get("VAR_INFO_MODEL"), "height": d.get("VAR_INFO_HEIGHT"),
                         "unit": d.get("VAR_INFO_UNIT")})
    return pd.DataFrame(rows, columns=["varname", "date", "model", "height", "unit"])


def read_oneflux(zip_path: Path, start_minute: int) -> Record:
    """A Shuttle archive: FLUXMET HH (or HR), ERA5 at the same resolution, BIF, BIFVARINFO."""
    with zipfile.ZipFile(zip_path) as z:
        hh = _member(z, r"_FLUXMET_H[HR]_")
        era = _member(z, r"_ERA5_H[HR]_")
        bif = _member(z, r"_BIF_\d")
        vinfo = _member(z, r"_BIFVARINFO_H[HR]_")
        d = _read_csv(z, hh)
        e = _read_csv(z, era) if era else None
        b = parse_bif(pd.read_csv(z.open(bif))) if bif else {}
        vi = parse_varinfo(pd.read_csv(z.open(vinfo))) if vinfo else pd.DataFrame()
    site = re.search(r"_([A-Z]{2}-[A-Za-z0-9]{3})_", Path(hh).name).group(1)
    t = minutes(d.TIMESTAMP_START)
    keep = t >= start_minute
    d, t = d.loc[keep].reset_index(drop=True), t[keep]
    step = Q.regular(t)["step"] or (60 if "_HR_" in hh else 30)
    rec = Record("shuttle", site, t, step, float(b["UTC_OFFSET"]) if "UTC_OFFSET" in b else None,
                 meta={"bif": b, "varinfo": vi, "file": Path(hh).name})
    for name, var in VARIABLES.items():
        if var.oneflux is None:
            continue
        ks = range(1, MAX_PROFILE + 1) if var.profile else [None]
        for k in ks:
            vc, qc = columns(name, k) if var.profile else var.oneflux
            if vc not in d:
                continue
            raw = Q.clean(d[vc].values)
            q = d[qc].values if qc and qc in d else None
            ok = Q.measured(raw, q)
            key = f"{name}_{k}" if var.profile else name
            rec.series[key] = Series(np.where(ok, raw, np.nan), np.isfinite(raw) & ~ok, vc)
    for name, col in UNCERTAINTY.items():
        if col in d and name in rec.series:
            u = Q.clean(d[col].values)
            rec.extra[f"{name}_RANDUNC"] = np.where(np.isfinite(rec.series[name].value), u, np.nan)
    if "SW_IN_POT" in d:
        rec.extra["SW_IN_POT"] = Q.clean(d.SW_IN_POT.values)
    if e is not None:
        te = minutes(e.TIMESTAMP_START)
        idx = pd.Index(te).get_indexer(t)
        for c in ERA5_COLUMNS:
            if c in e:
                v = Q.clean(e[c].values)
                rec.extra[f"ERA5_{c[:-4]}"] = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
        rec.meta["era5_years"] = (int(str(e.TIMESTAMP_START.iloc[0])[:4]), int(str(e.TIMESTAMP_START.iloc[-1])[:4]))
    return rec


# ---- FLUXNET-CH4 V1.0 -------------------------------------------------------------------------------------------

CH4_COLUMNS = {"FCH4": "FCH4", "NEE": "NEE", "LE": "LE", "H": "H", "G": "G", "TA": "TA", "SW_IN": "SW_IN",
               "NETRAD": "NETRAD", "PPFD_IN": "PPFD_IN", "PPFD_OUT": "PPFD_OUT", "VPD": "VPD", "RH": "RH",
               "PA": "PA", "P": "P", "WS": "WS", "WD": "WD", "USTAR": "USTAR", "WTD": "WTD",
               "TS_1": "TS_1", "TS_2": "TS_2", "TS_3": "TS_3", "TS_4": "TS_4", "TS_5": "TS_5"}
"""The unfilled FLUXNET-CH4 columns (Delwiche et al. 2021, Table B1). `_F`, `_F_ANNOPTLM`, `_DT`, `_NT` and
`_RANDUNC` columns are fills or model products and are never read."""


def read_fluxnet_ch4(zip_path: Path, start_minute: int) -> Record:
    with zipfile.ZipFile(zip_path) as z:
        hh = _member(z, r"_H[HR]_\d")
        d = _read_csv(z, hh)
    site = re.search(r"FLX_([A-Z]{2}-[A-Za-z0-9]{3})_", Path(hh).name).group(1)
    t = minutes(d.TIMESTAMP_START)
    keep = t >= start_minute
    d, t = d.loc[keep].reset_index(drop=True), t[keep]
    rec = Record("fluxnet_ch4", site, t, Q.regular(t)["step"] or 30, None, meta={"file": Path(hh).name})
    for key, col in CH4_COLUMNS.items():
        if col in d:
            v = Q.clean(d[col].values)
            rec.series[key] = Series(v, np.zeros(v.shape, bool), col)
    return rec


# ---- AmeriFlux BASE ---------------------------------------------------------------------------------------------

BASE_SKIP = re.compile(r"(_F$|_F_|_PI_F|_SD$|_N$|_QC$|_SSITC_TEST$|_IU$|_ROT$|_RANDUNC|_MDS|_ANN|_DT|_NT)")


def base_column(cols: List[str], name: str) -> Optional[str]:
    """The measured BASE column of a variable: the unqualified name, else the lowest positional qualifier
    (NAME_1_1_1 before NAME_1_2_1), else a layer-integrated NAME_1. PI gap-fills (_F), statistics (_SD, _N) and
    test columns are never chosen; a PI-provided `_PI` variant is used only when nothing else exists."""
    if name in cols:
        return name
    pat = re.compile(rf"^{re.escape(name)}(_\d+)+$")
    cand = sorted([c for c in cols if pat.match(c)], key=lambda c: [int(x) for x in c[len(name) + 1:].split("_")])
    if cand:
        return cand[0]
    pi = [c for c in cols if c == f"{name}_PI" or re.match(rf"^{re.escape(name)}_PI(_\d+)+$", c)]
    return sorted(pi)[0] if pi else None


def read_amf_base(zip_path: Path, start_minute: int) -> Record:
    with zipfile.ZipFile(zip_path) as z:
        hh = _member(z, r"_BASE_H[HR]_")
        d = _read_csv(z, hh, skip_comments=True)
        badm = _member(z, r"BIF.*\.(xlsx|csv)$")
    site = re.search(r"AMF_([A-Z]{2}-[A-Za-z0-9]{3})_", Path(hh).name).group(1)
    t = minutes(d.TIMESTAMP_START)
    keep = t >= start_minute
    d, t = d.loc[keep].reset_index(drop=True), t[keep]
    rec = Record("amf_base", site, t, Q.regular(t)["step"] or (60 if "_HR_" in hh else 30), None,
                 meta={"file": Path(hh).name, "badm": badm})
    cols = list(d.columns)
    names = [n for n in VARIABLES if n not in ("TS", "SWC")] + ["FC", "SC"]
    for name in names:
        c = base_column(cols, name)
        if c is None:
            continue
        v = Q.clean(d[c].values)
        flag = d.get(f"{c}_SSITC_TEST")
        bad = (Q.clean(flag.values) == 2) if flag is not None else None
        rec.series[name] = Series(v, np.zeros(v.shape, bool), c, bad)
    for name in ("TS", "SWC"):
        for k in range(1, MAX_PROFILE + 1):
            c = next((x for x in (f"{name}_{k}", f"{name}_1_{k}_1", f"{name}_PI_1_{k}_1") if x in cols), None)
            if c:
                v = Q.clean(d[c].values)
                rec.series[f"{name}_{k}"] = Series(v, np.zeros(v.shape, bool), c)
    # NEE as ONEFlux forms it: FC plus the storage flux SC where SC is measured, FC alone where the site has no
    # storage measurement (short canopies, Pastorello et al. 2020). A site-submitted NEE (often NEE_PI, whose u*
    # and storage treatment is the PI's) is used only when the site submits no FC.
    if "FC" in rec.series:
        fc = rec.series["FC"]
        sc = rec.series.get("SC")
        if sc is not None and np.isfinite(sc.value).mean() > 0.5 * max(np.isfinite(fc.value).mean(), 1e-9):
            v = fc.value + sc.value
            col = f"{fc.column}+{sc.column}"
        else:
            v, col = fc.value, fc.column
        rec.series["NEE"] = Series(v, np.zeros(v.shape, bool), col, fc.quality)
    rec.series.pop("FC", None)
    rec.series.pop("SC", None)
    return rec


# ---- ICOS ETC L2 Fluxes -----------------------------------------------------------------------------------------

ICOS_COLUMNS = ("FCH4", "NEE", "LE", "H", "USTAR")
"""The cleaned flux columns. No ETC L2 Fluxes object carries FCH4 (checked on all 93 stations, 2026-10-26); the
column is read if a later release adds it."""


def read_icos_l2(zip_path: Path, start_minute: int) -> Record:
    with zipfile.ZipFile(zip_path) as z:
        name = _member(z, r"_FLUXES_L2\.csv$")
        d = _read_csv(z, name)
    site = re.search(r"ICOSETC_([A-Z]{2}-[A-Za-z0-9]{3})_", name).group(1)
    t = minutes(d.TIMESTAMP_START)
    keep = t >= start_minute
    d, t = d.loc[keep].reset_index(drop=True), t[keep]
    rec = Record("icos_l2", site, t, Q.regular(t)["step"] or 30, None, meta={"file": name})
    for c in ICOS_COLUMNS:
        if c in d:
            v = Q.clean(d[c].values)
            flag = d.get(f"{c}_SSITC_TEST")
            bad = (Q.clean(flag.values) == 2) if flag is not None else None
            rec.series[c] = Series(v, np.zeros(v.shape, bool), c, bad)
    return rec
