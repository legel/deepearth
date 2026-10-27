"""Build the tower dataset in two stages.

Stage 1, per source and site (`clean`): read the archive, keep measured values only, apply the quality rules in
`qc.py`, and write an interim table <out>/interim/<source>/<SITE>.parquet with values, flags and a JSON sidecar
(tests, counts, metadata). Shuttle archives are downloaded, cleaned and deleted one at a time (`shuttle`).

Stage 2, per site (`merge`): put every source on one UTC grid, choose one source per variable and year by a fixed
priority, and write <out>/hh/<SITE>.parquet. Then `tables` writes the coverage, QC-count, event and window tables
and the final site table, and `split.py` freezes the site split.

    python build.py shuttle --raw R --out O --workers 3      # download, clean and delete every Shuttle archive
    python build.py clean   --raw R --out O --source fluxnet_ch4|amf_base|icos_l2
    python build.py merge   --raw R --out O
    python build.py tables  --raw R --out O
"""

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import qc as Q
import readers as R
import sources as S
from variables import FLUX, VARIABLES, base

START = "2014-01-01"
"""First timestamp of the dataset (local standard time). Sentinel-2 starts in mid-2015, so 2014 to mid-2015 gives
windows without imagery."""
NO_SIGN_TEST_IGBP = {"WAT", "URB", "BSV", "SNO"}
"""Surfaces without a canopy, where NEE need not anti-correlate with light."""
USTAR_SOURCES = {"fluxnet_ch4": {"FCH4"}, "amf_base": {"FCH4", "NEE"}, "icos_l2": {"FCH4", "NEE"}}
"""Variables this dataset u*-filters, per source. ONEFlux NEE_VUT_REF is already u*-filtered (VUT reference
threshold, Pastorello et al. 2020). FLUXNET-CH4 NEE was u*-filtered with REddyProc (Delwiche et al. 2021); its
FCH4 is filtered here. BASE and ICOS L2 fluxes are filtered here. LE and H are never u*-filtered, following ONEFlux."""
PRIORITY = {
    "FCH4": ["fluxnet_ch4", "icos_l2", "amf_base"],
    "WTD": ["fluxnet_ch4", "amf_base"],
}
DEFAULT_PRIORITY = ["shuttle", "fluxnet_ch4", "amf_base", "icos_l2"]
"""One source per (site, variable, year): the first in this order with any measured value that year. ONEFlux is
first for CO2, energy and meteorology because it is the harmonised, u*-filtered product; FLUXNET-CH4 is first for
FCH4 because it is the published, QA/QC'd methane product."""


def start_minute(start: str = START) -> int:
    return int((np.datetime64(f"{start}T00:00") - R.EPOCH) // np.timedelta64(1, "m"))


def years_of(t: np.ndarray) -> np.ndarray:
    return (t.astype("datetime64[m]").astype("datetime64[Y]").astype(int) + 1970).astype(np.int32)


# ---- Stage 1: rules on one record --------------------------------------------------------------------------------

DARK_ELEVATION = 5.0
"""Below this solar elevation (degrees) a shortwave or PAR sensor may read a constant zero or offset (horizon,
terrain, cosine response), so its repeats there are not stuck. A run of zeros from night into twilight is common."""


def night_mask(rec: R.Record, lat: float, lon: float, offset_h: float, below: float = 0.0) -> np.ndarray:
    """Night where the sun is below `below` degrees at the step's midpoint (ONEFlux SW_IN_POT is 0 below 0)."""
    el = Q.solar_elevation(lat, lon, Q.step_midpoints_utc(rec.t, rec.step, offset_h))
    return el < below


def ustar_steps(t: np.ndarray, by_year: Dict[int, float]) -> np.ndarray:
    """Each step's u* threshold: the site-year's value, else the median over the site's years; NaN without any."""
    if not by_year:
        return np.full(t.shape, np.nan)
    med = float(np.median(list(by_year.values())))
    yr = years_of(t)
    return np.array([by_year.get(int(y), med) for y in yr], np.float64)


def infer_ustar(rec: R.Record, night: np.ndarray) -> Dict[int, float]:
    """For a FLUXNET-CH4 site outside ONEFlux: per year, the 0.5th percentile of night-time u* over the steps where
    the record's own u*-filtered NEE is present (REddyProc removed the steps below its threshold, so the lowest u*
    that survives estimates the threshold). Years with fewer than 200 such steps get none."""
    if "NEE" not in rec.series or "USTAR" not in rec.series:
        return {}
    nee, u = rec.series["NEE"].value, rec.series["USTAR"].value
    ok = np.isfinite(nee) & np.isfinite(u) & night
    out = {}
    yr = years_of(rec.t)
    for y in np.unique(yr):
        sel = ok & (yr == y)
        if sel.sum() >= 200:
            out[int(y)] = float(np.percentile(u[sel], 0.5))
    return out


def mpt_by_year(rec: R.Record, night: np.ndarray) -> Dict[int, float]:
    """Per-year moving-point-test thresholds from the record's own night-time NEE, u* and air temperature."""
    if not all(k in rec.series for k in ("NEE", "USTAR", "TA")):
        return {}
    yr = years_of(rec.t)
    month = (rec.t.astype("datetime64[m]").astype("datetime64[M]").astype(int) % 12) + 1
    out = {}
    for y in np.unique(yr):
        sel = yr == y
        thr = Q.ustar_mpt(rec.series["NEE"].value[sel], rec.series["USTAR"].value[sel], rec.series["TA"].value[sel],
                          night[sel], month[sel])
        if thr is not None:
            out[int(y)] = thr
    return out


def apply_rules(rec: R.Record, lat: float, lon: float, offset_h: float, igbp: Optional[str],
                ustar_by_year: Dict[int, float], ustar_vars: set) -> Tuple[Dict, List[Dict], Dict]:
    """Apply every value rule and whole-record test to one record.

    Returns ({key: (value float32, flag uint8)}, count rows, tests). Rules run in the order of qc.FLAG_NAMES; a
    value removed by one rule is not counted again by a later one."""
    night = night_mask(rec, lat, lon, offset_h)
    dark = night_mask(rec, lat, lon, offset_h, DARK_ELEVATION)
    per_day = 1440 // rec.step
    thr = ustar_steps(rec.t, ustar_by_year)
    ustar = rec.series["USTAR"].value if "USTAR" in rec.series else np.full(rec.t.shape, np.nan)
    out, counts = {}, []
    for key, s in rec.series.items():
        var = VARIABLES[base(key)]
        v = s.value.astype(np.float64).copy()
        flag = np.zeros(v.shape, np.uint8)
        n_meas = int(np.isfinite(v).sum())
        flag[s.filled] |= Q.FILLED

        def drop(mask, bit):
            m = mask & np.isfinite(v)
            flag[m] |= bit
            v[m] = np.nan
            return int(m.sum())

        c = {"filled": int(s.filled.sum())}
        c["quality"] = drop(s.quality if s.quality is not None else np.zeros(v.shape, bool), Q.QUALITY)
        c["range"] = drop(Q.out_of_range(v, var.lo, var.hi, var.tol), Q.RANGE)
        run = None if var.stuck_steps is None else max(2, var.stuck_steps * 30 // rec.step)  # thresholds are in half-hours
        c["stuck"] = drop(Q.stuck(v, run, var.stuck_exempt, dark if var.dark_repeats else None, var.exempt_atol),
                          Q.STUCK)
        c["spike"] = drop(Q.spikes(v, night, per_day), Q.SPIKE) if var.spike else 0
        c["ustar"] = drop(Q.low_ustar(ustar, thr), Q.USTAR) if key in ustar_vars else 0
        c["site"] = 0
        out[key] = [v, flag]
        counts.append({"source": rec.source, "site_id": rec.site_id, "variable": key, "column": s.column,
                       "steps": int(v.size), "measured_at_source": n_meas, **c})
    tests = site_tests(rec, out, night, igbp)
    for key, why in tests["removed"].items():
        v, flag = out[key]
        m = np.isfinite(v)
        flag[m] |= Q.SITE
        v[m] = np.nan
        next(c for c in counts if c["variable"] == key)["site"] = int(m.sum())
    for c in counts:
        c["final"] = int(np.isfinite(out[c["variable"]][0]).sum())
    return {k: (v.astype(np.float32), f) for k, (v, f) in out.items()}, counts, tests


def site_tests(rec: R.Record, out: Dict, night: np.ndarray, igbp: Optional[str]) -> Dict:
    """Whole-record tests: clock (per year), flux sign conventions, FCH4 units, timestamp regularity."""
    tests: Dict = {"removed": {}, "timestamps": Q.regular(rec.t), "step_minutes": rec.step}
    sw = out["SW_IN"][0] if "SW_IN" in out else None
    if sw is not None:
        el = Q.solar_elevation(*rec.meta["latlon"], Q.step_midpoints_utc(rec.t, rec.step, rec.meta["offset_h"]))
        yr = years_of(rec.t)
        per_year = {}
        for y in np.unique(yr):
            sel = yr == y
            per_year[int(y)] = Q.clock_offset_minutes(sw[sel], el[sel], rec.step)
        tests["clock_offset_min"] = {"all": Q.clock_offset_minutes(sw, el, rec.step), "by_year": per_year}
    day = ~night
    if sw is not None:
        day = day & (np.nan_to_num(sw, nan=0) > 50)
    drivers = {"NEE": ("SW_IN", -1), "H": ("NETRAD", +1), "LE": ("NETRAD", +1)}
    tests["sign"] = {}
    for flux, (drv, sign) in drivers.items():
        if flux not in out:
            continue
        d = out.get(drv) or out.get("SW_IN")
        if d is None or (flux == "NEE" and igbp in NO_SIGN_TEST_IGBP):
            continue
        res = Q.sign_check(np.where(day, out[flux][0], np.nan), d[0], sign)
        tests["sign"][flux] = res
        if res["flipped"]:
            tests["removed"][flux] = f"sign: r={res['r']:.2f} against {drv}, convention expects {'+' if sign > 0 else '-'}"
    if "FCH4" in out:
        res = Q.fch4_unit_check(out["FCH4"][0])
        tests["fch4_units"] = res
        if res["suspect_umol"]:
            tests["removed"]["FCH4"] = "units: median |FCH4| below 0.05 nmol m-2 s-1, likely umol"
    return tests


def write_interim(rec: R.Record, cleaned: Dict, counts: List[Dict], tests: Dict, out_dir: Path,
                  meta: Dict) -> None:
    """<out_dir>/<SITE>.parquet (t_lst, t_utc in minutes; per variable value and flag; extras) and .json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_utc = rec.t - int(round(rec.meta["offset_h"] * 60))
    cols = {"t_lst": rec.t, "t_utc": t_utc}
    for k, (v, f) in cleaned.items():
        cols[k], cols[f"{k}_flag"] = v, f
    for k, v in rec.extra.items():
        cols[k] = v.astype(np.float32)
    pd.DataFrame(cols).to_parquet(out_dir / f"{rec.site_id}.parquet", index=False, compression="zstd")
    side = {"site_id": rec.site_id, "source": rec.source, "step": rec.step, "offset_h": rec.meta["offset_h"],
            "offset_source": rec.meta.get("offset_source"), "counts": counts, "tests": tests, **meta}
    (out_dir / f"{rec.site_id}.json").write_text(json.dumps(side, default=_json, indent=1))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return str(o)


# ---- Stage 1 drivers ----------------------------------------------------------------------------------------------

def _site_row(out: Path, site: str) -> Dict:
    s = pd.read_parquet(out / "sites.parquet")
    r = s[s.site_id == site]
    return r.iloc[0].to_dict() if len(r) else {}


def clean_shuttle_zip(zp: Path, out: Path, raw: Path) -> Dict:
    rec = R.read_oneflux(zp, start_minute())
    b = rec.meta["bif"]
    loc = b["locations"][-1] if b.get("locations") else None
    row = _site_row(out, rec.site_id)
    lat, lon = (loc["lat"], loc["lon"]) if loc else (row["lat"], row["lon"])
    off = rec.utc_offset_h if rec.utc_offset_h is not None else round(lon / 15)
    rec.meta.update(latlon=(lat, lon), offset_h=off,
                    offset_source="BIF UTC_OFFSET" if rec.utc_offset_h is not None else "longitude / 15")
    cleaned, counts, tests = apply_rules(rec, lat, lon, off, b.get("IGBP"), {}, set())
    # The ONEFlux u* check needs the thresholds; run it on the already-cleaned NEE separately.
    if b.get("ustar_vut") and "NEE" in rec.series and "USTAR" in rec.series:
        night = night_mask(rec, lat, lon, off)
        thr = ustar_steps(rec.t, b["ustar_vut"])
        nee, u = rec.series["NEE"].value, rec.series["USTAR"].value
        low = np.isfinite(nee) & np.isfinite(u) & (u < thr)
        tests["oneflux_nee_below_ustar_threshold"] = {
            "night": int((low & night).sum()), "day": int((low & ~night).sum()),
            "measured_night": int((np.isfinite(nee) & night).sum()), "measured_day": int((np.isfinite(nee) & ~night).sum())}
    vi = rec.meta["varinfo"]
    meta = {"file": rec.meta["file"], "lat": lat, "lon": lon, "elev": loc.get("elev") if loc else None,
            "locations": b.get("locations", []), "igbp": b.get("IGBP"), "koppen": b.get("CLIMATE_KOEPPEN"),
            "site_name": b.get("SITE_NAME"), "time_resolution": b.get("PRODUCT_TIME_RESOLUTION"),
            "oneflux_version": b.get("PRODUCT_ONEFLUX_VERSION"), "source_network": b.get("PRODUCT_SOURCE_NETWORK"),
            "processing_date": b.get("PRODUCT_PROCESSING_DATE"), "ustar_vut": b.get("ustar_vut", {}),
            "ustar_cut": b.get("ustar_cut"), "era_fit": b.get("era_fit", {}), "team": b.get("team", []),
            "mat": b.get("MAT"), "map": b.get("MAP"), "era5_years": rec.meta.get("era5_years"),
            "instruments": instrument_changes(vi), "era5_monthly": era5_monthly(zp)}
    write_interim(rec, cleaned, counts, tests, out / "interim" / "shuttle", meta)
    return {"site_id": rec.site_id, "rows": int(rec.t.size)}


def instrument_changes(vi: pd.DataFrame) -> List[Dict]:
    """BIFVARINFO records of the measured variables where the instrument model or height changes over time."""
    if vi.empty:
        return []
    ev = []
    for name, g in vi.groupby("varname"):
        if base(str(name).replace("_F_MDS", "")) not in VARIABLES or name.endswith("_QC"):
            continue
        g = g.sort_values("date")
        prev = None
        for r in g.itertuples():
            cur = (r.model, r.height)
            if prev is not None and cur != prev:
                ev.append({"variable": name, "date": r.date, "from": list(map(str, prev)), "to": list(map(str, cur))})
            prev = cur
    return ev


def era5_monthly(zp: Path) -> Dict:
    """1991-2020 monthly means of the site-downscaled ERA5 TA (deg C) and monthly sums of P (mm), for the Koppen
    class where the network gives none."""
    with zipfile.ZipFile(zp) as z:
        name = R._member(z, r"_ERA5_MM_")
        if not name:
            return {}
        d = pd.read_csv(z.open(name), na_values=[-9999])
    d["y"], d["m"] = d.TIMESTAMP // 100, d.TIMESTAMP % 100
    d = d[(d.y >= 1991) & (d.y <= 2020)]
    if d.empty or "TA_ERA" not in d or "P_ERA" not in d:
        return {}
    # ERA5 MM P_ERA is the mean daily total (mm d-1, "average from daily data" in BIFVARINFO_MM).
    days = pd.to_datetime(dict(year=d.y, month=d.m, day=1)).dt.days_in_month
    g = d.assign(P_tot=d.P_ERA * days).groupby("m")
    return {"TA": g.TA_ERA.mean().round(3).tolist(), "P": g.P_tot.mean().round(2).tolist()}


def _shuttle_worker(args) -> Dict:
    site, snap, raw, out = args
    tmp = raw / "shuttle"
    try:
        if S.free_gb(out) < 30:
            return {"site_id": site, "error": "under 30 GB free"}
        t0 = S.utc_now()
        zp = S.shuttle_download(snap, site, tmp)
        with zipfile.ZipFile(zp) as z:
            if z.testzip() is not None:
                raise RuntimeError("corrupt archive")
            keep = [n for n in z.namelist() if any(k in n for k in ("_BIF_", "_BIFVARINFO_HH_", "_BIFVARINFO_HR_",
                                                                     "README", "DATA_POLICY"))]
            meta_dir = raw / "shuttle_meta" / site
            meta_dir.mkdir(parents=True, exist_ok=True)
            for n in keep:
                (meta_dir / Path(n).name).write_bytes(z.read(n))
        snaprow = pd.read_csv(snap).set_index("site_id").loc[site]
        prov = S.record(zp, S.SHUTTLE, site, snaprow.download_link, zp.stem.split("FLUXNET_")[-1].split("_", 1)[-1],
                        kept=False, downloaded_utc=t0)
        res = clean_shuttle_zip(zp, out, raw)
        S.append_provenance(raw / "provenance.jsonl", [prov])
        zp.unlink()
        return res
    except Exception as e:  # one site's failure must not stop the others; it is logged and retried next run
        return {"site_id": site, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}


def run_shuttle(raw: Path, out: Path, workers: int, sites: Optional[List[str]] = None) -> None:
    snap = sorted((raw / "catalog").glob("fluxnet_shuttle_snapshot_*.csv"))[-1]
    cat = pd.read_csv(snap)
    cat = cat[cat.last_year >= int(START[:4])]
    todo = [s for s in cat.site_id if (sites is None or s in sites)
            and not (out / "interim" / "shuttle" / f"{s}.json").exists()]
    print(f"{len(todo)} Shuttle sites to process", flush=True)
    log = out / "logs" / "shuttle.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(_shuttle_worker, [(s, snap, raw, out) for s in todo]):
            res["time"] = S.utc_now()
            with open(log, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(res.get("site_id"), res.get("error", "ok"), flush=True)
            if res.get("error") == "under 30 GB free":
                pool.terminate()
                raise SystemExit("stopped: under 30 GB free")


def fallback_offset(rec: R.Record, lat: float, lon: float) -> Tuple[float, str]:
    """UTC offset for a record whose network metadata gives none: longitude / 15, corrected by whole hours when the
    clock test on the record's own SW_IN finds the sun an hour or more away (time zones do not follow meridians:
    Alaska is UTC-9 at 150 W, the Northwest Territories UTC-7 at 121 W)."""
    off = float(round(lon / 15))
    if "SW_IN" not in rec.series:
        return off, "longitude / 15 (no SW_IN to check)"
    el = Q.solar_elevation(lat, lon, Q.step_midpoints_utc(rec.t, rec.step, off))
    o = Q.clock_offset_minutes(rec.series["SW_IN"].value, el, rec.step)
    if o is not None and abs(o) >= 60:
        return off + int(np.trunc(o / 60)), f"longitude / 15 = {off:+.0f} h, corrected by the clock test ({o:+.0f} min)"
    return off, "longitude / 15, confirmed by the clock test"


def base_utc_offset(zp: Path) -> Optional[float]:
    """UTC_OFFSET from the BADM workbook inside a BASE-BADM archive."""
    with zipfile.ZipFile(zp) as z:
        name = R._member(z, r"_BIF_.*\.xlsx$")
        if not name:
            return None
        df = pd.read_excel(z.open(name))
    hit = df[df.VARIABLE == "UTC_OFFSET"]
    return float(hit.DATAVALUE.iloc[0]) if len(hit) else None


def clean_other(source: str, raw: Path, out: Path, only: Optional[List[str]] = None) -> None:
    """Stage 1 for FLUXNET-CH4, AmeriFlux BASE or ICOS L2 (archives kept on disk)."""
    reader = {"fluxnet_ch4": R.read_fluxnet_ch4, "amf_base": R.read_amf_base, "icos_l2": R.read_icos_l2}[source]
    sites = pd.read_parquet(out / "sites.parquet").set_index("site_id")
    prov = {r["file_name"]: r for r in S.read_provenance(raw / "provenance.jsonl") if r["source"] == source}
    dest = out / "interim" / source
    icos_start = {}
    if source == "icos_l2":
        cat = pd.read_csv(raw / "catalog" / "icos_etcL2Fluxes.csv")
        utc = pd.to_datetime(cat.timeStart).dt.tz_localize(None)
        icos_start = dict(zip(cat.stationId, ((utc.values - R.EPOCH) // np.timedelta64(1, "m")).astype(np.int64)))
    for zp in sorted((raw / source).glob("*.zip")):
        if only and not any(f"_{x}_" in zp.name for x in only):
            continue
        try:
            rec = reader(zp, start_minute())
        except Exception as e:
            print(zp.name, "unreadable:", e, flush=True)
            continue
        if rec.t.size == 0 or rec.site_id not in sites.index:
            print(zp.name, "no rows from", START, "or unknown site", flush=True)
            continue
        sh = out / "interim" / "shuttle" / f"{rec.site_id}.json"
        shmeta = json.loads(sh.read_text()) if sh.exists() else {}
        row = sites.loc[rec.site_id]
        lat, lon = shmeta.get("lat", row.lat), shmeta.get("lon", row.lon)
        if source == "icos_l2" and rec.site_id in icos_start:
            first_lst = R.minutes([int(pd.read_csv(zipfile.ZipFile(zp).open(rec.meta["file"]), nrows=1)
                                       .TIMESTAMP_START.iloc[0])])[0]
            off, osrc = (first_lst - icos_start[rec.site_id]) / 60.0, "ICOS object acquisition start (UTC) vs file"
        elif shmeta.get("offset_h") is not None:
            off, osrc = shmeta["offset_h"], "ONEFlux BIF UTC_OFFSET of the same site"
        elif (bz := next((raw / "amf_base").glob(f"AMF_{rec.site_id}_BASE-BADM_*.zip"), None)) is not None \
                and (o := base_utc_offset(bz)) is not None:
            off, osrc = o, "BASE-BADM UTC_OFFSET of the same site"
        else:
            off, osrc = fallback_offset(rec, lat, lon)
        rec.meta.update(latlon=(lat, lon), offset_h=off, offset_source=osrc)
        night = night_mask(rec, lat, lon, off)
        thr = {int(k): v for k, v in shmeta.get("ustar_vut", {}).items()}
        thr_src = "ONEFlux VUT reference threshold of the same site" if thr else None
        if not thr and source == "fluxnet_ch4":
            thr = infer_ustar(rec, night)
            thr_src = "0.5th percentile of night u* under the record's u*-filtered NEE" if thr else None
        if not thr and source in ("amf_base", "icos_l2"):
            thr = mpt_by_year(rec, night)
            thr_src = "moving point test on the record's own night NEE (Papale et al. 2006)" if thr else None
        cleaned, counts, tests = apply_rules(rec, lat, lon, off, row.igbp, thr, USTAR_SOURCES.get(source, set()))
        p = prov.get(zp.name, {})
        meta = {"file": rec.meta["file"], "archive": zp.name, "lat": lat, "lon": lon, "ustar_threshold": thr,
                "ustar_source": thr_src, "url": p.get("url"), "sha256": p.get("sha256")}
        write_interim(rec, cleaned, counts, tests, dest, meta)
        print(rec.site_id, source, rec.t.size, "rows", flush=True)


# ---- Stage 2: merge ---------------------------------------------------------------------------------------------

def merge_site(site: str, out: Path) -> Optional[Dict]:
    """One canonical table per tower: a UTC grid at the primary record's step, one source per variable-year."""
    recs = {}
    for src in S.SOURCE_CODES:
        p = out / "interim" / src / f"{site}.parquet"
        if p.exists():
            recs[src] = (pd.read_parquet(p), json.loads((out / "interim" / src / f"{site}.json").read_text()))
    if not recs:
        return None
    primary = "shuttle" if "shuttle" in recs else next(iter(recs))
    step = recs[primary][1]["step"]
    offset = recs[primary][1]["offset_h"]
    notes = []
    targets = [k for k in set().union(*[set(m["counts"][i]["variable"] for i in range(len(m["counts"])))
                                        for _, m in recs.values()])]
    t_lo, t_hi = None, None
    for src, (d, m) in list(recs.items()):
        if m["step"] != step:
            notes.append(f"{src} step {m['step']} min differs from the primary {step} min; not merged")
            del recs[src]
            continue
        have = np.zeros(len(d), bool)
        for k in targets:
            if k in d and VARIABLES[base(k)].kind == FLUX:
                have |= np.isfinite(d[k].values)
        if have.any():
            lo, hi = int(d.t_utc.values[have].min()), int(d.t_utc.values[have].max())
            t_lo = lo if t_lo is None else min(t_lo, lo)
            t_hi = hi if t_hi is None else max(t_hi, hi)
    if t_lo is None:
        return {"site_id": site, "rows": 0, "notes": notes + ["no measured flux"]}
    day = 1440
    off_min = int(round(offset * 60))
    g0 = (t_lo + off_min) // day * day - off_min          # local midnight of the first flux day
    g1 = ((t_hi + off_min) // day + 1) * day - off_min    # local midnight after the last
    grid = np.arange(g0, g1, step, dtype=np.int64)
    n = grid.size
    yr = years_of(grid + off_min)
    table: Dict[str, np.ndarray] = {}
    aligned = {}
    for src, (d, m) in recs.items():
        pos = (d.t_utc.values - g0)
        ok = (pos % step == 0) & (pos >= 0) & (pos < n * step)
        idx = (pos[ok] // step).astype(np.int64)
        dup = np.zeros(idx.size, bool)
        _, first = np.unique(idx, return_index=True)
        dup[np.setdiff1d(np.arange(idx.size), first)] = True
        if (~ok).sum() or dup.sum():
            notes.append(f"{src}: {int((~ok & (pos >= 0) & (pos < n * step)).sum())} off-grid and {int(dup.sum())} "
                         f"duplicate timestamps dropped")
        aligned[src] = (d.loc[ok].iloc[~dup], idx[~dup])
    choice = []
    for k in sorted(targets):
        order = PRIORITY.get(base(k), DEFAULT_PRIORITY)
        val = np.full(n, np.nan, np.float32)
        flag = np.zeros(n, np.uint8)
        src_code = np.zeros(n, np.uint8)
        for y in np.unique(yr):
            sel_y = yr == y
            picked = None
            for src in order:
                if src not in aligned or k not in aligned[src][0]:
                    continue
                d, idx = aligned[src]
                v = d[k].values
                inyear = sel_y[idx]
                if np.isfinite(v[inyear]).any():
                    picked = src
                    break
            for src in ([picked] if picked else []) + [s for s in order if s != picked]:
                if src not in aligned or k not in aligned[src][0]:
                    continue
                d, idx = aligned[src]
                inyear = sel_y[idx]
                tgt = idx[inyear]
                if src == picked:
                    val[tgt] = d[k].values[inyear]
                    flag[tgt] = d[f"{k}_flag"].values[inyear]
                    src_code[tgt] = np.where(np.isfinite(d[k].values[inyear]), S.SOURCE_CODES[src], 0)
                    choice.append({"site_id": site, "variable": k, "year": int(y), "source": src,
                                   "measured": int(np.isfinite(d[k].values[inyear]).sum())})
                elif picked is None:
                    flag[tgt] |= d[f"{k}_flag"].values[inyear]
        table[k], table[f"{k}_flag"], table[f"{k}_src"] = val, flag, src_code
    overlaps = source_overlaps(site, aligned, n)
    if "shuttle" in aligned:
        d, idx = aligned["shuttle"]
        for c in d.columns:
            if c.startswith("ERA5_") or c == "SW_IN_POT" or c.endswith("_RANDUNC"):
                a = np.full(n, np.nan, np.float32)
                a[idx] = d[c].values
                table[c] = a
    times = pd.to_datetime(grid, unit="m", utc=True)
    frame = pd.DataFrame({"time_utc": times, "time_lst": (times + pd.Timedelta(minutes=off_min)).tz_localize(None),
                          **table})
    frame.to_parquet(out / "hh" / f"{site}.parquet", index=False, compression="zstd")
    return {"site_id": site, "rows": n, "step": step, "offset_h": offset, "primary": primary,
            "sources": sorted(recs), "notes": notes, "choice": choice, "overlaps": overlaps}


def source_overlaps(site: str, aligned: Dict, n: int) -> List[Dict]:
    """Agreement of two sources on the same steps (FCH4, NEE, LE, H): count, Pearson r, mean difference, and the
    slope of the second on the first. A slope far from 1 shows a unit, sign or processing difference."""
    rows = []
    srcs = sorted(aligned)
    for k in ("FCH4", "NEE", "LE", "H"):
        for i, a in enumerate(srcs):
            for b in srcs[i + 1:]:
                da, ia = aligned[a]
                db, ib = aligned[b]
                if k not in da or k not in db:
                    continue
                va, vb = np.full(n, np.nan), np.full(n, np.nan)
                va[ia], vb[ib] = da[k].values, db[k].values
                ok = np.isfinite(va) & np.isfinite(vb)
                if ok.sum() < 500:
                    continue
                x, y = va[ok], vb[ok]
                varies = np.std(x) > 0 and np.std(y) > 0
                slope = float(np.polyfit(x, y, 1)[0]) if varies else None
                rows.append({"site_id": site, "variable": k, "a": a, "b": b, "n": int(ok.sum()),
                             "r": float(np.corrcoef(x, y)[0, 1]) if varies else None,
                             "mean_b_minus_a": float(np.mean(y - x)),
                             "slope_b_on_a": slope})
    return rows


def run_merge(out: Path) -> None:
    (out / "hh").mkdir(parents=True, exist_ok=True)
    sites = sorted({p.stem for p in (out / "interim").glob("*/*.parquet")})
    results = []
    for s in sites:
        try:
            r = merge_site(s, out)
        except Exception as e:
            r = {"site_id": s, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
        if r:
            results.append(r)
            print(s, r.get("rows"), r.get("error", ""), flush=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "merge.json").write_text(json.dumps(results, default=_json))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["shuttle", "clean", "merge", "tables"])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", choices=["fluxnet_ch4", "amf_base", "icos_l2"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sites", nargs="*")
    a = ap.parse_args()
    if a.stage == "shuttle":
        run_shuttle(a.raw, a.out, a.workers, a.sites)
    elif a.stage == "clean":
        clean_other(a.source, a.raw, a.out, a.sites)
    elif a.stage == "merge":
        run_merge(a.out)
    else:
        import tables
        tables.build(a.raw, a.out)


if __name__ == "__main__":
    main()
