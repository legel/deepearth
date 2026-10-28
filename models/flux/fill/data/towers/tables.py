"""Dataset-level tables from the merged site tables and the stage-1 sidecars.

Writes under <out>:
- sites.parquet / sites.csv: the final site table (coordinates, descriptors, sources, coverage, inclusion, licence,
  citation, PIs, acknowledgement).
- coverage.parquet: measured fraction per site, variable and year, with the source chosen.
- qc_counts.parquet: values removed per site, source, variable and rule.
- events.parquet: site moves, instrument changes, clock offsets, sign or unit failures, hourly records, step
  mismatches, cross-source disagreements, co-located towers.
- windows.parquet: every 180-day window (30-day stride) of every site with its measured fraction per variable.
- provenance.csv: every source file with URL, version, download time and SHA-256.
- data_dictionary.csv: every stored column with unit and meaning.

Inclusion: a window is usable when at least MIN_WINDOW_FLUX of its steps have a measured flux (NEE, LE, H or FCH4);
a site is included when it has at least one usable window, which needs at least 180 days of record.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import koppen
import qc as Q
import sites as SI
import sources as S
from variables import ERA5_COLUMNS, VARIABLES, base

WINDOW_DAYS = 180
STRIDE_DAYS = 30
MIN_WINDOW_FLUX = 0.20
"""A 180-day window is usable when at least 20 % of its steps have a measured NEE, LE, H or FCH4. Typical ONEFlux
records measure NEE in 25 to 50 % of half-hours after u* filtering, and LE in 60 to 80 %."""
MIN_YEAR_FLUX = 0.10
"""A site-year counts as measured when a flux is measured in at least 10 % of its steps (1752 half-hours)."""
FLUXES = ("NEE", "LE", "H", "FCH4")
WINDOW_VARS = ("NEE", "LE", "H", "FCH4", "G", "TA", "SW_IN", "VPD", "P", "WS", "USTAR", "SWC_1", "TS_1", "WTD")
KOPPEN_DIR = "/home/photon/4tb/fluxfill/raw/towers/koppen"
"""Where the Beck et al. (2023) maps are unpacked (koppen_geiger_tif.zip, figshare file 61012822)."""
COLOCATED_KM = 1.0
"""Two site IDs closer than this are recorded as possibly the same tower or a paired tower."""


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sidecars(out: Path) -> Dict[str, Dict[str, Dict]]:
    res: Dict[str, Dict[str, Dict]] = {}
    for p in (out / "interim").glob("*/*.json"):
        m = json.loads(p.read_text())
        res.setdefault(m["site_id"], {})[m["source"]] = m
    return res


def windows_of(df: pd.DataFrame, site: str, step: int) -> List[Dict]:
    per_day = 1440 // step
    n_win, stride = WINDOW_DAYS * per_day, STRIDE_DAYS * per_day
    have = {v: np.isfinite(df[v].values) for v in WINDOW_VARS if v in df}
    anyflux = np.zeros(len(df), bool)
    for v in FLUXES:
        if v in have:
            anyflux |= have[v]
    era = np.isfinite(df["ERA5_TA"].values) if "ERA5_TA" in df else np.zeros(len(df), bool)
    rows = []
    for s0 in range(0, max(len(df) - n_win + 1, 0), stride):
        sl = slice(s0, s0 + n_win)
        r = {"site_id": site, "start_lst": df.time_lst.iloc[s0], "start_utc": df.time_utc.iloc[s0],
             "steps": n_win, "any_flux": float(anyflux[sl].mean()), "era5": float(era[sl].mean())}
        for v, h in have.items():
            r[v] = float(h[sl].mean())
        r["usable"] = r["any_flux"] >= MIN_WINDOW_FLUX
        rows.append(r)
    return rows


def build(raw: Path, out: Path) -> None:
    cards = sidecars(out)
    merge = {r["site_id"]: r for r in json.loads((out / "logs" / "merge.json").read_text())}
    base_sites = pd.read_parquet(out / "sites.parquet")
    cov, win, events, counts = [], [], [], []
    for site, srcs in sorted(cards.items()):
        for src, m in srcs.items():
            for c in m["counts"]:
                counts.append({**c, "source": src})
            t = m["tests"]
            if m.get("step") == 60:
                events.append({"site_id": site, "kind": "hourly", "source": src, "detail": "record is hourly (HR)"})
            co = t.get("clock_offset_min") or {}
            for y, o in (co.get("by_year") or {}).items():
                if o is not None and abs(o) >= max(60, m["step"]):
                    events.append({"site_id": site, "kind": "clock", "source": src, "year": int(y),
                                   "detail": f"SW_IN peaks {o:+.0f} min from solar noon at the stated UTC offset"})
            for var, why in t.get("removed", {}).items():
                events.append({"site_id": site, "kind": "site_test", "source": src, "variable": var, "detail": why})
            ts = t.get("timestamps", {})
            if ts.get("duplicates") or ts.get("irregular"):
                events.append({"site_id": site, "kind": "timestamps", "source": src,
                               "detail": f"{ts['duplicates']} duplicate and {ts['irregular']} irregular steps"})
            for ev in m.get("instruments", []):
                events.append({"site_id": site, "kind": "instrument", "source": src, "variable": ev["variable"],
                               "date": ev["date"], "detail": f"{' @ '.join(ev['from'])} -> {' @ '.join(ev['to'])} m"})
            locs = m.get("locations") or []
            for a, b in zip(locs[:-1], locs[1:]):
                d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                if d > 0.01:
                    events.append({"site_id": site, "kind": "site_move", "source": src, "date": b["date_start"],
                                   "detail": f"{d * 1000:.0f} m to {b['lat']}, {b['lon']}"
                                             + (f"; {b['comment']}" if b.get("comment") else "")})
        mr = merge.get(site)
        if not mr or not mr.get("rows"):
            continue
        for note in mr.get("notes", []):
            events.append({"site_id": site, "kind": "merge", "detail": note})
        for o in mr.get("overlaps", []):
            bad = (o["r"] is not None and o["r"] < 0.8) or (o["slope_b_on_a"] is not None
                                                             and abs(o["slope_b_on_a"] - 1) > 0.2)
            events.append({"site_id": site, "kind": "source_overlap" + ("_disagree" if bad else ""),
                           "variable": o["variable"],
                           "detail": f"{o['a']} vs {o['b']}: n={o['n']} r={o['r'] if o['r'] is None else round(o['r'], 3)} "
                                     f"slope={o['slope_b_on_a'] if o['slope_b_on_a'] is None else round(o['slope_b_on_a'], 3)} "
                                     f"mean diff={o['mean_b_minus_a']:.3g}"})
        df = pd.read_parquet(out / "hh" / f"{site}.parquet")
        yr = df.time_lst.dt.year.values
        for v in [c for c in df.columns if c.endswith("_src")]:
            name = v[:-4]
            fin = np.isfinite(df[name].values)
            for y in np.unique(yr):
                sel = yr == y
                srcs_y = np.unique(df[v].values[sel & fin])
                cov.append({"site_id": site, "variable": name, "year": int(y), "steps": int(sel.sum()),
                            "measured": int(fin[sel].sum()), "fraction": float(fin[sel].mean()),
                            "source": ",".join(k for k, c in S.SOURCE_CODES.items() if c in srcs_y)})
        win += windows_of(df, site, mr["step"])
    cov = pd.DataFrame(cov)
    win = pd.DataFrame(win)
    counts = pd.DataFrame(counts)
    # Co-located site IDs.
    s = base_sites
    d = haversine_km(s.lat.values[:, None], s.lon.values[:, None], s.lat.values[None], s.lon.values[None])
    i, j = np.where(np.triu(d < COLOCATED_KM, 1))
    for a, b in zip(i, j):
        events.append({"site_id": s.site_id.iloc[a], "kind": "colocated", "variable": None,
                       "detail": f"{s.site_id.iloc[b]} at {d[a, b] * 1000:.0f} m"})
    events = pd.DataFrame(events)
    sites = final_sites(base_sites, cards, merge, cov, win)
    SI.write(sites, out)
    cov.to_parquet(out / "coverage.parquet", index=False)
    win.to_parquet(out / "windows.parquet", index=False)
    counts.to_parquet(out / "qc_counts.parquet", index=False)
    events.astype({c: "string" for c in events.columns if events[c].dtype == object}).to_parquet(
        out / "events.parquet", index=False)
    pd.DataFrame(S.read_provenance(raw / "provenance.jsonl")).to_csv(out / "provenance.csv", index=False)
    dictionary().to_csv(out / "data_dictionary.csv", index=False)
    print(f"{int(sites.included.sum())} included sites of {len(sites)}; {int(win.usable.sum())} usable windows")


def final_sites(s: pd.DataFrame, cards: Dict, merge: Dict, cov: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    s = s.copy().set_index("site_id")
    for col in ("utc_offset_h", "step_min", "first_year", "last_year", "sources", "ustar_source", "era5",
                "oneflux_version", "n_years_measured", "nee_measured_hh", "le_measured_hh", "h_measured_hh",
                "fch4_measured_hh", "usable_windows"):
        s[col] = None
    for site, srcs in cards.items():
        if site not in s.index:
            continue
        sh = srcs.get("shuttle")
        if sh:
            s.loc[site, ["lat", "lon"]] = sh["lat"], sh["lon"]
            s.loc[site, "coord_source"] = "ONEFlux BIF LOCATION (latest LOCATION_DATE_START)"
            if sh.get("elev") is not None:
                s.loc[site, "elevation_m"] = sh["elev"]
            if sh.get("igbp"):
                s.loc[site, "igbp"] = sh["igbp"]
            if koppen.normalize(sh.get("koppen")):
                s.loc[site, ["koppen", "koppen_source"]] = koppen.normalize(sh["koppen"]), "ONEFlux BIF CLIMATE_KOEPPEN"
            s.loc[site, "oneflux_version"] = sh.get("oneflux_version")
            s.loc[site, "era5"] = True
            team = [t["name"] for t in sh.get("team", []) if t.get("role") == "PI" and t.get("name")]
            if team:
                s.loc[site, "pis"] = "; ".join(dict.fromkeys(team))
        any_m = next(iter(srcs.values()))
        s.loc[site, "utc_offset_h"] = (sh or any_m)["offset_h"]
        s.loc[site, "step_min"] = (sh or any_m)["step"]
        thr = [m.get("ustar_source") for m in srcs.values() if m.get("ustar_source")]
        s.loc[site, "ustar_source"] = "; ".join(sorted(set(thr))) if thr else None
    s["koppen"] = s.koppen.map(koppen.normalize)
    s.loc[s.koppen.isna(), "koppen_source"] = None
    need = s.index[s.koppen.isna()]
    tif = sorted(Path(KOPPEN_DIR).glob("**/1991_2020/koppen_geiger_0p00833333.tif"))
    if len(need) and tif:
        got = koppen.from_map(tif[0], s.loc[need, "lat"].astype(float).values, s.loc[need, "lon"].astype(float).values)
        for site, k in zip(need, got):
            if k:
                s.loc[site, ["koppen", "koppen_source"]] = k, "Beck et al. 2023 1-km map, 1991-2020"
    for site in s.index[s.koppen.isna()]:
        e = (cards.get(site, {}).get("shuttle") or {}).get("era5_monthly") or {}
        if e.get("TA"):
            s.loc[site, ["koppen", "koppen_source"]] = (koppen.classify(e["TA"], e["P"], float(s.loc[site, "lat"])),
                                                        "ERA5 1991-2020 normals (ONEFlux ERA5 MM), Beck et al. 2018 rules")
    for site, mr in merge.items():
        if site in s.index and mr.get("rows"):
            s.loc[site, "sources"] = ",".join(mr["sources"])
    if len(cov):
        fl = cov[cov.variable.isin(FLUXES)]
        yrs = fl.groupby(["site_id", "year"]).fraction.max()
        good = yrs[yrs >= MIN_YEAR_FLUX].reset_index()
        g = good.groupby("site_id").year
        s.loc[g.min().index, "first_year"] = g.min()
        s.loc[g.max().index, "last_year"] = g.max()
        s.loc[g.size().index, "n_years_measured"] = g.size()
        tot = cov.groupby(["site_id", "variable"]).measured.sum()
        for v in FLUXES:
            t = tot.xs(v, level="variable") if v in tot.index.get_level_values(1) else pd.Series(dtype=float)
            s.loc[t.index, f"{v.lower()}_measured_hh"] = t
    if len(win):
        u = win.groupby("site_id").usable.sum()
        s.loc[u.index, "usable_windows"] = u
    for col in ("usable_windows", "first_year", "last_year", "n_years_measured", "step_min", "nee_measured_hh",
                "le_measured_hh", "h_measured_hh", "fch4_measured_hh"):
        s[col] = pd.to_numeric(s[col], errors="coerce").astype("Int64")
    s["utc_offset_h"] = pd.to_numeric(s.utc_offset_h, errors="coerce")
    s["usable_windows"] = s.usable_windows.fillna(0)
    s["included"] = s.usable_windows > 0
    s["has_fch4"] = s.fch4_measured_hh.fillna(0) > 0
    s["koppen_main"] = s.koppen.fillna("").str[:1].replace("", None)
    s["citation"] = s.get("shuttle_citation")
    return s.reset_index()


def dictionary() -> pd.DataFrame:
    rows = [{"column": "time_utc", "unit": "UTC timestamp", "kind": "time",
             "meaning": "Start of the step (TIMESTAMP_START) in UTC"},
            {"column": "time_lst", "unit": "local standard time", "kind": "time",
             "meaning": "Start of the step in the site's local standard time (no daylight saving), as published"}]
    for name, v in VARIABLES.items():
        col = f"{name}_k (k = 1..5, shallow to deep)" if v.profile else name
        rows.append({"column": col, "unit": v.unit, "kind": v.kind,
                     "meaning": f"{v.description}. Measured value after every quality rule; NaN where not measured "
                                f"or removed. Physical range {v.lo} to {v.hi}."})
        rows.append({"column": f"{col}_flag", "unit": "bit field", "kind": "qc",
                     "meaning": "Why the value is absent: " + ", ".join(f"{b} {n}" for b, n in Q.FLAG_NAMES.items())
                                + ". 0 with a NaN value: the source has no value at this step."})
        rows.append({"column": f"{col}_src", "unit": "code", "kind": "provenance",
                     "meaning": "Source of the value: " + ", ".join(f"{c} {S.SOURCE_LABELS[k]}"
                                                                   for k, c in S.SOURCE_CODES.items())
                                + "; 0 none."})
    for c in ERA5_COLUMNS:
        name = c[:-4]
        rows.append({"column": f"ERA5_{name}", "unit": VARIABLES[name].unit, "kind": "reanalysis",
                     "meaning": f"ERA5 {name} downscaled to the site by ONEFlux (linear regression on the site's "
                                f"measured steps; Vuichard and Papale 2015). Reanalysis, never a measurement; "
                                f"present at every step of a Shuttle site."})
    rows.append({"column": "SW_IN_POT", "unit": "W m-2", "kind": "derived",
                 "meaning": "Potential (top-of-atmosphere) shortwave radiation computed by ONEFlux"})
    for n in ("NEE", "LE", "H"):
        rows.append({"column": f"{n}_RANDUNC", "unit": VARIABLES[n].unit, "kind": "uncertainty",
                     "meaning": f"ONEFlux random uncertainty of the measured {n} (present only where {n} is)"})
    return pd.DataFrame(rows)
