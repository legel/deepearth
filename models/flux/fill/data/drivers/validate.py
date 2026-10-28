"""Drivers against measured tower meteorology: bias, RMSE and Pearson r per variable and source, hourly and daily.

Tower data: measured-only half-hours (flag 0, finite), `t_utc` = TIMESTAMP_START in UTC minutes. A driver value
stamped at valid time T (UTC) is matched to:
- an instantaneous variable (temperature, humidity, wind, pressure, soil, HRRR radiation): the mean of the two
  tower half-hours starting at T - 30 min and T, i.e. the hour centred on T;
- a preceding-hour mean or sum (ERA5 shortwave, all precipitation): the half-hours starting at T - 60 min and
  T - 30 min (mean for radiation, sum for precipitation).
Both half-hours must be measured. Daily values are local-standard-time days (tower clock) with at least 20 matched
hours (all 24 for precipitation), means except precipitation (sums).

Every comparison between two sources uses the same site-hours: a row enters only where the tower and every source
in the comparison have a value.

Sources compared: ERA5-Land 0.1 deg (TA, VPD, SWC_1, TS_1), ERA5 0.25 deg (the ERA5-Land forcing: SW_IN, P, WS,
PA, also TA and VPD), ERA5-Land with a lapse-rate move to the tower elevation (Copernicus DEM 30 m), HRRR 3 km, and
the ONEFlux site-downscaled ERA5 carried in the tower files (fitted to the tower's own data, so not independent).

    python validate.py --towers /home/photon/4tb/fluxfill/towers/interim/shuttle --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import physics

ROOT = Path("/home/photon/4tb/fluxfill/drivers")
INSTANT = {"TA", "VPD", "WS", "PA", "SWC_1", "TS_1", "RH"}
PRECEDING = {"P", "SW_IN_ERA5"}
VARS = ["TA", "VPD", "SW_IN", "LW_IN", "P", "WS", "PA", "SWC_1", "TS_1"]


def tower_minutes(tw: pd.DataFrame) -> pd.DataFrame:
    """Accept both tower layouts: interim (`t_utc`, `t_lst` in minutes) and final `hh/` (`time_utc`, `time_lst`)."""
    if "t_utc" in tw:
        return tw
    tw = tw.copy()
    tw["t_utc"] = pd.to_datetime(tw["time_utc"]).values.astype("datetime64[m]").astype(np.int64)
    tw["t_lst"] = pd.to_datetime(tw["time_lst"]).values.astype("datetime64[m]").astype(np.int64)
    return tw


def tower_hourly(tw: pd.DataFrame, var: str) -> pd.DataFrame:
    """Tower value matched to driver valid hours T: columns 'inst' (centred on T) and 'prev' (hour ending at T)."""
    if var not in tw:
        return pd.DataFrame(columns=["inst", "prev"])
    ok = np.isfinite(tw[var].values)
    if var + "_flag" in tw:
        ok &= tw[var + "_flag"].values == 0
    s = pd.Series(np.where(ok, tw[var].values, np.nan), index=tw.t_utc.values.astype(np.int64))
    s = s[~s.index.duplicated()]
    h0 = int(np.ceil(s.index.min() / 60) * 60)
    h1 = int(np.floor(s.index.max() / 60) * 60)
    hours = np.arange(h0, h1 + 1, 60)
    a = s.reindex(hours - 30).values
    b = s.reindex(hours).values
    c = s.reindex(hours - 60).values
    agg = np.add if var == "P" else (lambda x, y: (x + y) / 2)
    return pd.DataFrame({"inst": agg(a, b), "prev": agg(c, a)}, index=pd.to_datetime(hours, unit="m"))


def metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """Driver x against tower y."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return {"n": int(len(x)), "bias": np.nan, "rmse": np.nan, "r": np.nan}
    d = x - y
    r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan
    return {"n": int(len(x)), "bias": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())), "r": float(r)}


def load_driver(source: str, site: str, root: Path = ROOT):
    f = root / source / f"{site}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    return d.set_index(pd.to_datetime(d.time_utc)).drop(columns="time_utc")


def site_frames(site: str, tw: pd.DataFrame, offset_min: int, roots: dict, terrain=None, used=None):
    """Per variable a frame of matched hourly values: 'tower_<source>' and '<source>' columns."""
    el = load_driver("era5_land", site, roots["era5_land"])
    e5 = load_driver("era5", site, roots["era5"])
    hr = load_driver("hrrr", site, roots["hrrr"])
    pr = {k: load_driver(k, site, roots[k]) for k in ("imerg", "mrms")}
    z_site = z_cell = z_e5 = np.nan
    if terrain is not None and site in terrain.index:
        z_site = terrain.loc[site, "dem_elevation_m"]
    if used is not None:
        if site in used["era5_land"].index:
            z_cell = used["era5_land"].loc[site, "bilinear_orography_m"]
        if site in used["era5"].index:
            z_e5 = used["era5"].loc[site, "bilinear_orography_m"]
    out = {}
    for v in VARS:
        th = tower_hourly(tw, v)
        if th.empty or th.inst.notna().sum() + th.prev.notna().sum() == 0:
            continue
        cols = {}
        if el is not None and v in el:
            cols["era5_land"] = (el[v], "inst")
        if e5 is not None and v in e5:
            cols["era5"] = (e5[v], "prev" if v in ("P", "SW_IN") else "inst")
        if v == "TA" and el is not None and np.isfinite(z_site) and np.isfinite(z_cell):
            cols["era5_land_dem"] = (pd.Series(physics.lapse_adjust_c(el["TA"].values, z_cell, z_site), el.index), "inst")
        if v == "PA" and e5 is not None and np.isfinite(z_site) and np.isfinite(z_e5):
            cols["era5_dem"] = (pd.Series(physics.pressure_to_height_kpa(e5["PA"].values, e5["TA"].values, z_e5,
                                                                         z_site), e5.index), "inst")
        if hr is not None and v in hr:
            cols["hrrr"] = (hr[v], "prev" if v == "P" else "inst")
        if v == "P":
            for k, d in pr.items():
                if d is not None:
                    cols[k] = (d["P"], "prev")
        # ONEFlux ERA5 in the tower file: half-hourly on the tower grid, matched the same way as the tower.
        if "ERA5_" + v in tw:
            t1 = tower_hourly(tw.rename(columns={"ERA5_" + v: "_e"}).assign(_e_flag=0), "_e")
            cols["oneflux_era5"] = (t1["prev" if v == "P" else "inst"], "self")
        if not cols:
            continue
        idx = th.index
        df = {}
        for k, (s, conv) in cols.items():
            df[k] = s.reindex(idx).values
            df["tower_" + k] = th["prev" if conv == "prev" or (conv == "self" and v == "P") else "inst"].values
        f = pd.DataFrame(df, index=idx)
        f["day"] = ((f.index + pd.Timedelta(minutes=offset_min)).floor("D"))
        out[v] = f
    return out


def compare(frames_by_site: dict, sources: list, var: str):
    """Hourly and daily metrics of each source on the site-hours where the tower and all `sources` exist."""
    rows = []
    for site, fr in frames_by_site.items():
        if var not in fr:
            continue
        f = fr[var]
        if not all(s in f for s in sources):
            continue
        m = np.ones(len(f), bool)
        for s in sources:
            m &= np.isfinite(f[s].values) & np.isfinite(f["tower_" + s].values)
        if m.sum() < 24:
            continue
        g = f[m].copy()
        g["site"] = site
        rows.append(g)
    if not rows:
        return None
    g = pd.concat(rows)
    res = {}
    for s in sources:
        res[(s, "hourly")] = metrics(g[s].values, g["tower_" + s].values)
        per = g.groupby("site").apply(lambda x: metrics(x[s].values, x["tower_" + s].values)["rmse"])
        res[(s, "hourly")]["rmse_site_median"] = float(per.median())
        how = "sum" if var == "P" else "mean"
        need = 24 if var == "P" else 20
        d = g.groupby(["site", "day"]).agg(n=(s, "size"), x=(s, how), y=("tower_" + s, how))
        d = d[d.n >= need]
        res[(s, "daily")] = metrics(d.x.values, d.y.values)
        res[(s, "hourly")]["sites"] = res[(s, "daily")]["sites"] = int(g.site.nunique())
    return res


def run(towers: Path, out: Path, root: Path = ROOT, sites=None, om_root: Path = None):
    """om_root: where the Open-Meteo sources (era5_land, era5) were assembled (default: root)."""
    om_root = om_root or root
    roots = {"era5_land": om_root, "era5": om_root, "hrrr": root, "imerg": root, "mrms": root}
    terrain = pd.read_csv(root / "dem" / "terrain.csv").set_index("site_id") if (root / "dem" / "terrain.csv").exists() else None
    used = {s: pd.read_csv(om_root / s / "sites_used.csv").set_index("site_id") for s in ("era5_land", "era5")
            if (om_root / s / "sites_used.csv").exists()}
    used = used if len(used) == 2 else None
    frames = {}
    files = sorted(towers.glob("*.parquet"))
    for f in files:
        site = f.stem
        if sites and site not in sites:
            continue
        if not (om_root / "era5_land" / f"{site}.parquet").exists():
            continue
        tw = tower_minutes(pd.read_parquet(f))
        offset = int(np.median(tw.t_lst.values - tw.t_utc.values)) if "t_lst" in tw else 0
        if np.median(np.diff(tw.t_utc.values[:1000])) != 30:
            continue  # hourly records are matched differently; not in this table
        frames[site] = site_frames(site, tw, offset, roots, terrain, used)
    tables = []
    sets = {"TA": [["era5_land", "era5", "era5_land_dem", "oneflux_era5"], ["era5_land", "era5_land_dem", "hrrr"]],
            "VPD": [["era5_land", "era5", "oneflux_era5"], ["era5_land", "hrrr"]],
            "SW_IN": [["era5", "oneflux_era5"], ["era5", "hrrr"]],
            "LW_IN": [["oneflux_era5"], ["oneflux_era5", "hrrr"]],
            "P": [["era5", "oneflux_era5"], ["era5", "imerg"], ["era5", "mrms"], ["era5", "hrrr"], ["era5", "hrrr", "mrms", "imerg"]],
            "WS": [["era5", "oneflux_era5"], ["era5", "hrrr"]],
            "PA": [["era5", "era5_dem", "oneflux_era5"], ["era5", "era5_dem", "hrrr"]],
            "SWC_1": [["era5_land"]], "TS_1": [["era5_land"]]}
    for v, groups in sets.items():
        for gi, srcs in enumerate(groups):
            res = compare(frames, srcs, v)
            if res is None:
                continue
            for (s, scale), m in res.items():
                tables.append({"variable": v, "comparison": "+".join(srcs), "source": s, "scale": scale, **m})
    t = pd.DataFrame(tables)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "validation.csv", index=False)
    (out / "validation.md").write_text(markdown(t) + "\n")
    (out / "validation_sites.json").write_text(json.dumps(sorted(frames)))
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--towers", type=Path, default=Path("/home/photon/4tb/fluxfill/towers/interim/shuttle"))
    ap.add_argument("--out", type=Path, default=ROOT / "validation")
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--om-root", type=Path, default=None)
    a = ap.parse_args(argv)
    t = run(a.towers, a.out, sites=a.sites, om_root=a.om_root)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.round(3).to_string(index=False))



def markdown(t: pd.DataFrame) -> str:
    """One row per variable, comparison and source: hourly and daily bias, RMSE, r; sites and hourly n."""
    rows = ["| variable | matched set | source | sites | hours | hourly bias | hourly RMSE | hourly r | daily bias | "
            "daily RMSE | daily r |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (v, comp), g in t.groupby(["variable", "comparison"], sort=False):
        for s in comp.split("+"):
            h = g[(g.source == s) & (g.scale == "hourly")].iloc[0]
            d = g[(g.source == s) & (g.scale == "daily")].iloc[0]
            rows.append(f"| {v} | {comp} | {s} | {int(h.sites)} | {int(h.n)} | {h.bias:.2f} | {h.rmse:.2f} | "
                        f"{h.r:.3f} | {d.bias:.2f} | {d.rmse:.2f} | {d.r:.3f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
