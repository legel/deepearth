"""Tower coordinates for the Sentinel-2 gather, and their reconciliation across sources.

Sources, in order of precedence:
1. The FluxFill tower table (`towers/sites.parquet` or `sites.csv`), when it exists.
2. The FLUXNET Shuttle snapshot (ONEFlux v1.3 catalog: `site_id, location_lat, location_long`).
3. The FLUXNET and AmeriFlux site metadata services (`amfcdn.lbl.gov/api/v1/site_display/{FLUXNET,AmeriFlux}`),
   for FLUXNET-CH4 V1.0 sites that the Shuttle does not list.

Every pair of sources that disagrees on a site's position by more than 30 m is written to `coordinate_diffs.csv`.

    python sites.py --src <dir with downloaded catalogs> --towers <towers dir> --out <sentinel2 root>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIFF_M = 30.0


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _site_display(path: Path, source: str) -> pd.DataFrame:
    rows = [{"site_id": d["SITE_ID"], "lat": float(d["GRP_LOCATION"]["LOCATION_LAT"]),
             "lon": float(d["GRP_LOCATION"]["LOCATION_LONG"]), "source": source}
            for d in json.load(open(path)) if d.get("GRP_LOCATION", {}).get("LOCATION_LAT") not in (None, "")]
    return pd.DataFrame(rows)


def catalog_sources(src: Path) -> dict:
    """Each catalog as a (site_id, lat, lon, source) frame."""
    out = {}
    sh = pd.read_csv(src / "shuttle_snapshot.csv")
    out["shuttle"] = pd.DataFrame({"site_id": sh.site_id, "lat": sh.location_lat, "lon": sh.location_long,
                                   "source": "shuttle", "last_year": sh.last_year, "first_year": sh.first_year})
    for name, f in (("fluxnet_site_display", "fluxnet_site_display.json"),
                    ("amf_site_display", "amf_site_display.json")):
        if (src / f).exists():
            out[name] = _site_display(src / f, name)
    ch4 = json.load(open(src / "fluxnet_ch4_availability.json"))
    out["ch4_ids"] = pd.DataFrame({"site_id": [d["SITE_ID"] for d in ch4],
                                   "ch4_last_year": [max(d["publish_years"] or [np.nan]) for d in ch4]})
    return out


def towers_table(towers: Path) -> pd.DataFrame | None:
    """The FluxFill tower table, normalised to (site_id, lat, lon, source), if it exists."""
    for f in ("sites.parquet", "sites.csv"):
        p = towers / f
        if p.exists():
            t = pd.read_parquet(p) if f.endswith("parquet") else pd.read_csv(p)
            cols = {c.lower(): c for c in t.columns}
            sid = cols.get("site_id") or cols.get("site")
            lat = next(cols[c] for c in ("lat", "latitude", "location_lat", "tower_lat") if c in cols)
            lon = next(cols[c] for c in ("lon", "longitude", "location_long", "location_lon", "tower_lon")
                       if c in cols)
            return pd.DataFrame({"site_id": t[sid], "lat": t[lat].astype(float), "lon": t[lon].astype(float),
                                 "source": "towers"})
    return None


def build(src: Path, towers: Path) -> tuple:
    """(site table, coordinate differences over DIFF_M between sources)."""
    cat = catalog_sources(src)
    sh = cat["shuttle"]
    ch4 = cat["ch4_ids"]
    frames = [sh[["site_id", "lat", "lon", "source"]]]
    for k in ("fluxnet_site_display", "amf_site_display"):
        if k in cat:
            frames.append(cat[k])
    tw = towers_table(towers)
    if tw is not None:
        frames.append(tw)
    allc = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])

    # Chosen coordinate per site: towers table, else Shuttle, else FLUXNET, else AmeriFlux metadata.
    order = {"towers": 0, "shuttle": 1, "fluxnet_site_display": 2, "amf_site_display": 3}
    wanted = set(sh.site_id) | set(ch4.site_id) | (set(tw.site_id) if tw is not None else set())
    cand = allc[allc.site_id.isin(wanted)].assign(o=lambda d: d.source.map(order)).sort_values(["site_id", "o"])
    chosen = cand.drop_duplicates("site_id").drop(columns="o").rename(columns={"source": "coord_source"})
    chosen = chosen.merge(sh[["site_id", "first_year", "last_year"]], on="site_id", how="left")
    chosen = chosen.merge(ch4, on="site_id", how="left")
    chosen["in_shuttle"] = chosen.site_id.isin(sh.site_id)
    chosen["in_fluxnet_ch4"] = chosen.site_id.isin(ch4.site_id)
    chosen["in_towers"] = chosen.site_id.isin(tw.site_id) if tw is not None else False
    chosen["last_year"] = chosen.last_year.fillna(chosen.ch4_last_year)

    diffs = []
    for sid, g in cand.groupby("site_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = float(haversine_m(g.lat[i], g.lon[i], g.lat[j], g.lon[j]))
                if d > DIFF_M:
                    diffs.append({"site_id": sid, "source_a": g.source[i], "lat_a": g.lat[i], "lon_a": g.lon[i],
                                  "source_b": g.source[j], "lat_b": g.lat[j], "lon_b": g.lon[j], "dist_m": round(d, 1)})
    return chosen.reset_index(drop=True), pd.DataFrame(diffs)


PILOT = ["US-Ha1", "DE-Tha", "FI-Hyy", "AU-How", "GF-Guy", "SN-Dhr", "US-Var", "IT-Lav", "CH-Dav", "US-Los"]
"""Ten pilot sites across biomes (temperate and boreal forest, savanna, tropical forest, grassland, alpine, wetland)."""


def processing_order(sites: pd.DataFrame) -> pd.DataFrame:
    """Pilot sites first, then sites whose records reach 2015 or whose last year is unknown, then the rest."""
    s = sites.copy()
    s["pilot"] = s.site_id.isin(PILOT)
    s["s2_era"] = s.last_year.isna() | (s.last_year >= 2015)
    s["pilot_rank"] = s.site_id.map({k: i for i, k in enumerate(PILOT)}).fillna(99)
    s = s.sort_values(["pilot", "pilot_rank", "s2_era", "last_year", "site_id"],
                      ascending=[False, True, False, False, True])
    return s.drop(columns=["pilot_rank"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--towers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sites, diffs = build(a.src, a.towers)
    sites = processing_order(sites)
    a.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(a.out / "sites.csv", index=False)
    diffs.to_csv(a.out / "coordinate_diffs.csv", index=False)
    print(f"{len(sites)} sites ({sites.in_shuttle.sum()} Shuttle, {sites.in_fluxnet_ch4.sum()} FLUXNET-CH4, "
          f"{sites.in_towers.sum()} towers table); {len(diffs)} source pairs differ by more than {DIFF_M:.0f} m")


if __name__ == "__main__":
    main()
