"""The site table: one row per tower, with coordinates and descriptors from network metadata.

Coordinates are the tower location from the network's own metadata, at the precision the network publishes:
- ONEFlux sites: the BIF `LOCATION_LAT` / `LOCATION_LONG` record with the latest `LOCATION_DATE_START` (the current
  tower position; earlier records are site moves, kept in `events.parquet`). Before a site's archive is processed,
  the Shuttle catalog value is used; it is the same BIF field as published by the network's metadata service.
- AmeriFlux and FLUXNET-CH4 sites outside the Shuttle: the AmeriFlux / FLUXNET `site_display` BADM location.
- ICOS stations outside the Shuttle: the ICOS Carbon Portal station metadata (meta.icos-cp.eu).

One row per FLUXNET site ID. The ID is global (CC-XXX), so a tower distributed by two networks (an ICOS station also
in FLUXNET-CH4, an AmeriFlux tower in both the Shuttle and BASE) is one row with several sources.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

IGBP_LABELS = {
    "Evergreen Needleleaf Forests": "ENF", "Evergreen Broadleaf Forests": "EBF", "Deciduous Needleleaf Forests": "DNF",
    "Deciduous Broadleaf Forests": "DBF", "Mixed Forests": "MF", "Closed Shrublands": "CSH", "Open Shrublands": "OSH",
    "Woody Savannas": "WSA", "Savannas": "SAV", "Grasslands": "GRA", "Permanent Wetlands": "WET", "Croplands": "CRO",
    "Urban and Built-Up Lands": "URB", "Cropland/Natural Vegetation Mosaics": "CVM", "Snow and Ice": "SNO",
    "Barren": "BSV", "Barren or Sparsely Vegetated": "BSV", "Water Bodies": "WAT",
}

LICENCE = "CC-BY-4.0"
ACKNOWLEDGEMENT = {
    "AMF": "Funding for the AmeriFlux data service was provided by the U.S. Department of Energy Office of Science.",
    "CNF": "ChinaFlux data are from the National ecosystem science data center https://www.nesdc.org.cn "
           "(cite Yu et al. 2006, doi:10.1016/j.agrformet.2006.02.011).",
    "JPF": "Cite Ueyama et al. 2025, doi:10.5194/essd-17-3807-2025.",
    "OZF": "OzFlux data are from the OzFlux Data Portal https://data.ozflux.org.au/home.jspx (cite Beringer et al. "
           "2016, doi:10.5194/bg-13-5895-2016; Isaac et al. 2017, doi:10.5194/bg-14-2903-2017).",
    "TERN": "TERN data are from the TERN Data Discovery Portal https://portal.tern.org.au/browse/theme (cite "
            "Beringer et al. 2016, doi:10.5194/bg-13-5895-2016; Isaac et al. 2017, doi:10.5194/bg-14-2903-2017).",
}
"""Network-specific acknowledgements required by the FLUXNET data policy (DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt,
version 20251226), in addition to the global FLUXNET acknowledgement in the dataset card."""


def _koppen_from_label(label: str) -> Optional[str]:
    m = re.search(r"\(([A-E][A-Za-z]{1,2})\)", str(label))
    return m.group(1) if m else None


def _pis(names: str, roles: str) -> str:
    n, r = str(names).split(";"), str(roles).split(";")
    pis = [a.strip() for a, b in zip(n, r) if b.strip() == "PI"]
    return "; ".join(dict.fromkeys(pis))


def catalog_sites(cat: Path) -> pd.DataFrame:
    """Every site of the four sources, from their catalogs, before any archive is opened."""
    snap = sorted(cat.glob("fluxnet_shuttle_snapshot_*.csv"))[-1]
    sh = pd.read_csv(snap)
    rows: Dict[str, Dict] = {}
    for r in sh.itertuples():
        rows[r.site_id] = dict(
            site_id=r.site_id, site_name=r.site_name, lat=float(r.location_lat), lon=float(r.location_long),
            elevation_m=np.nan, igbp=r.igbp, koppen=None, koppen_source=None, data_hub=r.data_hub,
            source_network=r.product_source_network, coord_source="shuttle catalog (network BIF LOCATION)",
            pis=_pis(r.team_member_name, r.team_member_role), shuttle_years=f"{r.first_year}-{r.last_year}",
            shuttle_product=r.fluxnet_product_name, shuttle_citation=r.product_citation,
            shuttle_pid=r.product_id, shuttle_url=r.download_link)
    amf = {d["SITE_ID"]: d for d in json.loads((cat / "amf_site_display.json").read_text())}
    flx = {d["SITE_ID"]: d for d in json.loads((cat / "fluxnet_site_display.json").read_text())}
    for sid, d in {**flx, **amf}.items():
        if sid in rows:
            row = rows[sid]
            loc = d.get("GRP_LOCATION") or {}
            if np.isnan(row["elevation_m"]) and loc.get("LOCATION_ELEV"):
                row["elevation_m"] = float(loc["LOCATION_ELEV"])
            k = (d.get("GRP_CLIM_AVG") or {}).get("CLIMATE_KOEPPEN")
            if k and not row["koppen"]:
                row["koppen"], row["koppen_source"] = k, "network BADM CLIMATE_KOEPPEN"
    ch4 = json.loads((cat / "fluxnet_ch4_availability.json").read_text())
    base = json.loads((cat / "amf_base_urls.json").read_text()) if (cat / "amf_base_urls.json").exists() else {}
    icos_l2 = pd.read_csv(cat / "icos_etcL2Fluxes.csv")
    icos_st = pd.read_csv(cat / "icos_stations.csv").drop_duplicates("id").set_index("id")
    extra = {x["SITE_ID"]: "FLUXNET-CH4" for x in ch4 if x["publish_years"]}
    extra.update({s: "AmeriFlux BASE" for s in base if s not in extra})
    extra.update({s: "ICOS L2" for s in icos_l2.stationId if s not in extra})
    for sid, why in extra.items():
        if sid in rows:
            continue
        d = amf.get(sid) or flx.get(sid)
        if d is not None:
            loc = d.get("GRP_LOCATION") or {}
            k = (d.get("GRP_CLIM_AVG") or {}).get("CLIMATE_KOEPPEN")
            rows[sid] = dict(site_id=sid, site_name=d.get("SITE_NAME"), lat=float(loc["LOCATION_LAT"]),
                             lon=float(loc["LOCATION_LONG"]),
                             elevation_m=float(loc["LOCATION_ELEV"]) if loc.get("LOCATION_ELEV") else np.nan,
                             igbp=d.get("IGBP"), koppen=k, koppen_source="network BADM CLIMATE_KOEPPEN" if k else None,
                             data_hub="AmeriFlux" if sid in amf else "FLUXNET",
                             source_network="AMF" if sid in amf else "FLX",
                             coord_source=f"{'AmeriFlux' if sid in amf else 'FLUXNET'} site_display BADM LOCATION",
                             pis="")
        elif sid in icos_st.index:
            st = icos_st.loc[sid]
            k = _koppen_from_label(st.get("clim"))
            rows[sid] = dict(site_id=sid, site_name=st["name"], lat=float(st["lat"]), lon=float(st["lon"]),
                             elevation_m=float(st["elev"]) if pd.notna(st.get("elev")) else np.nan,
                             igbp=IGBP_LABELS.get(st.get("igbp")), koppen=k,
                             koppen_source="ICOS station climate zone" if k else None, data_hub="ICOS",
                             source_network="ICOS", coord_source="ICOS Carbon Portal station metadata", pis="")
    df = pd.DataFrame(list(rows.values()))
    df["licence"] = LICENCE
    df["in_fluxnet_ch4"] = df.site_id.isin([x["SITE_ID"] for x in ch4 if x["publish_years"]])
    df["in_amf_base_fch4"] = df.site_id.isin(list(base))
    df["in_icos_l2"] = df.site_id.isin(icos_l2.stationId)
    df["in_shuttle"] = df.site_id.isin(sh.site_id)
    df["acknowledgement"] = df.source_network.map(ACKNOWLEDGEMENT).fillna("")
    return df.sort_values("site_id").reset_index(drop=True)


def write(df: pd.DataFrame, out: Path) -> None:
    """sites.parquet and sites.csv, written atomically so a reader never sees a partial file."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("parquet", "csv"):
        tmp = out / f".sites.{ext}.tmp"
        df.to_parquet(tmp, index=False) if ext == "parquet" else df.to_csv(tmp, index=False)
        tmp.replace(out / f"sites.{ext}")
