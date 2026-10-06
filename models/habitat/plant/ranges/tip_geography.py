"""Tip locations for phylogeographic dispersal fits: the area-weighted spherical centroid of each species' WCVP
native TDWG level-3 areas, joined to tree tips by accepted binomial."""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def l3_centroids(wgsrpd_level3: str | Path) -> pd.DataFrame:
    """Unit-sphere centroid vector and area (km^2) of every TDWG L3 area."""
    g = gpd.read_file(wgsrpd_level3)[["LEVEL3_COD", "geometry"]]
    eq = g.to_crs("+proj=cea +lon_0=0 +lat_ts=0 +datum=WGS84")         # equal-area for weights
    area = eq.area.values / 1e6
    c = eq.representative_point().to_crs(4326)                       # point inside each area
    lat, lon = np.radians(c.y.values), np.radians(c.x.values)
    return pd.DataFrame({"area_code_l3": g.LEVEL3_COD.values, "area_km2": area,
                         "ux": np.cos(lat) * np.cos(lon), "uy": np.cos(lat) * np.sin(lon), "uz": np.sin(lat)})


def species_native_centroids(wcvp_dir: str | Path, wgsrpd_level3: str | Path) -> pd.DataFrame:
    names = pd.read_csv(Path(wcvp_dir) / "wcvp_names.csv", sep="|", dtype=str,
                        usecols=["plant_name_id", "taxon_rank", "taxon_status", "family", "taxon_name"])
    acc = names[(names.taxon_status == "Accepted") & (names.taxon_rank == "Species")]
    dist = pd.read_csv(Path(wcvp_dir) / "wcvp_distribution.csv", sep="|", dtype=str,
                       usecols=["plant_name_id", "area_code_l3", "introduced", "extinct", "location_doubtful"])
    nat = dist[(dist.introduced == "0") & (dist.extinct == "0") & (dist.location_doubtful == "0")]
    nat = nat[nat.plant_name_id.isin(acc.plant_name_id)].merge(l3_centroids(wgsrpd_level3), on="area_code_l3")
    for k in ("ux", "uy", "uz"):
        nat[k] = nat[k] * nat.area_km2
    v = nat.groupby("plant_name_id")[["ux", "uy", "uz"]].sum()
    n = np.linalg.norm(v.values, axis=1, keepdims=True)
    u = v.values / np.where(n == 0, 1, n)
    out = pd.DataFrame({"plant_name_id": v.index, "latitude": np.degrees(np.arcsin(u[:, 2])),
                        "longitude": np.degrees(np.arctan2(u[:, 1], u[:, 0])),
                        "n_l3": nat.groupby("plant_name_id").size().values})
    return out.merge(acc[["plant_name_id", "taxon_name", "family"]], on="plant_name_id")


def tree_tips(newick_path: str | Path) -> pd.DataFrame:
    """Tip labels of a Carruthers-style tree (``Genus_epithet__index``) with their binomial."""
    labels = re.findall(r"[(,]([A-Za-z][^:(),]*)", Path(newick_path).read_text())
    return pd.DataFrame({"tip": labels, "taxon_name": [l.split("__")[0].replace("_", " ") for l in labels]})


if __name__ == "__main__":
    import sys
    wcvp, wgs, tree, out = sys.argv[1:5]
    cents = species_native_centroids(wcvp, wgs)
    tips = tree_tips(tree).merge(cents, on="taxon_name", how="left")
    tips.to_csv(out, index=False)
    print(f"tips {len(tips)}  with native centroid {tips.latitude.notna().sum()}  families {tips.family.nunique()}")
