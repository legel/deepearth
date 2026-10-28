"""The FluxFill baseline driver table of one tower: ERA5-Land joined with its ERA5 forcing, PA at the tower.

    from load import baseline
    df = baseline("US-Bi1")   # hourly, UTC valid time; TA, VPD, SWC_1, SWC_2, TS_1, TS_2, D_SNOW, SW_IN, P, WS, PA

PA is moved hydrostatically from the ERA5 bilinear cell orography to the Copernicus DEM tower elevation (validation:
RMSE 1.77 to 0.59 kPa at 33 towers). TA is not moved: the lapse-rate move made TA worse at the towers.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import physics

ROOT = Path("/home/photon/4tb/fluxfill/drivers")


def baseline(site: str, root: Path = ROOT, nearest: bool = False) -> pd.DataFrame:
    suf = "_NN" if nearest else ""
    el = pd.read_parquet(root / "era5_land" / f"{site}.parquet")
    e5 = pd.read_parquet(root / "era5" / f"{site}.parquet")
    keep_el = ["TA", "VPD", "SWC_1", "SWC_2", "TS_1", "TS_2", "D_SNOW"]
    keep_e5 = ["SW_IN", "P", "WS", "PA"]
    df = el[["time_utc"] + [v + suf for v in keep_el]].merge(
        e5[["time_utc", "TA" + suf] + [v + suf for v in keep_e5]].rename(columns={"TA" + suf: "_TA_E5"}),
        on="time_utc", how="outer")
    df.columns = [c[:-3] if c.endswith("_NN") else c for c in df.columns]
    terrain = pd.read_csv(root / "dem" / "terrain.csv").set_index("site_id")
    used = pd.read_csv(root / "era5" / "sites_used.csv").set_index("site_id")
    z_site = terrain.loc[site, "dem_elevation_m"]
    z_cell = used.loc[site, "cell_orography_m" if nearest else "bilinear_orography_m"]
    if np.isfinite(z_site) and np.isfinite(z_cell):
        df["PA"] = physics.pressure_to_height_kpa(df["PA"].values, df["_TA_E5"].values, z_cell, z_site)
    return df.drop(columns="_TA_E5")
