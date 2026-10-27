"""Tower coordinates used for driver extraction.

Everything is keyed by site id. The coordinates used for each extraction are written next to the data
(`<drivers root>/<source>/sites_used.csv`), so a later refinement of the tower table can be detected and only the
sites whose grid cells change are extracted again.
"""

from pathlib import Path

import pandas as pd

TOWERS = Path("/home/photon/4tb/fluxfill/towers/sites.parquet")


def load_sites(path: Path = TOWERS) -> pd.DataFrame:
    s = pd.read_parquet(path)[["site_id", "lat", "lon", "elevation_m"]].copy()
    s = s.dropna(subset=["lat", "lon"]).drop_duplicates("site_id").sort_values("site_id").reset_index(drop=True)
    return s
