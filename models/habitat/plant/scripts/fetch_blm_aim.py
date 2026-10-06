"""Download BLM AIM Terrestrial Species Indicators (public ArcGIS feature service) as CSV: one row per
plot visit x species, with plot coordinates (EPSG:4326)."""
import concurrent.futures as cf
import sys
import time

import pandas as pd
import requests

URL = ("https://services1.arcgis.com/KbxwQRRfWyEYLgp4/arcgis/rest/services/"
       "BLM_Natl_AIM_Terrestrial_Species_Indicators_Public/FeatureServer/6/query")
FIELDS = "PrimaryKey,PlotID,State,ScientificName,Species,CurrentPLANTSCode,DateVisited,AH_SpeciesCover,Nonnative,DBKey"
PAGE = 2000


def page(offset: int) -> list[dict]:
    params = {"where": "1=1", "outFields": FIELDS, "returnGeometry": "true", "outSR": 4326, "f": "json",
              "resultOffset": offset, "resultRecordCount": PAGE, "orderByFields": "OBJECTID"}
    for t in range(6):
        try:
            j = requests.get(URL, params=params, timeout=120).json()
            return [{**f["attributes"], "lon": f["geometry"]["x"], "lat": f["geometry"]["y"]}
                    for f in j["features"] if f.get("geometry")]
        except Exception:
            time.sleep(3 * (t + 1))
    raise RuntimeError(f"page {offset} failed")


if __name__ == "__main__":
    n = requests.get(URL, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, timeout=60).json()["count"]
    with cf.ThreadPoolExecutor(6) as ex:
        rows = [r for chunk in ex.map(page, range(0, n, PAGE)) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(sys.argv[1], index=False)
    print("records", len(df), "of", n, "plots", df.PrimaryKey.nunique())
