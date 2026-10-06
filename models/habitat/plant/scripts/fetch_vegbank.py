"""VegBank (Ecological Society of America plot archive, api.vegbank.org): vegetation plots with complete species
lists and coordinates, many in the eastern US where BLM AIM has none. Downloads all plot observations and all
taxon observations, page by page (resumable), into raw/validation/vegbank/{plots,taxa}.parquet."""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402

OUT = config.data_root() / "raw/validation/vegbank"
API = "https://api.vegbank.org"
PLOT_COLS = ["ob_code", "pl_code", "latitude", "longitude", "confidentiality_status", "location_accuracy", "country",
             "state_province", "obs_start_date", "effort_level", "floristic_quality", "area", "project_name"]
TAXON_COLS = ["ob_code", "int_curr_plant_sci_name_no_auth", "int_curr_plant_sci_full", "author_plant_name"]


def fetch(endpoint: str, cols: list[str], page: int = 20000) -> pd.DataFrame:
    d = OUT / endpoint
    d.mkdir(parents=True, exist_ok=True)
    total = json.load(urllib.request.urlopen(f"{API}/{endpoint}?limit=1", timeout=120))["count"]
    for off in range(0, total, page):
        f = d / f"{off:09d}.parquet"
        if f.exists():
            continue
        for attempt in range(6):
            try:
                rows = json.load(urllib.request.urlopen(f"{API}/{endpoint}?limit={page}&offset={off}", timeout=300))["data"]
                break
            except Exception as e:                                       # transient API errors: back off and retry
                print("retry", off, repr(e)[:100], flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"{endpoint} offset {off} failed")
        pd.DataFrame(rows).reindex(columns=cols).astype(str).to_parquet(f)
        print(endpoint, off + len(rows), "/", total, flush=True)
    return pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)


plots = fetch("plot-observations", PLOT_COLS)
plots.to_parquet(OUT / "plots.parquet")
taxa = fetch("taxon-observations", TAXON_COLS)
taxa.to_parquet(OUT / "taxa.parquet")
print("plots", len(plots), "taxon observations", len(taxa))
