"""GBIF species-match keys for every species of the species table (build_us_natives_table.py). Synonyms are kept with
their accepted key too: GBIF files a lumped taxon's records under the accepted key, and the WCVP name rules sort
them out at fit time. Writes work/us_natives_gbif_keys.csv (name, usage_key, accepted_key, gbif_name, status,
match, rank), the input of request_gbif_download.py.

usage: match_gbif_keys_all.py [--config configs/conus.json] [--threads 16]
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402

SPECIES_RANKS = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")


def match(name: str) -> dict:
    """GBIF backbone species match of one plant name (four tries)."""
    url = "https://api.gbif.org/v1/species/match?kingdom=Plantae&name=" + urllib.parse.quote(name)
    for _ in range(4):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=60))
            return {"name": name, "usage_key": d.get("usageKey"),
                    "accepted_key": d.get("acceptedUsageKey") or d.get("usageKey"), "gbif_name": d.get("canonicalName"),
                    "status": d.get("status"), "match": d.get("matchType"), "rank": d.get("rank")}
        except Exception:
            continue
    return {"name": name, "match": "ERROR"}


def species_keys(d: pd.DataFrame) -> set[str]:
    """Accepted and usage keys of the rows matched at species level or below."""
    ok = d["match"].isin(["EXACT", "FUZZY"]) & d["rank"].isin(SPECIES_RANKS)
    return (set(d.loc[ok, "accepted_key"].dropna().astype(int).astype(str))
            | set(d.loc[ok, "usage_key"].dropna().astype(int).astype(str)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    cfg = config.load(a.config)
    t = pd.read_csv(cfg.path(cfg["species"]["table"]))
    with ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(match, t.wcvp_accepted_name.tolist()))
    d = pd.DataFrame(rows)
    d.to_csv(cfg.path("work/us_natives_gbif_keys.csv"), index=False)
    ok = d["match"].isin(["EXACT", "FUZZY"]) & d["rank"].isin(SPECIES_RANKS)
    print(len(d), "names;", int(ok.sum()), "matched at species level;", d["match"].value_counts().to_dict())


if __name__ == "__main__":
    main()
