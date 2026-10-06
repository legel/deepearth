"""Request a GBIF occurrence download: SIMPLE_PARQUET; TAXON_KEY in a key list; HAS_COORDINATE = true;
OCCURRENCE_STATUS = PRESENT; worldwide (native ranges extend beyond the US), the predicate of every download this
product used (doi:10.15468/dl.wwa829, doi:10.15468/dl.pkunks, doi:10.15468/dl.52fhnr).

Keys: by default the species-level keys of work/us_natives_gbif_keys.csv (match_gbif_keys_all.py) not yet
requested; ``--keys <json>`` gives an explicit list instead, e.g. work/us_natives_synonym_download_keys.json from
match_gbif_synonym_keys.py. Every requested key is added to work/gbif_requested_keys.json.

Without ``--submit`` the request body is printed. With ``--submit`` it is sent with the credentials of a GBIF
account taken from the environment only (GBIF_USER, GBIF_PWD, GBIF_EMAIL; never written anywhere), and the
download key is printed; fetch the finished download with ``fetch_sources.sh gbif <key>``.

usage: request_gbif_download.py [--keys keys.json] [--submit] [--config configs/conus.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from match_gbif_keys_all import species_keys  # noqa: E402

API = "https://api.gbif.org/v1/occurrence/download/request"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", help="JSON list of GBIF taxon keys")
    ap.add_argument("--submit", action="store_true", help="submit with GBIF_USER / GBIF_PWD / GBIF_EMAIL")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    ledger = cfg.path("work/gbif_requested_keys.json")
    have = set(map(str, json.loads(ledger.read_text()))) if ledger.exists() else set()
    if a.keys:
        keys = set(map(str, json.loads(Path(a.keys).read_text())))
    else:
        keys = species_keys(pd.read_csv(cfg.path("work/us_natives_gbif_keys.csv"))) - have
    keys = sorted(keys, key=int)
    body = {"format": "SIMPLE_PARQUET", "predicate": {"type": "and", "predicates": [
        {"type": "in", "key": "TAXON_KEY", "values": keys},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"}]}}
    if not a.submit:
        print(json.dumps(body))
        return
    import requests
    user, pwd, email = (os.environ.get(k) for k in ("GBIF_USER", "GBIF_PWD", "GBIF_EMAIL"))
    if not (user and pwd):
        sys.exit("set GBIF_USER and GBIF_PWD (and GBIF_EMAIL for the completion notice) in the environment")
    body["creator"] = user
    if email:
        body["notificationAddresses"] = [email]
        body["sendNotification"] = True
    r = requests.post(API, json=body, auth=(user, pwd), timeout=120)
    r.raise_for_status()
    ledger.write_text(json.dumps(sorted(have | set(keys), key=int)))
    print(f"download {r.text.strip()} requested ({len(keys)} taxon keys)")


if __name__ == "__main__":
    main()
