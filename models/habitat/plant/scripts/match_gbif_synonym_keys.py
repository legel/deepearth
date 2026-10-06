"""GBIF keys of the WCVP synonyms of US native species that the existing downloads leave with few or no records.

The supplementary download (doi:10.15468/dl.pkunks) requested the GBIF species-match key of each species' WCVP
accepted name. Where GBIF's backbone does not know that name as a species (recently moved genera such as Anatherum,
Pyrrocoma, Senega: the match falls back to the genus or family, so no key was requested) or files the species under
another name, its records are missing although GBIF holds them under a WCVP synonym (e.g. Andropogon mohrii for
Anatherum mohrii). Here every species-rank WCVP synonym of such species (``names.names_of_accepted``) is matched
against the GBIF backbone, and the species-level keys not already requested (work/gbif_requested_keys.json) are
written for a further download. Records are then assigned at fit time by the usual WCVP resolution of GBIF names
(``names.gbif_names_by_accepted``), so a synonym key can only add records to the species WCVP sinks it under.

usage: match_gbif_synonym_keys.py [--max-records 19] [--config configs/conus.json]
       -> work/us_natives_synonym_keys.csv, work/us_natives_synonym_download_keys.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges import names  # noqa: E402
from match_gbif_keys_all import SPECIES_RANKS, match, species_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--max-records", type=int, default=19, help="species with at most this many records are searched")
    a = ap.parse_args()
    cfg = config.load(a.config)
    R = cfg.root
    inv = pd.read_csv(cfg.path(cfg["species"]["inventory"]))
    target = inv.loc[inv.n_records <= a.max_records, "wcvp_accepted_name"].tolist()
    wc = names.load_wcvp_names(R / "raw/wcvp")
    pairs = [(s, n) for s in target for n in names.names_of_accepted(s, wc) if n != s and len(n.split()) in (2, 3)]
    print(f"{len(target)} species with <= {a.max_records} records; {len(pairs)} WCVP synonyms to match", flush=True)
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(match, [n for _, n in pairs]))
    d = pd.DataFrame(rows)
    d.insert(0, "species", [s for s, _ in pairs])
    ok = d["match"].isin(["EXACT", "FUZZY"]) & d["rank"].isin(SPECIES_RANKS)
    d["species_level"] = ok
    d.to_csv(R / "work/us_natives_synonym_keys.csv", index=False)
    ledger = R / "work/gbif_requested_keys.json"
    have = set(map(str, json.loads(ledger.read_text()))) if ledger.exists() else set()
    new = sorted(species_keys(d) - have, key=int)
    (R / "work/us_natives_synonym_download_keys.json").write_text(json.dumps(new))
    print(f"{int(ok.sum())} synonyms matched at species level ({d.loc[ok, 'species'].nunique()} species); "
          f"{len(new)} keys not yet requested", flush=True)


if __name__ == "__main__":
    main()
