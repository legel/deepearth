"""Species table: every vascular plant species native to the United States per WCVP (accepted species with a native,
non-extinct occurrence in a US level-3 area), in the format run_fit.py reads: wcvp_accepted_name, gbif_name (= the
WCVP name; GBIF names are resolved through WCVP at fit time), wcvp_family, native_l3 (all native areas, worldwide),
us_regions (the configured US regions it is native to: CONUS, ASK = Alaska with the Aleutians, HAW = Hawaii) and
listed (on the optional listed-species table of the configuration, whose records follow the name rule of
build_name_reassignment.py). Writes the configured species table (work/us_natives_table.csv).

usage: build_us_natives_table.py [--config configs/conus.json]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    w = cfg.path(cfg["evaluation"]["wcvp_dir"])
    n = pd.read_csv(w / "wcvp_names.csv", sep="|", dtype=str,
                    usecols=["plant_name_id", "taxon_rank", "taxon_status", "taxon_name", "family"])
    d = pd.read_csv(w / "wcvp_distribution.csv", sep="|", dtype=str,
                    usecols=["plant_name_id", "region_code_l2", "area_code_l3", "introduced", "extinct"])
    acc = n[(n.taxon_status == "Accepted") & (n.taxon_rank == "Species")].set_index("plant_name_id")
    nat = d[(d.introduced == "0") & (d.extinct == "0") & d.plant_name_id.isin(acc.index)]
    region = pd.Series("", index=nat.index)
    for code, rule in cfg["species"]["regions"].items():          # later rules win, as listed
        if "l2" in rule:
            region[nat.region_code_l2.isin(rule["l2"])] = code
        if "l3" in rule:
            region[nat.area_code_l3.isin(rule["l3"])] = code
    us = nat[region != ""].assign(r=region[region != ""]).groupby("plant_name_id").r.apply(
        lambda x: ",".join(sorted(set(x))))
    l3 = nat[nat.plant_name_id.isin(us.index)].groupby("plant_name_id").area_code_l3.apply(
        lambda x: ",".join(sorted(set(x))))
    t = pd.DataFrame({"wcvp_accepted_name": acc.loc[us.index, "taxon_name"], "wcvp_family": acc.loc[us.index, "family"],
                      "native_l3": l3, "us_regions": us})
    t["gbif_name"] = t.wcvp_accepted_name
    listed = cfg.path(cfg["species"].get("listed_table"))
    t["listed"] = t.wcvp_accepted_name.isin(set(pd.read_csv(listed).wcvp_accepted_name)) if listed else False
    t = t.sort_values(["listed", "wcvp_accepted_name"], ascending=[False, True])
    out = cfg.path(cfg["species"]["table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print(len(t), "US native species;", int(t.listed.sum()), "listed;", t.us_regions.value_counts().to_dict())


if __name__ == "__main__":
    main()
