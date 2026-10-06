"""Which listed species each GBIF record counts toward (provenance 2026-10-04, ledger L17).

Default (closest to Daru): GBIF's interpreted species, resolved through WCVP. GBIF lumps some taxa that WCVP and
horticulture keep apart; a record is reassigned only when the name it was originally identified under
(verbatimScientificName, binomial resolved through WCVP) is itself a listed species and differs
from the default. Splits toward taxa the trade does not list (e.g. Pinus brachyptera, P. scopulorum
within Pinus ponderosa) are not applied: a listed name means the plant as it is known in horticulture.

The rule depends on a record only through its (interpreted species, verbatim name) pair, so it is decided once per
distinct pair and the records are then collected in batches (memory stays small for any number of downloads).

The listed species are those of the configuration's ``species.listed_table`` (columns wcvp_accepted_name and
gbif_name). The production run listed a horticultural priority list of 2,672 species that is not public; without a
listed table no record moves and every species takes its default records.

usage: build_name_reassignment.py [listed_table] [--config configs/conus.json] [--out <configured name_reassignment>]
Writes <out>.parquet (gbifid, species: the listed species the record moves to) and <out>_summary.csv (records
gained/lost per listed species)."""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges import names  # noqa: E402

KEY = ["species", "verbatimscientificname"]


def dataset(dirs) -> ds.Dataset:
    parts = [str(f) for d in dirs if Path(d).exists() for f in sorted(Path(d).rglob("*")) if f.is_file() and f.stat().st_size > 0]
    return ds.dataset(parts, format="parquet")          # GBIF downloads include zero-byte part files


def batches(dset: ds.Dataset):
    """(species, verbatim name, gbifid) in batches, missing names as "" so that every record keeps its pair key."""
    for b in dset.to_batches(columns=KEY + ["gbifid"], batch_size=5_000_000):
        t = pa.Table.from_batches([b])
        yield t.set_column(0, "species", pc.fill_null(t["species"], "")).set_column(
            1, "verbatimscientificname", pc.fill_null(t["verbatimscientificname"], ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("listed", nargs="?", help="listed species table (default: the configured one)")
    ap.add_argument("--config")
    ap.add_argument("--out", help="output path without suffix (default: the configured name_reassignment)")
    a = ap.parse_args()
    cfg = config.load(a.config)
    R = cfg.root
    listed_table = a.listed or cfg.path(cfg["species"].get("listed_table"))
    if listed_table is None:
        sys.exit("no listed species table configured: no record changes species")
    a.out = a.out or str(cfg.path(cfg["species"]["name_reassignment"])).removesuffix(".parquet")
    dset = dataset(cfg.path(v) / "parquet" for v in cfg["sources"]["gbif_downloads"].values())
    # distinct (interpreted species, verbatim name) pairs with their record counts
    pairs = pa.concat_tables(
        t.group_by(KEY).aggregate([("gbifid", "count")]) for t in batches(dset)
    ).group_by(KEY).aggregate([("gbifid_count", "sum")]).to_pandas().rename(columns={"gbifid_count_sum": "n"})
    tab = pd.read_csv(listed_table)
    listed = set(tab.wcvp_accepted_name)
    wc = names.load_wcvp_names(R / "raw/wcvp")
    # default rule: GBIF species name -> listed species (as run_fit selects records)
    resolved = names.gbif_names_by_accepted(pairs.species[pairs.species != ""].unique(), wc)
    default_of = {}
    for w, g in zip(tab.wcvp_accepted_name, tab.gbif_name):
        for n in {w, g} | set(resolved.get(w, [])):
            default_of.setdefault(n, w)
    pairs["default"] = pairs.species.map(default_of)          # "" (no interpreted species) maps to none
    # verbatim rule: original name -> WCVP accepted species. The full name with its infraspecific rank is resolved
    # first ("Rhus aromatica var. trilobata" -> Rhus trilobata), the binomial only if that fails.
    pairs["canonical"] = pairs.verbatimscientificname.map(names.canonical)
    pairs["binomial"] = pairs.verbatimscientificname.map(names.binomial)
    queries = pd.Series(sorted(set(pairs.canonical.dropna()) | set(pairs.binomial.dropna())))
    r = names.resolve(queries, wc).dropna(subset=["wcvp_accepted_name"])
    to_species = lambda x: " ".join(x.split()[:3]) if " × " in x else " ".join(x.split()[:2])
    verb = dict(zip(r["query"], r["wcvp_accepted_name"].map(to_species)))
    pairs["verbatim"] = pairs.canonical.map(verb).fillna(pairs.binomial.map(verb))
    move = pairs[pairs.verbatim.isin(listed) & (pairs.verbatim != pairs.default)]
    # the records of the moving pairs
    mv = pa.Table.from_pandas(move[KEY + ["verbatim"]].reset_index(drop=True))
    out = []
    for t in batches(dset):
        j = t.join(mv, keys=KEY, join_type="inner")
        out.append(j.select(["gbifid", "verbatim"]).rename_columns(["gbifid", "species"]).to_pandas())
    rec = pd.concat(out, ignore_index=True).sort_values("gbifid", ignore_index=True)
    rec.to_parquet(f"{a.out}.parquet")
    gained = move.groupby("verbatim").n.sum().rename("gained")
    lost = move.dropna(subset=["default"]).groupby("default").n.sum().rename("lost")
    base = pairs.groupby("default").n.sum().rename("default_records")
    summ = pd.concat([base, gained, lost], axis=1).fillna(0).astype(int)
    summ = summ[summ.index.isin(listed)]
    summ["change"] = (summ.gained - summ.lost) / summ.default_records.clip(lower=1)
    summ.sort_values("change").to_csv(f"{a.out}_summary.csv")
    print(f"{len(rec)} records reassigned between listed species; {int((summ.gained + summ.lost > 0).sum())} species touched; "
          f"|change| > 1%: {int((summ.change.abs() > 0.01).sum())}, > 5%: {int((summ.change.abs() > 0.05).sum())}")
    print(summ[summ.change.abs() > 0.05].sort_values("change").to_string())


if __name__ == "__main__":
    main()
