"""Occurrence coverage of every US native species (the configured species table) in the GBIF downloads present, under
the record-selection rules of run_fit.py: records whose interpreted GBIF species resolves through WCVP to the species
(default), else records identified under any WCVP name of the species (verbatim fallback, used only where the
default finds nothing). Name reassignment between listed species (ledger L17) is not applied here; it moves 0.1% of
records.

Writes work/national_name_counts.parquet (records per download, interpreted species and verbatim name) and the
configured inventory (work/national_inventory.csv; per species: n_default, n_verbatim, n_records, GBIF
species-match level). The inventory also gives the species without records their native areas (zero-shot maps).

usage: inventory_us_natives.py [--reuse-counts] [--config configs/conus.json]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges import names  # noqa: E402

KEY = ["species", "verbatimscientificname"]


def name_counts(downloads: dict) -> pd.DataFrame:
    """Records per (interpreted species, verbatim name) in each GBIF download directory (key -> parquet dir)."""
    out = []
    for tag, d in downloads.items():
        if not d.exists():
            continue
        parts = [str(f) for f in sorted(d.rglob("*")) if f.is_file() and f.stat().st_size > 0]
        aggs = [pa.Table.from_batches([b]).group_by(KEY).aggregate([("gbifid", "count")])
                for b in ds.dataset(parts, format="parquet").to_batches(columns=KEY + ["gbifid"], batch_size=5_000_000)]
        a = pa.concat_tables(aggs).group_by(KEY).aggregate([("gbifid_count", "sum")]).to_pandas()
        out.append(a.rename(columns={"gbifid_count_sum": "n"}).assign(download=tag))
        print(f"{tag}: {int(out[-1].n.sum()):,} records", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse-counts", action="store_true", help="read work/national_name_counts.parquet")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    R = cfg.root
    downloads = {k: cfg.path(v) / "parquet" for k, v in cfg["sources"]["gbif_downloads"].items()}
    cp = R / "work/national_name_counts.parquet"
    c = pd.read_parquet(cp) if a.reuse_counts else name_counts(downloads)
    if not a.reuse_counts:
        c.to_parquet(cp)
    t = pd.read_csv(cfg.path(cfg["species"]["table"]))
    wc = names.load_wcvp_names(R / "raw/wcvp")
    by_sp = c.groupby("species").n.sum()
    resolved = names.gbif_names_by_accepted(by_sp.index, wc)
    names_of = {w: sorted({w, g} | set(resolved.get(w, []))) for w, g in zip(t.wcvp_accepted_name, t.gbif_name)}
    t["n_default"] = [int(by_sp.reindex(names_of[w]).fillna(0).sum()) for w in t.wcvp_accepted_name]
    v = c.groupby("verbatimscientificname").n.sum()
    by_binomial = v.groupby(v.index.map(names.binomial).values).sum()
    t["n_verbatim"] = [0 if d else int(by_binomial.reindex(names.names_of_accepted(w, wc)).fillna(0).sum())
                       for w, d in zip(t.wcvp_accepted_name, t.n_default)]
    t["n_records"] = t.n_default + t.n_verbatim
    k = pd.read_csv(R / "work/us_natives_gbif_keys.csv")
    ok = k["match"].isin(["EXACT", "FUZZY"]) & k["rank"].isin(["SPECIES", "SUBSPECIES", "VARIETY", "FORM"])
    level = np.where(ok, "species", k["match"].fillna("NONE").astype(str) + ":" + k["rank"].fillna("").astype(str))
    t["gbif_match"] = t.wcvp_accepted_name.map(dict(zip(k.name, level)))
    out = cfg.path(cfg["species"]["inventory"])
    t.to_csv(out, index=False)
    print(f"{len(t)} species: 0 records {int((t.n_records == 0).sum())}, "
          f"verbatim only {int(((t.n_default == 0) & (t.n_verbatim > 0)).sum())}, "
          f"1-4 {int(t.n_records.between(1, 4).sum())}, 5-19 {int(t.n_records.between(5, 19).sum())}, "
          f">=20 {int((t.n_records >= 20).sum())}; median {t.n_records.median():.0f}")
    print(t.groupby("us_regions").n_records.agg(n="size", zero=lambda x: int((x == 0).sum()),
                                                under5=lambda x: int((x < 5).sum())).to_string())


if __name__ == "__main__":
    main()
