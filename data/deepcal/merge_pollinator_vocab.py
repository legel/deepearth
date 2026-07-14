"""Merge the cleaned + GBIF-resolved pollinator vocab into the FINAL clean vocab + the old->new index map that the
emb/distance/backbone/interaction rebuilds reindex through. NON-DESTRUCTIVE (writes new files). Inputs (from
clean_pollinator_vocab.py + gbif_resolve_pollinators.py):
  pollinator_vocab_clean.csv     — 6981 kept (new_idx, old_idx, taxon, genus, kingdom, count)
  pollinator_vocab_resolved.csv  — 392 recovered from the flagged 706 (old_idx, orig, canonical, rank, confidence)
Output:
  pollinator_vocab_final.csv     — final vocab (final_idx, old_idx, taxon, source)
  pollinator_vocab_final_map.json— {old_idx(str) -> final_idx | null(dropped)} over ALL original 8111 indices

    python data/deepcal/merge_pollinator_vocab.py
"""
import os, csv, json
from pathlib import Path

D = Path(os.environ.get("POLL_DIR",
    "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator"))


def main():
    clean = list(csv.DictReader(open(D / "pollinator_vocab_clean.csv")))
    resolved = list(csv.DictReader(open(D / "pollinator_vocab_resolved.csv")))
    n_orig = max([int(r["old_idx"]) for r in clean] + [int(r["old_idx"]) for r in resolved]) + 1

    final, mp = [], {str(i): None for i in range(n_orig)}
    for r in clean:                                            # clean binomials/genus keep their normalized taxon
        fi = len(final); mp[r["old_idx"]] = fi
        final.append({"final_idx": fi, "old_idx": r["old_idx"], "taxon": r["taxon"], "source": "clean"})
    seen = {int(r["old_idx"]) for r in clean}
    for r in resolved:                                         # GBIF-recovered use the canonical name; skip any dup
        if int(r["old_idx"]) in seen:
            continue
        fi = len(final); mp[r["old_idx"]] = fi
        final.append({"final_idx": fi, "old_idx": r["old_idx"], "taxon": r["canonical"], "source": "gbif"})

    with open(D / "pollinator_vocab_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["final_idx", "old_idx", "taxon", "source"]); w.writeheader()
        w.writerows(final)
    json.dump(mp, open(D / "pollinator_vocab_final_map.json", "w"))
    kept = sum(1 for v in mp.values() if v is not None)
    print(f"final pollinator vocab: {len(final)} taxa ({sum(r['source']=='clean' for r in final)} clean + "
          f"{sum(r['source']=='gbif' for r in final)} gbif-recovered) | dropped {n_orig - kept}/{n_orig}", flush=True)
    print(f"wrote pollinator_vocab_final.csv + pollinator_vocab_final_map.json (over {n_orig} original indices)", flush=True)


if __name__ == "__main__":
    main()
