"""Audit + clean the pollinator vocabulary (GloBI-sourced, ~8111 entries) — NON-DESTRUCTIVE: reads
pollinator_vocab.csv, writes pollinator_vocab_clean.csv + pollinator_vocab_map.json (old_idx -> new_idx | null) +
prints a full report. Does NOT touch the shipped emb/distance/backbone (those get rebuilt from the clean vocab in a
later, directed step). Purpose: real pollinators only (rules 27, phase-order data-quality), no crude approximations.

Cleaning, in order (each auditable):
  1. NAME NORMALIZATION (deterministic): underscore->space, collapse whitespace, strip trailing specimen digits on a
     bare genus (``Hylaeus1``->``Hylaeus``), de-duplicate a leading repeated genus (``Merodon Merodon equestris``->
     ``Merodon equestris``), strip open-nomenclature qualifiers (cf./aff./sp./nr.).
  2. DROP non-animals by kingdom (Plantae/Archaeplastida/Viridiplantae/Fungi/Chromista/Archaea) — GloBI noise.
  3. DROP family/subfamily-level junk (a single token ending -idae/-inae/-ini is a clade, not a taxon).
  4. RESOLVE empty-kingdom rows heuristically: keep if the (normalized) genus is a known animal genus in this vocab;
     else FLAG for GBIF/OTT re-resolution (NOT silently dropped — written to pollinator_vocab_unresolved.csv).
Kept = clean binomial or genus-level animal taxon. The map lets the emb/distance/interaction rebuild reindex.

    python -m deepearth.data.deepcal.clean_pollinator_vocab            # POLL_VOCAB env overrides the source path
"""
import os, csv, re, json
from pathlib import Path

SRC = Path(os.environ.get("POLL_VOCAB",
      "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab.csv"))
OUT = SRC.parent
NON_ANIMAL = {"Plantae", "Archaeplastida", "Viridiplantae", "Fungi", "Chromista", "Archaea"}
QUALIFIER = re.compile(r"\b(cf|aff|sp|nr|near|indet)\.?\b", re.I)
FAMILY = re.compile(r"(idae|inae|ini)$")


def normalize(t):
    t = t.strip().replace("_", " ")
    t = QUALIFIER.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    w = t.split()
    if len(w) >= 2 and w[0] == w[1]:                       # "Merodon Merodon equestris" -> "Merodon equestris"
        w = w[1:]
    if len(w) == 1:                                        # bare "Hylaeus1" -> "Hylaeus"
        w0 = re.sub(r"\d+$", "", w[0])
        return w0
    return " ".join(w[:2])                                 # genus + species (drop trailing authorities/codes)


def main():
    rows = list(csv.DictReader(open(SRC)))
    # known animal genera (from Animalia/Metazoa rows) — for empty-kingdom recovery
    anim_genera = {normalize(r["pollinator_taxon"]).split()[0]
                   for r in rows if r.get("kingdom", "") in ("Animalia", "Metazoa")
                   and normalize(r["pollinator_taxon"])}
    kept, dropped, unresolved, mp = [], [], [], {}
    reasons = {}
    for r in rows:
        oi = int(r["pollinator_idx"]); k = r.get("kingdom", "").strip()
        name = normalize(r["pollinator_taxon"])
        genus = name.split()[0] if name else ""
        if not name or len(genus) < 3:
            reason = "empty/degenerate"
        elif k in NON_ANIMAL:
            reason = f"non-animal:{k}"
        elif len(name.split()) == 1 and FAMILY.search(name):
            reason = "family-level"
        elif k in ("Animalia", "Metazoa"):
            reason = None
        elif not k or k == "Eukaryota":                    # empty or SUPER-kingdom (e.g. Apis mellifera is labeled
            #                                                'Eukaryota') -> recover if the genus is a known animal genus, else flag for GBIF/OTT
            reason = None if genus in anim_genera else "unresolved-kingdom"
        else:
            reason = f"non-animal:{k}"
        if reason is None:
            mp[oi] = len(kept)
            kept.append({"new_idx": len(kept), "old_idx": oi, "taxon": name,
                         "genus": genus, "kingdom": k or "Animalia?", "count": r.get("count", "")})
        else:
            mp[oi] = None
            (unresolved if reason == "unresolved-kingdom" else dropped).append((oi, r["pollinator_taxon"], reason))
            reasons[reason] = reasons.get(reason, 0) + 1

    with open(OUT / "pollinator_vocab_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["new_idx", "old_idx", "taxon", "genus", "kingdom", "count"]); w.writeheader()
        w.writerows(kept)
    json.dump(mp, open(OUT / "pollinator_vocab_map.json", "w"))
    with open(OUT / "pollinator_vocab_unresolved.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["old_idx", "taxon", "reason"]); w.writerows(unresolved)

    print(f"source {len(rows)} -> kept {len(kept)} | dropped {len(dropped)} | "
          f"unresolved(empty-kingdom, flagged for GBIF/OTT) {len(unresolved)}", flush=True)
    print("drop reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])), flush=True)
    print(f"wrote pollinator_vocab_clean.csv ({len(kept)}), _map.json, _unresolved.csv ({len(unresolved)})", flush=True)


if __name__ == "__main__":
    main()
