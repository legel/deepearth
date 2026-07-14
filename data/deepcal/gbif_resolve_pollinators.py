"""Resolve the pollinator_vocab_unresolved.csv (706 empty/super-kingdom names flagged by clean_pollinator_vocab.py)
against the GBIF backbone (species/match API) — recover the real animals (with canonical names), drop non-animals,
leave the genuinely-unmatchable flagged (mandate: resolve, don't silently drop). NON-DESTRUCTIVE: writes
pollinator_vocab_resolved.csv (recovered animals) + a report. Network task (~1 req/name).

    python data/deepcal/gbif_resolve_pollinators.py     # UNRESOLVED env overrides the input path
"""
import os, csv, json, re, time
import urllib.request, urllib.parse
from pathlib import Path

SRC = Path(os.environ.get("UNRESOLVED",
      "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab_unresolved.csv"))
OUT = SRC.parent
API = "https://api.gbif.org/v1/species/match?name="


def _variants(t):
    """Try the name, and a space-split for concatenated 'Bombusmixtus' -> 'Bombus mixtus' (lowercase run split)."""
    yield t
    m = re.match(r"^([A-Z][a-z]+)([a-z]{4,})$", t.replace(" ", ""))   # GenusSpecies concatenated
    if m:
        yield f"{m.group(1)} {m.group(2)}"


def match(name):
    for v in _variants(name):
        try:
            r = json.load(urllib.request.urlopen(API + urllib.parse.quote(v), timeout=12))
            if r.get("matchType") not in (None, "NONE") and r.get("kingdom"):
                return r.get("canonicalName") or v, r.get("kingdom"), r.get("rank"), r.get("confidence")
        except Exception:
            time.sleep(0.5)
    return None, None, None, None


def main():
    rows = list(csv.DictReader(open(SRC)))
    recovered, non_animal, unmatched = [], 0, 0
    for i, r in enumerate(rows):
        canon, kingdom, rank, conf = match(r["taxon"])
        if kingdom in ("Animalia",):
            recovered.append({"old_idx": r["old_idx"], "orig": r["taxon"], "canonical": canon, "rank": rank, "confidence": conf})
        elif kingdom is None:
            unmatched += 1
        else:
            non_animal += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} | recovered {len(recovered)} non-animal {non_animal} unmatched {unmatched}", flush=True)
    with open(OUT / "pollinator_vocab_resolved.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["old_idx", "orig", "canonical", "rank", "confidence"]); w.writeheader()
        w.writerows(recovered)
    print(f"DONE {len(rows)} flagged -> RECOVERED {len(recovered)} animals | non-animal {non_animal} | unmatched {unmatched}", flush=True)
    print(f"wrote pollinator_vocab_resolved.csv ({len(recovered)})", flush=True)


if __name__ == "__main__":
    main()
