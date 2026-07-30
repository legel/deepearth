"""Stage 2a HARVEST: pull candidate pollinator binomials (with full GBIF taxonomy) from the GBIF
backbone for the major pollinator clades, so we can embed them with BioCLIP-2.5 and NN-match to the
used pollinator_taxon_text_emb rows. Durable: writes derived/pollinator_candidates.jsonl incrementally."""
import sys, json, time, urllib.request, urllib.parse
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/deepearth/autoresearch/probes/biological")
from pathlib import Path
from ensue_log import log_stage

DER = Path("/workspace/deepearth/data/deepcal/derived")
OUT = DER / "pollinator_candidates.jsonl"
GBIF = "https://api.gbif.org/v1"

# Target higher taxa that dominate flower-visiting / pollinator interaction records.
CLADES = {
    "Anthophila": None,            # bees (epifamily) -> resolve to key via name lookup
    "Apidae": None, "Halictidae": None, "Megachilidae": None, "Andrenidae": None,
    "Colletidae": None, "Melittidae": None,          # bee families (Anthophila not a backbone rank)
    "Syrphidae": None,             # hoverflies
    "Lepidoptera": None,           # butterflies + moths (order)
    "Formicidae": None,            # ants
    "Trochilidae": None,           # hummingbirds
    "Vespidae": None, "Sphecidae": None, "Bombyliidae": None,   # wasps, bee-flies
    "Cetoniidae": None, "Scarabaeidae": None,        # flower beetles
    "Nymphalidae": None, "Pieridae": None, "Lycaenidae": None, "Hesperiidae": None,
    "Papilionidae": None, "Sphingidae": None, "Noctuidae": None, "Geometridae": None,
}


def _get(url, tries=4):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.7)
    return None


def resolve_key(name):
    r = _get(f"{GBIF}/species/match?name={urllib.parse.quote(name)}")
    if r and r.get("usageKey") and r.get("matchType") not in (None, "NONE"):
        return r["usageKey"], r.get("rank")
    return None, None


def harvest_species(key):
    """Page through all SPECIES descendants of a backbone key (rank=SPECIES, ACCEPTED)."""
    out = []
    offset = 0
    while True:
        url = (f"{GBIF}/species/search?highertaxonKey={key}&rank=SPECIES&status=ACCEPTED"
               f"&datasetKey=d7dddbf4-2cf0-4f39-9b2a-bb099caae36c&limit=1000&offset={offset}")
        r = _get(url)
        if not r or not r.get("results"):
            break
        for s in r["results"]:
            nm = s.get("canonicalName") or s.get("scientificName", "")
            if nm and len(nm.split()) >= 2:
                out.append({
                    "name": " ".join(nm.split()[:2]),
                    "kingdom": s.get("kingdom", ""), "phylum": s.get("phylum", ""),
                    "class": s.get("class", ""), "order": s.get("order", ""),
                    "family": s.get("family", ""), "genus": s.get("genus", ""),
                })
        offset += 1000
        if r.get("endOfRecords") or offset > 60000:
            break
    return out


def main():
    seen = set()
    n = 0
    with open(OUT, "w") as f:
        for name in CLADES:
            key, rank = resolve_key(name)
            if not key:
                print(f"  {name}: no GBIF key, skip", flush=True); continue
            sp = harvest_species(key)
            added = 0
            for s in sp:
                k = s["name"].lower()
                if k in seen:
                    continue
                seen.add(k); f.write(json.dumps(s) + "\n"); added += 1; n += 1
            f.flush()
            print(f"  {name} (key={key},{rank}): {len(sp)} species, {added} new -> total {n}", flush=True)
    msg = f"STAGE2a HARVEST ok. {n} unique candidate binomials with GBIF taxonomy -> {OUT.name}"
    print(msg, flush=True)
    log_stage("harvest", msg, "Stage2a: GBIF candidate pollinator binomials harvested")


if __name__ == "__main__":
    main()
