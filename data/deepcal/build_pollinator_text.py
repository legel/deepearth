"""Rebuild the pollinator BioCLIP-2.5 text prior for the CLEANED vocab (pollinator_vocab_final.csv, 7373 real animal
pollinators) — same recipe as the plant build_bioclip_text.py + add_pollinator_species.py: GBIF species/match ->
Kingdom..Genus, string "K P C O F G binomial" wrapped "a photo of {s}." -> frozen BioCLIP-2.5 ViT-H/14 text tower,
unit-normalized [N,1024]. NON-DESTRUCTIVE: writes pollinator_taxon_text_emb_clean.npy (the reindexed emb that
replaces the plant-contaminated one once the whole pollinator rebuild is validated). GBIF taxonomy cached to disk.

    python data/deepcal/build_pollinator_text.py --device cuda:1
"""
import os, csv, json, time, argparse
from pathlib import Path
import urllib.request, urllib.parse
import numpy as np

for _a, _t in [("float_", np.float64), ("int_", np.int64), ("unicode_", np.str_), ("complex_", np.complex128), ("bool_", np.bool_)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

D = Path(os.environ.get("POLL_DIR", "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator"))
CACHE = D / "gbif_taxonomy_cache.json"
API = "https://api.gbif.org/v1/species/match?name="


def gbif_taxonomy(name, cache):
    if name in cache:
        return cache[name]
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(API + urllib.parse.quote(name), timeout=12))
            t = [r.get(k, "") or "" for k in ("kingdom", "phylum", "class", "order", "family", "genus")]
            cache[name] = t; return t
        except Exception:
            time.sleep(0.6)
    cache[name] = ["", "", "", "", "", ""]; return cache[name]


def bioclip25_text(strings, dev):
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 64):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 64]]).to(dev))
            out.append(F.normalize(t, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    rows = list(csv.DictReader(open(D / "pollinator_vocab_final.csv")))              # final_idx order
    names = [r["taxon"] for r in rows]
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = [n for n in dict.fromkeys(names) if n not in cache]                       # unique, uncached
    print(f"{len(names)} names, {len(todo)} to fetch from GBIF (parallel)", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    import threading
    done = [0]; lock = threading.Lock()
    def fetch(n):
        gbif_taxonomy(n, cache)                                             # writes cache[n] (atomic under GIL)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0: print(f"  taxonomy {done[0]}/{len(todo)}", flush=True)   # count only; no dump mid-threads
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch, todo))
    tmp = CACHE.with_suffix(".tmp"); json.dump(cache, open(tmp, "w")); os.replace(tmp, CACHE)   # atomic dump
    strings, miss = [], 0
    for n in names:
        K, P, C, O, Fa, G = cache.get(n, ["", "", "", "", "", ""])
        if not K: miss += 1
        strings.append(f"{K} {P} {C} {O} {Fa} {G} {n}".replace("  ", " ").strip())
    print(f"{len(strings)} pollinators | {miss} no-GBIF-kingdom | e.g. {strings[0]!r}", flush=True)
    emb = bioclip25_text(strings, a.device)
    np.save(D / "pollinator_taxon_text_emb_clean.npy", emb)
    print(f"wrote pollinator_taxon_text_emb_clean.npy {emb.shape} norm mean {np.linalg.norm(emb, axis=1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
