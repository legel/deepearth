"""Register missing pollinator species (derived/pending_pollinator_missing.json) into the 8111-taxon pollinator
vocab with inductive placement -- the pollinator analog of add_species.py. For each new species:
  * GBIF species/match -> Kingdom..Genus (for the BioCLIP-2.5 text string).
  * BioCLIP-2.5 text seed (frozen text prior, rule 26) -> appended to pollinator_taxon_text_emb.npy.
  * inductive OU distance: pollinator_distance.npy IS the OU distance source (analog of plant E1). Congeners already
    in the vocab -> the new species' distance row = mean of its congeners' rows (relatives near relatives); else the
    BioCLIP-2.5-text-nearest species' row. Extended SYMMETRICALLY with a zero diagonal; new-new pairs get the typical
    within-genus distance if congeneric, else the global mean distance.
  * pollinator_vocab.csv extended consistently (pollinator_idx, pollinator_taxon, kingdom, count).
Idempotent (skips species already in the vocab). After running: re-run map_pollinator_species.py so the newly-covered
obs get species_local. Backs up the three modified artifacts first.
"""
import os, csv, json, glob, shutil
import numpy as np, requests
from pathlib import Path

D = Path(os.environ.get("DEEPCAL", "/home/legel/deepcal/data/deepcal"))

def gbif_taxonomy(name):
    for _ in range(3):
        try:
            r = requests.get("https://api.gbif.org/v1/species/match", params={"name": name}, timeout=30).json()
            return [r.get(k, "") or "" for k in ("kingdom", "phylum", "class", "order", "family", "genus")]
        except Exception:
            pass
    return ["", "", "", "", "", ""]

def bioclip25_text(strings, dev="cpu"):
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 64):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 64]]).to(dev))
            out.append(F.normalize(t, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def obs_counts(species_lower):
    """count obs per (lowercased) species across the pollinator obs shards."""
    cnt = {s: 0 for s in species_lower}
    for f in glob.glob(str(D / "gbif_pollinator_obs/*.npz")):
        z = np.load(f, allow_pickle=True)
        for s in z["species"]:
            k = str(s).strip().lower()
            if k in cnt: cnt[k] += 1
    return cnt

def main():
    pend = json.load(open(D / "derived/pending_pollinator_missing.json"))
    rows = list(csv.reader(open(D / "pollinator/pollinator_vocab.csv")))
    header, body = rows[0], [r for r in rows[1:] if r]
    have = {r[1].strip().lower(): int(r[0]) for r in body}
    n0 = len(body)                                                   # 8111
    genus_idx = {}
    for r in body:
        genus_idx.setdefault(r[1].split()[0].lower(), []).append(int(r[0]))
    new = [s for s in pend if s.strip().lower() not in have]
    print(f"{len(pend)} pending, {len(new)} to register (vocab {n0})", flush=True)
    if not new:
        return

    text = np.load(D / "pollinator_taxon_text_emb.npy").astype(np.float32)   # [n0,1024]
    dist = np.load(D / "pollinator_distance.npy")                            # [n0,n0] f32
    assert text.shape[0] == n0 and dist.shape == (n0, n0), (text.shape, dist.shape)
    for p in ("pollinator_taxon_text_emb.npy", "pollinator_distance.npy"):   # back up before overwrite
        if not (D / (p + ".pre46.bak")).exists():
            shutil.copy(D / p, D / (p + ".pre46.bak"))
    shutil.copy(D / "pollinator/pollinator_vocab.csv", D / "pollinator/pollinator_vocab.csv.pre46.bak")

    tax = [gbif_taxonomy(s) for s in new]
    strings = [f"{t[0]} {t[1]} {t[2]} {t[3]} {t[4]} {t[5]} {s}".replace("  ", " ").strip() for s, t in zip(new, tax)]
    txt = bioclip25_text(strings)                                            # [K,1024] BioCLIP-2.5 seed

    K = len(new); M = n0 + K
    newdist = np.zeros((M, M), np.float32); newdist[:n0, :n0] = dist
    # typical within-genus distance (for new-new congeneric pairs) + global mean (else)
    wg = [dist[np.ix_(g, g)][np.triu_indices(len(g), 1)].mean() for g in genus_idx.values() if len(g) > 1]
    within = float(np.median(wg)) if wg else float(dist[np.triu_indices(n0, 1)].mean())
    gmean = float(dist[np.triu_indices(n0, 1)].mean())
    placed = []
    for k, s in enumerate(new):
        g = s.split()[0].lower(); cong = genus_idx.get(g, [])
        if cong:
            row = dist[cong].mean(0); how = f"{len(cong)} congeners"
        else:
            j = int((text @ txt[k]).argmax()); row = dist[j].copy(); how = f"text-nearest #{j}"
        newdist[n0 + k, :n0] = row; newdist[:n0, n0 + k] = row
        placed.append((s, how))
    for a in range(K):                                                       # new-new block
        for b in range(a + 1, K):
            d = within if new[a].split()[0].lower() == new[b].split()[0].lower() else gmean
            newdist[n0 + a, n0 + b] = newdist[n0 + b, n0 + a] = d
    np.fill_diagonal(newdist, 0.0)

    cnt = obs_counts([s.strip().lower() for s in new])
    np.save(D / "pollinator_distance.npy", newdist)
    np.save(D / "pollinator_taxon_text_emb.npy", np.concatenate([text, txt]))
    with open(D / "pollinator/pollinator_vocab.csv", "a", newline="") as f:
        w = csv.writer(f)
        for k, s in enumerate(new):
            w.writerow([n0 + k, s, tax[k][0] or "Animalia", cnt.get(s.strip().lower(), 0)])
    print(f"registered {K} species -> vocab {M}; dist {newdist.shape}; text {n0+K} rows", flush=True)
    for s, how in placed[:12]:
        print(f"  {s:32} <- {how}", flush=True)

if __name__ == "__main__":
    main()
