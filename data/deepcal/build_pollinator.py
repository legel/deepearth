"""DeepCal POLLINATOR-modality build scripts, consolidated into one dispatcher.

Each subcommand is an original standalone run-once builder, preserved verbatim. Invoke as:
    python -m deepearth.data.deepcal.build_pollinator <cmd> [args]
  cmds: add_pollinator_species, build_pollinator_text, clean_pollinator_vocab, gbif_resolve_pollinators,
        map_pollinator_species, merge_pollinator_vocab, remap_pollinator_interactions, pollinator_dated_distance

See each cmd_*() docstring for the original script's purpose and usage.
"""
import os, csv, json, glob, re, time, shutil, argparse, threading
import urllib.request, urllib.parse
from pathlib import Path
import numpy as np

for _a, _t in [("float_", np.float64), ("int_", np.int64), ("uint", np.uint64),
               ("unicode_", np.str_), ("complex_", np.complex128), ("bool_", np.bool_)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)


# ============================================================================ add_pollinator_species
"""Register missing pollinator species (derived/pending_pollinator_missing.json) into the 8111-taxon pollinator
vocab with inductive placement -- the pollinator analog of add_species. GBIF taxonomy + BioCLIP-2.5 text seed +
inductive OU distance from congeners (else text-nearest). Idempotent. Backs up the three modified artifacts first.

    python -m deepearth.data.deepcal.build_pollinator add_pollinator_species
"""
_APS_D = Path(os.environ.get("DEEPCAL", "/home/legel/deepcal/data/deepcal"))

def _aps_gbif_taxonomy(name):
    import requests
    for _ in range(3):
        try:
            r = requests.get("https://api.gbif.org/v1/species/match", params={"name": name}, timeout=30).json()
            return [r.get(k, "") or "" for k in ("kingdom", "phylum", "class", "order", "family", "genus")]
        except Exception:
            pass
    return ["", "", "", "", "", ""]

def _aps_bioclip25_text(strings, dev="cpu"):
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 64):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 64]]).to(dev))
            out.append(F.normalize(t, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def _aps_obs_counts(species_lower):
    """count obs per (lowercased) species across the pollinator obs shards."""
    cnt = {s: 0 for s in species_lower}
    for f in glob.glob(str(_APS_D / "gbif_pollinator_obs/*.npz")):
        z = np.load(f, allow_pickle=True)
        for s in z["species"]:
            k = str(s).strip().lower()
            if k in cnt: cnt[k] += 1
    return cnt

def cmd_add_pollinator_species(argv):
    argparse.ArgumentParser(prog="build_pollinator add_pollinator_species").parse_args(argv)
    D = _APS_D
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

    tax = [_aps_gbif_taxonomy(s) for s in new]
    strings = [f"{t[0]} {t[1]} {t[2]} {t[3]} {t[4]} {t[5]} {s}".replace("  ", " ").strip() for s, t in zip(new, tax)]
    txt = _aps_bioclip25_text(strings)                                      # [K,1024] BioCLIP-2.5 seed

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

    cnt = _aps_obs_counts([s.strip().lower() for s in new])
    np.save(D / "pollinator_distance.npy", newdist)
    np.save(D / "pollinator_taxon_text_emb.npy", np.concatenate([text, txt]))
    with open(D / "pollinator/pollinator_vocab.csv", "a", newline="") as f:
        w = csv.writer(f)
        for k, s in enumerate(new):
            w.writerow([n0 + k, s, tax[k][0] or "Animalia", cnt.get(s.strip().lower(), 0)])
    print(f"registered {K} species -> vocab {M}; dist {newdist.shape}; text {n0+K} rows", flush=True)
    for s, how in placed[:12]:
        print(f"  {s:32} <- {how}", flush=True)


# ============================================================================ build_pollinator_text
"""Rebuild the pollinator BioCLIP-2.5 text prior for the CLEANED vocab (pollinator_vocab_final.csv). Writes
pollinator_taxon_text_emb_clean.npy. GBIF taxonomy cached to disk.

    python -m deepearth.data.deepcal.build_pollinator build_pollinator_text --device cuda:1
"""
_BPT_D = Path(os.environ.get("POLL_DIR", "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator"))
_BPT_CACHE = _BPT_D / "gbif_taxonomy_cache.json"
_BPT_API = "https://api.gbif.org/v1/species/match?name="

def _bpt_gbif_taxonomy(name, cache):
    if name in cache:
        return cache[name]
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(_BPT_API + urllib.parse.quote(name), timeout=12))
            t = [r.get(k, "") or "" for k in ("kingdom", "phylum", "class", "order", "family", "genus")]
            cache[name] = t; return t
        except Exception:
            time.sleep(0.6)
    cache[name] = ["", "", "", "", "", ""]; return cache[name]

def _bpt_bioclip25_text(strings, dev):
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 64):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 64]]).to(dev))
            out.append(F.normalize(t, dim=-1).float().cpu())
    return torch.cat(out).numpy()

def cmd_build_pollinator_text(argv):
    ap = argparse.ArgumentParser(prog="build_pollinator build_pollinator_text"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    D = _BPT_D
    rows = list(csv.DictReader(open(D / "pollinator_vocab_final.csv")))              # final_idx order
    names = [r["taxon"] for r in rows]
    cache = json.load(open(_BPT_CACHE)) if _BPT_CACHE.exists() else {}
    todo = [n for n in dict.fromkeys(names) if n not in cache]                       # unique, uncached
    print(f"{len(names)} names, {len(todo)} to fetch from GBIF (parallel)", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    done = [0]; lock = threading.Lock()
    def fetch(n):
        _bpt_gbif_taxonomy(n, cache)                                        # writes cache[n] (atomic under GIL)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0: print(f"  taxonomy {done[0]}/{len(todo)}", flush=True)   # count only; no dump mid-threads
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch, todo))
    tmp = _BPT_CACHE.with_suffix(".tmp"); json.dump(cache, open(tmp, "w")); os.replace(tmp, _BPT_CACHE)   # atomic dump
    strings, miss = [], 0
    for n in names:
        K, P, C, O, Fa, G = cache.get(n, ["", "", "", "", "", ""])
        if not K: miss += 1
        strings.append(f"{K} {P} {C} {O} {Fa} {G} {n}".replace("  ", " ").strip())
    print(f"{len(strings)} pollinators | {miss} no-GBIF-kingdom | e.g. {strings[0]!r}", flush=True)
    emb = _bpt_bioclip25_text(strings, a.device)
    np.save(D / "pollinator_taxon_text_emb_clean.npy", emb)
    print(f"wrote pollinator_taxon_text_emb_clean.npy {emb.shape} norm mean {np.linalg.norm(emb, axis=1).mean():.4f}", flush=True)


# ============================================================================ clean_pollinator_vocab
"""Audit + clean the pollinator vocabulary (GloBI-sourced, ~8111 entries) — NON-DESTRUCTIVE: reads
pollinator_vocab.csv, writes pollinator_vocab_clean.csv + pollinator_vocab_map.json + prints a full report.

    python -m deepearth.data.deepcal.build_pollinator clean_pollinator_vocab   # POLL_VOCAB env overrides the source path
"""
_CPV_NON_ANIMAL = {"Plantae", "Archaeplastida", "Viridiplantae", "Fungi", "Chromista", "Archaea"}
_CPV_QUALIFIER = re.compile(r"\b(cf|aff|sp|nr|near|indet)\.?\b", re.I)
_CPV_FAMILY = re.compile(r"(idae|inae|ini)$")

def _cpv_normalize(t):
    t = t.strip().replace("_", " ")
    t = _CPV_QUALIFIER.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    w = t.split()
    if len(w) >= 2 and w[0] == w[1]:                       # "Merodon Merodon equestris" -> "Merodon equestris"
        w = w[1:]
    if len(w) == 1:                                        # bare "Hylaeus1" -> "Hylaeus"
        w0 = re.sub(r"\d+$", "", w[0])
        return w0
    return " ".join(w[:2])                                 # genus + species (drop trailing authorities/codes)

def cmd_clean_pollinator_vocab(argv):
    argparse.ArgumentParser(prog="build_pollinator clean_pollinator_vocab").parse_args(argv)
    SRC = Path(os.environ.get("POLL_VOCAB",
          "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab.csv"))
    OUT = SRC.parent
    rows = list(csv.DictReader(open(SRC)))
    # known animal genera (from Animalia/Metazoa rows) — for empty-kingdom recovery
    anim_genera = {_cpv_normalize(r["pollinator_taxon"]).split()[0]
                   for r in rows if r.get("kingdom", "") in ("Animalia", "Metazoa")
                   and _cpv_normalize(r["pollinator_taxon"])}
    kept, dropped, unresolved, mp = [], [], [], {}
    reasons = {}
    for r in rows:
        oi = int(r["pollinator_idx"]); k = r.get("kingdom", "").strip()
        name = _cpv_normalize(r["pollinator_taxon"])
        genus = name.split()[0] if name else ""
        if not name or len(genus) < 3:
            reason = "empty/degenerate"
        elif k in _CPV_NON_ANIMAL:
            reason = f"non-animal:{k}"
        elif len(name.split()) == 1 and _CPV_FAMILY.search(name):
            reason = "family-level"
        elif k in ("Animalia", "Metazoa"):
            reason = None
        elif not k or k == "Eukaryota":                    # empty or SUPER-kingdom (e.g. Apis mellifera is labeled
            #                                                'Eukaryota') -> recover if the genus is a known animal genus, else flag
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


# ============================================================================ gbif_resolve_pollinators
"""Resolve the pollinator_vocab_unresolved.csv (706 empty/super-kingdom names flagged by clean_pollinator_vocab)
against the GBIF backbone. Writes pollinator_vocab_resolved.csv (recovered animals) + a report.

    python -m deepearth.data.deepcal.build_pollinator gbif_resolve_pollinators   # UNRESOLVED env overrides input path
"""
_GRP_API = "https://api.gbif.org/v1/species/match?name="

def _grp_variants(t):
    """Try the name, and a space-split for concatenated 'Bombusmixtus' -> 'Bombus mixtus' (lowercase run split)."""
    yield t
    m = re.match(r"^([A-Z][a-z]+)([a-z]{4,})$", t.replace(" ", ""))   # GenusSpecies concatenated
    if m:
        yield f"{m.group(1)} {m.group(2)}"

def _grp_match(name):
    for v in _grp_variants(name):
        try:
            r = json.load(urllib.request.urlopen(_GRP_API + urllib.parse.quote(v), timeout=12))
            if r.get("matchType") not in (None, "NONE") and r.get("kingdom"):
                return r.get("canonicalName") or v, r.get("kingdom"), r.get("rank"), r.get("confidence")
        except Exception:
            time.sleep(0.5)
    return None, None, None, None

def cmd_gbif_resolve_pollinators(argv):
    argparse.ArgumentParser(prog="build_pollinator gbif_resolve_pollinators").parse_args(argv)
    SRC = Path(os.environ.get("UNRESOLVED",
          "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab_unresolved.csv"))
    OUT = SRC.parent
    rows = list(csv.DictReader(open(SRC)))
    recovered, non_animal, unmatched = [], 0, 0
    for i, r in enumerate(rows):
        canon, kingdom, rank, conf = _grp_match(r["taxon"])
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


# ============================================================================ map_pollinator_species
"""Assign species_local to the pollinator obs from the staged 8111-taxon pollinator vocab. Species not in the vocab
are written to pending_pollinator_missing.json for add_pollinator_species. Idempotent."""

def cmd_map_pollinator_species(argv):
    argparse.ArgumentParser(prog="build_pollinator map_pollinator_species").parse_args(argv)
    D = Path("/home/legel/deepcal/data/deepcal")
    pv = {}
    for r in csv.reader(open(D/"pollinator/pollinator_vocab.csv")):
        if r and r[0].isdigit(): pv[r[1].strip().lower()] = int(r[0])
    shards = sorted(glob.glob(str(D/"gbif_pollinator_obs/*.npz")))
    mapped = missing = 0; missing_sp = {}
    for f in shards:
        z = dict(np.load(f, allow_pickle=True))
        sl = np.full(len(z["gbifID"]), -1, np.int32)
        for i, s in enumerate(z["species"]):
            k = str(s).strip().lower(); j = pv.get(k)
            if j is not None: sl[i] = j; mapped += 1
            else:
                missing += 1
                if k and k != "nan": missing_sp[k] = z
        z["species_local"] = sl
        np.savez(f, **z)                                  # overwrite shard with the pollinator species_local
    print(f"mapped {mapped} obs to pollinator vocab | {missing} obs of {len(missing_sp)} species NOT in vocab")
    json.dump(sorted(missing_sp), open(D/"derived/pending_pollinator_missing.json", "w"))
    print("wrote pending_pollinator_missing.json (species for add_pollinator_species)")


# ============================================================================ merge_pollinator_vocab
"""Merge the cleaned + GBIF-resolved pollinator vocab into the FINAL clean vocab + the old->new index map. Writes
pollinator_vocab_final.csv + pollinator_vocab_final_map.json.

    python -m deepearth.data.deepcal.build_pollinator merge_pollinator_vocab
"""
_MPV_D = Path(os.environ.get("POLL_DIR",
    "/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator"))

def cmd_merge_pollinator_vocab(argv):
    argparse.ArgumentParser(prog="build_pollinator merge_pollinator_vocab").parse_args(argv)
    D = _MPV_D
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


# ============================================================================ remap_pollinator_interactions
"""Reindex the plant->pollinator GloBI interaction labels (gbif_pollinator_dist.npz) from the OLD pollinator vocab
to the CLEANED one, via pollinator_vocab_final_map.json. Writes gbif_pollinator_dist_clean.npz.

    python -m deepearth.data.deepcal.build_pollinator remap_pollinator_interactions
"""
_RPI_MAP = Path("/home/photon/ecological/sandbox/deepearth/data/deepcal/pollinator/pollinator_vocab_final_map.json")

def cmd_remap_pollinator_interactions(argv):
    argparse.ArgumentParser(prog="build_pollinator remap_pollinator_interactions").parse_args(argv)
    D = Path(os.environ.get("DEEPCAL_DATA_DIR", "/home/photon/4tb/deepcal_dogfood/data/deepcal"))
    z = dict(np.load(D / "gbif_pollinator_dist.npz", allow_pickle=True))
    m = {int(k): v for k, v in json.load(open(_RPI_MAP)).items()}             # old_idx -> final_idx | None
    W = z["marg_poll_idx"].shape[1]                                          # top-K width (40)
    pidx, pfrq, npoll = z["marg_poll_idx"], z["marg_poll_frq"], z["marg_npoll"]
    n_plants = len(npoll)
    new_idx = np.full_like(pidx, -1); new_frq = np.zeros_like(pfrq); new_np = np.zeros_like(npoll)
    kept = dropped = 0
    for p in range(n_plants):
        row = []
        for k in range(min(int(npoll[p]), W)):
            fi = m.get(int(pidx[p, k]))
            if fi is not None:
                row.append((fi, pfrq[p, k])); kept += 1
            else:
                dropped += 1
        for j, (fi, fr) in enumerate(row[:W]):
            new_idx[p, j] = fi; new_frq[p, j] = fr
        new_np[p] = len(row)
    z["marg_poll_idx"] = new_idx; z["marg_poll_frq"] = new_frq; z["marg_npoll"] = new_np

    # NOTE: loc_*/spacetime_coo left UNCHANGED — their column semantics aren't referenced in data.py/evaluate.py
    # (likely builder intermediates, not model inputs); do not remap what we don't understand (a col-1=pollinator
    # assumption dropped 99.5%, clearly wrong). Clarify + remap-or-drop them before they're used as model inputs.

    np.savez(D.parent / "deepcal" / "gbif_pollinator_dist_clean.npz", **z)
    print(f"marg interactions: {kept} kept / {dropped} dropped | wrote gbif_pollinator_dist_clean.npz", flush=True)


# ============================================================================ pollinator_dated_distance
"""Assemble the pollinator distance from DATED clade patristics + the topological/text backbone. For each clade with
a published chronogram, replace its within-clade block of pollinator_distance.npy with the clade's dated patristic,
rescaled to preserve the global scale. Backs up the topological matrix first.

    python -m deepearth.data.deepcal.build_pollinator pollinator_dated_distance
"""
_PDD_COPHEN = Path("/home/photon/4tb/deepcal_data/trees/dated_cophen")
_PDD_CLADES = ["bees", "ants", "butterflies", "moths", "birds"]

def _pdd_binom_to_index(here):
    m = {}
    for r in csv.reader(open(here / "pollinator" / "pollinator_vocab.csv")):
        if r and r[0].isdigit():
            m[r[1].strip().lower().replace(" ", "_")] = int(r[0])
    return m

def cmd_pollinator_dated_distance(argv):
    argparse.ArgumentParser(prog="build_pollinator pollinator_dated_distance").parse_args(argv)
    HERE = Path(__file__).resolve().parent
    DIST = HERE / "pollinator_distance.npy"
    b2i = _pdd_binom_to_index(HERE)
    topo = HERE / "pollinator_distance_topo.npy"
    if not topo.exists():                                             # preserve the ORIGINAL topological+text matrix once
        np.save(topo, np.load(DIST).astype(np.float32))
    D = np.load(topo).astype(np.float64)                              # always rebuild from the topological original (idempotent)
    print(f"distance {D.shape}, vocab-mapped binomials {len(b2i)}")
    total_overwritten = 0
    for clade in _PDD_CLADES:
        f = _PDD_COPHEN / f"{clade}_cophen.csv"
        if not f.exists():
            print(f"  {clade}: no cophen, skip"); continue
        rows = list(csv.reader(open(f)))
        labels = [c.strip().lower() for c in rows[0][1:]]              # header: "",tip1,tip2,...
        M = np.array([[float(x) for x in r[1:]] for r in rows[1:]], np.float64)   # dated patristic (Myr)
        idx = np.array([b2i.get(l, -1) for l in labels])
        keep = idx >= 0
        if keep.sum() < 3:
            print(f"  {clade}: <3 mapped, skip"); continue
        sub = np.where(keep)[0]; gi = idx[keep]                        # rows/cols to use; their global indices
        Msub = M[np.ix_(sub, sub)]
        cur = D[np.ix_(gi, gi)]                                        # current (topological) block
        cm, dm = cur[~np.eye(len(gi), dtype=bool)].mean(), Msub[~np.eye(len(sub), dtype=bool)].mean()
        Msub = Msub * (cm / dm) if dm > 0 else Msub                   # rescale dated block -> match current block mean
        np.fill_diagonal(Msub, 0.0)
        D[np.ix_(gi, gi)] = Msub                                       # overwrite within-clade with dated structure
        total_overwritten += len(gi)
        print(f"  {clade}: {len(gi)} taxa dated | block mean {cm:.3f} kept | congener range [{Msub[Msub>0].min():.4f}, {Msub.max():.3f}]")
    D = 0.5 * (D + D.T)                                                # enforce symmetry
    D = np.clip(D, 0.0, None)                                          # clip tiny pre-existing float noise (~-1e-7)
    np.save(DIST, D.astype(np.float32))
    print(f"wrote {DIST} ({total_overwritten} taxa now on dated patristics; topological backup -> pollinator_distance_topo.npy)")


if __name__ == "__main__":
    import sys
    cmds = {
        "add_pollinator_species": cmd_add_pollinator_species,
        "build_pollinator_text": cmd_build_pollinator_text,
        "clean_pollinator_vocab": cmd_clean_pollinator_vocab,
        "gbif_resolve_pollinators": cmd_gbif_resolve_pollinators,
        "map_pollinator_species": cmd_map_pollinator_species,
        "merge_pollinator_vocab": cmd_merge_pollinator_vocab,
        "remap_pollinator_interactions": cmd_remap_pollinator_interactions,
        "pollinator_dated_distance": cmd_pollinator_dated_distance,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("usage: python -m deepearth.data.deepcal.build_pollinator <cmd> [args]\n  cmds: " + ", ".join(cmds))
        sys.exit(1)
    cmds[sys.argv[1]](sys.argv[2:])
