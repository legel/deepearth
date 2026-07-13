"""Production species registration: add a species to the DeepCal vocabulary so the model can train/infer on it,
whether or not it already has a position in the phylogeny. Idempotent.

For each new binomial:
  * BioCLIP-2.5 text seed  — encode "Kingdom Phylum Class Order Family Genus species" (rule 26); appended to the frozen
    text prior so the species graph's probe gives it a phylogenomically-structured embedding immediately.
  * E1 evolutionary vector — the OU species-graph distance source. In-tree: reuse. Out-of-tree: INDUCTIVE PLACEMENT —
    borrow from relatives (E1 clusters by genus at cos .85 vs .00 across), i.e. mean E1 of congeners in the vocab, else
    the E1 of the BioCLIP-2.5-nearest known species. The new species lands right next to its relatives in the OU space.
  * vocab / species_index / traits — extended so n_classes grows by one, consistent across every per-species array.

A newly-added (out-of-tree) species needs only E1 at runtime: it takes the BioCLIP-embedding-shadow distance
(distance_from_embedding), so no tree surgery is required. Tree-covered species instead use the REAL dated cophenetic in
gbif_plant_dist.npz (blended by fusion.py); grafting a new species into ca_subtree.dated.nwk and rebuilding
gbif_plant_dist.npz (via plant_dated_distance.py) so it too gets a dated position is a documented follow-on.

Usage:
  python -m deepearth.data.deepcal.add_species --species "Clarkia williamsonii" --taxonomy "Plantae,Tracheophyta,Magnoliopsida,Myrtales,Onagraceae,Clarkia"
  python -m deepearth.data.deepcal.add_species --pending pending_species.json     # from add_observation.py
"""
import os, csv, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def _norm(s):
    p = str(s).split(); return (p[0] + " " + p[1]).lower() if len(p) >= 2 else str(s).strip().lower()

def bioclip25_text(strings, dev=None):
    """Encode taxonomy strings with the frozen BioCLIP-2.5 text tower (rule 26), unit-normalized [N,1024]."""
    import torch, open_clip, torch.nn.functional as F
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 128]]).to(dev))
            out.append(F.normalize(t, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=HERE)
    ap.add_argument("--species", default=None); ap.add_argument("--taxonomy", default=None,
                    help="Kingdom,Phylum,Class,Order,Family,Genus for --species")
    ap.add_argument("--pending", default=None, help="JSON list of observation records (from add_observation.py)")
    a = ap.parse_args(); cache = a.cache

    vocab = np.load(os.path.join(cache, "gbif_vocab.npz"), allow_pickle=True)
    E1 = vocab["E1"].astype(np.float32); binos = [str(b) for b in vocab["binomial"]]; gidx = list(vocab["global_idx"])
    have = {_norm(b): i for i, b in enumerate(binos)}
    genus_of = np.array([b.split()[0] for b in binos])
    text = np.load(os.path.join(cache, "bioclip_taxon_text_emb.npy")).astype(np.float32)  # [n,1024], vocab order
    sidx = list(csv.DictReader(open(os.path.join(cache, "derived", "species_index.csv"))))
    next_global = max(int(r["idx"]) for r in sidx) + 1

    # ---- gather requested species with full taxonomy ----
    reqs = {}                                            # norm binomial -> (binomial, [K,P,C,O,F,G])
    if a.species:
        tx = (a.taxonomy or "").split(",")
        reqs[_norm(a.species)] = (a.species, (tx + [""] * 6)[:6])
    if a.pending and os.path.exists(a.pending):
        for r in json.load(open(a.pending)):
            reqs.setdefault(_norm(r["species"]), (r["species"], [r.get("kingdom", ""), r.get("phylum", ""),
                            r.get("klass", ""), r.get("order", ""), r.get("family", ""), r.get("genus", "")]))
    new = {k: v for k, v in reqs.items() if k not in have}
    print(f"{len(reqs)} requested, {len(new)} not yet in the vocab", flush=True)
    if not new:
        return

    keys = list(new); strings = [f"{tx[0]} {tx[1]} {tx[2]} {tx[3]} {tx[4]} {tx[5]} {b}".replace("  ", " ").strip()
                                 for b, tx in (new[k] for k in keys)]
    txt = bioclip25_text(strings)                        # BioCLIP-2.5 seed per new species

    # Per-genus inductive position for genera ABSENT from the existing vocab: place ALL the batch's congeners of a
    # novel genus at ONE shared E1 (the BioCLIP-2.5-nearest known species to the genus's MEAN text seed), so congeners
    # cluster (cos~1) instead of scattering to independent nearest species -- we have no within-genus resolution for a
    # novel genus, so genus-level placement is the honest, principled seed (relatives near relatives).
    from collections import defaultdict
    gmembers = defaultdict(list)
    for j, k in enumerate(keys):
        bino, tx = new[k]; gmembers[tx[5] or bino.split()[0]].append(j)
    novel_genus_e = {}
    for genus, members in gmembers.items():
        if not len(np.where(genus_of == genus)[0]):      # novel genus -> one shared position from the genus text centroid
            novel_genus_e[genus] = E1[int((text @ (txt[members].mean(0))).argmax())]
    add_E1, add_text, add_rows = [], [], []
    for j, k in enumerate(keys):
        bino, tx = new[k]; genus = tx[5] or bino.split()[0]
        cong = np.where(genus_of == genus)[0]            # congeners already in the vocab -> mean E1 (best placement)
        e = E1[cong].mean(0) if len(cong) else novel_genus_e[genus]   # else the shared novel-genus inductive position
        add_E1.append(e); add_text.append(txt[j])
        add_rows.append({"idx": next_global + j, "tip_label": f"{bino.replace(' ', '_')}__{next_global + j}",
                         "binomial": bino, "genus": genus, "family": tx[4], "order": tx[3]})

    # ---- persist: extend every per-species array consistently ----
    E1_new = np.concatenate([E1, np.stack(add_E1)]); text_new = np.concatenate([text, np.stack(add_text)])
    gidx_new = gidx + [r["idx"] for r in add_rows]; bino_new = binos + [new[k][0] for k in keys]
    np.savez(os.path.join(cache, "gbif_vocab.npz"), global_idx=np.array(gidx_new),
             binomial=np.array(bino_new, object), E1=E1_new)
    np.save(os.path.join(cache, "bioclip_taxon_text_emb.npy"), text_new)
    with open(os.path.join(cache, "derived", "species_index.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sidx[0].keys()))
        for r in add_rows:
            w.writerow({c: r.get(c, "") for c in sidx[0].keys()})
    # traits: append an all-unknown (0) row per new species so the trait heads treat them as unobserved
    tp = os.path.join(cache, "derived", "traits_syn.npz")
    if os.path.exists(tp):
        z = dict(np.load(tp, allow_pickle=True))
        need = next_global + len(keys)                    # traits are indexed by global_idx (species_index rows), not vocab
        for kk in list(z):
            if kk.startswith("cat_") and z[kk].ndim >= 1 and z[kk].shape[0] < need:
                z[kk] = np.concatenate([z[kk], np.zeros((need - z[kk].shape[0],) + z[kk].shape[1:], z[kk].dtype)])
        np.savez(tp, **z)
    print(f"registered {len(keys)} species -> vocab now {len(gidx_new)} (global_idx {next_global}..{next_global+len(keys)-1})", flush=True)
    print("delete prepared_*.pt so the dataset rebuilds with the new species.", flush=True)

if __name__ == "__main__":
    main()
