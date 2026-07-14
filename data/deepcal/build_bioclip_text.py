"""Regenerate bioclip_taxon_text_emb.npy — the frozen BioCLIP-2.5 taxon-string prior (science.md rule 26), one
unit-normalized [1024] vector per vocab species, in gbif_vocab order. Each species is encoded from its 7-level
taxonomy string "Kingdom Phylum Class Order Family Genus binomial" wrapped as "a photo of {s}." and passed through
the FROZEN BioCLIP-2.5 ViT-H/14 text tower — identical to add_species.bioclip25_text (the per-species path), so the
full-vocab prior and the add-species path stay byte-compatible.

Taxonomy sources: O/F/G from derived/species_index.csv; K/P/C from the GBIF observation metadata
(observations_meta.parquet, one row per species). Vocab order + membership from gbif_vocab.npz.

    BIOCLIP_META=/home/photon/4tb/deepearth_gbif/observations_meta.parquet \
    python -m deepearth.data.deepcal.build_bioclip_text --device cuda:1     # audits cosine vs the shipped emb
"""
import os, csv, argparse
from pathlib import Path
from collections import Counter
import numpy as np

# open_clip (via a dep) references np.float_ etc., removed in numpy 2.0; restore the aliases before importing it.
for _a, _t in [("float_", np.float64), ("int_", np.int64), ("uint", np.uint64),
               ("unicode_", np.str_), ("complex_", np.complex128), ("bool_", np.bool_)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

HERE = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
META = Path(os.environ.get("BIOCLIP_META", "/home/photon/4tb/deepearth_gbif/observations_meta.parquet"))
MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"


def bioclip25_text(strings, dev):
    """Encode taxonomy strings with the frozen BioCLIP-2.5 text tower (rule 26), unit-normalized [N,1024]."""
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms(MODEL); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer(MODEL)
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 128]]).to(dev))
            out.append(F.normalize(t, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write", action="store_true", help="write the emb (default: audit only)")
    a = ap.parse_args()

    bino = np.load(HERE / "gbif_vocab.npz", allow_pickle=True)["binomial"]          # [N], vocab order (authority)
    ofg = {r["binomial"]: (r["order"], r["family"], r["genus"])
           for r in csv.DictReader(open(HERE / "derived/species_index.csv"))}       # O/F/G

    import pandas as pd
    df = pd.read_parquet(META, columns=["species", "kingdom", "phylum", "class"]).dropna(subset=["species"])
    kpc = {}                                                                        # species -> most-common (K,P,C)
    for sp, g in df.groupby("species"):
        kpc[sp] = (Counter(g["kingdom"].dropna()).most_common(1) or [("",)])[0][0], \
                  (Counter(g["phylum"].dropna()).most_common(1) or [("",)])[0][0], \
                  (Counter(g["class"].dropna()).most_common(1) or [("",)])[0][0]

    strings, miss = [], 0
    for b in bino:
        K, P, C = kpc.get(b, ("", "", ""))
        O, Fa, G = ofg.get(b, ("", "", ""))
        if b not in kpc or b not in ofg: miss += 1
        s = f"{K} {P} {C} {O} {Fa} {G} {b}".replace("  ", " ").strip()
        strings.append(s)
    print(f"{len(strings)} vocab species | {miss} missing taxonomy | e.g. {strings[0]!r}", flush=True)

    emb = bioclip25_text(strings, a.device)
    print(f"encoded {emb.shape} norm mean {np.linalg.norm(emb, axis=1).mean():.4f}", flush=True)

    ref = HERE / "bioclip_taxon_text_emb.npy"
    if ref.exists():
        D = np.load(ref).astype(np.float32)
        cos = (emb * D).sum(1) if D.shape == emb.shape else None
        if cos is not None:
            print(f"audit vs shipped {D.shape}: cosine min {cos.min():.4f} mean {cos.mean():.4f} "
                  f"median {np.median(cos):.4f} | <0.99: {int((cos < 0.99).sum())}", flush=True)
    if a.write:
        np.save(ref, emb); print(f"wrote {ref}", flush=True)


if __name__ == "__main__":
    main()
