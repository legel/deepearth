"""B42 plant-fungal SYMBIOSIS label from the FungalRoot database (Soudzilovskaia et al. 2020, GBIF dataset
744edc21-8dd2-474e-8a0b-b8c3d56a3c2d, direct DwCA). Per occurrence, FungalRoot records a "Mycorrhiza type"
measurement + the taxon "Name"; we take the majority mycorrhizal type per plant GENUS (its recommended genus-level
association) and map it onto the DeepCal plant vocab. Output: gbif_mycorrhiza.npz {myco[n_species] int, has_myco,
classes}. Genus-level is the FungalRoot-recommended resolution. Real published data — no heuristic assumption."""
import numpy as np, csv, zipfile, io, os
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP = Path(os.environ.get("FUNGALROOT_ZIP", "/home/photon/4tb/deepcal_data/fungalroot.zip"))
CLASSES = ["AM", "EcM", "ErM", "OM", "NM"]                       # arbuscular / ecto / ericoid / orchid / non-mycorrhizal


# Explicit map over the 12 distinct FungalRoot "Mycorrhiza type" values — substring matching is unsafe
# ("undetermined" contains "erm"). Dual types resolve to the more specialized/defining partner; genuinely
# undetermined / non-vascular / "Other" are left UNLABELED (None) rather than guessed.
_MYCO_MAP = {
    "am": "AM",
    "non-mycorrhizal": "NM",
    "ecm, am undetermined": "EcM", "ecm, no am colonization": "EcM", "ecm,am": "EcM",
    "erm": "ErM", "erm,ecm": "ErM", "erm,am": "ErM",
    "om": "OM",
    "other": None, "non-ectomycorrhizal (am undetermined)": None, "am-like (non-vascular plants)": None,
}


def canon(v):
    return _MYCO_MAP.get(v.strip().lower())


def main():
    z = zipfile.ZipFile(ZIP)
    # occurrences.csv: ID -> genus (the real taxon; the measurements' "Name" field is a record id, not a name)
    id_genus = {}
    with z.open("occurrences.csv") as f:
        r = csv.reader(io.TextIOWrapper(f, "utf-8")); h = next(r)
        idi, gi = h.index("ID"), h.index("genus")
        for row in r:
            if len(row) > max(idi, gi) and row[gi]: id_genus[row[idi]] = row[gi].strip().lower()
    # measurements.csv: Core ID -> Mycorrhiza type ; link by id
    gen = defaultdict(Counter)
    with z.open("measurements.csv") as f:
        r = csv.reader(io.TextIOWrapper(f, "utf-8")); h = next(r)
        cid, ti, vi = h.index("Core ID"), h.index("measurementType"), h.index("measurementValue")
        for row in r:
            if len(row) <= max(cid, ti, vi) or row[ti] != "Mycorrhiza type": continue
            g = id_genus.get(row[cid]); cl = canon(row[vi])
            if g and cl: gen[g][cl] += 1
    genus_type = {g: cnt.most_common(1)[0][0] for g, cnt in gen.items() if cnt}
    print(f"FungalRoot: {len(id_genus)} occ with genus; {len(genus_type)} genera with a majority mycorrhizal type")

    # map onto the DeepCal plant vocab (species -> genus -> type)
    vocab = np.load(HERE / "gbif_vocab.npz", allow_pickle=True)
    binomials = vocab["binomial"]
    myco_idx = np.full(len(binomials), -1, np.int64)
    for i, b in enumerate(binomials):
        g = str(b).strip().split()[0].lower()
        t = genus_type.get(g)
        if t is not None: myco_idx[i] = CLASSES.index(t)
    have = myco_idx >= 0
    from collections import Counter as C
    dist = C(CLASSES[j] for j in myco_idx[have])
    print(f"plant vocab {len(binomials)}: {int(have.sum())} species with a mycorrhizal type | dist {dict(dist)}")
    np.savez(HERE / "gbif_mycorrhiza.npz", myco=myco_idx, has_myco=have, classes=np.array(CLASSES, object))
    print(f"wrote {HERE/'gbif_mycorrhiza.npz'}")


if __name__ == "__main__":
    main()
