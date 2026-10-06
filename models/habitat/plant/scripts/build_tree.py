#!/usr/bin/env python3
"""Dated phylogeny of the species table: the dated megatree of Carruthers et al. (OSF 9tbha; ferns and seed plants,
128,270 tips labelled "Genus_epithet__id") pruned to our species, with the species it lacks grafted in.

A species is placed, in order of preference:
  1. exact     — its binomial is a tip;
  2. synonym   — a species-rank WCVP synonym (or the accepted name, if ours is a synonym) is a tip that is not itself
                 one of our species;
  3. genus     — grafted at the crown of its genus (a polytomy child, ultrametric). A genus with a single tip in the tree
                 is split at the midpoint of that tip's terminal branch, so the two congeners are each other's sisters;
  4. family    — grafted at the crown of its WCVP family (genera of the tree mapped to families through WCVP);
  5. dropped   — no relative in the tree (reported).
Grafting follows V.PhyloMaker's "scenario 3" logic (Jin & Qian 2019), the standard for adding missing species.

Lycophytes (Lycopodiaceae, Isoetaceae, Selaginellaceae) are absent from the tree, whose root is the euphyllophyte crown.
They are added as one clade, sister to the tree: a new root at the crown age of the tracheophytes, 435 Ma, and every
lycophyte species attached at the lycophyte crown, 413 Ma (midpoints of the Morris et al. 2018 PNAS 115:E2274
intervals 450.8-419.3 and 432.5-392.8 Ma). No dates inside the lycophytes are available, so lycophytes share only their
stem with each other (placement "lycophyte"); a dated lycophyte tree would refine this.

Five species belong to families with no tip in the tree (Joinvilleaceae, Mayacaceae, Apodanthaceae, Tetrachondraceae,
Surianaceae); each is grafted at the crown of its APG IV order, located as the MRCA of the order's families in the
tree (placement "order:<order>").

usage: build_tree.py [--config configs/conus.json] [--tree T] [--out DIR]
Writes <out>/natives.dated.nwk (tips = species labels, e.g. "Abies_amabilis") and <out>/tree_placement.csv
(default out: the configured joint.tree_dir).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.joint.tree import parse_newick  # noqa: E402
from ranges.names import load_wcvp_names  # noqa: E402

sys.setrecursionlimit(100000)
LYCOPHYTES = {"Lycopodiaceae", "Isoetaceae", "Selaginellaceae"}
TRACHEOPHYTE_CROWN_MA, LYCOPHYTE_CROWN_MA = 435.0, 413.0
# APG IV orders of the US families absent from the tree, each located by families of the order that the tree holds
ORDER_OF = {"Joinvilleaceae": "Poales", "Mayacaceae": "Poales", "Apodanthaceae": "Cucurbitales",
            "Tetrachondraceae": "Lamiales", "Surianaceae": "Fabales"}
ORDER_FAMILIES = {"Poales": ["Poaceae", "Cyperaceae", "Bromeliaceae", "Typhaceae"],
                  "Cucurbitales": ["Cucurbitaceae", "Begoniaceae", "Datiscaceae", "Coriariaceae"],
                  "Lamiales": ["Oleaceae", "Lamiaceae", "Plantaginaceae"],
                  "Fabales": ["Fabaceae", "Polygalaceae"]}


def binomial_of_tip(lab):
    return "_".join(lab.split("__")[0].split("_")[:2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--tree", help="dated megatree (default: the configured sources.carruthers_tree)")
    ap.add_argument("--species-table")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = config.load(a.config)
    tree = Path(a.tree) if a.tree else cfg.path(cfg["sources"]["carruthers_tree"])
    species_table = a.species_table or cfg.path(cfg["species"]["table"])
    wcvp = cfg.path(cfg["evaluation"]["wcvp_dir"])
    out = Path(a.out) if a.out else cfg.path(cfg["joint"]["tree_dir"])
    out.mkdir(parents=True, exist_ok=True)

    parent, blen, label = parse_newick(tree.read_text())
    N = len(parent)
    children = defaultdict(list)
    for c in range(N):
        if parent[c] >= 0: children[parent[c]].append(c)
    depth = np.zeros(N)
    for c in range(1, N): depth[c] = depth[parent[c]] + blen[c]          # parent id < child id
    leaves = [c for c in range(N) if not children[c]]
    root_age = depth[leaves].max()
    print(f"tree: {N:,} nodes, {len(leaves):,} tips, root age {root_age:.1f}; tip-depth spread "
          f"{depth[leaves].min():.2f}..{root_age:.2f}")
    tip_of = {}
    for c in leaves: tip_of.setdefault(binomial_of_tip(label[c]), c)
    by_genus = defaultdict(list)
    for b, c in tip_of.items(): by_genus[b.split("_")[0]].append(c)

    names = load_wcvp_names(wcvp)
    acc = names[names.taxon_status == "Accepted"]
    genus_family = (names.assign(g=names.taxon_name.str.split(" ").str[0]).groupby("g").family
                    .agg(lambda x: x.mode().iat[0]).to_dict())
    by_family = defaultdict(list)
    for g, cs in by_genus.items():
        if g in genus_family: by_family[genus_family[g]] += cs
    id2name = names.set_index("plant_name_id").taxon_name.to_dict()
    syn = names.groupby("accepted_id").taxon_name.apply(list).to_dict()
    name2acc = names.sort_values("status_rank").drop_duplicates("taxon_name").set_index("taxon_name").accepted_id.to_dict()

    table = pd.read_csv(species_table)
    species = sorted(table.wcvp_accepted_name.str.replace(" ", "_").unique())
    fam_of = dict(zip(table.wcvp_accepted_name.str.replace(" ", "_"), table.wcvp_family))
    species_set = set(species)

    def mrca(nodes):
        m = nodes[0]
        for nd in nodes[1:]:                                       # m <- lca(m, nd)
            anc = set(); y = m
            while y >= 0: anc.add(y); y = parent[y]
            while nd not in anc: nd = parent[nd]
            m = nd
        return m

    # placement
    new_parent, new_blen, new_label = list(parent), list(blen), list(label)
    placed, rows, lyco = {}, [], []
    for sp in species:
        name = sp.replace("_", " ")
        how, node = None, None
        if sp in tip_of:
            how, node = "exact", tip_of[sp]
        else:
            accid = name2acc.get(name)
            cands = syn.get(accid, []) + ([id2name[accid]] if accid in id2name else [])
            for c in cands:
                b = "_".join(c.split(" ")[:2])
                # species-rank synonyms only (an infraspecific synonym names a different species' variety), and never
                # a tip that is itself one of our species (exact matches own their tip)
                if len(c.split(" ")) == 2 and b in tip_of and "×" not in c and b not in species_set:
                    how, node = f"synonym:{c}", tip_of[b]; break
        if node is not None and node in placed.values():         # two of ours on one tip: graft the second as sister
            how = how + "+sister"
            node_t = node; node = None
            mid = len(new_parent); new_parent.append(new_parent[node_t]); new_blen.append(new_blen[node_t] / 2)
            new_label.append(""); new_parent[node_t] = mid; new_blen[node_t] /= 2
            node = len(new_parent); new_parent.append(mid); new_blen.append(new_blen[node_t]); new_label.append(sp)
            placed[sp] = node; rows.append((sp, how)); continue
        if node is None:
            genus = sp.split("_")[0]
            pool = by_genus.get(genus)
            level = "genus"
            if not pool:
                accid = name2acc.get(name)
                fam = acc.set_index("plant_name_id").family.get(accid) if accid else genus_family.get(genus)
                fam = fam or genus_family.get(genus)
                pool = by_family.get(fam); level = f"family:{fam}"
            if not pool and fam_of.get(sp) in LYCOPHYTES:
                lyco.append(sp); continue
            if not pool and fam_of.get(sp) in ORDER_OF:
                order = ORDER_OF[fam_of[sp]]
                pool = [mrca([c for f in ORDER_FAMILIES[order] for c in by_family.get(f, [])])] * 2
                level = f"order:{order}"
            if not pool:
                rows.append((sp, "dropped")); continue
            if len(pool) == 1:                                     # split the lone tip's terminal branch
                t = pool[0]
                mid = len(new_parent); new_parent.append(new_parent[t]); new_blen.append(new_blen[t] / 2)
                new_label.append(""); new_parent[t] = mid; new_blen[t] /= 2
                node = len(new_parent); new_parent.append(mid); new_blen.append(new_blen[t]); new_label.append(sp)
            else:
                m = mrca(pool)
                node = len(new_parent); new_parent.append(m); new_blen.append(root_age - depth[m])
                new_label.append(sp)
            how = level
        else:
            new_label[node] = sp
        placed[sp] = node; rows.append((sp, how))

    if lyco:                                                       # lycophyte clade, sister to the tree
        top = int(np.where(np.array(new_parent) < 0)[0][0])
        new_root = len(new_parent); new_parent.append(-1); new_blen.append(0.0); new_label.append("")
        new_parent[top] = new_root; new_blen[top] = TRACHEOPHYTE_CROWN_MA - root_age
        crown = len(new_parent); new_parent.append(new_root); new_blen.append(TRACHEOPHYTE_CROWN_MA - LYCOPHYTE_CROWN_MA)
        new_label.append("")
        for sp in lyco:
            node = len(new_parent); new_parent.append(crown); new_blen.append(LYCOPHYTE_CROWN_MA); new_label.append(sp)
            placed[sp] = node; rows.append((sp, "lycophyte"))
    P = np.array(new_parent); B = np.array(new_blen)
    keep = np.zeros(len(P), bool)
    for nd in placed.values():
        x = nd
        while x >= 0 and not keep[x]: keep[x] = True; x = P[x]
    kids = defaultdict(list)
    for c in np.where(keep)[0]:
        if P[c] >= 0: kids[P[c]].append(c)
    root = int(np.where(keep & (P < 0))[0][0])
    leafset = set(placed.values()); labmap = {v: k for k, v in placed.items()}

    def write(nd, acc_len):
        ks = kids.get(nd, [])
        if nd in leafset:
            return f"{labmap[nd]}:{acc_len + B[nd]:.6f}"
        if len(ks) == 1:                                           # collapse unary nodes, carrying the length down
            return write(ks[0], acc_len + B[nd])
        return "(" + ",".join(write(k, 0.0) for k in ks) + f"):{acc_len + B[nd]:.6f}"

    while len(kids.get(root, [])) == 1 and root not in leafset: root = kids[root][0]
    nwk = "(" + ",".join(write(k, 0.0) for k in kids[root]) + ");"
    (out / "natives.dated.nwk").write_text(nwk)
    df = pd.DataFrame(rows, columns=["species", "placement"])
    df.to_csv(out / "tree_placement.csv", index=False)
    kind = df.placement.str.split(":").str[0].str.replace(r"\+sister", "", regex=True)
    print(kind.value_counts().to_string())
    print(f"wrote {out/'natives.dated.nwk'}: {len(placed)} tips")


if __name__ == "__main__":
    main()
