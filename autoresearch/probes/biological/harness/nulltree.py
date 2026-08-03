"""The fair control for a phylogenetic operator: a tree that is not the phylogeny.

WHAT THIS IS FOR. `bio_gain` has always been `graph(ON) - graph._seed()`, i.e. the operator against its
own input. That is an ABLATION, not a control, and it confounds four things at once: the parameter
count, the training loop, the reconstruction objective, and the tree. It is a hard bar here only by
accident of this dataset -- the champion E1 text prior already scores ~0.89 family-NN on its own, so
almost anything looks weak against it. Swap in the vision seed (~0.47) and `vs seed` becomes trivially
beatable by any neighbourhood smoothing at all: a plain k-NN average would "win". That is the same
class of weak baseline the spacetime loop collapsed `FAIR_ORDER` to a single entry to eliminate.

The invariant that makes `fair_rff` fair is not that it is a Fourier feature. It is that the control
gets everything the encoder gets EXCEPT the one structural claim under test, and gets the same
courtesies -- matched width, the same normalization, and a swept free parameter chosen in the
control's favour. `SpeciesGraph`'s claim (science.md rules 7/8/29) is narrow and specific:

    the REAL dated phylogeny's topology and branch lengths carry biology the seed does not.

So the control holds the seed, `d_model`, layers, heads, parameter count, the rule-25 mask protocol,
the training budget, the split and the NN readout all fixed, and deletes only the REALNESS of the tree.

THE FAMILY. Two kinds of null, because they falsify different things:

  * `tip-permutation` (5 draws) -- the identical tree buffers with the species-to-leaf assignment
    permuted. Topology, branch lengths, level slices and node count are byte-identical; the module is
    literally the same module with the same parameter count and the same compute. Only clade membership
    stops meaning relatedness. This is the standard tip-shuffle randomization for phylogenetic signal,
    so it is defensible outside this repository too.
  * `seed-dendrogram` (1) -- an average-linkage tree built from seed cosine distance, made ultrametric
    and rescaled to unit mean branch length, then run through the SAME operator. This is the strong
    one: it asks whether the dated phylogeny beats a tree the seed could have drawn by itself. If E1
    really is phylogeny-derived, this control is what will say so.

We deliberately do NOT generate a random Yule/coalescent tree instead. That changes the shape
distribution as well as the labelling, so a loss to it would not localize to the phylogeny.

BEST-OF, NOT AVERAGE. `null_baseline = max` over the family, which is generous to the control by
design -- exactly as `fair_rff` sweeps six bandwidths and keeps the one that scores BEST for the
baseline. Taking the max inside the probe, rather than publishing one label per member and relying on
`ProbeResult.fair_gain`'s min, is deliberate: fair-gain matches labels by substring and
`definitions.py`'s audit requires exactly one fair label per declare site, so a multi-label encoding
would be fragile in a way this is not.

TWO PROPERTIES, BOTH INTENTIONAL:

  * The seed-dendrogram is built over ALL species' seeds, held-out ones included. No label is used, so
    there is no target leak -- but a held-out species' seed does reach the control through the
    topology even while its node state is masked. That leaks in the CONTROL's favour, the same
    direction as `fair_rff` selecting its bandwidth on the evaluation split, and is acceptable for the
    same reason: a control we bend toward is a control that cannot flatter the encoder.
  * If the redundancy diagnosis in `program.md` is right, this will report `fair_gain <= 0` and the
    board will read INPUT-LIMITED. That is not the control failing. That is the honest number the
    `vs seed` ablation was hiding, and it names the fix (rule 26: reseed orthogonal to topology).

NOT EDITABLE. This is what the loop is scored against.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from deepearth.autoresearch.probes.biological.editable_files.phylogenomic import build_tree_buffers

# The control's budget, fixed like FAIR_CONTROL_DIM rather than tracking anything the encoder does.
# Five permutations is enough to see the spread of the shuffle null without making the control the
# dominant cost of a run; the probe reports every member's score so the spread stays visible.
FAIR_CONTROL_DRAWS = 5
FAIR_CONTROL_MEMBERS = ("tip-permutation", "seed-dendrogram")


def permuted_tip_row(tip_row: torch.Tensor, seed: int) -> torch.Tensor:
    """The same tree, with the species-to-leaf assignment shuffled.

    `tip_row[j]` is the vocabulary row that occupies leaf `j`. Permuting it rather than rebuilding the
    tree is what makes this control exact: `build_tree_buffers` is never called again, so the topology,
    the branch lengths, the level-synchronous sweep buffers and the node count cannot drift by even a
    float. The operator sees a tree of identical shape in which relatedness means nothing.
    """
    rng = np.random.default_rng(seed)
    order = torch.as_tensor(rng.permutation(tip_row.numel()), dtype=torch.long, device=tip_row.device)
    return tip_row[order]


def _linkage_to_newick(Z: np.ndarray, labels: Sequence[str]) -> str:
    """Render a SciPy average-linkage matrix as an ultrametric Newick string.

    Branch length is the drop in merge height from a node to its child, so every leaf sits at the same
    total distance from the root -- an ultrametric, which is the same shape class as the dated tree the
    real arm uses. `build_tree_buffers` rescales to unit mean afterwards, so the absolute heights here
    only need to be internally consistent.
    """
    n = len(labels)
    parts: Dict[int, str] = {i: str(labels[i]) for i in range(n)}
    height: Dict[int, float] = {i: 0.0 for i in range(n)}
    for k, (a, b, dist, _) in enumerate(Z):
        a, b, node = int(a), int(b), n + k
        h = float(dist) / 2.0
        parts[node] = f"({parts[a]}:{h - height[a]:.8f},{parts[b]}:{h - height[b]:.8f})"
        height[node] = h
        parts.pop(a, None); parts.pop(b, None)
    return parts[n + len(Z) - 1] + ";"


def seed_dendrogram_tree(seed_emb: torch.Tensor, tip_row: torch.Tensor) -> Dict:
    """An average-linkage tree over SEED cosine distance, through the same buffer builder.

    The strongest member of the family: a real, well-formed ultrametric tree carrying real structure --
    just structure the seed already had. Beating the tip-shuffles only shows the operator can use A
    tree; beating this shows the DATED PHYLOGENY carries something the seed does not.

    Leaf names are synthetic (`t0..tN`) and positional: leaf `j` is the species at `tip_row[j]`, the
    same convention the real tree's buffers already use. Nothing downstream reads a label -- they exist
    only so the Newick round-trip through `build_tree_buffers` can match leaves to model rows -- so
    generating them here avoids threading the real tip labels through a `load_species` signature that
    fourteen call sites already unpack positionally.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    labels = [f"t{i}" for i in range(int(tip_row.numel()))]
    x = F.normalize(seed_emb[tip_row].float(), dim=-1)
    d = (1.0 - x @ x.t()).detach().cpu().numpy().astype(np.float64)
    d = np.clip((d + d.T) / 2.0, 0.0, None)                  # exact symmetry; squareform is strict
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    with tempfile.NamedTemporaryFile("w", suffix=".nwk", delete=False) as fh:
        fh.write(_linkage_to_newick(Z, labels))
        path = fh.name
    try:
        return build_tree_buffers(path, labels)
    finally:
        Path(path).unlink(missing_ok=True)


def null_family(seed_emb: torch.Tensor, tree: Dict, tip_row: torch.Tensor,
                draws: int = FAIR_CONTROL_DRAWS) -> List[tuple]:
    """Every member of the control family as ``(label, tree_buffers, tip_row)``, ready to fit.

    The caller fits each with the SAME trainer, budget, split and readout it used for the real tree --
    that identity is the whole control, and it cannot be enforced from here.
    """
    members = [(f"null-tree/perm{k}", tree, permuted_tip_row(tip_row, seed=k)) for k in range(draws)]
    members.append(("null-tree/seed-dendrogram", seed_dendrogram_tree(seed_emb, tip_row), tip_row))
    return members


def fair_gain(real: float, null_scores: Dict[str, float]) -> tuple:
    """``(gain, best_label, best_score)`` against the STRONGEST member of the family.

    Strongest, not average: a gain that survives the best null is the only one that means the real tree
    did the work. Reporting the mean would let a lucky-bad shuffle carry the claim.
    """
    if not null_scores:
        return (None, None, None)
    label, best = max(null_scores.items(), key=lambda kv: kv[1])
    return (real - best, label, best)
