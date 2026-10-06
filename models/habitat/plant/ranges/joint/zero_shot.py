"""Maps for species with no usable occurrence record ("zero-shot"), inferred from their relatives.

The model's tree is the full species tree (all 18,600 US natives), recorded or not, so the path matrix A has a
column for every branch, and every branch with a recorded species below it has a learned vector z_e. A species m
without records is a tip of that tree whose own branches carry no data. Climbing from m towards the root, the
first node x whose clade also contains trained species is where m joins its closest trained relatives R_m (all
trained species under x). Under the Brownian-motion prior (model.py) the expected niche vector of m is the value
at x, i.e. the path sum from the root down to x,

    w_m = sum over the branches e between the root and x of sqrt(l_e / l_mean) z_e,

and the branches from x down to m, which no record informs, contribute their prior mean, zero (as does the
species term u_m). Its offset is the mean offset of R_m. The calibration area, which for a recorded species comes
from the ecoregions its records occupy, comes from WCVP instead: the RESOLVE ecoregions that overlap m's native
WGSRPD level-3 regions (botanical countries/states) by at least 10% of the ecoregion's area. Its background and
presence quantiles (store.py) are computed from its relatives' training points.

Every other quantity under the same Brownian prior follows the same rule: a model with a place pathway gives m the
place vector v_m = sum over the same branches of sqrt(l_e / l_mean) z_p,e, and a model with a learned calibration
penalty gives it pi_m = softplus(sum over the same branches of sqrt(l_e / l_mean) z_c,e + c_0).

Leave-one-out check (2026-10-05; 3,605 trained species with >= 20 VegBank presence plots, each treated as
unrecorded): the path to the joining node x scores median AUC 0.9062, against 0.9003 for an earlier rule that took
the branches shared by all of R_m (the path down to R_m's own common ancestor, which can lie below x and, for a
single relative, includes that relative's terminal branch); better for 72% of the species where the rules differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import JointRangeModel
from .tree import Tree


def l3_ecoregions(table: str | Path, l3_geojson: str | Path | None = None, ecoregions_shp: str | Path | None = None,
                  min_share: float = 0.10) -> dict[str, list[int]]:
    """WGSRPD level-3 code -> ids of the RESOLVE ecoregions with >= ``min_share`` of their area inside it.
    Read from the CSV ``table`` (columns l3, eco); computed from the two polygon layers (equal-area projection
    EPSG:6933) and written there when it does not exist."""
    table = Path(table)
    if not table.exists():
        if l3_geojson is None or ecoregions_shp is None:
            raise FileNotFoundError(f"{table} does not exist and no polygon layers were given to compute it")
        import geopandas as gpd
        l3 = gpd.read_file(l3_geojson)[["LEVEL3_COD", "geometry"]].to_crs(6933)
        eco = gpd.read_file(ecoregions_shp)[["ECO_ID", "geometry"]].to_crs(6933)
        eco["eco_area"] = eco.area
        ov = gpd.overlay(eco, l3, how="intersection", keep_geom_type=True)
        ov = ov[ov.area / ov.eco_area >= min_share]
        groups = ov.groupby("LEVEL3_COD").ECO_ID.apply(lambda x: sorted(set(int(i) for i in x))).to_dict()
        pd.DataFrame({"l3": list(groups), "eco": [" ".join(map(str, v)) for v in groups.values()]}
                     ).to_csv(table, index=False)
    d = pd.read_csv(table)
    return {k: [int(x) for x in str(v).split()] for k, v in zip(d.l3, d.eco)}


@dataclass
class ZeroShot:
    names: list[str]                 # tip labels of the inferred species
    W: torch.Tensor                  # [n, width] niche vectors
    b: np.ndarray                    # [n] offsets
    calibration: list[str]           # space-separated ecoregion ids
    relatives: list[list[int]]       # model indices of each species' closest trained relatives
    V: torch.Tensor | None = None    # [n, place_dim] place vectors (models with a place pathway)
    penalty: torch.Tensor | None = None   # [n] outside-area penalties (models with a learned calibration)


@torch.no_grad()
def infer_species(model: JointRangeModel, trained: list[str], tree: Tree, inventory: pd.DataFrame,
                  l3eco: dict[str, list[int]]) -> ZeroShot:
    """Vectors, offsets and calibration areas for the species of ``inventory`` (columns ``wcvp_accepted_name``,
    ``native_l3``: comma-separated WGSRPD level-3 codes) that are not in ``trained`` (the model's species, in
    order). ``tree`` is the tree the model's path matrix was built on (its node ids are the matrix columns), with
    the species without records among its tips. Species with no tip in ``tree``, no trained relative or no mapped
    native region are left out."""
    if len(tree) != model.A.shape[1]:
        raise ValueError(f"tree has {len(tree)} nodes, the model's path matrix {model.A.shape[1]} branch columns: "
                         f"not the tree the model was trained on")
    index = {s: i for i, s in enumerate(trained)}
    node = tree.node_of()
    A = model.A.coalesce()
    rows, cols = A.indices().cpu().numpy()
    vals = A.values()
    starts = np.searchsorted(rows, np.arange(A.shape[0] + 1))       # coalesced: sorted by row
    Z, bvec = model.z, model.b
    labels = inventory.wcvp_accepted_name.str.replace(" ", "_")
    names, W, b, calib, relatives, V, pen = [], [], [], [], [], [], []
    for label, native in zip(labels, inventory.native_l3):
        if label in index or label not in node:
            continue
        x, rel = node[label], []
        while tree.parent[x] >= 0 and not rel:
            x = tree.parent[x]
            rel = [index[t] for t in tree.tips_under(x) if t in index]
        if not rel:
            continue
        eco = sorted({e for c in str(native).split(",") for e in l3eco.get(c.strip(), [])})
        if not eco:
            continue
        path = []                                                      # branches root -> x (column = child node)
        n = x
        while tree.parent[n] >= 0:
            path.append(n)
            n = tree.parent[n]
        own = slice(starts[rel[0]], starts[rel[0] + 1])               # a relative's loadings on those branches
        hit = np.flatnonzero(np.isin(cols[own], path))
        if len(hit) != len(path):
            raise ValueError(f"{label}: branches above its joining node are not on its relative's path; the path "
                             f"matrix was not built on this tree")
        a = torch.zeros(A.shape[1], device=Z.device)
        a[torch.from_numpy(cols[own][hit]).to(Z.device)] = vals[own][torch.from_numpy(hit).to(vals.device)].to(Z.device)
        W.append(a @ Z)
        if model.place is not None:
            V.append(a @ model.zp.float())
        if model.has_penalty:
            pen.append(torch.nn.functional.softplus(a @ model.zc.float() + model.c0))
        b.append(float(bvec[torch.tensor(rel, device=bvec.device)].mean()))
        names.append(label)
        calib.append(" ".join(map(str, eco)))
        relatives.append(rel)
    dev = Z.device
    return ZeroShot(names, torch.stack(W) if W else torch.zeros(0, model.width, device=dev),
                    np.array(b, np.float32), calib, relatives,
                    (torch.stack(V) if V else torch.zeros(0, model.place_dim, device=dev)) if model.place is not None
                    else None,
                    (torch.stack(pen) if pen else torch.zeros(0, device=dev)) if model.has_penalty else None)
