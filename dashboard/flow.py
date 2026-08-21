"""Bottom-up data + architecture census -> state/flow.json. Nothing hand-written.

Modalities: every cache file under data/deepcal, its REAL keys/shapes/dtypes (npz headers,
no data loaded), size, gbifID coverage against the canonical observation set, and the
data.py/train.py source lines that load it (absent = stray file, flagged).
Architecture: dims and counts read from the champion yaml; stages are live defs pulled
from state/callgraph.json with their file:line spans and reach labels.

    python -m dashboard.flow [--config autoresearch/deepcal.yaml]
"""
import argparse, json, time, zipfile
from pathlib import Path

import numpy as np
import yaml
from numpy.lib import format as npf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "data" / "deepcal"


def npz_headers(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            with z.open(name) as f:
                ver = npf.read_magic(f)
                shape, _, dtype = npf._read_array_header(f, ver)
                out[name[:-4]] = [list(shape), str(dtype)]
    return out


def npy_header(path):
    with open(path, "rb") as f:
        ver = npf.read_magic(f)
        shape, _, dtype = npf._read_array_header(f, ver)
    return {path.stem: [list(shape), str(dtype)]}


def gid_key(keys):
    return next((k for k in keys if k.lower() == "gbifid"), None)


def census(canon):
    src = {p: (REPO / p).read_text(errors="replace").splitlines()
           for p in ("autoresearch/data.py", "autoresearch/train.py", "autoresearch/prepare.py")}
    entries = []

    def loader_refs(basename):
        return [f"{p}:{i + 1}" for p, lines in src.items()
                for i, l in enumerate(lines) if basename in l]

    def add(name, files, keys, nbytes, ids=None):
        cov = None
        if ids is not None:
            cov = round(float(np.isin(canon, ids).mean()), 4)
        refs = loader_refs(name)
        entries.append({"name": name, "files": files, "keys": keys, "bytes": int(nbytes),
                        "coverage": cov, "loaders": refs, "stray": not refs})

    for d in sorted(DATA.glob("gbif_*_tokens")) + [DATA / "gbif_tokens"]:
        if not d.is_dir():
            continue
        chunks = sorted(d.glob("chunk*.npz"))
        if not chunks:
            continue
        keys = npz_headers(chunks[0])
        gk = gid_key(keys)
        ids = np.concatenate([np.load(c, allow_pickle=True)[gk] for c in chunks]) if gk else None
        add(d.name, len(chunks), keys, sum(c.stat().st_size for c in chunks), ids)
    for f in sorted(DATA.glob("*.npz")):
        keys = npz_headers(f)
        gk = gid_key(keys)
        ids = np.load(f, allow_pickle=True)[gk] if gk else None
        add(f.name, 1, keys, f.stat().st_size, ids)
    for f in sorted(DATA.glob("*.npy")):
        add(f.name, 1, npy_header(f), f.stat().st_size)
    return sorted(entries, key=lambda e: -(e["coverage"] or 0))


def architecture(config, callg):
    m, d = config.get("model", {}), config.get("data", {})
    byid = {x["id"]: x for x in callg["defs"]}

    def ref(did):
        x = byid.get(did)
        return {"id": did, "path": x["path"], "start": x["start"], "end": x["end"],
                "reach": x["reach"], **({"gate": x["gate"]} if x and x.get("gate") else {})} if x else None

    variables = list(config.get("variables", []))
    n_manifolds = len(m.get("manifolds", {}))
    K = d.get("n_neighbors", 24)
    V = len(variables)
    return {
        "dims": {"d_model": m.get("d_model", 256), "n_latents": m.get("n_latents", 24),
                 "n_layers": m.get("n_layers", 2), "rounds": m.get("rounds", 1),
                 "n_neighbors": K, "n_manifolds": n_manifolds, "n_variables": V,
                 "context_tokens": V + 1 + (1 + n_manifolds) * K,
                 "batch": config.get("training", {}).get("batch")},
        "variables": variables,
        "stages": [
            {"stage": "source", "title": "batch assembly",
             "defs": [ref("autoresearch/data.py::California.batch")]},
            {"stage": "spacetime", "title": "Earth4D space-time encoding",
             "defs": [ref("encoders/spacetime/earth4d.py::Earth4D.forward"),
                      ref("encoders/spacetime/hashencoder/hashgrid.py::HashEncoder.forward"),
                      ref("core/fusion.py::SpaceTimeField.forward"),
                      ref("core/fusion.py::NeighborContext.forward")]},
            {"stage": "species", "title": "phylogenomic species encoding",
             "defs": [ref("encoders/biological/phylogenomic.py::SpeciesGraph.forward"),
                      ref("encoders/biological/phylogenomic.py::LatentCladeAttention.forward"),
                      ref("encoders/biological/phylogenomic.py::TreeMessagePassing.forward")]},
            {"stage": "tokens", "title": "token assembly (value+type+position)",
             "defs": [ref("core/fusion.py::DeepEarth.context"),
                      ref("core/fusion.py::DeepEarth.encode")]},
            {"stage": "fuse", "title": "latent fusion (read + blocks)",
             "defs": [ref("core/fusion.py::LatentBlock.forward"),
                      ref("core/fusion.py::DeepEarth._refine")]},
            {"stage": "decode", "title": "per-variable decoders",
             "defs": [ref("core/fusion.py::DeepEarth._decode_loss"),
                      ref("core/fusion.py::DeepEarth.infer")]},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "autoresearch" / "deepcal.yaml"))
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    callg = json.loads((ROOT / "state" / "callgraph.json").read_text())
    canon = np.load(ROOT / "state" / "observations.npz")["gbifID"]
    flow = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": args.config,
            "n_observations": int(len(canon)),
            "modalities": census(canon), "arch": architecture(config, callg)}
    (ROOT / "state" / "flow.json").write_text(json.dumps(flow) + "\n")
    m = flow["modalities"]
    print(f"flow: {len(m)} cache artifacts ({sum(x['stray'] for x in m)} stray), "
          f"{flow['arch']['dims']['context_tokens']} context tokens "
          f"({flow['arch']['dims']['n_variables']} vars, {flow['arch']['dims']['n_neighbors']} neighbors)")


if __name__ == "__main__":
    main()
