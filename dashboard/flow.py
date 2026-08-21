"""Bottom-up data + architecture census -> state/flow.json. Nothing hand-written.

Modalities: every cache file under data/deepcal, its REAL keys/shapes/dtypes (npz headers,
no data loaded), size, gbifID coverage against the canonical observation set, and the
data.py/train.py source lines that load it (absent = stray file, flagged).
Architecture: dimensions come from the production dataclasses; stages are live defs
pulled from state/callgraph.json with their file:line spans and reach labels.

    python -m dashboard.flow [--cache /path/to/deepcal]
"""
import argparse, ast, json, time, zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_DATA = REPO / "data" / "deepcal"


def npz_headers(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            with z.open(name) as f:
                ver = npf.read_magic(f)
                reader = npf.read_array_header_1_0 if ver == (1, 0) \
                         else npf.read_array_header_2_0
                shape, _, dtype = reader(f)
                out[name[:-4]] = [list(shape), str(dtype)]
    return out


def npy_header(path):
    with open(path, "rb") as f:
        ver = npf.read_magic(f)
        reader = npf.read_array_header_1_0 if ver == (1, 0) \
                 else npf.read_array_header_2_0
        shape, _, dtype = reader(f)
    return {path.stem: [list(shape), str(dtype)]}


def gid_key(keys):
    return next((k for k in keys if k.lower() == "gbifid"), None)


def census(canon, data):
    src = {p: (REPO / p).read_text(errors="replace").splitlines()
           for p in ("core/data.py", "core/train.py")}
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

    for d in sorted(data.glob("gbif_*_tokens")) + [data / "gbif_tokens"]:
        if not d.is_dir():
            continue
        chunks = sorted(d.glob("chunk*.npz"))
        if not chunks:
            continue
        keys = npz_headers(chunks[0])
        gk = gid_key(keys)
        ids = np.concatenate([np.load(c, allow_pickle=True)[gk] for c in chunks]) if gk else None
        add(d.name, len(chunks), keys, sum(c.stat().st_size for c in chunks), ids)
    for f in sorted(data.glob("*.npz")):
        keys = npz_headers(f)
        gk = gid_key(keys)
        ids = np.load(f, allow_pickle=True)[gk] if gk else None
        add(f.name, 1, keys, f.stat().st_size, ids)
    for f in sorted(data.glob("*.npy")):
        add(f.name, 1, npy_header(f), f.stat().st_size)
    return sorted(entries, key=lambda e: -(e["coverage"] or 0))


def architecture(callg):
    byid = {x["id"]: x for x in callg["defs"]}

    def ref(did):
        x = byid.get(did)
        if not x:
            return None
        return {"id": did, "path": x["path"], "start": x["start"], "end": x["end"],
                "reach": x["reach"], **({"gate": x["gate"]} if x.get("gate") else {})}

    tree = ast.parse((REPO / "core" / "fusion.py").read_text())
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MeshConfig")
    dims = {n.target.id: ast.literal_eval(n.value) for n in config.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    return {
        "dims": {"d_model": dims["width"], "n_latents": dims["latents"],
                 "n_layers": dims["layers"], "levels": dims["levels"],
                 "world_cells": 2, "lenses": 4,
                 "mesh_slots": 2 * dims["levels"] * 4},
        "stages": [
            {"stage": "source", "title": "batch assembly",
             "defs": [ref("core/data.py::California.batch")]},
            {"stage": "spacetime", "title": "Earth4D space-time encoding",
             "defs": [ref("encoders/spacetime/earth4d.py::to_ecef"),
                      ref("encoders/spacetime/hashencoder/hashgrid.py::HashEncoder.forward"),
                      ref("core/fusion.py::WorldMesh.raw"),
                      ref("core/fusion.py::RelativeField.forward")]},
            {"stage": "species", "title": "phylogenomic species encoding",
             "defs": [ref("encoders/biological/phylogenomic.py::SpeciesGraph.forward"),
                      ref("encoders/biological/phylogenomic.py::LatentCladeAttention.forward"),
                      ref("encoders/biological/phylogenomic.py::TreeMessagePassing.forward")]},
            {"stage": "write", "title": "typed residual mesh writes",
             "defs": [ref("core/fusion.py::DeepEarth._write"),
                      ref("core/fusion.py::DeepEarth._fiber_write")]},
            {"stage": "read", "title": "query-conditioned reader and fusion",
             "defs": [ref("core/fusion.py::DeepEarth._pool"),
                      ref("core/fusion.py::DeepEarth._prime_pool_cache")]},
            {"stage": "decode", "title": "per-variable predictions",
             "defs": [ref("core/fusion.py::DeepEarth.decode"),
                      ref("core/fusion.py::DeepEarth.infer")]},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = args.cache.expanduser().resolve()
    callg = json.loads((ROOT / "state" / "callgraph.json").read_text())
    chunks = sorted((data / "gbif_tokens").glob("chunk*.npz"))
    canon = np.concatenate([np.load(path)["gbifID"] for path in chunks]) \
            if chunks else np.array([], np.int64)
    flow = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": "core",
            "n_observations": int(len(canon)),
            "modalities": census(canon, data), "arch": architecture(callg)}
    (ROOT / "state" / "flow.json").write_text(json.dumps(flow) + "\n")
    m = flow["modalities"]
    print(f"flow: {len(m)} cache artifacts ({sum(x['stray'] for x in m)} stray), "
          f"{flow['arch']['dims']['mesh_slots']} mesh slots")


if __name__ == "__main__":
    main()
