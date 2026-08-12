"""Real masked reconstructions vs ground truth for sampled test observations.

Loads a trained checkpoint through train.py's own eval path (no duplicated model
plumbing), then for K held-out observations masks each target variable and
reconstructs it from everything else — the model's actual posterior, per example.

    python -m dashboard.reconstruct data/deepcal/ckpt_dash-e2e.pt [--config autoresearch/deepcal.yaml] [--k 64]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO.parent))                   # deepearth.* imports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default=str(REPO / "autoresearch" / "deepcal.yaml"))
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from deepearth.autoresearch import train as T
    from deepearth.core.fusion import DeepEarth
    _load = DeepEarth.load_state_dict                    # checkpoints may lack aux-head keys
    def _tolerant(self, sd, strict=True):                # (prepared-cache vs fresh-assembly asymmetry)
        r = _load(self, sd, strict=False)
        if r.missing_keys:
            print(f"[reconstruct] {len(r.missing_keys)} keys left at init: "
                  f"{sorted({k.split('.')[0] for k in r.missing_keys})}")
        return r
    DeepEarth.load_state_dict = _tolerant
    from dashboard._shared import normalize_config, prepared_path
    config = normalize_config(yaml.safe_load(open(args.config)))
    config["_eval_ckpt"] = args.ckpt
    config["_tag"] = "reconstruct"
    model, scores = T.train_and_evaluate(config, args.device)   # builds model, loads ckpt, scores suite
    model.eval()

    d = config["data"]
    import hashlib
    keyparts = {k: d.get(k) for k in ("adapter", "cache_dir", "n_neighbors", "holdout", "subset", "time_axis", "time_km")}
    tag = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    from deepearth.autoresearch import data as data_module
    source = data_module.build(d["adapter"], cache_dir=d["cache_dir"], n_neighbors=d.get("n_neighbors", 24),
                               device=args.device, holdout=d.get("holdout", "spatial"), subset=d.get("subset"),
                               time_axis=d.get("time_axis", False), meta_path=d.get("meta_path"),
                               time_km=d.get("time_km", 50.0),
                               prepared=prepared_path(config))

    obs = np.load(ROOT / "state" / "observations.npz")
    species = json.loads((ROOT / "state" / "species.json").read_text())["binomial"]
    rng = np.random.default_rng(0)
    pick = rng.choice(source.test, size=min(args.k, len(source.test)), replace=False)
    idx = torch.tensor(pick, device=args.device)
    values, observed, coords, nbr, mani, nbrv = source.batch(idx)
    ctx = model.context(coords, nbr, mani, nbrv)
    torch.cuda.synchronize()

    names = [v.name for v in model.variables]
    kinds = {v.name: v.kind for v in model.variables}
    targets = [v.name for v in model.variables if v.reconstruct]   # every reconstructable variable
    out = {}
    lat_ok = np.allclose(obs["lat"][pick], coords[:, 0].cpu().numpy(), atol=1e-3)
    assert lat_ok, "observation index misaligned with adapter order"
    with torch.no_grad():
        for t in targets:
            preds = model.infer(values, [n for n in names if n != t], [t], ctx)[t]
            truth = values[t]
            if kinds[t] == "categorical":
                p = preds.softmax(-1)
                top = p.topk(min(5, p.shape[-1]), dim=-1)
                rank = (p.argsort(-1, descending=True) == truth[:, None]).float().argmax(-1)
                for j, g in enumerate(pick.tolist()):
                    r = out.setdefault(str(int(obs["gbifID"][g])), {})
                    label = (lambda i: species[i] if t in ("species", "identity") else str(i))
                    r[t] = {"top": [[label(int(i)), round(float(v), 4)] for i, v in
                                    zip(top.indices[j].tolist(), top.values[j].tolist())],
                            "true": label(int(truth[j])), "rank": int(rank[j])}
            else:
                cos = F.cosine_similarity(preds, truth, dim=-1)
                for j, g in enumerate(pick.tolist()):
                    out.setdefault(str(int(obs["gbifID"][g])), {})[t] = {"cos": round(float(cos[j]), 4)}

    with torch.no_grad():                                # R23's own invariant: pluralism conserved iff
        mf = model.marginal_fidelity(values, observed, ctx)   # marginals hold as coupling K rises
    (ROOT / "state" / "reconstructions.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "net_score": scores.get("net_score"), "n": len(pick),
         "targets": targets, "rows": out,
         "marginal_fidelity": {k: {kk: round(v, 4) for kk, v in d.items()} for k, d in mf.items()}}) + "\n")
    print(f"reconstructions: {len(pick)} observations x {len(targets)} targets -> state/reconstructions.json")


if __name__ == "__main__":
    main()
