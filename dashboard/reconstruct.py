"""Real masked reconstructions vs ground truth for sampled test observations.

Loads a trained checkpoint through the production model path, then masks each
target variable for K held-out observations and reconstructs it from the rest.

    python -m dashboard.reconstruct checkpoint.pt --cache /path/to/cache [--k 64]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO.parent))                   # deepearth.* imports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from dashboard._shared import load_checkpoint
    model, source = load_checkpoint(args.cache, args.ckpt, args.device)
    model.eval()

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

    (ROOT / "state" / "reconstructions.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "n": len(pick), "targets": targets, "rows": out}) + "\n")
    print(f"reconstructions: {len(pick)} observations x {len(targets)} targets -> state/reconstructions.json")


if __name__ == "__main__":
    main()
