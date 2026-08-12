"""Execution trace of the real model on a real batch -> state/trace.json.

Loads the checkpoint, hooks every nn.Module, runs the actual eval path (context + infer)
and one training-loss pass, and records what executed: module tree, classes, source lines,
parameter counts, real input/output tensor shapes, sampled values, and the batch's real
provenance (gbifIDs, species, coordinates). The Flow diagram renders this file verbatim.

    python -m dashboard.trace data/deepcal/ckpt_dash-e2e-noae.pt [--config ...] [--n 8]
"""
import argparse, inspect, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO.parent))


def rel_source(cls):
    try:
        f = Path(inspect.getsourcefile(cls)).resolve().relative_to(REPO)
        return str(f), inspect.getsourcelines(cls)[1]
    except Exception:
        return None, None


def shapes(x):
    if torch.is_tensor(x):
        return [list(x.shape)]
    if isinstance(x, (tuple, list)):
        return [s for e in x for s in shapes(e)]
    if isinstance(x, dict):
        return [s for e in x.values() for s in shapes(e)]
    return []


def sample(x):
    while isinstance(x, (tuple, list)) and x:
        x = x[0]
    if torch.is_tensor(x) and x.is_floating_point() and x.numel():
        f = x.detach().float().flatten()
        return {"first": [round(v, 4) for v in f[:4].tolist()],
                "mean": round(float(f.mean()), 4), "std": round(float(f.std()), 4)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default=str(REPO / "autoresearch" / "deepcal.yaml"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from deepearth.autoresearch import train as T, evaluate as ev
    from deepearth.core.fusion import DeepEarth
    ev.evaluate_benchmarks = lambda *a, **k: {}           # trace run: skip the suite
    T.evaluate = lambda *a, **k: {}
    _load = DeepEarth.load_state_dict
    DeepEarth.load_state_dict = lambda self, sd, strict=True: _load(self, sd, strict=False)

    config = yaml.safe_load(open(args.config))
    config["_eval_ckpt"] = args.ckpt
    config["_tag"] = "trace"
    model, _ = T.train_and_evaluate(config, args.device)
    model.eval()

    import hashlib
    d = config["data"]
    keyparts = {k: d.get(k) for k in ("adapter", "cache_dir", "n_neighbors", "holdout", "subset", "time_axis", "time_km")}
    tag = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    from deepearth.autoresearch import data as data_module
    source = data_module.build(d["adapter"], cache_dir=d["cache_dir"], n_neighbors=d.get("n_neighbors", 24),
                               device=args.device, holdout=d.get("holdout", "spatial"), subset=d.get("subset"),
                               time_axis=d.get("time_axis", False), meta_path=d.get("meta_path"),
                               time_km=d.get("time_km", 50.0),
                               prepared=str(REPO / "data" / "deepcal" / f"prepared_{tag}.pt"))

    obs = np.load(ROOT / "state" / "observations.npz")
    species = json.loads((ROOT / "state" / "species.json").read_text())["binomial"]
    recon = json.loads((ROOT / "state" / "reconstructions.json").read_text())
    gids = [int(g) for g in list(recon["rows"])[:args.n]]
    pick = np.where(np.isin(obs["gbifID"], gids))[0][:args.n]
    idx = torch.tensor(pick, device=args.device)
    values, observed, coords, nbr, mani, nbrv = source.batch(idx)

    events, phase = [], ["inference"]
    names = {m: n for n, m in model.named_modules()}

    def hook(mod, inp, out):
        cls = type(mod).__name__
        f, line = rel_source(type(mod))
        events.append({"i": len(events), "name": names.get(mod, "?"), "cls": cls,
                       "file": f, "line": line, "phase": phase[0],
                       "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                       "in": shapes(inp)[:4], "out": shapes(out)[:4], "sample": sample(out)})

    handles = [m.register_forward_hook(hook) for n, m in model.named_modules() if n]
    with torch.no_grad():
        ctx = model.context(coords, nbr, mani, nbrv)
        torch.cuda.synchronize()
        target = "identity" if "identity" in values else list(values)[0]
        preds = model.infer(values, [n for n in values if n != target], [target], ctx)
    phase[0] = "training-loss"
    model.train()
    try:
        loss = model.masked_loss(values, observed, observed, ctx)
    except Exception as e:
        print(f"[trace] training-loss pass skipped: {e}")
        loss = float("nan")
    for h in handles:
        h.remove()

    p = preds[target].softmax(-1)
    top = p.topk(3, dim=-1)
    batch = {
        "gbifIDs": [int(obs["gbifID"][i]) for i in pick],
        "species": [species[int(obs["sp"][i])] for i in pick],
        "coords": [[round(float(obs["lat"][i]), 5), round(float(obs["lon"][i]), 5),
                    round(float(obs["elev"][i]), 1), None] for i in pick],
        "variables": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", ""),
                          "sample": sample(v),
                          "observed": round(float(observed[k].float().mean()), 3) if k in observed else None}
                      for k, v in values.items()},
        "context_shape": list(ctx.shape) if torch.is_tensor(ctx) else shapes(ctx)[:2],
        "outputs": {"target": target,
                    "top3": [[[species[j] if target == "identity" else str(j), round(float(v), 4)]
                              for j, v in zip(ti.tolist(), tv.tolist())]
                             for ti, tv in zip(top.indices, top.values)],
                    "loss": round(float(loss), 4)},
    }
    n_params = sum(p.numel() for p in model.parameters())
    (ROOT / "state" / "trace.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "n_params": n_params, "batch": batch, "events": events}) + "\n")
    print(f"trace: {len(events)} module executions, batch of {len(pick)}, "
          f"{n_params/1e6:.1f}M params -> state/trace.json")


if __name__ == "__main__":
    main()
