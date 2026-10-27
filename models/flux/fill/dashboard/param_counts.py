"""Parameter counts and checkpoint facts for the Methods tab, read from the checkpoints themselves.

Run in the FluxFill environment (needs torch): build.py calls it and caches the output, so a build without torch
still shows the last counts.

Usage: python param_counts.py --model-root /home/photon/4tb/fluxfill/model --out param_counts.json [--cache old.json]
Counts:
- each FluxFill run: tensors of the `model` state dict of best.pt (or last.pt); the non-persistent buffers (rotary
  tables, provenance index) are not in the state dict, so this is the parameter count;
- the scored FluxFill checkpoint (bench/val/fluxfill_checkpoint_used.pt): its run name and step;
- SAITS and CSDI: the network built from the checkpoint's configuration, `parameters()`; best step, split, sites;
- Chronos-2: the tensor shapes in the safetensors header of the pinned snapshot.
A run whose checkpoint is unchanged (same size and modification time) keeps its cached count.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHRONOS = Path("/home/photon/4tb/hf/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c")


def stamp(p: Path) -> str:
    s = p.stat()
    return f"{s.st_size}:{int(s.st_mtime)}"


def safetensors_params(path: Path) -> int:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    tot = 0
    for k, v in head.items():
        if k == "__metadata__":
            continue
        c = 1
        for s in v["shape"]:
            c *= s
        tot += c
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="/home/photon/4tb/fluxfill/model")
    ap.add_argument("--scored", default="/home/photon/4tb/fluxfill/bench/val/fluxfill_checkpoint_used.pt")
    ap.add_argument("--pypots", default="/home/photon/4tb/fluxfill/baselines/pypots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    old = json.loads(Path(a.cache).read_text()) if a.cache and Path(a.cache).exists() else {}
    out = {"runs": {}, "baselines": {}}
    for d in sorted(Path(a.model_root).iterdir()):
        ck = next((d / n for n in ("best.pt", "last.pt") if (d / n).exists()), None)
        if ck is None:
            continue
        prev = old.get("runs", {}).get(d.name)
        if prev and prev.get("stamp") == stamp(ck):
            out["runs"][d.name] = prev
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        sd = c.get("model", c)
        out["runs"][d.name] = {"n_params": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
                               "checkpoint": ck.name, "step": int(c.get("step", -1)), "stamp": stamp(ck)}
    sp = Path(a.scored)
    if sp.exists():
        c = torch.load(sp, map_location="cpu", weights_only=False)
        out["scored"] = {"run": c["config"]["name"], "step": int(c["step"]),
                         "n_params": int(sum(v.numel() for v in c["model"].values()))}
    sys.path.insert(0, str(HERE.parent / "baselines"))
    try:
        import pypots_imputers as P
        for k in ("saits", "csdi"):
            p = Path(a.pypots) / f"{k}.pt"
            if not p.exists():
                continue
            c = torch.load(p, map_location="cpu", weights_only=False)
            net = P.build(k, c["V"])
            hist = Path(a.pypots) / f"{k}_history.json"
            steps = json.loads(hist.read_text())[-1]["step"] if hist.exists() else None
            out["baselines"][k] = {"n_params": int(sum(x.numel() for x in net.parameters())), "best_step": int(c["step"]),
                                   "steps": steps, "split": c.get("split"), "n_train_sites": len(c.get("train_sites", [])),
                                   "n_calib_sites": len(c.get("calib_sites", [])), "n_variables": int(c["V"])}
    except Exception as e:                       # PyPOTS missing: keep the cached values
        out["baselines"].update({k: v for k, v in old.get("baselines", {}).items() if k in ("saits", "csdi")})
        out["pypots_error"] = repr(e)[:200]
    st = CHRONOS / "model.safetensors"
    if st.exists():
        out["baselines"]["chronos2"] = {"n_params": safetensors_params(st), "snapshot": CHRONOS.name}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {len(out['runs'])} runs")


if __name__ == "__main__":
    main()
