"""How much a trained model uses each input: the score with a modality removed minus the score with it present, on the
same validation windows and the same weights (the diagnostic of Wang, Tran and Feiszli 2020 and Wu et al. 2022). It needs
no retraining, so it separates what a model reads from what training made of it; the ablation (ablate.py) measures the
second. Positive: the modality lowers the CRPS. Near zero: the model does not read it.

Usage: python utilisation.py <checkpoint> [--out utilisation.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import IS_FLUX, IS_LAND, IS_REAN, IS_TOWER_MET, Stats  # noqa: E402
from net import FluxFillNet  # noqa: E402
from recipe import net_kwargs  # noqa: E402
from train import evaluate, setup, val_examples  # noqa: E402


def drop_rows(e, rows):
    e = dict(e)
    keep = (~rows)[:, None, None]
    e["z"] = e["z"] * keep
    e["obs"] = e["obs"] & keep
    e["present"] = e["present"] & ~rows
    return e


def drop_imagery(e):
    e = dict(e)
    e["img_obs"] = np.zeros_like(e["img_obs"])
    e["img_present"] = np.array(False)
    return e


def drop_memory(e):
    e = dict(e)
    e["mem_h"] = np.zeros_like(e["mem_h"])
    return e


def drop_location(e):
    e = dict(e)
    e["site"] = np.zeros_like(e["site"])
    return e


def drop_season(e):
    e = dict(e)
    e["doy"] = np.zeros_like(e["doy"])
    return e


REMOVALS = {"tower meteorology": lambda e: drop_rows(e, IS_TOWER_MET), "reanalysis": lambda e: drop_rows(e, IS_REAN),
            "reanalysis-land": lambda e: drop_rows(e, IS_LAND), "imagery": drop_imagery, "site memory": drop_memory,
            "location": drop_location, "time of year": drop_season,
            "other fluxes": lambda e: drop_rows(e, IS_FLUX)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck["config"]
    net = FluxFillNet(**net_kwargs(c))
    net.load_state_dict(ck["model"])
    net.to(a.device).eval()
    args = argparse.Namespace(max_sites=0, n_train_sites=0, imagery_sites=c.get("imagery_sites", "all"), asinh=c.get("asinh", False))
    records, train, val, _, stats, imagery = setup(args)
    stats = Stats.from_dict(c["stats"])
    exs = val_examples(records, val, stats, imagery, memory=c.get("memory", False))
    keys = ("val_crps", "val_crps_NEE", "val_crps_LE", "val_crps_H", "val_crps_FCH4")
    base, _ = evaluate(net, exs, torch.device(a.device), False, asinh=c.get("asinh", False))
    out = {"present": {k: base[k] for k in keys if k in base}}
    print(out, flush=True)
    for name, fn in REMOVALS.items():
        s, _ = evaluate(net, [fn(e) for e in exs], torch.device(a.device), False, asinh=c.get("asinh", False))
        out[name] = {k: 100 * (s[k] / base[k] - 1) for k in keys if k in s and k in base}
        print(name, {k: round(v, 1) for k, v in out[name].items()}, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
