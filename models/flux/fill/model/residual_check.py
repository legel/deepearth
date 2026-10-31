"""What information does a FluxFill network leave unused? Gradient-boosted trees are fitted to its errors on hidden flux
values from what is visible, on validation towers, fitted on other towers or on other months of the same tower, with
the absolute error as objective (errors are heavy-tailed). A gain means the information is present in the inputs and the
network does not reach it. It costs no training run, so it comes before building a path.

  dump:  python residual_check.py dump CKPT OUT.npz      (one network, on the CPU unless --device is given)
  score: python residual_check.py score OUT.npz          (needs xgboost)

Feature sets: `drivers` every row's visible value and flag at the same half-hour, with hour and day of year; `memory`
adds the tower's other years at that half-hour; `imagery` adds 16 principal components of the nearest visible Clay
embedding; `imagery only` is those components with hour and day of year."""

import json
import sys
from argparse import Namespace

import numpy as np

FLUX = ("NEE", "LE", "H", "FCH4")
MAX_CELLS = 1500                # hidden values kept per example and flux


class Config(Namespace):
    """A run's saved arguments; an option newer than the run reads as off."""

    def __getattr__(self, name):
        return False


def dump(ckpt, out, device="cpu"):
    import torch
    from data import V_TAB, VARS
    from net import FluxFillNet, marginal_mean
    from recipe import net_kwargs
    from sampler import collate
    from train import regime_examples, setup, val_examples
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["config"]
    a = Config(**c)
    records, train, val, _, stats, imagery = setup(a)
    net = FluxFillNet(**net_kwargs(c)).to(device).eval()
    net.load_state_dict(ck["model"])
    vex = val_examples(records, val, stats, imagery, memory=True)
    exs = vex + regime_examples(vex, records, val, stats, imagery, True)
    cls = np.concatenate([e["img_cls"][e["img_obs"]] for e in vex if e["img_obs"].any()])
    mu = cls.mean(0)
    pcs = np.linalg.svd((cls - mu)[:: max(1, len(cls) // 4000)], full_matrices=False)[2][:16]
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("site", "fam", "flux", "y", "m", "x", "mem", "img", "t")}
    for n, ex in enumerate(exs):
        with torch.no_grad():
            mean = marginal_mean(net(collate([{k: v for k, v in ex.items() if k not in ("family", "_id")}], device))[1])[0].float().cpu().numpy()
        z, obs = ex["z"] * ex["obs"], ex["obs"].astype(np.float32)
        days = np.flatnonzero(ex["img_obs"])
        for f, k in enumerate(FLUX):
            v = VARS.index(k)
            d_, h_ = np.nonzero(ex["tmask"][v])
            if d_.size == 0:
                continue
            if d_.size > MAX_CELLS:
                keep = rng.choice(d_.size, MAX_CELLS, replace=False)
                d_, h_ = d_[keep], h_[keep]
            near = days[np.abs(days[None] - d_[:, None]).argmin(1)] if days.size else None
            rows["site"] += [ex["_id"][0]] * d_.size
            rows["fam"] += [ex["family"]] * d_.size
            rows["flux"].append(np.full(d_.size, f))
            rows["y"].append(ex["tgt"][v, d_, h_])
            rows["m"].append(mean[v, d_, h_])
            rows["x"].append(np.concatenate([z[:, d_, h_].T, obs[:, d_, h_].T], 1).astype(np.float16))
            rows["mem"].append(np.stack([ex["mem_m"][v, d_, h_], ex["mem_s"][v, d_, h_], ex["mem_h"][v, d_, h_]], 1))
            rows["img"].append(((ex["img_cls"][near] - mu) @ pcs.T).astype(np.float16) if near is not None else np.zeros((d_.size, 16), np.float16))
            rows["t"].append(np.stack([h_, d_, ex["doy"][d_, 0] if ex["doy"].ndim == 2 else np.zeros(d_.size), ex["doy"][d_, 1] if ex["doy"].ndim == 2 else np.zeros(d_.size)], 1).astype(np.float32))
        if n % 20 == 0:
            print(f"{n} of {len(exs)}", flush=True)
    np.savez_compressed(out, **{k: (np.array(v) if k in ("site", "fam") else np.concatenate(v)) for k, v in rows.items()})


def score(path):
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    z = np.load(path)
    hour = 2 * np.pi * z["t"][:, 0] / 48
    time = np.column_stack([np.sin(hour), np.cos(hour), z["t"][:, 2:]])
    base = np.column_stack([z["x"], time])
    sets = {"drivers": base, "memory": np.column_stack([base, z["mem"]]), "imagery": np.column_stack([base, z["mem"], z["img"]]),
            "imagery only": np.column_stack([time, z["img"]])}
    fams = np.array([f if f.startswith("val") else "M2 1-180 d" for f in z["fam"]])
    e = z["y"] - z["m"]
    days = z["t"][:, 1].astype(int) // 30
    towers = z["site"]
    within = np.array([f"{s}|{d}" for s, d in zip(towers, days)])
    out = []
    for f, flux in enumerate(FLUX):
        for fam in np.unique(fams):
            sel = (z["flux"] == f) & (fams == fam)
            if sel.sum() < 300 or len(set(towers[sel])) < 2:
                continue
            row = {"flux": flux, "form": fam, "n": int(sel.sum()), "towers": len(set(towers[sel])), "mae": float(np.abs(e[sel]).mean())}
            for how, groups in (("other towers", towers[sel]), ("same tower, other months", within[sel])):
                for name, X in sets.items():
                    pred = np.zeros(sel.sum())
                    for tr, te in GroupKFold(min(4, len(set(groups)))).split(X[sel], e[sel], groups):
                        b = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=120, max_depth=4, learning_rate=0.05, subsample=0.7,
                                             colsample_bytree=0.5, n_jobs=3, tree_method="hist").fit(X[sel][tr], e[sel][tr])
                        pred[te] = b.predict(X[sel][te])
                    row[f"{how}: {name}"] = float(100 * (np.abs(e[sel] - pred).mean() / row["mae"] - 1))
            out.append(row)
            print(json.dumps(row), flush=True)
    json.dump(out, open(str(path) + ".json", "w"), indent=1)


if __name__ == "__main__":
    dump(sys.argv[2], sys.argv[3], *sys.argv[4:]) if sys.argv[1] == "dump" else score(sys.argv[2])
