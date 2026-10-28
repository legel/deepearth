"""Train FluxFill on training sites; select and monitor on validation sites; never read test sites.

Modes:
- `overfit`: a fixed set of 1 to 4 windows with fixed masks, trained until the hidden values are reproduced. A bug
  in tokens, masks, targets or heads shows up here as a loss that does not go to its floor.
- `train`: windows drawn by `sampler.WindowSampler` with fresh training masks every step.

Every `--eval-every` steps: validation NLL, CRPS, MAE and 95 % coverage on fixed validation windows with the benchmark's
M2 all-tower blocks (1, 7, 30, 90, 180 d), a reconstruction figure (PNG and the arrays behind it), and the last and best
checkpoints (by validation CRPS). Training stops on a non-finite loss.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import masking  # noqa: E402
from data import (scaling_order, IS_TOWER, STEPS_PER_DAY, T_WIN, V_TAB, VARS, fit_stats, heldout_year, load_records, load_split,  # noqa: E402
                  table_dir, valid_starts, window_values)
from imagery import load_imagery  # noqa: E402
from net import FluxFillNet, imagery_scores, marginal_mean, marginal_nll, mixture_cdf, sample_marginal_params  # noqa: E402
from sampler import Stream, WindowSampler, collate, make_example  # noqa: E402

sys.path.insert(0, str(HERE.parent / "bench"))
from masks import BLOCK_DAYS, contiguous_block, seed_for  # noqa: E402
from metrics import crps_samples, crps_student_t  # noqa: E402

OUT = Path("/home/photon/4tb/fluxfill/model")


def setup(args):
    sites = sorted(p.stem for p in table_dir().glob("*.parquet"))
    split, split_name = load_split(sites)
    train = [s for s, v in split.items() if v == "train"]
    val = [s for s, v in split.items() if v == "val"]
    if args.max_sites:
        train, val = train[: args.max_sites], val[: max(2, args.max_sites // 4)]
    if args.n_train_sites:        # data-scaling study: a nested, stratified prefix of the training sites
        train = scaling_order(train)[: args.n_train_sites]
    records = load_records(train + val)
    for s in train:
        if s in records:
            records[s].heldout_year = heldout_year(records[s])
    stats = fit_stats(records, [s for s in train if s in records])
    imagery = {}
    allowed = None if args.imagery_sites in ("", "all") else set(args.imagery_sites.split(","))
    for s, r in records.items():
        if allowed is not None and s not in allowed:
            continue
        im = load_imagery(s, r.utc_offset_h)
        if im is not None:
            imagery[s] = im
    return records, train, val, split_name, stats, imagery


def site_weights():
    """Per-site sampling weights from the dataset's weights.parquet (data/towers/README.md, Balance and sampling
    weights), or None (uniform over sites) before the build writes it."""
    f = Path("/home/photon/4tb/fluxfill/towers/weights.parquet")
    if not f.exists():
        return None
    w = pd.read_parquet(f)
    col = next((c for c in ("site_weight", "weight") if c in w), None)
    if col is None or "site_id" not in w:
        return None
    return w.groupby("site_id")[col].first().astype(float).to_dict()


def val_examples(records, val, stats, imagery, n_per_site: int = 2, max_n: int = 60, memory: bool = False,
                 memory_skew: bool = False):
    """Fixed validation windows: up to `n_per_site` usable windows per site (evenly spaced starts), each with the
    M2 all-tower block of one length in turn, seeded as in the benchmark."""
    out = []
    k = 0
    msmp = WindowSampler(records, val, stats, imagery, train=False, memory=True, memory_skew=memory_skew) if memory else None
    for s in val:
        if s not in records:
            continue
        st = valid_starts(records[s], exclude_heldout=False)
        if st.size == 0:
            continue
        for j in np.linspace(0, st.size - 1, n_per_site).astype(int)[: n_per_site]:
            d = int(st[j])
            rec = records[s]
            present = np.isfinite(window_values(rec, d)).any(1)
            days = BLOCK_DAYS[k % len(BLOCK_DAYS)]
            start = str((rec.t0 + pd.Timedelta(days=d)).date())
            hide = contiguous_block(np.flatnonzero(present & IS_TOWER), V_TAB, T_WIN, days,
                                    np.random.default_rng(seed_for(s, start, "M2", days)))
            mem = msmp.window_memory(s, d) if memory else None
            ex = make_example(rec, d, stats, hide, np.zeros(180, bool), imagery.get(s), mem)
            if ex["tmask"].any():
                ex["family"] = f"M2-{days}d"
                ex["_id"] = (s, start, days)
                out.append(ex)
                k += 1
        if len(out) >= max_n:
            break
    return out


def imagery_val_examples(records, val, stats, imagery, memory: bool, per_site: int = 4, frac: float = 0.3,
                         memory_skew: bool = False):
    """Validation windows at validation sites with Clay embeddings, for imagery scores only: up to `per_site` windows
    with at least 5 acquisition days, a seeded 30 % of those days hidden (the targets) and the M2 7-day all-tower
    block. Kept apart from `val_examples`, so the main validation numbers do not change with imagery coverage."""
    from imagery import window_imagery
    out = []
    if sum(s in imagery for s in val) > 5:      # imagery at many validation sites: one window per site keeps it fast
        per_site = 1
    msmp = WindowSampler(records, val, stats, imagery, train=False, memory=True, memory_skew=memory_skew) if memory else None
    for s in val:
        if s not in imagery or s not in records:
            continue
        rec = records[s]
        st = valid_starts(rec, exclude_heldout=False)
        good = [int(d) for d in st if window_imagery(imagery[s], rec.t0 + pd.Timedelta(days=int(d)), 180)[2].sum() >= 5]
        for d in [good[i] for i in np.unique(np.linspace(0, len(good) - 1, per_site).astype(int))] if good else []:
            start = str((rec.t0 + pd.Timedelta(days=d)).date())
            rng = np.random.default_rng(seed_for(s, start, "img", 0))
            present = np.isfinite(window_values(rec, d)).any(1)
            hide = contiguous_block(np.flatnonzero(present & IS_TOWER), V_TAB, T_WIN, 7,
                                    np.random.default_rng(seed_for(s, start, "M2", 7)))
            ex = make_example(rec, d, stats, hide, rng.random(180) < frac, imagery[s],
                              msmp.window_memory(s, d) if memory else None)
            if ex["img_tgt"].any() and ex["tmask"].any():
                ex["family"], ex["_id"] = "img", (s, start, 7)
                out.append(ex)
    return out


@torch.no_grad()
def evaluate_imagery(net, exs, device) -> dict:
    """Mean Gaussian NLL (per dimension), MSE and cosine of the hidden Clay CLS days of the imagery validation set."""
    if not exs:
        return {}
    net.eval()
    tot = {"val_img_nll": [], "val_img_mse": [], "val_img_cos": [], "n": []}
    for e in exs:
        b = collate([{k: v for k, v in e.items() if not k.startswith("_")}], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = net.encode(b)
        tgt = net.img_norm(b["img_tgt_cls"][b["img_tgt"]]).float()
        nll, mse, cos = imagery_scores(net, h[:, V_TAB][b["img_tgt"]].float(), tgt)
        tot["val_img_nll"].append(float(nll) if nll is not None else np.nan)
        tot["val_img_mse"].append(float(mse)); tot["val_img_cos"].append(float(cos)); tot["n"].append(int(tgt.shape[0]))
    net.train()
    w = np.array(tot.pop("n"), float)
    return {k: float(np.average(v, weights=w)) for k, v in tot.items()} | {"val_img_days": int(w.sum())}


def losses(net, b, use_flow: bool, img_w: float = 0.1, flow_tokens: int = 2048):
    h, params = net(b)
    tm = b["tmask"]
    nll = marginal_nll(b["tgt"], params)[tm].mean()
    out = {"nll": nll}
    total = nll
    if b["img_tgt"].any():
        tgt = net.img_norm(b["img_tgt_cls"][b["img_tgt"]]).float()
        nll, out["img_mse"], out["img_cos"] = imagery_scores(net, h[:, V_TAB][b["img_tgt"]], tgt)
        if nll is not None:
            out["img_nll"] = nll
        total = total + img_w * (nll if nll is not None else out["img_mse"])
    if use_flow:
        tok = tm.any(-1)                                                  # (B, V, D) tokens with a target
        idx = tok.nonzero()
        if idx.shape[0] > flow_tokens:
            idx = idx[torch.randperm(idx.shape[0], device=idx.device)[:flow_tokens]]
        bi, vi, di = idx.unbind(1)
        cond = h[bi, vi, di]
        x1 = b["tgt"][bi, vi, di] - net.offset(b)[bi, vi, di]          # residual from site memory (mem_residual)
        out["flow"] = net.head_flow.loss(cond, x1, tm[bi, vi, di].float(), b["z"][bi, vi, di],
                                         b["obs"][bi, vi, di].float())
        total = total + out["flow"]
    out["total"] = total
    return out


@torch.no_grad()
def evaluate(net, exs, device, use_flow: bool, batch: int = 4):
    from scipy import stats as sst
    net.eval()
    rows, recs, flows = [], [], []
    for i in range(0, len(exs), batch):
        chunk = exs[i:i + batch]
        b = collate([{k: v for k, v in e.items() if not k.startswith("_")} for e in chunk], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L = losses(net, b, use_flow)
            _, params = net(b)
            mean_all = marginal_mean(params).float()
            if len(params) == 4:                      # mixture: scores and intervals from 64 draws
                draws = sample_marginal_params(params, 64).float()
        for j, e in enumerate(chunk):
            tm = e["tmask"]
            y = e["tgt"][tm]
            mean_j = mean_all[j].cpu().numpy()
            if len(params) == 3:
                mu, sg, nu = (u[j].float().cpu().numpy()[tm] for u in params)
                crps = crps_student_t(mu, sg, nu, y)
                q = sst.t.ppf(0.975, nu)
                cov = np.abs(y - mu) <= q * sg
                full = [u[j].float().cpu().numpy() for u in params]
                half = sst.t.ppf(0.975, full[2]) * full[1]
                lo_j, hi_j = full[0] - half, full[0] + half
            else:
                dj = draws[:, j].cpu().numpy()
                mu = mean_j[tm]
                crps = crps_samples(dj[:, tm], y)                 # fair CRPS estimator from 64 draws
                lo_j, hi_j = np.quantile(dj, [0.025, 0.975], axis=0)   # for the reconstruction figure only
                pit = mixture_cdf([u[j].float().cpu().numpy()[tm] for u in params], y)
                cov = (pit >= 0.025) & (pit <= 0.975)             # exact coverage from the mixture CDF
            vi = np.nonzero(tm)[0]
            flux = np.isin(vi, [VARS.index(v) for v in ("NEE", "LE", "H", "FCH4")])
            rows.append(dict(id="|".join(map(str, e.get("_id", ("", "", "")))), fam=e["family"], n=int(tm.sum()),
                             n_flux=int(flux.sum()), crps=float(crps.mean()), mae=float(np.abs(mu - y).mean()),
                             cov95=float(cov.mean()), crps_flux=float(crps[flux].mean()) if flux.any() else np.nan,
                             nll=float(L["nll"])))
            recs.append((e, mean_j, lo_j, hi_j))
        if use_flow and "flow" in L:
            flows.append(float(L["flow"]))
    net.train()
    df_ = pd.DataFrame(rows)
    w = df_["n"].to_numpy()
    summ = {"val_crps": float(np.average(df_.crps, weights=w)), "val_mae": float(np.average(df_.mae, weights=w)),
            "val_cov95": float(np.average(df_.cov95, weights=w)), "val_crps_flux": float(np.nanmean(df_.crps_flux)),
            "val_nll": float(np.average(df_.nll, weights=w))}
    if flows:
        summ["val_flow"] = float(np.mean(flows))
    for fam, g in df_.groupby("fam"):
        summ[f"crps_{fam}"] = float(np.average(g.crps, weights=g.n))
        summ[f"cov95_{fam}"] = float(np.average(g.cov95, weights=g.n))
    summ["_windows"] = df_[["id", "fam", "n", "n_flux", "crps", "crps_flux"]].to_dict("records")
    return summ, recs


def plot_recon(recs, stats, path: Path, step: int, n: int = 3, var_keys=("NEE", "LE", "H", "TA", "SW_IN", "VPD")):
    """Masked truth vs prediction (median and 95 % interval) for the first `n` validation windows; arrays to .npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sst
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    vks = [VARS.index(v) for v in var_keys]
    n = min(n, len(recs))
    fig, axes = plt.subplots(len(vks), n, figsize=(5.2 * n, 1.7 * len(vks)), squeeze=False)
    for c, (e, mean_s, lo_s, hi_s) in enumerate(recs[:n]):
        tm = e["tmask"].reshape(V_TAB, -1)
        cols = np.flatnonzero(tm[IS_TOWER].any(0))
        a, bnd = (cols.min(), cols.max() + 1) if cols.size else (0, T_WIN)
        a, bnd = max(0, a - 96), min(T_WIN, bnd + 96)
        x = np.arange(a, bnd) / STEPS_PER_DAY
        for r, vi in enumerate(vks):
            ax = axes[r, c]
            truth = stats.denorm(e["tgt"][vi].reshape(1, -1).astype(np.float64), np.array([vi]))[0]
            meas = (e["obs"][vi] | e["tmask"][vi]).reshape(-1)
            den = lambda z: stats.denorm(z[vi].reshape(1, -1).astype(np.float64), np.array([vi]))[0]
            mu, lo95, hi95 = den(mean_s), den(lo_s), den(hi_s)
            hm = tm[vi, a:bnd]                         # this variable's hidden, measured values
            tr = np.where(meas, truth, np.nan)[a:bnd]
            ax.plot(x, tr, color="0.55", lw=0.6)
            ax.fill_between(x, np.where(hm, lo95[a:bnd], np.nan), np.where(hm, hi95[a:bnd], np.nan),
                            color="#2b7bba", alpha=0.25, lw=0)
            ax.plot(x, np.where(hm, mu[a:bnd], np.nan), color="#2b7bba", lw=0.7)
            ax.set_ylabel(VARS[vi], fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0:
                s, st, d = e["_id"]
                ax.set_title(f"{s} {st} M2 {d} d" if d else f"{s} window {st}: {e['family']}", fontsize=8)
            arrays[f"w{c}_{VARS[vi]}_truth"] = tr.astype(np.float32)
            arrays[f"w{c}_{VARS[vi]}_mean"] = mu[a:bnd].astype(np.float32)
            arrays[f"w{c}_{VARS[vi]}_lo95"] = lo95[a:bnd].astype(np.float32)
            arrays[f"w{c}_{VARS[vi]}_hi95"] = hi95[a:bnd].astype(np.float32)
            arrays[f"w{c}_{VARS[vi]}_hidden"] = hm
    fig.suptitle(f"step {step}: grey truth, blue prediction and 95 % interval on hidden spans", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=90)
    plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)


def main():
    import torch
    torch.set_num_threads(2)            # machine rule: at most 2 CPU threads per job
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["overfit", "train"], default="train")
    ap.add_argument("--n-overfit", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000, help="length of the learning-rate schedule")
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this step (checkpoint written), schedule kept")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step (batch = batch x accum)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout in every encoder block")
    ap.add_argument("--site-drop", type=float, default=0.0, help="probability of hiding the site location per window")
    ap.add_argument("--memory", action="store_true", help="site memory channels (memory.py)")
    ap.add_argument("--mem-residual", action="store_true", help="heads predict relative to the site-memory mean")
    ap.add_argument("--img-nll", action="store_true", help="imagery head: diagonal Gaussian trained on NLL")
    ap.add_argument("--imagery-sites", default="all",
                    help="'all', or a comma list: Clay imagery only at these sites (training and validation)")
    ap.add_argument("--local-base", action="store_true",
                    help="residual base = local mean diurnal course (+/- 3 days) where it exists, memory elsewhere")
    ap.add_argument("--gated-base", action="store_true",
                    help="E20: two Student-t anchored at the local course and the memory mean, gated per half-hour")
    ap.add_argument("--gate-scale", action="store_true",
                    help="E21: the gate's evidence features also scale each component's spread (with --gated-base)")
    ap.add_argument("--driver-anomaly", action="store_true",
                    help="E25: both anchors follow the ERA5 SW_IN, TA, VPD anomalies with per-token learned slopes")
    ap.add_argument("--memory-skew-fluxes", action="store_true",
                    help="E23b: site-memory skewness for the flux rows only (NEE, LE, H, FCH4)")
    ap.add_argument("--memory-skew", action="store_true", help="E23: site-memory skewness as a fourth memory channel")
    ap.add_argument("--edge-continue", action="store_true",
                    help="E24: local anchor continues the nearest visible anomaly with a learned per-variable decay")
    ap.add_argument("--components", type=int, default=1, help="Student-t mixture components per half-hour (1: v0)")
    ap.add_argument("--flow", action="store_true", help="train the v1 flow head jointly")
    ap.add_argument("--no-imagery", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-sites", type=int, default=0)
    ap.add_argument("--n-train-sites", type=int, default=0, help="data-scaling: first n sites of data.scaling_order")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    t0 = time.time()
    records, train, val, split_name, stats, imagery = setup(a)
    print(f"split {split_name}: {len(train)} train, {len(val)} val sites; {len(records)} loaded; imagery at "
          f"{sorted(imagery)}; {time.time() - t0:.0f} s", flush=True)
    vex = val_examples(records, val, stats, imagery, memory=a.memory, memory_skew=a.memory_skew or a.memory_skew_fluxes)
    vimg = imagery_val_examples(records, val, stats, imagery, memory=a.memory, memory_skew=a.memory_skew or a.memory_skew_fluxes)
    print(f"imagery validation: {len(vimg)} windows at {sorted({e['_id'][0] for e in vimg})}", flush=True)
    cfg = dict(vars(a), split=split_name, train_sites=train, val_sites=val, stats=stats.to_dict(),
               imagery_sites_used=sorted(imagery))
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    net = FluxFillNet(d=a.d, depth=a.depth, heads=a.heads, dropout=a.dropout, use_imagery=not a.no_imagery,
                      memory=a.memory, mem_residual=a.mem_residual, img_nll=a.img_nll,
                      components=a.components, local_base=a.local_base, gated_base=a.gated_base,
                      gate_scale=a.gate_scale, edge_continue=a.edge_continue, memory_skew=a.memory_skew,
                      driver_anomaly=a.driver_anomaly, memory_skew_fluxes_only=a.memory_skew_fluxes).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params {n_params / 1e6:.2f} M", flush=True)
    cfg["n_params"] = int(n_params)
    cfg["tokens_per_step"] = int(a.batch * a.accum * (V_TAB + 1) * 180)        # day-tokens per optimiser step
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.98))
    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0 = ck["step"]
    if a.mode == "overfit":
        rng = np.random.default_rng(a.seed)
        smp = WindowSampler(records, train, stats, imagery, train=True, memory=a.memory, memory_skew=a.memory_skew or a.memory_skew_fluxes)
        fixed = []
        while len(fixed) < a.n_overfit:
            e = smp.draw(rng)
            if "r3" not in e["family"]:
                fixed.append(e)
        fixed_b = collate(fixed, device)
        it = iter(lambda: fixed_b, None)
        vex_use = []
        for j, e in enumerate(fixed):
            e = dict(e); e["_id"] = ("train-overfit", str(j), 0); vex_use.append(e)
    else:
        sw = site_weights()
        print(f"site weights: {'weights.parquet' if sw else 'uniform'}", flush=True)
        smp = WindowSampler(records, train, stats, imagery, train=True, site_weights=sw, memory=a.memory,
                            memory_skew=a.memory_skew or a.memory_skew_fluxes)
        dl = torch.utils.data.DataLoader(Stream(smp, a.batch, a.seed), batch_size=None, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
        it = (({k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}) for b in dl)
        vex_use = vex
    print(f"{len(smp.sites)} training sites with usable windows; {len(vex)} validation windows", flush=True)
    cfg["n_train_sites_used"] = len(smp.sites)
    cfg["n_train_windows"] = int(sum(v.size for v in smp.starts.values()))   # usable daily window starts
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    log = open(out / "log.jsonl", "a")
    best = math.inf
    if a.resume and (out / "log.jsonl").exists():       # keep best.pt unless a resumed evaluation beats it
        prev = [json.loads(x) for x in open(out / "log.jsonl") if '"eval"' in x]
        best = min([p.get("val_crps", math.inf) for p in prev], default=math.inf)
    tl = time.time()
    acc = {}
    from gpu_pause import wait_if_paused
    for step in range(step0 + 1, a.steps + 1):
        if step % 50 == 0:      # cooperative pause of the shared GPU, after writing the current state
            wait_if_paused(f"{a.name} step {step}", on_pause=lambda: torch.save(
                {"model": net.state_dict(), "opt": opt.state_dict(), "step": step - 1, "config": cfg}, out / "last.pt"))
        lr = a.lr * min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        stop = False
        for _ in range(a.accum):
            b = next(it)
            if a.site_drop > 0:      # zero location = the "unknown location" encoding (data.site_features)
                keep = (torch.rand(b["site"].shape[0], 1, device=device) >= a.site_drop).float()
                b["site"] = b["site"] * keep
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L = losses(net, b, a.flow)
            if not torch.isfinite(L["total"]):
                stop = True
                break
            (L["total"] / a.accum).backward()
        if stop:
            print(f"non-finite loss at step {step}: { {k: float(v) for k, v in L.items()} }; stopping", flush=True)
            log.write(json.dumps({"step": step, "event": "nan"}) + "\n"); log.flush()
            break
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        acc["gn"] = acc.get("gn", 0.0) + float(gn)
        acc["n"] = acc.get("n", 0) + 1
        if step % 50 == 0:
            n = acc.pop("n")
            rec = {"step": step, "lr": float(f"{lr:.3e}"), "s_per_step": (time.time() - tl) / 50, **{k: v / n for k, v in acc.items()},
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            tl = time.time(); acc = {}
            print(json.dumps({k: (v if k == "lr" else round(v, 4) if isinstance(v, float) else v)
                              for k, v in rec.items()}), flush=True)        # lr unrounded (it is 3e-5 at the end)
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps or step == a.stop_at:
            summ, recs = evaluate(net, vex_use if a.mode == "overfit" else vex, device, a.flow)
            if a.mode != "overfit":
                summ.update(evaluate_imagery(net, vimg, device))
            per_window = summ.pop("_windows", [])
            if a.mode != "overfit":      # per-window scores for bootstrap uncertainty (scaling fits)
                (out / "val_windows").mkdir(exist_ok=True)
                (out / "val_windows" / f"step{step:06d}.json").write_text(json.dumps(per_window))
            if a.mode == "overfit":
                summ = {k.replace("val_", "fit_"): v for k, v in summ.items()}
            rec = {"step": step, "event": "eval", "tokens_seen": int(step * cfg["tokens_per_step"]), **summ}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            plot_recon(recs, stats, out / "recon" / f"step{step:06d}", step)
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg}
            torch.save(ck, out / "last.pt")
            score = summ.get("val_crps", summ.get("fit_crps"))
            if score < best:
                best = score
                torch.save({"model": net.state_dict(), "step": step, "config": cfg}, out / "best.pt")
            torch.save({"model": net.state_dict(), "step": step, "config": cfg}, out / f"step{step:06d}.pt")
        if a.stop_at and step >= a.stop_at:
            print(f"stopped at step {step} (--stop-at)", flush=True)
            break
    log.close()


if __name__ == "__main__":
    main()
