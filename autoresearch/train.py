"""Train a DeepEarth model from a config file (model, data source, variables, size, budget).

A data adapter (``data.py``) supplies variable widths; evaluation reports each reconstructed variable's transfer
to held-out regions when conditioned on the config's widely-available variables.

Usage:  python train.py configs/deepcal.yaml [--device cuda] [--steps N]
"""
from __future__ import annotations
import argparse, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # reduce fragmentation for large models
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from deepearth.autoresearch import data as data_module
from deepearth.core.fusion import DeepEarth, Variable


def build_variables(spec, dims):
    """Turn the config's variable entries into :class:`Variable`s, filling widths from the data adapter."""
    variables = []
    for v in spec:
        if v.get("expand") == "trait_classes":
            for name, classes in dims["trait_classes"].items():
                variables.append(Variable(name, "categorical", num_classes=classes,
                                          reconstruct=v.get("reconstruct", True)))
            continue
        kind = v["kind"]
        variables.append(Variable(
            v["name"], kind,
            dim=dims.get(v["name"], 0) if kind == "continuous" else 0,
            num_classes=dims["identity_classes"] if v.get("classes") == "identity" else v.get("num_classes", 0),
            reconstruct=v.get("reconstruct", True), neighbor=v.get("neighbor", False)))
    return variables


def train_and_evaluate(config, device):
    seed = config.get("training", {}).get("seed", 0)   # fixed seed -> matched-init A/B: backbone benchmarks bit-identical across runs, so a detached head's causal effect is isolated (no run-to-run noise masquerading as regression).
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    d = config["data"]
    # Prepared-dataset cache: run the glob + KD-tree neighbor build once, reuse across runs; keyed by the data settings that change the assembled set.
    import hashlib, json
    keyparts = {k: d.get(k) for k in ("adapter", "cache_dir", "n_neighbors", "holdout", "subset", "time_axis", "time_km")}
    tag = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    prepared = str(Path(__file__).resolve().parents[1] / "data" / "deepcal" / f"prepared_{tag}.pt")
    source = data_module.build(d["adapter"], cache_dir=d["cache_dir"], n_neighbors=d.get("n_neighbors", 24),
                               device=device, holdout=d.get("holdout", "spatial"), subset=d.get("subset"),
                               time_axis=d.get("time_axis", False), meta_path=d.get("meta_path"),
                               time_km=d.get("time_km", 50.0), prepared=prepared)
    dims = source.variable_dims()
    variables = build_variables(config["variables"], dims)
    m = config["model"]
    manifolds = {name: dims[dim_key] for name, dim_key in m.get("manifolds", {}).items()}
    sg = m.get("species_graph")
    species = dict(species_variable=sg["variable"], species_embedding=source.phylo,
                   species_layers=sg.get("layers", 2), species_heads=sg.get("heads", 4),
                   species_top_k=sg.get("top_k"), species_flex=sg.get("flex", False),
                   species_operator=sg.get("operator", "ou-attention"),
                   species_tree=source.tree if sg.get("operator") == "tree" else None,
                   species_text=getattr(source, "species_text", None) if sg.get("bioclip_init") else None) if sg else {}
    if m.get("species_conditioned_decode") and species:
        # route the refined species state into the species-linked heads (traits + phylo composition)
        _vnames = {v.name for v in variables}
        species["species_conditioned"] = list(dims["trait_classes"]) + (["phylo"] if "phylo" in _vnames else [])
    rel_extra = {}
    if "relative_finest" in m:
        rel_extra["relative_finest"] = tuple(m["relative_finest"])
    if "relative_log2_hashmap_size" in m:
        rel_extra["relative_log2_hashmap_size"] = m["relative_log2_hashmap_size"]
    if "n_heads" in m:
        rel_extra["n_heads"] = m["n_heads"]
    poll_kw = {}                                           # rule 27: pollinator species graph (bilinear cross-tree interaction)
    if m.get("poll_weight", 0.0) > 0 and getattr(source, "pollinator_text", None) is not None and m.get("pollinator_graph", True):
        from deepearth.encoders.biological.phylogenomic import SpeciesGraph
        cdir = Path(d["cache_dir"]); cdir = cdir if cdir.is_absolute() else Path(__file__).resolve().parents[1] / d["cache_dir"]
        pdp = cdir / "pollinator_distance.npy"             # real OpenTree patristic (tree-covered) + BioCLIP-2.5 text shadow (rest)
        pdist = torch.tensor(np.load(pdp), device=device) if pdp.exists() \
            else SpeciesGraph.distance_from_embedding(source.pollinator_text)   # fallback: pure text shadow
        poll_kw = dict(pollinator_text=source.pollinator_text, pollinator_top_k=m.get("pollinator_top_k", 64),
                       pollinator_distance=pdist)
    model = DeepEarth(variables, d_model=m.get("d_model", 256), n_latents=m.get("n_latents", 24),
                      n_layers=m.get("n_layers", 4), capacity=m.get("capacity", 16),
                      relative_window=tuple(m.get("relative_window", (8000., 8000., 300., 130.))), **rel_extra,
                      manifolds=manifolds, compile_processor=m.get("compile") == "processor",
                      rounds=m.get("rounds", 1), write_back=m.get("write_back", True), revise=m.get("revise", False),
                      round_loss=m.get("round_loss", "final"), learned_mask=m.get("learned_mask"),
                      feedback_detach=m.get("feedback_detach", False), flex_attention=m.get("flex_attention", False),
                      decoder_hidden=m.get("decoder_hidden"), loss_weights=m.get("loss_weights"),
                      contrastive_weight=m.get("contrastive_weight", 0.0), contrastive_vars=m.get("contrastive_vars"),
                      smooth_geo=m.get("smooth_geo", False),
                      smooth_geo_sigmas=m.get("smooth_geo_sigmas"),
                      smooth_geo_per_scale=m.get("smooth_geo_per_scale", 32),
                      n_pollinators=getattr(source, "n_pollinators", 0) if m.get("poll_weight", 0.0) > 0 else 0, **poll_kw,
                      reference_latitude_deg=source.reference_latitude_deg, **species).to(device)
    model._sdist_weight = m.get("sdist_weight", 0.0)        # distribution-matching aux loss (U->species toward local community)
    model._poll_weight = m.get("poll_weight", 0.0)          # plant->pollinator distribution aux loss (GloBI); enables B41/B51-B54
    model._phylo_mask_weight = m.get("phylo_mask_weight", 0.0)   # rule 25: mask-and-reconstruct species embedding from relatives
    model._lfmc_weight = m.get("lfmc_weight", 0.0)               # B34 ecophysiology head (live fuel moisture)
    if model._sdist_weight > 0 and hasattr(source, "gbifID"):
        sdp = Path(d["cache_dir"]); sdp = (sdp if sdp.is_absolute() else Path(__file__).resolve().parents[1] / d["cache_dir"]) / "gbif_species_dist.npz"
        if sdp.exists():
            zz = np.load(sdp); mrow = {int(g): i for i, g in enumerate(zz["gbifID"])}
            rows = np.array([mrow.get(int(g), -1) for g in source.gbifID]); ok = rows >= 0
            idx3 = np.where(ok[:, None], zz["idx_3km"][rows.clip(0)], 0); frq3 = np.where(ok[:, None], zz["frq_3km"][rows.clip(0)], 0.0)
            source.sdist_idx = torch.tensor(idx3, dtype=torch.long, device=device)
            source.sdist_frq = torch.tensor(frq3, dtype=torch.float32, device=device)
            print(f"sdist loaded: {int(ok.sum())}/{len(rows)} obs have local distribution", flush=True)
    if m.get("compile", False) or config["training"].get("precision") == "bf16":
        from hashencoder.hashgrid import HashEncoder      # route the hash through its compile/autocast-safe op
        HashEncoder.use_custom_op = True
    print(f"{config['name']}: {source.n} observations, {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, "
          f"train {len(source.train)} / held-out regions {len(source.test)}", flush=True)

    t = config["training"]
    lr0, wd0 = t.get("lr", 3e-4), t.get("weight_decay", 3e-4)
    # Learnable frequency params get much larger/noisier gradients than the rest, so give them their own lower LR and their own gradient clip.
    freq_keys = ("freq_log_scale", "freq_center", "per_level_scale")
    freq_lr = t.get("freq_lr", lr0 * 0.3)   # swept: 9e-5 (0.3x) beat 3e-5 (0.1x)
    freq_params = [p for n, p in model.named_parameters() if any(k in n for k in freq_keys)]
    freq_ids = {id(p) for p in freq_params}
    sparse_hash = m.get("sparse_hash", False)
    if sparse_hash:
        model.enable_sparse_hash(source.coords, lr=lr0, weight_decay=wd0)
        freq_ids |= {id(p) for p in model.absolute_hash_params()}
    rest_params = [p for p in model.parameters() if id(p) not in freq_ids]
    opt = torch.optim.AdamW([{"params": rest_params, "lr": lr0},
                             {"params": freq_params, "lr": freq_lr}], weight_decay=wd0, fused=True)
    steps = t["steps"]; batch = t.get("batch", 512)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bf16 = t.get("precision", "fp32") == "bf16"
    hide_prob = t.get("hide_prob", 0.35)
    # Hash gradients are well-behaved; clip only the non-hash params (where instability comes from) to avoid a huge per-step reduction.
    clip_params = [p for n, p in model.named_parameters()
                   if id(p) not in freq_ids and not any(k in n.lower() for k in ("earth4d", "hash_encoder", "hashgrid", "comm_head", "poll_head", "poll_emb", "lfmc_head"))]
    hash_encoders = [mod for mod in model.modules() if hasattr(mod, "clamp_per_level_scale")]
    def clamp_res():                                     # keep learnable per-level resolutions in the safe (scale>0) region
        for he in hash_encoders:
            he.clamp_per_level_scale()
    # Full-step compile (~2.4x): fuse context + masked_loss into one graph, random reveal mask computed eagerly so capture stays deterministic.
    # Modes: "graph"/"full" -> reduce-overhead (CUDA graphs, biggest win but its pool can alias live buffers on large models); "compiled"/True -> default fusion (stable at any size); "processor" -> narrow Processor-only compile.
    _cm = m.get("compile")
    compile_full = _cm in (True, "full", "graph", "compiled")
    cuda_graphs = _cm in ("full", "graph")
    tc_mode = "reduce-overhead" if cuda_graphs else "default"
    # bf16 covers the Processor and decoders; the Earth4D hash stays fp32 (bf16 rounding near the [-0.9,0.9] coord edge reads OOB), so context() is fp32 and only masked_loss casts.
    if sparse_hash and compile_full:
        def _sparse_step(values, observed, present, flat, coords, nbr, mani, nbrv):
            ctx = model.context_from_flat(flat, coords, nbr, mani, nbrv)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                return model.masked_loss(values, observed, present, ctx)
        sparse_step_fn = torch.compile(_sparse_step, mode=tc_mode)
    elif compile_full:
        def _step(values, observed, present, coords, nbr, mani, nbrv):
            ctx = model.context(coords, nbr, mani, nbrv)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                return model.masked_loss(values, observed, present, ctx)
        step_fn = torch.compile(_step, mode=tc_mode)

    eval_ckpt = config.get("_eval_ckpt")
    if eval_ckpt:                                        # score an existing checkpoint: load weights, skip training
        sd = torch.load(eval_ckpt, map_location=device)
        model.load_state_dict(sd)
        print(f"loaded checkpoint {eval_ckpt} for eval-only scoring", flush=True)
        steps = 0
    model.train(); t0 = time.time()
    # Optional wall-clock training budget (s), measured from step 10 (excludes startup/compile), so architectures compare at equal time. Absent -> train the full ``steps``.
    time_budget = t.get("time_budget_s")
    t_budget_start = None
    steps_done = steps                                   # actual steps run (the budget usually stops us well short of `steps`)
    for step in range(steps):
        if step == 10:
            t_budget_start = time.time()
        if time_budget is not None and t_budget_start is not None and (time.time() - t_budget_start) >= time_budget:
            print(f"  [time budget {time_budget}s reached at step {step}]", flush=True); steps_done = step; break
        idx = source.train_index[torch.randint(0, len(source.train_index), (batch,), device=device)]
        values, observed, coords, nbr_coords, manifold_coords, nbr_values = source.batch(idx)
        if sparse_hash:
            # Refresh the cached discrete cell every K steps so the fast-path hit rate tracks slow resolution drift
            # (correctness holds regardless — the kernel recomputes on a cache miss). Kept outside the compiled step.
            if step > 0 and step % 200 == 0:
                model.absolute_encoder.precompute(source.coords)
            flat = model.read_absolute_leaf(idx)   # detached leaf; also stashes dy_dx/inputs for the resolution grad
            if compile_full:
                if cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                present = {n: (torch.rand(batch, device=device) > hide_prob) & observed[n] for n in model.names}
                blank = torch.rand(batch, device=device) < 0.15   # match reconstruction_loss: 15% fully-blank queries train the position->variable pathway (else A1 species-from-geo collapses under compile)
                for n in model.names: present[n] = present[n] & ~blank
                loss = sparse_step_fn(values, observed, present, flat, coords, nbr_coords,
                                      manifold_coords, nbr_values)
                if cuda_graphs:
                    loss = loss.clone()
            else:
                ctx = model.context_from_flat(flat, coords, nbr_coords, manifold_coords, nbr_values)
                loss = model.reconstruction_loss(values, observed, ctx, hide_prob=hide_prob)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                model.sparse_hash_step(flat, idx)                        # sparse Adam on the absolute hash table
                torch.nn.utils.clip_grad_norm_(clip_params, 5.0)
                if freq_params: torch.nn.utils.clip_grad_norm_(freq_params, 2.0)
                opt.step(); clamp_res()   # AdamW on everything else
            model.set_sparse_lr(sched.get_last_lr()[0]); sched.step()
            if step % 500 == 0:
                print(f"  step {step} loss {float(loss):.3f} [{time.time()-t0:.0f}s]", flush=True)
            continue
        if compile_full:
            # CUDA-graph mode: mark the new iteration and clone the loss so the graph's recycled static buffers are never read afterward.
            if cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            present = {n: (torch.rand(batch, device=device) > hide_prob) & observed[n] for n in model.names}
            blank = torch.rand(batch, device=device) < 0.15   # match reconstruction_loss: 15% fully-blank queries train the position->variable pathway (else A1 species-from-geo collapses under compile)
            for n in model.names: present[n] = present[n] & ~blank
            loss = step_fn(values, observed, present, coords, nbr_coords, manifold_coords, nbr_values)
            if cuda_graphs:
                loss = loss.clone()
        else:
            ctx = model.context(coords, nbr_coords, manifold_coords, nbr_values)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = model.reconstruction_loss(values, observed, ctx, hide_prob=hide_prob)
        if not torch.isfinite(loss):                    # backstop: never let one bad step poison the weights
            opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, 5.0)
        if freq_params: torch.nn.utils.clip_grad_norm_(freq_params, 2.0)
        opt.step(); clamp_res(); sched.step()
        if step % 500 == 0:
            print(f"  step {step} loss {float(loss):.3f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"trained {steps_done} steps in {time.time()-t0:.0f}s", flush=True)

    given = config.get("condition_on", [])
    targets = [v.name for v in variables if v.reconstruct and v.name not in given]
    scores = evaluate(model, source, given, targets, device)
    line = " | ".join(f"{k} {v:.3f}" for k, v in scores.items())
    print(f"held-out regions (conditioning on {given}): {line}", flush=True)
    from deepearth.autoresearch import evaluate as ev      # the frozen benchmark suite -> net-score north star
    _t_eval = time.time()
    raw = ev.evaluate_benchmarks(model, source, device)
    _eval_s = time.time() - _t_eval
    print(ev.format_benchmarks(raw), flush=True)
    print(f"benchmark_suite_seconds: {_eval_s:.1f} ({len(source.test)} held-out rows, {len(ev.normalized(raw))} active)", flush=True)
    ns = ev.net_score(raw)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.startswith("cuda") else 0.0
    if config.get("_tag"):
        print(f"tag:              {config['_tag']}", flush=True)   # parseable run label (matches --tag), so run.log self-identifies
    print(f"net_score:        {ns:.6f}", flush=True)          # parseable north star
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}", flush=True)
    scores["net_score"] = ns
    return model, scores


@torch.no_grad()
def evaluate(model, source, given, targets, device):
    """For each target variable, transfer on held-out regions: accuracy (categorical) or cosine (continuous)."""
    model.eval()
    correct = {t: 0.0 for t in targets}; total = 0
    kinds = {v.name: v.kind for v in model.variables}
    for c0 in range(0, len(source.test), 2048):
        idx = torch.tensor(source.test[c0:c0 + 2048], device=device)
        values, observed, coords, nbr_coords, manifold_coords, nbr_values = source.batch(idx)
        ctx = model.context(coords, nbr_coords, manifold_coords, nbr_values)
        preds = model.infer(values, given, targets, ctx)
        for t in targets:
            if kinds[t] == "categorical":
                correct[t] += (preds[t].argmax(-1) == values[t]).float().sum().item()
            else:
                correct[t] += F.cosine_similarity(preds[t], values[t], dim=-1).sum().item()
        total += len(idx)
    return {t: correct[t] / total for t in targets}


def main():
    ap = argparse.ArgumentParser(description="Train a DeepEarth model from a config (default deepcal.yaml).")
    ap.add_argument("config", nargs="?", default=str(Path(__file__).with_name("deepcal.yaml")))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--eval_ckpt", default=None, help="score an existing checkpoint (skip training)")
    ap.add_argument("--time_budget", type=float, default=None, help="stop training after N seconds (experiment budget)")
    ap.add_argument("--tag", default=None, help="a label for this run (recorded, e.g. the experiment id)")
    ap.add_argument("--save", action="store_true", help="save the checkpoint (off by default; on only for champion runs)")
    a = ap.parse_args()
    config = yaml.safe_load(open(a.config))
    if a.steps is not None:
        config["training"]["steps"] = a.steps
    if a.cache_dir is not None:
        config["data"]["cache_dir"] = a.cache_dir
    # Portability: a relative cache_dir is resolved against the repo root (where prepare.py downloads the cache), so a fresh clone on any device works without editing absolute paths.
    _cd = config["data"].get("cache_dir")
    if _cd and not os.path.isabs(_cd):
        config["data"]["cache_dir"] = str(Path(__file__).resolve().parents[1] / _cd)
    if a.eval_ckpt is not None:
        config["_eval_ckpt"] = a.eval_ckpt
    if a.time_budget is not None:
        config["training"]["time_budget_s"] = a.time_budget
    if a.tag is not None:
        config["_tag"] = a.tag
    model, _ = train_and_evaluate(config, a.device)
    if a.save and a.eval_ckpt is None:
        torch.save(model.state_dict(), Path(a.config).with_suffix(".pt"))


if __name__ == "__main__":
    main()
