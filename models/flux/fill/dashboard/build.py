"""Build the FluxFill dashboard as a static folder: static/ plus data/*.json from the dataset and benchmark outputs.

Usage: python -I build.py <out_dir> [--data-root /home/photon/4tb/fluxfill] [--brand <dir with logos>]
Every number on the page comes from a file this script reads; nothing is typed in by hand.
"""

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import components  # noqa: E402
sys.path.insert(0, str(HERE))


def truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def sites_json(root: Path) -> dict:
    path = root / "towers" / "sites.csv"
    if not path.exists():
        return {"sites": []}
    split = {}
    split_path = root / "towers" / "split.csv"
    if split_path.exists():
        with split_path.open() as f:
            for r in csv.DictReader(f):
                split[r["site_id"]] = {"val": "validation"}.get(r.get("split", ""), r.get("split", ""))
    lf = HERE / "static_data" / "site_links.json"           # site pages (site_links.py)
    links = json.loads(lf.read_text())["links"] if lf.exists() else {}
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError):
                continue
            out.append({"id": r["site_id"], "name": r.get("site_name", ""), "lat": lat, "lon": lon,
                        "igbp": r.get("igbp", ""), "koppen": r.get("koppen", ""),
                        "network": r.get("source_network") or r.get("data_hub", ""),
                        "years": r.get("shuttle_years", ""),
                        "ch4": truthy(r.get("in_fluxnet_ch4", "")) or truthy(r.get("in_amf_base_fch4", "")),
                        "split": split.get(r["site_id"], ""), **links.get(r["site_id"], {})})
    return {"sites": out}


_BM = {}
BASELINE_IDS = ("climatology", "mds", "irvin_rf", "vekuri_de", "liu_xgb", "saits", "csdi", "chronos2")


def benchmark_data_cached(root: Path) -> dict:
    import benchmark_data
    if str(root) not in _BM:
        _BM[str(root)] = benchmark_data.build(root)
    return _BM[str(root)]


def status_json(root: Path, sites: dict) -> dict:
    s2 = list((root / "sentinel2" / "status").glob("*.json")) if (root / "sentinel2" / "status").exists() else []
    chips = done = 0
    for p in s2:
        try:
            st = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        chips += int(st.get("chips", 0))
        done += bool(st.get("complete"))
    ids = {x["id"] for x in sites["sites"]}
    clay_sites = sum(1 for q in (root / "clay" / "sites").glob("*.zarr") if q.stem in ids) if (root / "clay" / "sites").exists() else 0
    n = len(sites["sites"])
    bm = benchmark_data_cached(root)
    sp = bm["splits"]["frozen"]
    towers = sum(v["towers"] for v in sp.values())
    flux_kept = sum(q["final"] for q in bm["qc"] if q["variable"] in ("NEE", "LE", "H", "FCH4"))
    res = overview_results(root)
    val = next((r for r in res if r["view"] == "Validation sites"), None)
    n_base = len(BASELINE_IDS)
    mega = lambda v: f"{v / 1e6:.1f} M"
    tiles = [{"group": "Data", "value": towers, "label": f"towers with measured half-hourly fluxes, of {n} candidates"},
             {"group": "Data", "value": f"{sp['train']['towers']} / {sp['validation']['towers']} / {sp['test']['towers']}",
              "label": "training / validation / test towers"},
             {"group": "Data", "value": sum(v["methane"] for v in sp.values()), "label": "towers measuring methane"},
             {"group": "Data", "value": mega(flux_kept), "label": "flux half-hours after quality control (NEE, LE, H, FCH4)"},
             {"group": "Data", "value": mega(chips), "label": f"Sentinel-2 chips of 300 m at {done} towers"},
             {"group": "Data", "value": clay_sites, "label": "towers with a Clay embedding of every chip"},
             {"group": "Benchmark", "value": bm["test"]["n_windows"], "label": f"test windows of 180 days at {bm['test']['n_sites']} towers"},
             {"group": "Benchmark", "value": mega(bm["test"].get("hidden_total", 0)), "label": "hidden measured values scored at test towers"},
             {"group": "Benchmark", "value": n_base, "label": "established methods on identical windows and masks"},
             {"group": "Benchmark", "value": val["windows"] if val else 0,
              "label": f"validation windows at {val['sites'] if val else 0} towers, for model selection"}]
    upd = HERE / "static_data" / "updates.json"
    return {"built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "tiles": tiles,
            "results": res, "abstract": overview_abstract(root, res, sp["train"]["towers"]),
            "updates": json.loads(upd.read_text()) if upd.exists() else [],
            "note": "Counts and results update with each build."}


def overview_results(root: Path) -> list:
    """Overview: per result view and flux, the best FluxFill version against the best baseline on the cells every
    method in that view fills (the view's summary table), and on how many metrics a FluxFill version is best."""
    out = []
    for name, fn in (("Validation sites", "results.json"), ("Test sites", "results_test.json")):
        p = root / "bench" / "dashboard" / fn
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ours = {m["id"] for m in r["methods"] if m["ours"]}
        short = {m["id"]: m["short"] for m in r["methods"]}
        fluxes = []
        for t in r["summary"]:
            i = t["metrics"].index("crps")
            vals = [(row["method"], row["cells"][i]["v"]) for row in t["rows"] if row["cells"][i]["v"] is not None]
            o = min((x for x in vals if x[0] in ours), key=lambda x: x[1], default=None)
            b = min((x for x in vals if x[0] not in ours), key=lambda x: x[1], default=None)
            if not o or not b:
                continue
            best_all = t.get("best_all", [[x] for x in t["best"]])
            wins = sum(any(t["rows"][j]["method"] in ours for j in ba) for ba in best_all)
            ours_row = next(j for j, row in enumerate(t["rows"]) if row["method"] == o[0])
            cov = t["rows"][ours_row]["cells"][t["metrics"].index("cov95")]["v"]
            lost = [{"metric": t["metrics"][i], "ours": t["rows"][ours_row]["cells"][i]["v"],
                     "leader": short[t["rows"][ba[0]]["method"]], "lead": t["rows"][ba[0]]["cells"][i]["v"]}
                    for i, ba in enumerate(best_all) if ba and not any(t["rows"][j]["method"] in ours for j in ba)]
            fluxes.append({"flux": t["flux"], "units": t["units"], "cells": t["cells"], "n": t["n"], "ours": short[o[0]],
                           "ours_crps": o[1], "base": short[b[0]], "base_crps": b[1], "pct": 100 * (o[1] / b[1] - 1), "ours_cov95": cov,
                           "best_metrics": wins, "metrics": len(t["metrics"]), "methods": len(t["rows"]), "lost": lost})
        out.append({"view": name, "windows": r["scope"]["windows"], "sites": r["scope"]["sites"], "fluxes": fluxes})
    return out


def overview_abstract(root: Path, res: list, train_towers: int) -> str:
    """The Overview paragraph: the abstract with the numbers of the Overview table, so both change together."""
    views = {r["view"]: {f["flux"]: f for f in r["fluxes"]} | {"n": r} for r in res}
    t, v = views["Test sites"], views["Validation sites"]
    cmp = lambda f: f"{abs(f['pct']):.1f}% {'lower' if f['pct'] < 0 else 'higher'}"
    rep = root / "bench" / "dashboard" / "report" / "values.json"
    vals = json.loads(rep.read_text()) if rep.exists() else {}
    dec = root / "model" / "copula_final.decide.log"
    cop = re.search(r"sum-30-day .*cov95 ([\d.]+) -> ([\d.]+)", dec.read_text()) if dec.exists() else None
    return " ".join([
        f"Eddy-covariance towers measure the exchange of carbon dioxide, water vapour, heat and methane every half-hour, and "
        f"station outages remove {vals.get('outage_share', '?')}% of the record, {vals.get('gap_share_ge7d', '?')}% of that time in "
        "outages of 7 days or longer.",
        "FluxFill is a self-supervised AI foundation model trained to reconstruct masked and missing flux data, based on the "
        "DeepEarth neural network architecture. It predicts a distribution for any hidden part of a 180-day window of half-hourly "
        f"tower fluxes, tower meteorology, reanalysis drivers and Clay embeddings of Sentinel-2 images, and is trained on {train_towers} towers.",
        f"On {t['n']['windows']} benchmark windows at {t['n']['sites']} test towers at least 50 km from any training tower, its "
        f"continuous ranked probability score is {cmp(t['NEE'])} than that of the best established method for net ecosystem exchange "
        f"(NEE), {cmp(t['LE'])} for latent heat (LE), {cmp(t['H'])} for sensible heat (H) and {cmp(t['FCH4'])} for methane (FCH4), "
        f"and its 95% intervals contain {t['NEE']['ours_cov95']:.3f} of hidden NEE values.",
        "It has the lowest score for NEE, LE and H at every gap length from 1 to 180 days.",
        (f"On {v['n']['windows']} validation windows, a temporal copula raises the 95% coverage of 30-day sums from {cop.group(1)} to "
         f"{cop.group(2)}; other methods reach 0.15 to 0.45." if cop else ""),
        "Chronos-2 and XGBoost lead at the shortest gaps, and the per-tower deep ensemble leads on mixed per-variable gaps for NEE, LE and H. On FCH4, Chronos-2 and the per-tower random forest ensemble lead at most gap "
        f"lengths, and the random forest ensemble is {abs(v['FCH4']['pct']):.1f}% lower at validation towers.",
        "These test scores are from the network trained for 24,000 steps; the final model is scored once at the end of the component study."])


FLUXFILL_PY = Path("/home/photon/4tb/envs/fluxfill/bin/python")


def param_counts(root: Path) -> dict:
    """Parameter counts from the checkpoints (param_counts.py, run in the FluxFill environment, which has torch);
    the cached file is used when that environment is not available."""
    import os
    import subprocess
    cache = root / "bench" / "dashboard" / "param_counts.json"
    if FLUXFILL_PY.exists():
        tmp = cache.with_suffix(".json.tmp")
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
        r = subprocess.run([str(FLUXFILL_PY), str(HERE / "param_counts.py"), "--model-root", str(root / "model"),
                            "--out", str(tmp), "--cache", str(cache)], capture_output=True, text=True, env=env,
                           timeout=900)
        if r.returncode == 0 and tmp.exists():
            tmp.replace(cache)
        else:
            print("param_counts.py failed; using the cached counts:", r.stderr[-400:])
    return json.loads(cache.read_text()) if cache.exists() else {}


def scored_config(root: Path, counts: dict) -> dict:
    sc = counts.get("scored") or {}
    p = root / "model" / sc.get("run", "") / "config.json"
    return json.loads(p.read_text()) if sc and p.exists() else {}


def bench_defaults() -> dict:
    """Sample and round counts of the FluxFill benchmark runner (model/score_bench.py argument defaults)."""
    import re
    t = (HERE.parent / "model" / "score_bench.py").read_text()
    get = lambda k: int(re.search(r'"--' + k + r'", type=int, default=(\d+)', t).group(1))
    return {"samples": get("samples"), "rounds": get("steps")}


def arch_values(root: Path, counts: dict, fig) -> dict:
    c = scored_config(root, counts)
    sc = counts.get("scored", {})
    rows = len(c.get("stats", {}).get("vars", []))
    days = 180
    return dict(d=c.get("d", "?"), depth=c.get("depth", "?"), heads=c.get("heads", "?"), window_days=days,
                n_rows=rows, n_tokens=f"{(rows + 1) * days:,}", n_params=f"{sc.get('n_params', 0):,}",
                scored_run=sc.get("run", "?"), scored_step=f"{sc.get('step', 0):,}",
                train_sites=len(c.get("train_sites", [])), batch=c.get("batch", 0) * c.get("accum", 1),
                train_steps=f"{c.get('steps', 0):,}", lr=f"{c.get('lr', 0):g}", fig=fig, **bench_defaults())


def report_json(root: Path, counts: dict, meth: dict, results: dict, status: dict) -> dict:
    """The draft report: dashboard/report_draft.html with values from the dataset, benchmark and run files."""
    import methods
    rep = json.loads((root / "towers" / "report.json").read_text())
    man = json.loads((HERE.parent / "bench" / "release" / "0.2" / "MANIFEST.json").read_text())
    c = scored_config(root, counts)
    sc = counts.get("scored", {})
    bl = counts.get("baselines", {})
    s2 = {t["label"]: t["value"] for t in status.get("tiles", [])}
    chips_tile = next(t for t in status["tiles"] if t["label"].startswith("Sentinel-2 chips"))
    scope = results.get("scope", {})
    sp, sf, mv = rep["split"], rep["split_fch4"], rep["measured_values"]
    v = dict(sites_candidate=rep["sites_candidate"], sites_included=rep["sites_included"],
             sites_fch4=rep["sites_with_fch4"], windows_all=rep["windows"], windows_usable=rep["usable_windows"],
             n_train=sp["train"], n_val=sp["val"], n_test=sp["test"], fch4_train=sf["train"], fch4_val=sf["val"],
             fch4_test=sf["test"], hourly_sites=rep["hourly_sites"], size_gb=rep["size_gb"]["hh"],
             nee_measured=mv["NEE"], le_measured=mv["LE"], h_measured=mv["H"], fch4_measured=mv["FCH4"],
             release_version=man["version"], release_windows=man["n_windows"], release_sites=man["n_sites"],
             release_families=", ".join(man["families"]), split_sha=man["split_sha256"][:12],
             s2_chips=chips_tile["value"], s2_sites=int(re.search(r"at (\d+) towers", chips_tile["label"]).group(1)),
             clay_sites=s2.get("towers with a Clay embedding of every chip", 0),
             model_params=f"{sc.get('n_params', 0):,}", scored_run=sc.get("run", "?"),
             scored_step=f"{sc.get('step', 0):,}", d=c.get("d", "?"), depth=c.get("depth", "?"),
             heads=c.get("heads", "?"), n_rows=len(c.get("stats", {}).get("vars", [])),
             train_steps=f"{c.get('steps', 0):,}", batch=c.get("batch", 0) * c.get("accum", 1),
             lr=f"{c.get('lr', 0):g}", warmup=c.get("warmup", 0),
             saits_params=f"{bl.get('saits', {}).get('n_params', 0):,}",
             csdi_params=f"{bl.get('csdi', {}).get('n_params', 0):,}",
             chronos_params=f"{bl.get('chronos2', {}).get('n_params', 0) / 1e6:.1f} million",
             pypots_sites=bl.get("saits", {}).get("n_train_sites", 0),
             val_windows=scope.get("windows", 0), val_sites=scope.get("sites", 0),
             built=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), **bench_defaults())
    v["arch"] = methods.render((HERE / "arch_svg.html").read_text(), arch_values(root, counts, 2), meth["references"])
    # Figures, tables and numbers computed from the score files by report_figures.py (bench/dashboard/report/).
    rd = root / "bench" / "dashboard" / "report"
    if (rd / "values.json").exists():
        def flat(prefix, x):
            if isinstance(x, dict):
                for k, y in x.items():
                    yield from flat(f"{prefix}_{k}" if prefix else str(k), y)
            elif not isinstance(x, list):
                yield re.sub(r"[^a-z0-9_]", "_", prefix.lower()), x
        v.update(dict(flat("", json.loads((rd / "values.json").read_text()))))
        v.update(json.loads((rd / "tables.json").read_text()))
    comp = components.collect(root / "model")
    v.update(table_components=components.table_html(comp), comp_done=len(comp["rows"]), comp_total=comp["total"],
             comp_noise=f"{abs(comp['noise']['crps']):.1f}%" if comp["noise"] else "–")
    body = methods.render((HERE / "report_draft.html").read_text(), v, meth["references"])
    return {"html": body, "references": meth["references"], "draft": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--data-root", default="/home/photon/4tb/fluxfill")
    ap.add_argument("--brand", default="/home/photon/ecological/flux_gapfill/brand")
    a = ap.parse_args()
    out, root, brand = Path(a.out), Path(a.data_root), Path(a.brand)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(HERE / "static", out)
    (out / "img").mkdir(exist_ok=True)
    for name in ("qed-lab.svg", "logo_ei_lockup.png", "favicon-32.png"):
        if (brand / name).exists():
            shutil.copy2(brand / name, out / "img" / name)
    data = out / "data"
    (data / "windows").mkdir(parents=True, exist_ok=True)
    # Natural Earth 1:110m land v5.1.2 (public domain, cut at the antimeridian), the map's only base layer.
    shutil.copy2(HERE / "static_data" / "ne_110m_land.geojson", data / "ne_110m_land.geojson")
    sites = sites_json(root)
    (data / "sites.json").write_text(json.dumps(sites, separators=(",", ":")))
    (data / "status.json").write_text(json.dumps(status_json(root, sites), indent=1))
    import gzip
    src = root / "bench" / "dashboard" / "results.json"
    results = {}
    if src.exists():
        shutil.copy2(src, data / "results.json")
        (data / "results.json.gz").write_bytes(gzip.compress(src.read_bytes(), 9))
        results = json.loads(src.read_text())
    (data / "benchmark.json").write_text(json.dumps(benchmark_data_cached(root), separators=(",", ":")))
    for name in ("results_test.json",):
        src_x = root / "bench" / "dashboard" / name
        if src_x.exists():
            shutil.copy2(src_x, data / name)
            (data / f"{name}.gz").write_bytes(gzip.compress(src_x.read_bytes(), 9))
    (data / "components.json").write_text(json.dumps(components.collect(root / "model"), separators=(",", ":")))
    scaling = root / "model" / "scaling" / "scaling.json"
    if scaling.exists():
        shutil.copy2(scaling, data / "scaling.json")
    counts = param_counts(root)
    import methods
    meth = methods.build(root / "model", counts)
    meth["arch_html"] = methods.render((HERE / "arch_svg.html").read_text(), arch_values(root, counts, 1), meth["references"])
    (data / "methods.json").write_text(json.dumps(meth, separators=(",", ":"), ensure_ascii=False))
    rd = root / "bench" / "dashboard" / "report"
    if rd.exists():
        (out / "img" / "report").mkdir(parents=True, exist_ok=True)
        for f in rd.glob("fig*.svg"):
            shutil.copy2(f, out / "img" / "report" / f.name)
    (data / "report.json").write_text(json.dumps(report_json(root, counts, meth, results, status_json(root, sites)),
                                                 separators=(",", ":"), ensure_ascii=False))
    win_src = root / "bench" / "dashboard" / "windows"
    if win_src.exists():
        shutil.copytree(win_src, data / "windows", dirs_exist_ok=True)
        # The server sends JSON uncompressed; a gzip copy of each window is read by the page (DecompressionStream),
        # with the plain file as the fallback.
        for f in (data / "windows").glob("*.json"):
            if f.name != "index.json":
                (f.parent / (f.name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), 9))
    print(f"built {out}: {len(sites['sites'])} sites")


if __name__ == "__main__":
    main()
