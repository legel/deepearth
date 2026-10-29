"""Build the FluxFill dashboard as a static folder: static/ plus data/*.json from the dataset and benchmark outputs.

Usage: python -I build.py <out_dir> [--data-root /home/photon/4tb/fluxfill] [--brand <dir with logos>]
Every number on the page comes from a file this script reads; nothing is typed in by hand.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
                        "split": split.get(r["site_id"], "")})
    return {"sites": out}


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
    clay_sites = len(list((root / "clay" / "sites").glob("*"))) if (root / "clay" / "sites").exists() else 0
    n = len(sites["sites"])
    rep_path = root / "towers" / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    if rep:
        sp = rep.get("split", {})
        tiles = [{"value": rep["sites_included"], "label": f"towers in the dataset (of {rep['sites_candidate']} candidates)"},
                 {"value": rep["usable_windows"], "label": "usable 180-day windows"},
                 {"value": sp.get("train", 0), "label": "training towers"},
                 {"value": sp.get("val", 0), "label": "validation towers"},
                 {"value": sp.get("test", 0), "label": "test towers (held out)"},
                 {"value": rep["sites_with_fch4"], "label": "towers measuring methane"}]
    else:
        tiles = [{"value": n, "label": "candidate towers"},
                 {"value": sum(s["ch4"] for s in sites["sites"]), "label": "towers measuring methane"}]
    tiles += [
             {"value": done, "label": "towers with Sentinel-2 complete"},
             {"value": chips, "label": "Sentinel-2 chips gathered"},
             {"value": clay_sites, "label": "towers with Clay embeddings"}]
    return {"built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "tiles": tiles,
            "note": "The dataset, benchmark and model are in construction; counts update with each build."}


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
    man = json.loads((HERE.parent / "bench" / "release" / "MANIFEST.json").read_text())
    c = scored_config(root, counts)
    sc = counts.get("scored", {})
    bl = counts.get("baselines", {})
    s2 = {t["label"]: t["value"] for t in status.get("tiles", [])}
    scope = results.get("scope", {})
    sp, sf, mv = rep["split"], rep["split_fch4"], rep["measured_values"]
    v = dict(sites_candidate=rep["sites_candidate"], sites_included=rep["sites_included"],
             sites_fch4=rep["sites_with_fch4"], windows_all=rep["windows"], windows_usable=rep["usable_windows"],
             n_train=sp["train"], n_val=sp["val"], n_test=sp["test"], fch4_train=sf["train"], fch4_val=sf["val"],
             fch4_test=sf["test"], hourly_sites=rep["hourly_sites"], size_gb=rep["size_gb"]["hh"],
             nee_measured=mv["NEE"], le_measured=mv["LE"], h_measured=mv["H"], fch4_measured=mv["FCH4"],
             release_version=man["version"], release_windows=man["n_windows"], release_sites=man["n_sites"],
             release_families=", ".join(man["families"]), split_sha=man["split_sha256"][:12],
             s2_chips=s2.get("Sentinel-2 chips gathered", 0), s2_sites=s2.get("towers with Sentinel-2 complete", 0),
             clay_sites=s2.get("towers with Clay embeddings", 0),
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
    v["arch"] = methods.render((HERE / "arch_svg.html").read_text(), arch_values(root, counts, 1), meth["references"])
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
    src_test = root / "bench" / "dashboard" / "results_test.json"
    if src_test.exists():
        shutil.copy2(src_test, data / "results_test.json")
        (data / "results_test.json.gz").write_bytes(gzip.compress(src_test.read_bytes(), 9))
    scaling = root / "model" / "scaling" / "scaling.json"
    if scaling.exists():
        shutil.copy2(scaling, data / "scaling.json")
    counts = param_counts(root)
    import methods
    meth = methods.build(root / "model", counts)
    meth["arch_html"] = methods.render((HERE / "arch_svg.html").read_text(), arch_values(root, counts, 1), meth["references"])
    (data / "methods.json").write_text(json.dumps(meth, separators=(",", ":"), ensure_ascii=False))
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
