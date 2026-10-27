"""Build the FluxFill dashboard as a static folder: static/ plus data/*.json from the dataset and benchmark outputs.

Usage: python -I build.py <out_dir> [--data-root /home/photon/4tb/fluxfill] [--brand <dir with logos>]
Every number on the page comes from a file this script reads; nothing is typed in by hand.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
    for name in ("results.json", "report.json"):
        src = root / "bench" / "dashboard" / name
        if src.exists():
            shutil.copy2(src, data / name)
    win_src = root / "bench" / "dashboard" / "windows"
    if win_src.exists():
        shutil.copytree(win_src, data / "windows", dirs_exist_ok=True)
        # The server sends JSON uncompressed; a gzip copy of each window is read by the page (DecompressionStream),
        # with the plain file as the fallback.
        import gzip
        for f in (data / "windows").glob("*.json"):
            if f.name != "index.json":
                (f.parent / (f.name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), 9))
    print(f"built {out}: {len(sites['sites'])} sites")


if __name__ == "__main__":
    main()
