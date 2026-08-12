"""Thin Flask reader over state/, runs/, and repo file content.

    python -m dashboard.server [--port 8321]
"""
import argparse, json, subprocess
from pathlib import Path
from flask import Flask, Response, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STATE, RUNS, STATIC = ROOT / "state", ROOT / "runs", ROOT / "static"

app = Flask(__name__, static_folder=None)


def _state(name):
    p = STATE / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_file(p):
    return send_from_directory(STATIC, p)


@app.get("/api/meta")
def meta():
    head = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h|%s|%ci"],
                          capture_output=True, text=True).stdout.strip().split("|")
    reg = _state("registry") or {}
    return jsonify({
        "head": {"sha": head[0], "subject": head[1], "date": head[2]} if len(head) == 3 else None,
        "counts": reg.get("counts"),
        "audited": (_state("status") or {}).get("audited"),
    })


@app.get("/api/registry")
def registry():
    return jsonify(_state("registry") or abort(404))


@app.get("/api/graph")
def graph():
    return jsonify(_state("graph") or abort(404))


@app.get("/api/status")
def status():
    return jsonify(_state("status") or abort(404))


@app.get("/api/verification")
def verification():
    return jsonify(_state("verification") or abort(404))


@app.get("/api/callgraph")
def callgraph():
    return jsonify(_state("callgraph") or abort(404))


@app.get("/api/flow")
def flow():
    return jsonify(_state("flow") or abort(404))


@app.get("/api/findings")
def findings():
    p = ROOT / "seed" / "findings.json"
    return jsonify(json.loads(p.read_text())) if p.exists() else abort(404)


@app.get("/api/reconstructions")
def reconstructions():
    return jsonify(_state("reconstructions") or abort(404))


@app.get("/api/code/<path:p>")
def code(p):
    reg = _state("registry") or abort(404)
    if p not in {f["path"] for f in reg["files"]}:
        abort(404)                                    # only registry-listed files are readable
    return Response((REPO / p).read_text(errors="replace"), mimetype="text/plain")


_OBS = {}


def _obs():
    if not _OBS:
        import numpy as np
        p = STATE / "observations.npz"
        p.exists() or abort(404)
        _OBS.update(np.load(p))
        _OBS["species"] = json.loads((STATE / "species.json").read_text())
    return _OBS


MODS = ["daymet", "naip", "clay", "soil", "topo", "chm", "hydro", "flower", "species_dist"]


@app.get("/api/observations")
def observations():
    import numpy as np
    o = _obs()
    n = len(o["lat"])
    keep = np.ones(n, bool)
    if (split := request.args.get("split")) in ("train", "test"):
        keep &= o["test"] == (split == "test")
    if (sp := request.args.get("sp", type=int)) is not None:
        keep &= o["sp"] == sp
    idx = np.where(keep)[0]
    cap = request.args.get("n", 12000, type=int)
    if len(idx) > cap:
        idx = idx[:: len(idx) // cap + 1]                 # deterministic thinning
    return jsonify({"total": int(keep.sum()), "shown": len(idx),
                    "id": o["gbifID"][idx].tolist(), "lat": o["lat"][idx].round(5).tolist(),
                    "lon": o["lon"][idx].round(5).tolist(), "sp": o["sp"][idx].tolist(),
                    "test": o["test"][idx].astype(int).tolist()})


@app.get("/api/observation/<int:gid>")
def observation(gid):
    import numpy as np
    o = _obs()
    hits = np.where(o["gbifID"] == gid)[0]
    len(hits) or abort(404)
    i = int(hits[0])
    days = float(o["days"][i])
    date = None if np.isnan(days) else (np.datetime64("1970-01-01") + np.timedelta64(int(days), "D")).item().isoformat()
    return jsonify({
        "gbifID": int(gid), "lat": float(o["lat"][i]), "lon": float(o["lon"][i]),
        "elev": float(o["elev"][i]), "date": date,
        "species": o["species"]["binomial"][int(o["sp"][i])], "sp": int(o["sp"][i]),
        "split": "test" if o["test"][i] else "train",
        "modalities": [m for b, m in enumerate(MODS) if int(o["mods"][i]) >> b & 1],
        "source": f"https://www.gbif.org/occurrence/{gid}"})


DAYMET_COLS = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
TOPO_COLS = ["elev", "slope_deg", "northness", "eastness", "TRI", "curvature", "VRM", "HLI",
             "TPI8", "TPI24", "TPI72", "TPI140"]
SOIL_COLS = ["pH", "organic_matter", "clay_%", "sand_%", "silt_%", "awc", "ksat", "cec7", "bulk_density"]
HYDRO_COLS = ["TWI", "HAND_m", "ln_SCA", "Sx_west", "Sx_max", "TPI"]          # build_hydrowind_torch.py:108
CHM_COLS = ["mean_m", "std_m", "max_m", "p50_m", "p90_m", "p95_m", "cover>2m",
            "gap<0.5m", "shrub_0.5-2m", "rumple", "heterogeneity"]            # build_chm.py:26-37
DATA = REPO / "data" / "deepcal"
_RAW = {}


def _kv(npz_name, key):
    if npz_name not in _RAW:
        import numpy as np
        z = np.load(DATA / npz_name, allow_pickle=True)
        _RAW[npz_name] = (dict(zip(z["gbifID"].tolist(), range(len(z["gbifID"])))), z[key])
    return _RAW[npz_name]


def _daymet(gid):
    import numpy as np
    if "daymet_idx" not in _RAW:                        # gbifID -> (chunk, row) over all shards
        idx = {}
        for p in sorted((DATA / "gbif_daymet_tokens").glob("chunk*.npz")):
            for r, g in enumerate(np.load(p, allow_pickle=True)["gbifID"].tolist()):
                idx[g] = (p, r)
        _RAW["daymet_idx"] = idx
    loc = _RAW["daymet_idx"].get(gid)
    return None if loc is None else np.load(loc[0], allow_pickle=True)["daymet"][loc[1]].astype(float)


@app.get("/api/observation/<int:gid>/raw")
def observation_raw(gid):
    import math
    r3 = lambda x: round(float(x), 3) if math.isfinite(float(x)) else None   # NaN is invalid JSON
    out = {}
    d = _daymet(gid)
    if d is not None:
        out["climate"] = {"cols": DAYMET_COLS, "rows": [[r3(v) for v in r] for r in d]}
    for npz, key, names in [("gbif_soil_tokens.npz", "soil", SOIL_COLS),
                            ("gbif_topo_tokens.npz", "topo", TOPO_COLS),
                            ("gbif_hydro_tokens.npz", "hydro", HYDRO_COLS),
                            ("gbif_chm_tokens.npz", "chm", CHM_COLS)]:
        idx, arr = _kv(npz, key)
        if (i := idx.get(gid)) is not None:
            v = [r3(x) for x in arr[i]]
            out[key] = dict(zip(names, v)) if names else v
    return jsonify(out)


@app.get("/api/species")
def species():
    o = _obs()
    q = request.args.get("q", "").lower()
    s = o["species"]
    hits = [{"sp": i, "name": nm, "n": c} for i, (nm, c) in enumerate(zip(s["binomial"], s["count"]))
            if q in nm.lower() and c > 0]
    hits.sort(key=lambda x: -x["n"])
    return jsonify(hits[:30])


@app.get("/api/runs")
def runs():
    out = []
    for p in sorted(RUNS.glob("*.jsonl"), reverse=True) if RUNS.exists() else []:
        first = last = None
        with open(p) as f:
            for line in f:
                if line.strip():
                    last = line
                    first = first or line
        out.append({"id": p.stem, "config": json.loads(first) if first else None,
                    "last": json.loads(last) if last else None})
    return jsonify(out)


@app.get("/api/runs/<rid>")
def run_events(rid):
    p = RUNS / f"{rid}.jsonl"
    p.exists() or abort(404)
    offset = request.args.get("offset", 0, type=int)   # byte offset for live tailing
    with open(p) as f:
        f.seek(offset)
        text = f.read()
    events = [json.loads(l) for l in text.splitlines() if l.strip()]
    return jsonify({"events": events, "offset": offset + len(text.encode())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
