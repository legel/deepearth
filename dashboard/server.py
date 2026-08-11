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


@app.get("/api/code/<path:p>")
def code(p):
    reg = _state("registry") or abort(404)
    if p not in {f["path"] for f in reg["files"]}:
        abort(404)                                    # only registry-listed files are readable
    return Response((REPO / p).read_text(errors="replace"), mimetype="text/plain")


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
