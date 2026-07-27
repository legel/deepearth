"""agents/earth4d/trace.py — FIXED harness: run the Earth4D ENCODER PROBE, emit a consistent trace.

Surface = ONLY the spacetime encoder. Every experiment is a big architectural swing on Earth4D
(encoders/spacetime/earth4d.py) measured by the fast encoder probe
(autoresearch/programs/spacetime/probe.py & friends) — NOT full-model training. The probe trains only
the encoder + a light head on ~65k obs in minutes and reports st_gain (Earth4D vs the fair coordinate /
RFF / MLP baseline) = the encoder's isolated marginal on the SAME capability the scorecard/science measures.

--metric (required) declares which scorecard capability this run targets (keeps the loop on track).
--probe  (required) is the probe invocation flags = the architectural lever. Native probe metrics only.
--ensue  auto-logs the trace to Ensue (token from /workspace/.env, never committed).

Usage:
  python -m deepearth.agents.earth4d.trace --metric family_from_spacetime \
      --probe "--n_shards 8 --steps 800" --tag baseline --device cuda:0 --ensue
  python -m deepearth.agents.earth4d.trace --metric species_from_env \
      --probe "--sdm_presence --n_shards 8" --tag sdm_recurrence --device cuda:1 --ensue
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # .../deepearth

# The scorecard capabilities — the objective must be one (keeps the loop on track). Same capabilities as
# the science / evaluate.py, scoped to the encoder probe. The probe MODE/architecture is the loop's choice.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "lfmc_from_env", "mycorrhiza_from_env", "pollinator_from_env",
    "calibration", "flowering_auc", "flowering_fidelity", "flowering_peak_month",
    "infer_clay", "infer_soil", "infer_climate", "infer_hydro",
]


def _run(module: str, probe_args: str, device: str, log_path: str) -> int:
    cmd = [sys.executable, "-m", module] + shlex.split(probe_args) + ["--device", device]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
    with open(log_path, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode


def _parse(text: str):
    header = next((l.strip() for l in text.splitlines() if l.strip().startswith("=== SPACETIME")), "")
    st_gains = {}
    for m in re.finditer(r"st_gain(?:\(([^)]*)\))?\s*([+\-]?\d+\.\d+)", text):
        st_gains[(m.group(1) or "default").strip()] = float(m.group(2))
    metrics = [l.strip() for l in text.splitlines()
               if re.search(r"\b(acc|micro-AP|MAE|absR2|GAIN|Spearman|top5|prop)\b", l) and l.strip()]
    return header, st_gains, metrics[:24]


def _ensue_token() -> str:
    t = os.environ.get("ENSUE_API_TOKEN")
    if t:
        return t.strip()
    f = Path("/workspace/.env")
    if f.exists():
        for line in f.read_text(errors="ignore").splitlines():
            if line.strip().startswith("ENSUE_API_TOKEN"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def post_ensue(trace: dict) -> None:
    tok = _ensue_token()
    if not tok:
        print("[trace] --ensue set but no ENSUE_API_TOKEN (env or /workspace/.env); skipping", flush=True)
        return
    best = max(trace["st_gain"].values()) if trace["st_gain"] else None
    val = (f"Earth4D ENCODER-PROBE '{trace['tag']}' | capability {trace['metric']} | probe='{trace['probe']}'. "
           f"st_gain={trace['st_gain']} (best {best}). {trace['header']}. "
           f"metrics: {' || '.join(trace['metrics'][:8])}")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "create_memory", "arguments": {
        "items": [{"key_name": f"earth4d_probe_{trace['tag']}_{trace['metric']}", "value": val,
                   "description": f"Earth4D probe {trace['metric']} via '{trace['tag']}' (best st_gain {best})"}]}}}
    req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[trace] Ensue logged ({r.status}): earth4d_probe_{trace['tag']}_{trace['metric']}", flush=True)
    except Exception as e:
        print(f"[trace] Ensue POST failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Earth4D encoder-probe trace harness — big architectural swings, measured fast.")
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--probe", required=True, help="probe flags = the architectural lever (quote the whole string)")
    ap.add_argument("--probe-module", default="deepearth.autoresearch.programs.spacetime.probe",
                    help="probe module to run (e.g. ...spacetime.calib_probe for calibration)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ensue", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    if a.metric not in CAPABILITIES:
        sys.exit("[trace] --metric %r is not a scorecard capability. one of:\n  %s"
                 % (a.metric, "\n  ".join(CAPABILITIES)))

    tag = a.tag or ("e4d_" + re.sub(r"\W+", "_", a.probe)[:24].strip("_"))
    log_path = a.log or str(Path(__file__).resolve().parent / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] OBJECTIVE={a.metric}  probe='{a.probe}'  tag={tag}", flush=True)
    rc = _run(a.probe_module, a.probe, a.device, log_path)
    text = Path(log_path).read_text(errors="ignore")
    header, st_gains, metrics = _parse(text)
    if rc != 0 and not st_gains and not header:
        print(text[-1800:])
        sys.exit(f"[trace] probe FAILED (rc={rc}); see {log_path}")

    trace = {"metric": a.metric, "tag": tag, "probe": a.probe, "probe_module": a.probe_module,
             "st_gain": st_gains, "header": header, "metrics": metrics, "rc": rc}

    print("\n" + "=" * 74)
    print(f"OBJECTIVE {a.metric}   probe='{a.probe}'")
    print(header or "(no '=== SPACETIME' header parsed — check the log)")
    print("-" * 74)
    print("st_gain (encoder marginal vs fair baseline):", st_gains or "(none parsed)")
    print("metrics:")
    for m in metrics:
        print("  " + m)
    print("=" * 74)
    out = Path(log_path).with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2))
    print(f"[trace] wrote {out}")
    if a.ensue:
        post_ensue(trace)


if __name__ == "__main__":
    main()
