"""agents/earth4d/trace.py — FIXED harness: run the Earth4D ENCODER PROBE, emit a consistent trace.

Surface = ONLY the spacetime encoder. Every experiment is a big architectural swing on Earth4D
(encoders/spacetime/earth4d.py) measured by the fast encoder probe
(autoresearch/programs/spacetime/probe.py & friends) — NOT full-model training. The probe trains only
the encoder + a light head on ~65k obs in minutes and reports the encoder-isolated marginal vs a FAIR
baseline (RFF / MLP / best generic PE) on the SAME capability the scorecard/science measures.

Every run produces the SAME consistent trace:
  - OBJECTIVE block: the declared --metric's primary score + fair st_gain + RECORD verdict.
  - BOTTLENECK read: ARCHITECTURE-LIMITED (Earth4D loses to a generic PE) vs EARNING vs EARNING-BUT-LOW.
  - the parsed probe header + native metric lines.
  - RECORD tracking in agents/earth4d/records.json (fill the scorecard by breaking records).

--metric (required) declares the objective capability (keeps the loop on track).
--probe  (required) is the probe flags = the architectural lever.  --ensue auto-logs to Ensue.

Usage:
  python -m deepearth.agents.earth4d.trace --metric family_from_spacetime \
      --probe "--forecast --n_shards 8" --tag forecast --device cuda:0 --ensue
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

REPO = Path(__file__).resolve().parents[2]                 # .../deepearth
RECORDS = Path(__file__).resolve().parent / "records.json"  # the machine record (fill scorecard by breaking these)

# The scorecard capabilities — the objective must be one (keeps the loop on track). Same capabilities as
# the science / evaluate.py, scoped to the encoder probe. The probe MODE/architecture is the loop's choice.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "lfmc_from_env", "mycorrhiza_from_env", "pollinator_from_env",
    "calibration", "flowering_auc", "flowering_fidelity", "flowering_peak_month",
    "infer_clay", "infer_soil", "infer_climate", "infer_hydro",
]

# How to read the capability's PRIMARY absolute score from the probe output (native metric, first match wins).
PRIMARY_RE = {
    "species_from_env": [r"micro-AP\(feat\)\s+([\d.]+)"],
    "community_from_env": [r"micro-AP\(feat\)\s+([\d.]+)"],
    "family_from_env": [r"Earth4D\+ENV\s+([\d.]+)", r"\bEarth4D\s+([\d.]+)"],
    "family_from_spacetime": [r"\bEarth4D\s+([\d.]+)"],
    "species_from_spacetime": [r"\bEarth4D\s+([\d.]+)"],
    "flowering_peak_month": [r"acc\s+([\d.]+)"],
    "flowering_auc": [r"acc\s+([\d.]+)"],
}
# Fair-baseline preference: Earth4D must beat a TRAINED generic PE, not just raw coords.
FAIR_ORDER = ["best-ctrl", "RFF", "mlp", "GAIN", "prop_acc", "best-coord", "raw"]


def _run(module: str, probe_args: str, device: str, log_path: str) -> int:
    cmd = [sys.executable, "-m", module] + shlex.split(probe_args) + ["--device", device]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
    with open(log_path, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode


def _parse(text: str):
    header = next((l.strip() for l in text.splitlines() if l.strip().startswith("=== SPACETIME")), "")
    gains = {}
    for m in re.finditer(r"st_gain(?:\(([^)]*)\))?\s*([+\-]?\d+\.\d+)", text):   # coord/env/forecast modes
        gains[(m.group(1) or "default").strip()] = float(m.group(2))
    for m in re.finditer(r"\bGAIN\s+([+\-]?\d+\.\d+)", text):                    # sdm / cooccur modes report GAIN
        gains["GAIN"] = float(m.group(1))
    for line in text.splitlines():                                              # phenology: within-tol-acc propagator gain
        if "propagator_gain(within-tol acc" in line:
            nums = [float(x) for x in re.findall(r"([+\-]\d+\.\d+)", line)]
            if nums:
                gains["prop_acc"] = max(nums)
    metrics = [l.strip() for l in text.splitlines()
               if re.search(r"\b(acc|micro-AP|MAE|absR2|GAIN|Spearman|top5|prop|ECE|AUROC|Brier|SUMMARY)\b", l) and l.strip()]
    return header, gains, metrics[:24]


def _fair_gain(gains: dict):
    """The honest encoder marginal = gain vs the strongest fair baseline present (best-ctrl > RFF > mlp > ...)."""
    for pref in FAIR_ORDER:
        for k, v in gains.items():
            if pref.lower() in k.lower():
                return v, k
    return (None, None)


def _primary(text: str, cap: str):
    if cap == "calibration":                              # calib_probe: AUROC of confidence->correctness (0.5=useless)
        m = re.search(r"conf_auroc=([\d.]+)", text)
        return float(m.group(1)) if m else None
    if cap.startswith("flowering"):                       # phenology modes report within-tol acc (0..1), take the best
        accs = [float(x) for x in re.findall(r"\bacc\s+([\d.]+)", text)]
        return max(accs) if accs else None
    for p in PRIMARY_RE.get(cap, [r"\bEarth4D\s+([\d.]+)"]):
        m = re.search(p, text)
        if m:
            return float(m.group(1))
    return None


def _bottleneck(fair, primary) -> str:
    if fair is None:
        return "NO-FAIR-BASELINE (probe reported no vs-generic-PE gain — check output)"
    if fair <= 0:
        return "ARCHITECTURE-LIMITED: Earth4D loses to a generic trained PE → swing bigger on the architecture"
    if primary is not None and primary < 0.20:
        return "EARNING-BUT-LOW: encoder beats the PE but the absolute ceiling is elsewhere (capacity/objective/data)"
    return "EARNING: the architecture is carrying real signal → push it further"


def _print_net_scorecard(recs: dict, current: str) -> None:
    """The NET scorecard: every capability's current best encoder-probe record. Printed after every run."""
    print("\n" + "#" * 76)
    print("# NET SCORECARD  —  Earth4D encoder-probe records so far")
    print("#" * 76)
    print(f"{'capability':<26}{'record':>9}{'fair_gain':>11}  best-lever")
    earning = 0
    for cap in CAPABILITIES:
        r = recs.get(cap)
        mark = "  <— this run" if cap == current else ""
        if not r:
            print(f"{cap:<26}{'—':>9}{'—':>11}  —{mark}")
            continue
        fg = r.get("fair_st_gain")
        if fg is not None and fg > 0:
            earning += 1
        sc = r.get("score")
        print(f"{cap:<26}{(f'{sc:.3f}' if sc is not None else '—'):>9}"
              f"{(f'{fg:+.3f}' if fg is not None else '—'):>11}  {r.get('tag', '')}{mark}")
    probed = sum(1 for c in CAPABILITIES if recs.get(c))
    print("-" * 76)
    print(f"probed {probed}/{len(CAPABILITIES)}   |   earning (fair_gain > 0): {earning}")
    print("#" * 76, flush=True)


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
    o = trace["objective"]
    led = trace.get("ledger", {}) or {}
    hist = led.get("records", [])
    best = hist[-1] if hist else {"tag": trace["tag"], "score": o.get("record_value"), "gain": o.get("fair_st_gain")}
    dead = led.get("deadends", {})
    rec_str = " -> ".join(f"{r['tag']}:{r['score']}" for r in hist[-8:]) or "(none)"
    dead_str = "; ".join(f"{t}={d['score']}({(d.get('why') or '')[:34]})" for t, d in list(dead.items())[-12:]) or "(none)"
    # ONE upserted key per capability (LOOP-<program>-<capability> taxonomy): running best + record history +
    # this run's outcome + deduped dead-ends WITH their bottleneck reason. Win or dead-end, every run captured.
    val = (f"LOOP-earth4d {trace['metric']}: BEST {best.get('score')} (gain {best.get('gain')}, {o.get('fair_baseline')}) "
           f"via '{best.get('tag')}'. runs={led.get('runs')}. record-history: {rec_str}. "
           f"THIS RUN '{trace['tag']}': primary={o['primary']} gain={o['fair_st_gain']} RECORD={o['record']} "
           f"bottleneck={trace['bottleneck']}. dead-ends-tried: {dead_str}.")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "create_memory", "arguments": {
        "items": [{"key_name": f"LOOP-earth4d-{trace['metric']}", "value": val,
                   "description": f"Earth4D encoder-probe loop {trace['metric']}: best {best.get('score')} "
                                  f"gain {best.get('gain')} over {led.get('runs')} runs"}]}}}
    req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                                          "Accept": "application/json, text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[trace] Ensue logged LOOP-earth4d-{trace['metric']} ({r.status})", flush=True)
    except Exception as e:
        print(f"[trace] Ensue POST failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Earth4D encoder-probe trace harness — big architectural swings, measured fast.")
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--probe", required=True, help="probe flags = the architectural lever (quote the whole string)")
    ap.add_argument("--probe-module", default="deepearth.autoresearch.programs.spacetime.probe")
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
    header, gains, metrics = _parse(text)
    if rc != 0 and not gains and not header:
        print(text[-1800:])
        sys.exit(f"[trace] probe FAILED (rc={rc}); see {log_path}")

    primary = _primary(text, a.metric)
    fair, fair_base = _fair_gain(gains)
    bottleneck = _bottleneck(fair, primary)

    # RECORD tracking + full run LEDGER (taxonomy: never lose a run's result; publish win OR dead-end w/ reason) --
    recs = json.loads(RECORDS.read_text()) if RECORDS.exists() else {}
    key_val = primary if primary is not None else fair
    cur = recs.get(a.metric, {})
    prev = cur.get("score")
    is_record = key_val is not None and (prev is None or key_val > prev)
    ledger = cur.get("ledger", {"runs": 0, "records": [], "deadends": {}})
    ledger["runs"] = ledger.get("runs", 0) + 1
    if is_record:
        cur = {"score": key_val, "primary": primary, "fair_st_gain": fair,
               "fair_baseline": fair_base, "tag": tag, "probe": a.probe}
        ledger["records"] = (ledger.get("records", []) + [{"tag": tag, "score": key_val, "gain": fair}])[-20:]
    elif key_val is not None:
        # dead-end: a lever below record — kept WITH its reason, deduped by tag (no noise-floor spam)
        ledger.setdefault("deadends", {})[tag] = {"score": key_val, "gain": fair, "why": bottleneck}
        if len(ledger["deadends"]) > 40:
            ledger["deadends"] = dict(list(ledger["deadends"].items())[-40:])
    cur["ledger"] = ledger
    recs[a.metric] = cur
    RECORDS.write_text(json.dumps(recs, indent=2, sort_keys=True))

    objective = {"primary": primary, "fair_st_gain": fair, "fair_baseline": fair_base,
                 "record": bool(is_record), "prev_record": prev, "record_value": key_val}
    trace = {"metric": a.metric, "tag": tag, "probe": a.probe, "probe_module": a.probe_module,
             "objective": objective, "gains": gains, "header": header, "metrics": metrics,
             "bottleneck": bottleneck, "rc": rc, "ledger": ledger}

    # one-screen consistent summary ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"OBJECTIVE  {a.metric}   probe='{a.probe}'")
    print(header or "(no '=== SPACETIME' header parsed — check the log)")
    print("-" * 76)
    print(f"  primary(score) = {primary}   fair_st_gain = {fair} (vs {fair_base})   all_gains = {gains}")
    print(f"  RECORD = {'YES (new best!)' if is_record else 'no'}   prev_record = {prev}")
    print(f"  BOTTLENECK: {bottleneck}")
    print("  metrics:")
    for m in metrics:
        print("    " + m)
    print("=" * 76)
    out = Path(log_path).with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2))
    print(f"[trace] wrote {out}" + ("  |  RECORDS.json updated" if is_record else ""))

    _print_net_scorecard(recs, a.metric)   # show the whole board after every run

    if a.ensue:
        post_ensue(trace)


if __name__ == "__main__":
    main()
