"""agents/earth4d/trace.py — the FIXED experimentation harness for the Earth4D scorecard.

Every Earth4D experiment runs through this ONE unchanged entrypoint. It runs the candidate
config through the native run (`programs/run_experiment --st-gain`, so `evaluate.py`'s own
metrics AND the Earth4D-ablation `*_spacetime_gain` deltas land in the log), then parses the
log with `programs/score.parse_log` and emits a CONSISTENT trace, identical shape every run:

  - OBJECTIVE block — the declared `--metric`'s before/after/Δ-vs-champion/spacetime_gain + verdict.
  - the 16 scorecard rows: {score, Δ vs champion, spacetime_gain}.
  - BOTTLENECK read — weakest rows classified: earth4d-limited (gain≈0 -> encoder/data ceiling)
    vs earth4d-contributing vs no-ablation-probe (gain not wired for that row).
  - HIGH-SIGNAL scorecard metric — mean of 16, worst row, FAIL/mid/PASS counts, Σ spacetime_gain
    (= "how much Earth4D is earning"), full-suite net_score.

Rules baked in: `--metric` is REQUIRED and must be one of the 16 (no aimless runs). Metrics come
from `evaluate.py` via the log — NEVER reimplemented. Measurement only; this never promotes.

Usage:
  python -m deepearth.agents.earth4d.trace --config <exp.yaml> --metric <Bxx_key> \
         [--tag id] [--device cuda:0] [--budget 4000] [--fresh-data]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <parent-of-deepearth> on path (mirror score.py)
from deepearth.autoresearch.programs.score import parse_log, load_scores  # native log parser + baseline loader
from deepearth.autoresearch.programs.hooks import ST_GAIN_MAP             # capability -> Earth4D-ablation gain key

REPO = Path(__file__).resolve().parents[2]                 # .../deepearth
CHAMPION = REPO / "autoresearch" / "champion_scores.json"

# The 16 Earth4D scorecard metrics — the ONLY valid objectives (must match scorecard.md).
EARTH4D_KEYS = [
    "B1_species_from_env_top10", "B5_species_from_spacetime_top10", "B6_family_from_env",
    "B8_family_from_spacetime", "B20_community_from_env_recall", "B34_lfmc_from_env",
    "B42_mycorrhiza_from_env", "B51_pollinator_from_env_recall", "B23_species_calibration_mrr",
    "B26_flowering_auc", "B27_flowering_fidelity", "B28_flowering_peak_month_mrr",
    "B16_infer_clay_cos", "B17_infer_soil_cos", "B18_infer_climate_cos", "B43_infer_hydro_cos",
]


def _status(v: float) -> str:
    return "PASS" if v >= 0.70 else ("mid" if v >= 0.45 else "FAIL")


def _run(config: str, tag: str, device: str, budget, steps, log_path: str) -> int:
    """Run the candidate natively with the Earth4D spacetime-gain instrument, tee stdout to log_path."""
    cmd = [sys.executable, "-m", "deepearth.autoresearch.programs.run_experiment",
           config, "--st-gain", "--device", device, "--tag", tag]
    if budget:
        cmd += ["--time_budget", str(budget)]
    if steps:
        cmd += ["--steps", str(steps)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[trace] $ {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as lf:
        return subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO.parent)).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Earth4D fixed trace harness — every experiment runs through here.")
    ap.add_argument("--config", required=True, help="full experiment yaml (champion.yaml + ONE lever change)")
    ap.add_argument("--metric", required=True, help="the OBJECTIVE: one of the 16 Earth4D scorecard keys")
    ap.add_argument("--tag", default=None, help="run label (default: derived from config filename)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget", type=float, default=None, help="time budget seconds (fixed across an A/B)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fresh-data", action="store_true",
                    help="rm data/deepcal/prepared_*.pt first — REQUIRED for any data-lever change (lossy cache)")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    # --- the objective must be declared and valid (keeps every run on track) ---
    if a.metric not in EARTH4D_KEYS:
        sys.exit("[trace] --metric %r is not an Earth4D scorecard metric. choose one of:\n  %s"
                 % (a.metric, "\n  ".join(EARTH4D_KEYS)))

    config = str(Path(a.config).resolve())
    tag = a.tag or ("e4d_" + Path(config).stem)
    log_path = a.log or str(Path(__file__).resolve().parent / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    if a.fresh_data:
        for f in (REPO / "data" / "deepcal").glob("prepared_*.pt"):
            f.unlink()
            print(f"[trace] cleared {f} (data-lever run)", flush=True)

    print(f"[trace] OBJECTIVE={a.metric}  config={config}  tag={tag}", flush=True)
    rc = _run(config, tag, a.device, a.budget, a.steps, log_path)
    if rc != 0:
        sys.exit(f"[trace] run FAILED (rc={rc}); see {log_path}")

    raw, diag, meta = parse_log(log_path)
    if not any(k in raw for k in EARTH4D_KEYS):
        sys.exit(f"[trace] no benchmark rows parsed from {log_path} — run likely crashed.")
    champ = load_scores(str(CHAMPION)) if CHAMPION.exists() else {}
    mnet = re.search(r"net_score:\s+([\d.]+)", Path(log_path).read_text(errors="ignore"))
    net = float(mnet.group(1)) if mnet else None

    # --- the consistent trace ---
    rows = []
    for k in EARTH4D_KEYS:
        after, before = raw.get(k), champ.get(k)
        gain = raw.get(ST_GAIN_MAP.get(k, ""))   # Earth4D ablation marginal (wired for 6 of the 16)
        rows.append({"metric": k, "score": after, "before": before,
                     "delta": (None if after is None or before is None else round(after - before, 4)),
                     "spacetime_gain": gain})

    def _classify(r):
        g = r["spacetime_gain"]
        if g is None:
            return "no-ablation-probe"
        return "earth4d-limited" if g < 0.01 else "earth4d-contributing"

    scored = sorted((r for r in rows if r["score"] is not None), key=lambda r: r["score"])
    bottleneck = [{"metric": r["metric"], "score": r["score"], "spacetime_gain": r["spacetime_gain"],
                   "class": _classify(r)} for r in scored[:5]]
    present = [r["score"] for r in scored]
    mean16 = round(sum(present) / len(present), 4) if present else None
    counts = {"FAIL": sum(v < 0.45 for v in present),
              "mid": sum(0.45 <= v < 0.70 for v in present),
              "PASS": sum(v >= 0.70 for v in present)}
    sum_gain = round(sum(r["spacetime_gain"] for r in rows if r["spacetime_gain"] is not None), 4)

    obj = next(r for r in rows if r["metric"] == a.metric)
    verdict = ("no-baseline" if obj["delta"] is None
               else ("UP" if obj["delta"] > 0 else ("FLAT" if obj["delta"] == 0 else "DOWN")))

    trace = {"objective": a.metric, "tag": tag, "config": config,
             "objective_result": {**obj, "verdict": verdict},
             "rows": rows, "bottleneck": bottleneck,
             "high_signal": {"mean16": mean16, "worst": bottleneck[0] if bottleneck else None,
                             "counts": counts, "sum_spacetime_gain": sum_gain, "full_suite_net": net},
             "diagnostics": diag, "run": meta}

    # --- one-screen summary ---
    print("\n" + "=" * 74)
    print(f"OBJECTIVE  {a.metric}  ->  {obj['score']}  (was {obj['before']}, Δ {obj['delta']}, "
          f"gain {obj['spacetime_gain']})   VERDICT: {verdict}")
    print("-" * 74)
    print(f"{'metric':<34}{'score':>8}{'Δchamp':>9}{'st_gain':>9}  status")
    for r in rows:
        s = "" if r["score"] is None else f"{r['score']:.3f}"
        d = "" if r["delta"] is None else f"{r['delta']:+.3f}"
        g = "" if r["spacetime_gain"] is None else f"{r['spacetime_gain']:.3f}"
        st = "" if r["score"] is None else _status(r["score"])
        print(f"{r['metric']:<34}{s:>8}{d:>9}{g:>9}  {st}")
    print("-" * 74)
    print(f"mean16={mean16}  FAIL={counts['FAIL']} mid={counts['mid']} PASS={counts['PASS']}  "
          f"Σst_gain={sum_gain}  net={net}")
    print("BOTTLENECK (weakest 5):")
    for b in bottleneck:
        print(f"  {b['metric']:<34} {b['score']:.3f}  [{b['class']}]")
    print("=" * 74)

    out = Path(log_path).with_suffix(".trace.json")
    out.write_text(json.dumps(trace, indent=2))
    print(f"[trace] wrote {out}")


if __name__ == "__main__":
    main()
