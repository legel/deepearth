"""Per-encoder scoring lens + per-experiment trace for the two science-first autoresearch programs.

Each program (spacetime, biological) maximises ONE scalar -- the encoder's MARGINAL scientific
contribution (its ablation-gain), which is ~0 today and is exactly what science.md's ablation-delta
benchmarks measure. This module reads the benchmark table an eval run already prints (evaluate.format_benchmarks),
restricts it to the encoder's benchmark subset, and reports:

  - the objective scalar:  bio_gain = mean(B56..B62 phylo-graph-gain);  st_gain = mean(*_spacetime_gain)
  - the capability no-regression floor (scoped arithmetic mean of the non-gain capabilities)
  - the encoder-scoped harmonic + arithmetic (via evaluate's OWN _net_value -- scoring is NOT redefined)
  - a per-experiment TRACE json {scalar, delta-vs-champion, per-benchmark deltas, loss components,
    throughput, stability, encoder_diagnostics} + a one-screen summary, optionally pushed to Ensue.

The full-suite net_score is untouched -- this is an added lens, not a replacement (science.md rule 32).

Usage:
  python -m deepearth.autoresearch.programs.score --log run.log [--encoder biological|spacetime|both]
         [--champion autoresearch/champion_scores.json] [--json out.json] [--ensue-tag biological]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- reuse evaluate's OWN scoring maps so the encoder lens never redefines scoring (autoresearch.md rule) ---
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # <parent-of-deepearth> on path
    from deepearth.autoresearch.evaluate import _net_value, is_diagnostic  # type: ignore
except Exception:  # pragma: no cover - standalone log scoring without the heavy (torch) import
    _SCORE_FLOOR = 1e-3
    def is_diagnostic(k: str) -> bool:            # MIRROR of evaluate.is_diagnostic -- keep byte-identical
        return k.endswith("_gain")
    def _net_value(k: str, v: float) -> float:    # MIRROR of evaluate._net_value -- keep byte-identical
        if is_diagnostic(k):
            return 0.5 + 0.5 * float(max(-1.0, min(1.0, v)))
        return max(v, _SCORE_FLOOR)

# ---------------------------------------------------------------------------------------------------------
# Benchmark -> encoder partition (see autoresearch/programs/README.md). Each program scores ONLY its set.
# ---------------------------------------------------------------------------------------------------------

# BIOLOGICAL (phylogenomic species-graph): the seven graph-gain deltas ARE the isolated GNN contribution.
BIO_GAIN = [
    "B56_family_phylo_graph_gain", "B57_flowering_phylo_graph_gain", "B58_lfmc_phylo_graph_gain",
    "B59_pollinator_phylo_graph_gain", "B60_community_phylo_graph_gain", "B61_trait_phylo_graph_gain",
    "B62_mycorrhiza_phylo_graph_gain",
]
BIO_CAP = [  # pure-biological capabilities (species-graph is the primary encoder) -- the no-regression floor
    "B7_family_from_phylo", "B21_community_from_species_recall", "B41_pollinator_from_species_recall",
    "B53_pollinator_calibration_mrr", "B54_pollinator_dist_kl", "B55_pollinator_phylo_transfer_recall",
    "B63_myco_from_species_f1",
]
BIOLOGICAL = BIO_CAP + BIO_GAIN

# SPACETIME (Earth4D / environment): env->biology SDM + phenology + (once wired) *_spacetime_gain deltas.
ST_GAIN = [  # the honest "does Earth4D add over a null spatial prior" deltas -- wired by task S0 (_ablate_spacetime).
    "B1_species_spacetime_gain", "B6_family_spacetime_gain", "B34_lfmc_spacetime_gain",
    "B42_mycorrhiza_spacetime_gain", "B51_pollinator_spacetime_gain", "B23_calibration_spacetime_gain",
]  # NOTE: B24_geo_information_gain (B2-B1, photo-vs-env) is deliberately excluded -- it is a capability
   # difference, not an Earth4D ablation marginal, so it must never be the objective. Until S0, st_gain = None.
ST_CAP = [  # env->biology SDM + phenology (the causal env->life map, all failing today)
    "B1_species_from_env_top10", "B5_species_from_spacetime_top10", "B6_family_from_env",
    "B8_family_from_spacetime", "B23_species_calibration_mrr", "B29_species_dist_30m_skill",
    "B39_species_dist_3km_skill", "B40_species_dist_300m_skill", "B34_lfmc_from_env",
    "B42_mycorrhiza_from_env", "B50_pollinator_from_spacetime_recall", "B51_pollinator_from_env_recall",
    "B26_flowering_auc", "B27_flowering_fidelity", "B28_flowering_peak_month_mrr",
]
ST_SECONDARY = [  # env->modality reconstruction (Earth4D as the positional backbone of the field)
    "B16_infer_clay_cos", "B17_infer_soil_cos", "B18_infer_climate_cos", "B43_infer_hydro_cos",
    "B44_infer_topo_cos", "B46_infer_chm_cos", "B47_infer_naip_ir_cos",
]
SPACETIME = ST_CAP + ST_GAIN + ST_SECONDARY

ENCODERS = {
    "biological": {"caps": BIO_CAP, "gains": BIO_GAIN, "objective": "bio_gain"},
    "spacetime":  {"caps": ST_CAP + ST_SECONDARY, "gains": ST_GAIN, "objective": "st_gain"},
}

# ---------------------------------------------------------------------------------------------------------
# Parsing: recover the {benchmark: raw_score} dict + encoder diagnostics from an eval run log.
# ---------------------------------------------------------------------------------------------------------

_BENCH_RE = re.compile(r"^\s+(B\d+_[A-Za-z0-9_]+)\s+(-?\d+\.\d+)")          # capability + ablation-delta rows
_PROFILE_RE = re.compile(r"\[profile\]\s+([A-Za-z0-9_.:/-]+)\s*[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_STEP_RE = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+|nan)")
_LOSSCOMP_RE = re.compile(r"\[loss\]\s+([A-Za-z0-9_]+)\s*[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def parse_log(path: str) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """Return (raw benchmark scores, encoder diagnostics, run-meta) from a training/eval run log.
    Benchmark rows come from evaluate.format_benchmarks; `[profile] k=v` and `[loss] k=v` are the
    optional encoder-internal diagnostics + loss components (emitted when the run sets profile:true)."""
    raw: Dict[str, float] = {}
    diag: Dict[str, float] = {}
    losses: Dict[str, float] = {}
    last_step, last_loss, saw_nan = 0, None, False
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = _BENCH_RE.match(line)
        if m:
            raw[m.group(1)] = float(m.group(2))
            continue
        m = _PROFILE_RE.search(line)
        if m:
            diag[m.group(1)] = float(m.group(2)); continue
        m = _LOSSCOMP_RE.search(line)
        if m:
            losses[m.group(1)] = float(m.group(2)); continue
        m = _STEP_RE.search(line)
        if m:
            last_step = int(m.group(1))
            if m.group(2) == "nan": saw_nan = True
            else: last_loss = float(m.group(2))
    meta = {"steps": last_step, "final_loss": last_loss, "nan": saw_nan, "loss_components": losses}
    return raw, diag, meta


def load_scores(path: str) -> Dict[str, float]:
    """Load a champion_scores.json-style baseline: {..., 'scores': {benchmark: value}} or a flat dict."""
    d = json.loads(Path(path).read_text())
    return d.get("scores", d) if isinstance(d, dict) else {}

# ---------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------

def _present(raw: Dict[str, float], ids: List[str]) -> List[str]:
    return [k for k in ids if k in raw]


def gain_scalar(raw: Dict[str, float], gains: List[str]) -> Optional[float]:
    """The loop's OBJECTIVE: mean raw ablation-gain delta over the encoder's gain benchmarks (~0 today,
    maximise positive). None if no gain benchmark is present yet (e.g. spacetime-gain not wired)."""
    vals = [raw[k] for k in gains if k in raw]
    return float(sum(vals) / len(vals)) if vals else None


def subset_score(raw: Dict[str, float], ids: List[str]) -> Dict[str, float]:
    """Encoder-scoped harmonic + arithmetic using evaluate's OWN _net_value (gains affine-mapped, caps floored).
    Arithmetic is over CAPABILITIES only (the no-regression floor); harmonic over all present (caps+gains)."""
    present = _present(raw, ids)
    caps = [raw[k] for k in present if not is_diagnostic(k)]
    netv = [_net_value(k, raw[k]) for k in present]
    harm = float(len(netv) / sum(1.0 / max(v, 1e-9) for v in netv)) if netv else 0.0
    arith = float(sum(caps) / len(caps)) if caps else 0.0
    return {"harmonic": harm, "arithmetic": arith, "n": len(present)}


def evaluate_encoder(encoder: str, raw: Dict[str, float]) -> Dict:
    """The full per-encoder score: objective scalar + capability floor + scoped harmonic/arith + per-benchmark."""
    spec = ENCODERS[encoder]
    ids = spec["caps"] + spec["gains"]
    scalar = gain_scalar(raw, spec["gains"])
    floor = subset_score(raw, spec["caps"])
    scoped = subset_score(raw, ids)
    per_bench = {k: raw[k] for k in _present(raw, ids)}
    return {
        "encoder": encoder,
        "objective": spec["objective"],
        "scalar": scalar,                         # the ONE number the loop maximises (None if gains not wired)
        "capability_floor_arith": floor["arithmetic"],   # must-not-regress constraint
        "scoped_harmonic": scoped["harmonic"],
        "scoped_arith": scoped["arithmetic"],
        "n_benchmarks": scoped["n"],
        "per_benchmark": per_bench,
    }

# ---------------------------------------------------------------------------------------------------------
# Per-experiment trace + one-screen summary
# ---------------------------------------------------------------------------------------------------------

def build_trace(encoder: str, raw: Dict[str, float], diag: Dict[str, float], meta: Dict,
                champion: Optional[Dict[str, float]] = None, noise_floor: float = 0.003) -> Dict:
    """The machine-readable feedback signal for ONE experiment. Its most valuable part is encoder_diagnostics
    (the bottleneck profile, e.g. ||refined-seed|| for biological) that tells the NEXT hypothesis where it's stuck."""
    res = evaluate_encoder(encoder, raw)
    per_delta = {}
    scalar_delta = None
    if champion:
        for k, v in res["per_benchmark"].items():
            if k in champion:
                per_delta[k] = round(v - champion[k], 4)
        if res["scalar"] is not None:
            base = gain_scalar(champion, ENCODERS[encoder]["gains"])
            if base is not None:
                scalar_delta = round(res["scalar"] - base, 4)
    return {
        "encoder": encoder,
        "objective": res["objective"],
        "scalar": res["scalar"],
        "scalar_delta_vs_champion": scalar_delta,
        "noise_flag": (scalar_delta is not None and abs(scalar_delta) < noise_floor),
        "capability_floor_arith": round(res["capability_floor_arith"], 4),
        "scoped_harmonic": round(res["scoped_harmonic"], 4),
        "scoped_arith": round(res["scoped_arith"], 4),
        "per_benchmark_delta": per_delta,
        "per_benchmark": {k: round(v, 4) for k, v in res["per_benchmark"].items()},
        "loss_components": meta.get("loss_components", {}),
        "throughput_steps": meta.get("steps", 0),
        "final_loss": meta.get("final_loss"),
        "stability": {"nan": meta.get("nan", False)},
        "encoder_diagnostics": diag,     # <-- the bottleneck profile (||refined-seed||, OU rate, loss ratio, ...)
    }


def summary(trace: Dict) -> str:
    """One-screen human read: objective + delta + the binding-weakest benchmarks + the bottleneck diagnostic."""
    L = []
    s = trace["scalar"]
    sd = trace["scalar_delta_vs_champion"]
    flag = " (within noise)" if trace["noise_flag"] else ""
    L.append(f"=== {trace['encoder'].upper()} experiment | objective {trace['objective']} ===")
    L.append(f"  {trace['objective']} = {s if s is not None else 'n/a (gains not wired)'}"
             + (f"   Δ vs champion {sd:+.4f}{flag}" if sd is not None else ""))
    L.append(f"  capability floor (arith) {trace['capability_floor_arith']:.4f}   scoped harmonic {trace['scoped_harmonic']:.4f}")
    L.append(f"  steps@budget {trace['throughput_steps']}   final_loss {trace['final_loss']}   nan={trace['stability']['nan']}")
    if trace["per_benchmark_delta"]:
        moved = sorted(trace["per_benchmark_delta"].items(), key=lambda kv: kv[1])
        downs = [f"{k} {d:+.3f}" for k, d in moved if d < -0.002][:4]
        ups = [f"{k} {d:+.3f}" for k, d in moved[::-1] if d > 0.002][:4]
        if ups:   L.append("  ↑ " + " · ".join(ups))
        if downs: L.append("  ↓ REGRESSED " + " · ".join(downs))
    if trace["encoder_diagnostics"]:
        L.append("  bottleneck: " + " · ".join(f"{k}={v:g}" for k, v in trace["encoder_diagnostics"].items()))
    return "\n".join(L)

# ---------------------------------------------------------------------------------------------------------
# Ensue coordination (optional): publish the trace so the loop + swarm build on it, tagged by encoder.
# ---------------------------------------------------------------------------------------------------------

def push_ensue(trace: Dict, tag: str, key: str, label: str) -> Optional[str]:
    import urllib.request
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "create_memory",
        "arguments": {"items": [{"key_name": label,
            "description": f"[{tag}] {trace['objective']}={trace['scalar']} Δ{trace.get('scalar_delta_vs_champion')}"
                           f" | floor {trace['capability_floor_arith']} | steps {trace['throughput_steps']}",
            "value": json.dumps(trace), "embed": True, "embed_source": "description"}]}}}
    req = urllib.request.Request("https://api.ensue-network.ai/", data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    r = urllib.request.urlopen(req, timeout=40).read().decode()
    for line in r.splitlines():
        if line.startswith("data:"):
            return line[5:].strip()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-encoder scoring lens + experiment trace")
    ap.add_argument("--log", help="training/eval run log (parsed for the benchmark table + [profile]/[loss] lines)")
    ap.add_argument("--scores", help="a champion_scores.json-style dict to score directly (alternative to --log)")
    ap.add_argument("--encoder", default="both", choices=["biological", "spacetime", "both"])
    ap.add_argument("--champion", help="champion_scores.json baseline for deltas")
    ap.add_argument("--json", help="write the trace json here")
    ap.add_argument("--ensue-tag", help="publish trace to Ensue with this encoder tag (needs ENSUE_KEY env)")
    a = ap.parse_args(argv)

    if a.log:
        raw, diag, meta = parse_log(a.log)
    elif a.scores:
        raw, diag, meta = load_scores(a.scores), {}, {}
    else:
        ap.error("one of --log or --scores is required")
    champion = load_scores(a.champion) if a.champion else None

    encoders = ["biological", "spacetime"] if a.encoder == "both" else [a.encoder]
    out_traces = {}
    for enc in encoders:
        tr = build_trace(enc, raw, diag, meta, champion)
        out_traces[enc] = tr
        print(summary(tr))
        print()
        if a.ensue_tag or (a.encoder != "both" and a.ensue_tag is None):
            import os
            key = os.environ.get("ENSUE_KEY")
            if key and a.ensue_tag:
                label = f"EXP-{enc}-{meta.get('steps',0)}steps"
                try:
                    print("  ensue:", push_ensue(tr, a.ensue_tag, key, label))
                except Exception as e:
                    print("  ensue push failed:", e)
    if a.json:
        Path(a.json).write_text(json.dumps(out_traces if len(out_traces) > 1 else next(iter(out_traces.values())), indent=2))
    return out_traces


if __name__ == "__main__":
    main()
