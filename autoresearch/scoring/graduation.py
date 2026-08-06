"""graduation.py — the bridge a probe record crosses to become a champion result.

Three loops have been measuring for months and NOTHING has crossed. The reason is structural, not
effort: no code in `main/` reads either probe's `records.json`. The only coupling between the loops is
two import lines in `fusion.py` (`earth4d.Earth4D`, `phylogenomic.SpeciesGraph`), so the risky thing --
encoder SOURCE -- propagates instantly and silently, while the safe thing -- measured evidence --
propagates not at all. This module is the missing half.

It does NOT pipe probe scores into the benchmark suite. It cannot: a probe score and a benchmark score
are different instruments on different models (scorecard.md: "Never compare Layer 1 to Layer 2 ...
Neither bounds the other"). What crosses is a PREDICTION THAT GETS TESTED:

    probe says capability X improved  ->  champion re-measures benchmark B(X)  ->  did it move?

Every crossing appends one row to `main/records/graduation.jsonl`. After a handful of rows that ledger
answers the question the whole three-loop design is premised on and which nothing currently measures:
**when a probe finds a gain, does the benchmark move?** A capability whose probe gains never transfer is
a capability the probe is mis-measuring -- and you learn that from ten rows instead of five hundred runs.

Lifecycle, because this tool cannot train (a fusion run needs a GPU and the rule-20 budget):

    graduation.py --status                     what is eligible, what is blocked, and why   [no GPU]
    graduation.py --open spacetime:family_from_spacetime
                                             snapshot bench_before + the probe record -> PENDING row
    < run paired CONTROL and CANDIDATE fusion jobs at time_budget_s=600 >
    graduation.py --close <id> \
      --control promoted-base \
      --log candidate.seed0.log --log candidate.seed1.log

Register ``promoted-base`` once from its two receipt-bearing control logs. A later crossing may reuse it
only when the receipt proves the candidate is still paired to that exact base and judge.

The close step compares the two live logs. It never compares a candidate against a stale champion
JSON written under another seed, suite, or protocol.

Eligibility is deliberately strict. A probe record that cannot be REPLAYED cannot be graduated, because
step 2 replays it: `code.dirty` means the CONFIG/earth4d diff that produced the number is unrecoverable
(all three spacetime records carry dirty=true today), and `provisional` means fewer than 2 matched seeds,
which program.md's evidence standard already refuses to call a claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Paths by NAME, not by counting parents -- the same discipline harness.py adopted after a parents[N]
# off-by-one pointed the board outside its own loop and minted a record against an empty file.
_HERE = Path(__file__).resolve()
AUTORESEARCH = next(p for p in _HERE.parents if p.name == "autoresearch")
sys.path.insert(0, str(AUTORESEARCH.parent.parent))   # dir holding the deepearth package

from deepearth.autoresearch.scoring.definitions import capability_to_benchmark  # noqa: E402
from deepearth.autoresearch.main.harness.evaluate import BENCHMARK_PROTOCOL  # noqa: E402
from deepearth.autoresearch.probes.biological.harness.board import PROTOCOL as BIO_PROTOCOL  # noqa: E402
from deepearth.autoresearch.probes.spacetime.harness import PROTOCOL as ST_PROTOCOL  # noqa: E402
CHAMPION = AUTORESEARCH / "main" / "records" / "champion_scores.json"
LEDGER = AUTORESEARCH / "main" / "records" / "graduation.jsonl"
CONTROLS = AUTORESEARCH / "main" / "records" / "fusion_controls.json"
CURRENT_PROBE_PROTOCOL = {"biological": BIO_PROTOCOL, "spacetime": ST_PROTOCOL}


# ==============================================================================================
# THE JOIN
# ==============================================================================================
#
# A probe capability and a benchmark are two measurements of the same requirement. That mapping has
# lived only in scorecard.md prose; main's benchmark names already CONTAIN the capability strings
# ("B8_family_from_spacetime"), so the join was always there and was never once used in code.
#
# This dict is the single owner. `--status` verifies every benchmark named here exists in
# evaluate.BENCHMARKS, so the map cannot silently rot the way probe_registry's mode table did.
# DERIVED, not declared. Every probe capability that names a champion metric does so in exactly one
# place -- `scoring.METRICS` -- so the routing an agent follows, the scoring it is judged by, and the
# crossing it eventually makes are literally the same rows. A second hand-written copy of this map is
# how `program.md`'s LEVER_SITES ended up pointing at a `lib/gnn.py` that had been deleted.
_JOIN = capability_to_benchmark()          # capability -> benchmark, from the registry

# Which loop owns which capability. The only thing not derivable from the metric registry, because a
# capability name does not say which probe measures it.
LOOP_CAPABILITIES: Dict[str, Tuple[str, ...]] = {
    "spacetime": ("species_from_env", "species_from_spacetime", "family_from_env",
                  "family_from_spacetime", "community_from_env", "flowering_peak_month"),
    "biological": ("family_from_phylo", "community_from_species", "pollinator_from_species",
                   "pollinator_transfer", "myco_from_species"),
}

CAPABILITY_BENCH: Dict[str, Dict[str, str]] = {
    loop: {c: _JOIN[c] for c in caps if c in _JOIN}
    for loop, caps in LOOP_CAPABILITIES.items()
}


def records_path(loop: str) -> Path:
    return AUTORESEARCH / "probes" / loop / "records" / "records.json"


def read_records(loop: str) -> Dict[str, Any]:
    p = records_path(loop)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError as exc:
        raise SystemExit(f"[graduate] {p} is not valid JSON: {exc}")


def read_champion() -> Dict[str, Any]:
    if not CHAMPION.exists():
        raise SystemExit(f"[graduate] no champion record at {CHAMPION} — run champion_report --save first")
    return json.loads(CHAMPION.read_text())


# ==============================================================================================
# ELIGIBILITY
# ==============================================================================================

def _probe_before(rec: Dict[str, Any]) -> Optional[float]:
    """Previous comparable probe record, if the board retained one."""
    rows = (rec.get("ledger") or {}).get("records") or []
    current = rec.get("score", rec.get("primary"))
    protocol = rec.get("protocol")
    comparable = [r.get("score") for r in rows
                  if r.get("protocol") == protocol and r.get("score") is not None]
    if comparable and current is not None and comparable[-1] == current:
        comparable = comparable[:-1]
    return float(comparable[-1]) if comparable else None


def blockers(loop: str, capability: str, rec: Dict[str, Any], bench_ok: bool = True,
             champion_protocol: Optional[str] = None) -> List[str]:
    """Why this record may NOT cross. Empty list = eligible.

    Each rule exists because its absence has already corrupted this board once."""
    out: List[str] = []
    if capability not in CAPABILITY_BENCH.get(loop, {}):
        out.append(f"no benchmark mapped for {loop}:{capability} — add it to CAPABILITY_BENCH or "
                   f"record why the capability is not champion-reachable")
    expected_probe = CURRENT_PROBE_PROTOCOL.get(loop)
    if rec.get("protocol") != expected_probe:
        out.append(f"probe protocol {rec.get('protocol')!r} is superseded; current {loop} protocol is "
                   f"{expected_probe!r} — re-baseline this capability before graduation")
    code = rec.get("code") or {}
    if code.get("dirty"):
        out.append("code.dirty — the tree that produced this number had uncommitted changes, so the "
                   "CONFIG/encoder diff cannot be replayed and the champion run would not be the "
                   "same experiment")
    if not code.get("commit"):
        out.append("no code.commit — nothing to check out")
    if rec.get("provisional", True):
        n = rec.get("n_seeds")
        out.append(f"provisional ({n if n is not None else '?'} seeds) — program.md's evidence standard "
                   f"needs >=2 matched seeds before a probe record is a claim")
    if rec.get("score") is None and rec.get("primary") is None:
        out.append("record carries no score")
    before = _probe_before(rec)
    after = rec.get("score", rec.get("primary"))
    if before is None:
        out.append("no prior same-protocol probe record — this is a baseline, not an improvement")
    elif after is None or after <= before:
        out.append(f"probe did not improve its prior record ({before} -> {after})")
    return out


def survey(loop: str) -> List[Dict[str, Any]]:
    """Every capability on this loop's board, with its verdict."""
    champion = read_champion()
    champ = champion.get("scores", {})
    rows = []
    for capability, rec in sorted(read_records(loop).items()):
        bench = CAPABILITY_BENCH.get(loop, {}).get(capability)
        rows.append({
            "loop": loop,
            "capability": capability,
            "bench": bench,
            "probe_score": rec.get("score", rec.get("primary")),
            "probe_gain": rec.get("fair_st_gain"),
            "tag": rec.get("tag"),
            "n_seeds": rec.get("n_seeds"),
            "commit": (rec.get("code") or {}).get("commit"),
            "bench_before": champ.get(bench) if bench else None,
            "probe_before": _probe_before(rec),
            "blockers": blockers(loop, capability, rec),
        })
    return rows


# ==============================================================================================
# LEDGER
# ==============================================================================================

def read_ledger() -> List[Dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def append_ledger(row: Dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def rewrite_ledger(rows: List[Dict[str, Any]]) -> None:
    LEDGER.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def crossing_id(loop: str, capability: str, tag: Optional[str]) -> str:
    """Stable and readable — the same probe record always names the same crossing."""
    return f"{loop}:{capability}:{tag or 'untagged'}"


# ==============================================================================================
# OPEN / CLOSE
# ==============================================================================================

def do_open(target: str, force: bool) -> None:
    if ":" not in target:
        raise SystemExit("[graduate] --open takes <loop>:<capability>, e.g. spacetime:family_from_spacetime")
    loop, capability = target.split(":", 1)
    if loop not in CAPABILITY_BENCH:
        raise SystemExit(f"[graduate] unknown loop {loop!r}; one of: {', '.join(CAPABILITY_BENCH)}")
    rec = read_records(loop).get(capability)
    if rec is None:
        raise SystemExit(f"[graduate] {loop} has no record for {capability!r}")

    bench = CAPABILITY_BENCH[loop].get(capability)
    stop = blockers(loop, capability, rec)
    if stop and not force:
        print(f"[graduate] {target} is NOT eligible:")
        for b in stop:
            print(f"    - {b}")
        raise SystemExit("[graduate] refusing to open. Fix the blockers, or pass --force to open a "
                         "crossing that is explicitly marked unreproducible.")

    cid = crossing_id(loop, capability, rec.get("tag"))
    if any(r["id"] == cid and r["state"] == "pending" for r in read_ledger()):
        raise SystemExit(f"[graduate] {cid} is already pending; close it with --close before reopening")

    probe_after = rec.get("score", rec.get("primary"))
    probe_before = _probe_before(rec)
    row = {
        "id": cid,
        "state": "pending",
        "loop": loop,
        "capability": capability,
        "bench": bench,
        "probe": {"tag": rec.get("tag"), "score": probe_after,
                  "score_before": probe_before,
                  "score_delta": (probe_after - probe_before
                                  if probe_after is not None and probe_before is not None else None),
                  "fair_gain": rec.get("fair_st_gain"), "fair_baseline": rec.get("fair_baseline"),
                  "mode": rec.get("mode"), "n_shards": rec.get("n_shards"),
                  "protocol": rec.get("protocol"), "n_seeds": rec.get("n_seeds"),
                  "seed_std": rec.get("seed_std"), "provisional": rec.get("provisional"),
                  "commit": (rec.get("code") or {}).get("commit"),
                  "branch": (rec.get("code") or {}).get("branch")},
        "forced": bool(stop),
        "blockers_at_open": stop,
    }
    append_ledger(row)
    print(f"[graduate] OPENED {cid}")
    print(f"    predicts   {bench} moves up against its paired fusion control")
    print(f"    probe      {row['probe']['score']} (fair-gain {row['probe']['fair_gain']} vs "
          f"{row['probe']['fair_baseline']}, {row['probe']['n_seeds']} seeds)")
    print(f"    replay     git checkout {row['probe']['commit']} -- the encoder + CONFIG diff")
    print(f"\n    Now run matched control and candidate fusion jobs at the rule-20 budget, then:\n"
          f"      python -m deepearth.autoresearch.scoring.graduation "
          f"--close {cid} --baseline-log control.seed0.log --baseline-log control.seed1.log "
          f"--log candidate.seed0.log --log candidate.seed1.log")


def compare_fusion(control: Dict[str, Any], candidate: Dict[str, Any], bench: str,
                   probe_delta: Optional[float] = None) -> Dict[str, Any]:
    """Paired encoder-to-fusion propagation scorecard. Scoring definitions stay untouched."""
    before, after = control["scores"], candidate["scores"]
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    if missing or added:
        raise ValueError("benchmark suites differ\n"
                         f"    added: {', '.join(added) or '(none)'}\n"
                         f"    missing: {', '.join(missing) or '(none)'}")
    bench_before, bench_after = before.get(bench), after.get(bench)
    bench_delta = (bench_after - bench_before
                   if bench_before is not None and bench_after is not None else None)
    deltas = {name: after[name] - value for name, value in before.items()}
    regressions = sorted(name for name, delta in deltas.items() if delta < -0.005)
    harmonic_delta = candidate["harmonic"] - control["harmonic"]
    arithmetic_delta = candidate["arithmetic"] - control["arithmetic"]
    transferred = bool(bench_delta is not None and bench_delta > 0 and not regressions)
    return {
        "bench_before": bench_before,
        "bench_after": bench_after,
        "bench_delta": bench_delta,
        "propagation_ratio": (bench_delta / probe_delta
                              if bench_delta is not None and probe_delta is not None and probe_delta > 0 else None),
        "harmonic_delta": harmonic_delta,
        "arithmetic_delta": arithmetic_delta,
        "worst_delta": min(deltas.items(), key=lambda item: item[1]) if deltas else None,
        "regressions": regressions,
        "transferred": transferred,
        "fusion_breakthrough": bool(transferred and harmonic_delta > 0 and arithmetic_delta > 0),
    }


def _mean_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    names = set(runs[0]["scores"])
    if any(set(run["scores"]) != names for run in runs[1:]):
        raise ValueError("benchmark suites differ across seeds")
    n = len(runs)
    return {
        "scores": {name: sum(run["scores"][name] for run in runs) / n for name in names},
        "harmonic": sum(run["harmonic"] for run in runs) / n,
        "arithmetic": sum(run["arithmetic"] for run in runs) / n,
        "steps": [run.get("steps") for run in runs],
        "peak_vram_mb": max((run.get("peak_vram_mb") or 0.0) for run in runs),
    }


def _receipt_mismatch(control: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    """Return reasons a frozen control cannot be paired with this candidate."""
    left, right = control.get("receipt"), candidate.get("receipt")
    if not left or not right:
        return ["missing fusion-run-v1 receipt"]
    out = []
    if left.get("schema") != "fusion-run-v1" or right.get("schema") != "fusion-run-v1":
        out.append("receipt schema")
    if left.get("source", {}).get("dirty") or right.get("source", {}).get("dirty"):
        out.append("dirty source tree")
    control_tree = left.get("source", {}).get("tree")
    candidate_source = right.get("source", {})
    if control_tree not in (candidate_source.get("tree"), candidate_source.get("parent_tree")):
        out.append("candidate is not a one-commit child or config variant of the control")
    for section, keys in {
        "judge": ("protocol", "evaluate_sha256", "definitions_sha256"),
        "data": ("identity", "prepared_sha256"),
        "training": ("steps", "time_budget_s", "batch", "precision"),
        "runner": ("hooks_sha256",),
        "runtime": ("torch", "cuda", "gpu"),
    }.items():
        for key in keys:
            if left.get(section, {}).get(key) != right.get(section, {}).get(key):
                out.append(f"{section}.{key}")
    if left.get("training", {}).get("seed") != right.get("training", {}).get("seed"):
        out.append("training.seed")
    for section, key in (("source", "tree"), ("judge", "evaluate_sha256"),
                         ("judge", "definitions_sha256"), ("data", "prepared_sha256")):
        if left.get(section, {}).get(key) is None or right.get(section, {}).get(key) is None:
            out.append(f"missing {section}.{key}")
    return out


def _validate_pairs(controls: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
    for control, candidate in zip(controls, candidates):
        mismatch = _receipt_mismatch(control, candidate)
        if mismatch:
            raise SystemExit("[graduate] frozen control is not provenance-matched: " + ", ".join(mismatch))


def _read_controls() -> Dict[str, Any]:
    if not CONTROLS.exists():
        return {}
    return json.loads(CONTROLS.read_text())


def register_control(label: str, paths: List[str]) -> None:
    from deepearth.autoresearch.main.harness.champion_report import parse_run

    if len(paths) != 2:
        raise SystemExit("[graduate] a frozen control requires exactly two logs")
    runs = [parse_run(path) for path in paths]
    seeds = [run.get("training_seed") for run in runs]
    if None in seeds or len(set(seeds)) != 2:
        raise SystemExit("[graduate] control logs need two distinct declared seeds")
    for run, path in zip(runs, paths):
        if not run.get("scores") or not run.get("receipt"):
            raise SystemExit(f"[graduate] {path} lacks scores or a fusion-run-v1 receipt")
        if run.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
            raise SystemExit(f"[graduate] {path} uses a superseded benchmark protocol")
    # The two arms differ only by seed; compare each receipt with seed zeroed.
    first = json.loads(json.dumps(runs[0]["receipt"]))
    second = json.loads(json.dumps(runs[1]["receipt"]))
    first["training"]["seed"] = second["training"]["seed"] = None
    first["config"].pop("effective_sha256", None)
    second["config"].pop("effective_sha256", None)
    if first != second:
        raise SystemExit("[graduate] control receipts differ beyond seed")
    board = _read_controls()
    board[label] = {
        "logs": [str(Path(path).resolve()) for path in paths],
        "runs": runs,
    }
    CONTROLS.write_text(json.dumps(board, indent=2, sort_keys=True))
    print(f"[graduate] FROZE control {label!r}: seeds {seeds}, tree {runs[0]['receipt']['source']['tree'][:12]}")


def do_close(cid: str, log_paths: List[str], baseline_log_paths: List[str], control_label: str = "") -> None:
    from deepearth.autoresearch.main.harness.champion_report import parse_run

    rows = read_ledger()
    match = [r for r in rows if r["id"] == cid and r["state"] == "pending"]
    if not match:
        raise SystemExit(f"[graduate] no pending crossing {cid!r}. Open ones:\n  "
                         + "\n  ".join(r["id"] for r in rows if r["state"] == "pending"))
    row = match[-1]

    if len(log_paths) != 2:
        raise SystemExit("[graduate] fusion confirmation requires exactly two candidate logs")
    if control_label:
        frozen = _read_controls().get(control_label)
        if not frozen:
            raise SystemExit(f"[graduate] no frozen control {control_label!r}")
        controls = frozen["runs"]
        baseline_log_paths = frozen["logs"]
    elif len(baseline_log_paths) == 2:
        controls = [parse_run(path) for path in baseline_log_paths]
    else:
        raise SystemExit("[graduate] provide two --baseline-log values or one --control label")
    candidates = [parse_run(path) for path in log_paths]
    for label, parsed_runs, paths in (("control", controls, baseline_log_paths),
                                      ("candidate", candidates, log_paths)):
        for parsed, path in zip(parsed_runs, paths):
            if not parsed["scores"]:
                raise SystemExit(f"[graduate] no Bxx scores found in {path}")
            if parsed.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
                raise SystemExit(f"[graduate] {label} protocol {parsed.get('benchmark_protocol')!r} does not "
                                 f"match the live benchmark protocol {BENCHMARK_PROTOCOL!r}")
    control_seeds = [run.get("training_seed") for run in controls]
    candidate_seeds = [run.get("training_seed") for run in candidates]
    if None in control_seeds + candidate_seeds or control_seeds != candidate_seeds or len(set(control_seeds)) != 2:
        raise SystemExit("[graduate] logs must declare the same two distinct training_seed values in paired order")
    if control_label:
        _validate_pairs(controls, candidates)
    try:
        control, after = _mean_runs(controls), _mean_runs(candidates)
    except ValueError as exc:
        raise SystemExit(f"[graduate] refusing to close: {exc}") from exc

    bench = row["bench"]
    try:
        result = compare_fusion(control, after, bench, row["probe"].get("score_delta"))
    except ValueError as exc:
        raise SystemExit(f"[graduate] refusing to close: {exc}") from exc

    row.update({
        "state": "closed",
        "fusion_control": {"harmonic": control["harmonic"], "arithmetic": control["arithmetic"],
                           "bench_value": result["bench_before"],
                           "n_benchmarks": len(control["scores"]),
                           "steps": control["steps"], "peak_vram_mb": control["peak_vram_mb"],
                           "seeds": control_seeds,
                           "logs": [str(Path(path).resolve()) for path in baseline_log_paths]},
        "fusion_candidate": {"harmonic": after["harmonic"], "arithmetic": after["arithmetic"],
                             "bench_value": result["bench_after"],
                             "n_benchmarks": len(after["scores"]),
                             "steps": after["steps"], "peak_vram_mb": after["peak_vram_mb"],
                             "seeds": candidate_seeds,
                             "logs": [str(Path(path).resolve()) for path in log_paths]},
        **result,
    })
    rewrite_ledger(rows)

    print(f"[graduate] CLOSED {cid}")
    print(f"    probe: {row['probe'].get('score_before')} -> {row['probe'].get('score')} "
          f"({_fmt(row['probe'].get('score_delta'), signed=True)})")
    print(f"    {bench}: {_fmt(result['bench_before'])} -> {_fmt(result['bench_after'])} "
          f"({_fmt(result['bench_delta'], signed=True)})")
    print(f"    propagation ratio: {_fmt(result['propagation_ratio'])} (diagnostic; probe/fusion scales differ)")
    print(f"    net harmonic {_fmt(control['harmonic'])} -> {_fmt(after['harmonic'])} "
          f"({_fmt(result['harmonic_delta'], signed=True)})")
    print(f"    net arithmetic {_fmt(control['arithmetic'])} -> {_fmt(after['arithmetic'])} "
          f"({_fmt(result['arithmetic_delta'], signed=True)})")
    print(f"    steps: control {control['steps']}  candidate {after['steps']}")
    print(f"    peak VRAM MB: control {control['peak_vram_mb']:.1f}  candidate {after['peak_vram_mb']:.1f}")
    worst = result["worst_delta"]
    worst_text = f"{worst[0]} {_fmt(worst[1], signed=True)}" if worst else "n/a"
    print(f"    worst delta: {worst_text}")
    print(f"    regressions (>0.005): {', '.join(result['regressions']) if result['regressions'] else 'none'}")
    print(f"    TRANSFERRED: {row['transferred']}   FUSION BREAKTHROUGH: {row['fusion_breakthrough']}")
    if not row["transferred"] and result["bench_delta"] is not None and result["bench_delta"] <= 0:
        print(f"    A probe gain that does not move its benchmark is information about the PROBE, "
              f"not a failed experiment. Log it against the capability in Ensue.")


def _fmt(v, signed=False):
    if v is None:
        return "  -  "
    return f"{v:+.4f}" if signed else f"{v:.4f}"


# ==============================================================================================
# STATUS
# ==============================================================================================

def do_status() -> None:
    try:
        from deepearth.autoresearch.main.harness.evaluate import BENCHMARKS
    except Exception:
        BENCHMARKS = []

    # The map must not rot. probe_registry's mode table drifted the moment lib/gnn.py was deleted;
    # this check is why that cannot happen here silently.
    if BENCHMARKS:
        bad = [(loop, cap, b) for loop, m in CAPABILITY_BENCH.items()
               for cap, b in m.items() if b not in BENCHMARKS]
        if bad:
            print("*** CAPABILITY_BENCH names benchmarks that do not exist in evaluate.BENCHMARKS:")
            for loop, cap, b in bad:
                print(f"      {loop}:{cap} -> {b}")
            print()

    champ = read_champion()
    print(f"champion: {champ.get('label', '(unlabeled)')[:70]}")
    print(f"          harmonic {_fmt(champ.get('harmonic'))}   arithmetic {_fmt(champ.get('arithmetic'))}"
          f"   {len(champ.get('scores', {}))} benchmarks scored\n")

    eligible = 0
    for loop in CAPABILITY_BENCH:
        rows = survey(loop)
        if not rows:
            print(f"=== {loop}: no records\n")
            continue
        print(f"=== {loop}")
        for r in rows:
            ok = not r["blockers"]
            eligible += ok
            mark = "OK " if ok else "-- "
            print(f"  {mark}{r['capability']:<24} probe {_fmt(r['probe_score'])}"
                  f"  gain {_fmt(r['probe_gain'], signed=True)}"
                  f"  -> {r['bench'] or '(unmapped)'} {_fmt(r['bench_before'])}")
            for b in r["blockers"]:
                print(f"        blocked: {b}")
        print()

    print(f"{eligible} capability record(s) eligible to graduate today.")

    ledger = read_ledger()
    if not ledger:
        print("\ngraduation ledger is EMPTY — no probe result has ever been tested against the champion.")
        return
    closed = [r for r in ledger if r["state"] == "closed"]
    pending = [r for r in ledger if r["state"] == "pending"]
    print(f"\nledger: {len(closed)} closed, {len(pending)} pending")
    for r in pending:
        print(f"  PENDING  {r['id']}")
    if closed:
        moved = sum(1 for r in closed if r.get("transferred"))
        print(f"\n  TRANSFER RATE: {moved}/{len(closed)} probe results moved their benchmark")
        for r in closed:
            print(f"    {'MOVED ' if r.get('transferred') else 'no    '}{r['id']:<52}"
                  f" {r['bench']} {_fmt(r.get('bench_delta'), signed=True)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--status", action="store_true", help="what is eligible, what is blocked, transfer rate")
    ap.add_argument("--open", dest="open_target", default="", metavar="LOOP:CAPABILITY",
                    help="snapshot the champion 'before' and open a pending crossing")
    ap.add_argument("--close", dest="close_id", default="", metavar="ID",
                    help="fill in the 'after' from a champion run log and score the crossing")
    ap.add_argument("--log", action="append", default=[], help="candidate fusion run log (repeat twice)")
    ap.add_argument("--baseline-log", action="append", default=[], help="paired control fusion run log (repeat twice)")
    ap.add_argument("--control", default="", help="reuse a frozen, provenance-matched two-seed control")
    ap.add_argument("--register-control", default="", metavar="LABEL",
                    help="freeze the two --baseline-log runs for later crossings")
    ap.add_argument("--force", action="store_true",
                    help="open a crossing despite blockers; the row is marked forced with its reasons")
    a = ap.parse_args(argv)

    if a.register_control:
        register_control(a.register_control, a.baseline_log)
    elif a.open_target:
        do_open(a.open_target, a.force)
    elif a.close_id:
        if not a.log or (not a.baseline_log and not a.control):
            raise SystemExit("[graduate] --close needs two --log values and --control or two --baseline-log values")
        do_close(a.close_id, a.log, a.baseline_log, a.control)
    else:
        do_status()


if __name__ == "__main__":
    main()
