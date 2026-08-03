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

Lifecycle, because this tool cannot train (a champion run needs a GPU and the rule-20 budget):

    graduation.py --status                     what is eligible, what is blocked, and why   [no GPU]
    graduation.py --open spacetime:family_from_spacetime
                                             snapshot bench_before + the probe record -> PENDING row
    < run the champion at time_budget_s=600 with the probe's commit applied, capture run.log >
    graduation.py --close <id> --log run.log   fill bench_after -> transferred: true/false

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

def blockers(loop: str, capability: str, rec: Dict[str, Any], bench_ok: bool,
             champion_protocol: Optional[str] = None) -> List[str]:
    """Why this record may NOT cross. Empty list = eligible.

    Each rule exists because its absence has already corrupted this board once."""
    out: List[str] = []
    if capability not in CAPABILITY_BENCH.get(loop, {}):
        out.append(f"no benchmark mapped for {loop}:{capability} — add it to CAPABILITY_BENCH or "
                   f"record why the capability is not champion-reachable")
    elif not bench_ok:
        out.append(f"mapped benchmark {CAPABILITY_BENCH[loop][capability]!r} is absent from the "
                   f"champion record — it was never scored, so there is no 'before' to move")
    expected_probe = CURRENT_PROBE_PROTOCOL.get(loop)
    if rec.get("protocol") != expected_probe:
        out.append(f"probe protocol {rec.get('protocol')!r} is superseded; current {loop} protocol is "
                   f"{expected_probe!r} — re-baseline this capability before graduation")
    if champion_protocol != BENCHMARK_PROTOCOL:
        out.append(f"champion benchmark protocol {champion_protocol!r} is not the current "
                   f"{BENCHMARK_PROTOCOL!r} — establish a fresh champion baseline")
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
            "blockers": blockers(loop, capability, rec, bench in champ if bench else False,
                                 champion.get("benchmark_protocol")),
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

    champ = read_champion()
    bench = CAPABILITY_BENCH[loop].get(capability)
    stop = blockers(loop, capability, rec, bench in champ.get("scores", {}) if bench else False,
                    champ.get("benchmark_protocol"))
    if stop and not force:
        print(f"[graduate] {target} is NOT eligible:")
        for b in stop:
            print(f"    - {b}")
        raise SystemExit("[graduate] refusing to open. Fix the blockers, or pass --force to open a "
                         "crossing that is explicitly marked unreproducible.")

    cid = crossing_id(loop, capability, rec.get("tag"))
    if any(r["id"] == cid and r["state"] == "pending" for r in read_ledger()):
        raise SystemExit(f"[graduate] {cid} is already pending; close it with --close before reopening")

    row = {
        "id": cid,
        "state": "pending",
        "loop": loop,
        "capability": capability,
        "bench": bench,
        "probe": {"tag": rec.get("tag"), "score": rec.get("score", rec.get("primary")),
                  "fair_gain": rec.get("fair_st_gain"), "fair_baseline": rec.get("fair_baseline"),
                  "mode": rec.get("mode"), "n_shards": rec.get("n_shards"),
                  "protocol": rec.get("protocol"), "n_seeds": rec.get("n_seeds"),
                  "seed_std": rec.get("seed_std"), "provisional": rec.get("provisional"),
                  "commit": (rec.get("code") or {}).get("commit"),
                  "branch": (rec.get("code") or {}).get("branch")},
        "champion_before": {"label": champ.get("label"), "harmonic": champ.get("harmonic"),
                            "arithmetic": champ.get("arithmetic"),
                            "benchmark_protocol": champ.get("benchmark_protocol"),
                            "bench_value": champ.get("scores", {}).get(bench) if bench else None,
                            "n_benchmarks": len(champ.get("scores", {}))},
        "forced": bool(stop),
        "blockers_at_open": stop,
    }
    append_ledger(row)
    print(f"[graduate] OPENED {cid}")
    print(f"    predicts   {bench} moves up from {row['champion_before']['bench_value']}")
    print(f"    probe      {row['probe']['score']} (fair-gain {row['probe']['fair_gain']} vs "
          f"{row['probe']['fair_baseline']}, {row['probe']['n_seeds']} seeds)")
    print(f"    replay     git checkout {row['probe']['commit']} -- the encoder + CONFIG diff")
    print(f"\n    Now run the champion at the rule-20 budget, then:\n"
          f"      python -m deepearth.autoresearch.scoring.graduation "
          f"--close {cid} --log run.log")


def do_close(cid: str, log_path: str) -> None:
    from deepearth.autoresearch.main.harness.champion_report import parse_run

    rows = read_ledger()
    match = [r for r in rows if r["id"] == cid and r["state"] == "pending"]
    if not match:
        raise SystemExit(f"[graduate] no pending crossing {cid!r}. Open ones:\n  "
                         + "\n  ".join(r["id"] for r in rows if r["state"] == "pending"))
    row = match[-1]

    after = parse_run(log_path)
    if not after["scores"]:
        raise SystemExit(f"[graduate] no Bxx scores found in {log_path}")

    before_champ = read_champion()
    before_scores = before_champ.get("scores", {})

    if after.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
        raise SystemExit(f"[graduate] run protocol {after.get('benchmark_protocol')!r} does not match "
                         f"the live benchmark protocol {BENCHMARK_PROTOCOL!r}")
    if before_champ.get("benchmark_protocol") != BENCHMARK_PROTOCOL:
        raise SystemExit(f"[graduate] champion protocol {before_champ.get('benchmark_protocol')!r} does not match "
                         f"the live benchmark protocol {BENCHMARK_PROTOCOL!r}; establish a new baseline first")
    if row.get("champion_before", {}).get("benchmark_protocol") != BENCHMARK_PROTOCOL:
        raise SystemExit("[graduate] this crossing was opened under a different benchmark protocol; "
                         "discard it and open a new crossing")

    # THE SUITE MUST BE THE SAME SUITE.  Earth4D gains are canonical now, but optional labels and
    # holdout-specific endpoints can still make a run incomplete.  A crossing is a paired experiment;
    # changing the measured set invalidates the pair.
    missing = sorted(set(before_scores) - set(after["scores"]))
    added = sorted(set(after["scores"]) - set(before_scores))
    suite_changed = bool(missing or added)
    if suite_changed:
        raise SystemExit("[graduate] refusing to close: before/after benchmark suites differ\n"
                         f"    added: {', '.join(added) or '(none)'}\n"
                         f"    missing: {', '.join(missing) or '(none)'}")

    bench = row["bench"]
    b_before = row["champion_before"]["bench_value"]
    b_after = after["scores"].get(bench) if bench else None
    delta = (b_after - b_before) if (b_after is not None and b_before is not None) else None

    regressions = sorted(n for n, v in after["scores"].items()
                         if n in before_scores and v < before_scores[n] - 0.005)

    row.update({
        "state": "closed",
        "champion_after": {"harmonic": after["harmonic"], "arithmetic": after["arithmetic"],
                           "bench_value": b_after, "n_benchmarks": len(after["scores"]),
                           "log": str(Path(log_path).resolve())},
        "bench_delta": delta,
        # The question the whole ledger exists to answer. Deliberately strict: the mapped benchmark
        # must move UP, and no other benchmark may regress (science.md rule 30).
        "transferred": bool(not suite_changed and delta is not None and delta > 0 and not regressions),
        "regressions": regressions,
        "suite_changed": suite_changed,
        "suite_added": added,
        "suite_missing": missing,
    })
    rewrite_ledger(rows)

    print(f"[graduate] CLOSED {cid}")
    print(f"    {bench}: {_fmt(b_before)} -> {_fmt(b_after)} ({_fmt(delta, signed=True)})")
    print(f"    net harmonic {_fmt(row['champion_before']['harmonic'])} -> {_fmt(after['harmonic'])}"
          f"   arithmetic {_fmt(row['champion_before']['arithmetic'])} -> {_fmt(after['arithmetic'])}")
    print(f"    regressions (>0.005): {', '.join(regressions) if regressions else 'none'}")
    print(f"    TRANSFERRED: {row['transferred']}")
    if not row["transferred"] and delta is not None and delta <= 0:
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
    ap.add_argument("--log", default="", help="champion run log (required with --close)")
    ap.add_argument("--force", action="store_true",
                    help="open a crossing despite blockers; the row is marked forced with its reasons")
    a = ap.parse_args(argv)

    if a.open_target:
        do_open(a.open_target, a.force)
    elif a.close_id:
        if not a.log:
            raise SystemExit("[graduate] --close needs --log <champion run log>")
        do_close(a.close_id, a.log)
    else:
        do_status()


if __name__ == "__main__":
    main()
