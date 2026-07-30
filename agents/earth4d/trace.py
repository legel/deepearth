"""agents/earth4d/trace.py — the Earth4D probe harness and record ledger.

The surface is the spacetime encoder (encoders/spacetime/earth4d.py) and the data channels feeding it,
measured by the fast encoder probe (autoresearch/programs/spacetime/probe.py & friends) — NOT full-model
training. The probe trains the encoder + a light head on ~65k obs in minutes and reports the
encoder-isolated marginal vs a FAIR baseline (RFF / MLP / best generic PE) on the SAME capability the
scorecard measures.

The agent declares one capability from scorecard.md and may then change anything the hypothesis needs.
What this harness enforces is measurement identity, not a menu of permitted edits: a run is comparable to
a record only when capability, mode, split, n_shards and protocol all match. A mismatch is recorded as a
re-baseline or withheld — never as a win.

Every successful run produces the same trace:
  - OBJECTIVE block: the declared --metric's primary score + fair gain + RECORD verdict.
  - BOTTLENECK read: INPUT-LIMITED (→ DATA lever) vs ENCODER-LIMITED (→ ARCHITECTURE lever) vs EARNING.
  - the parsed probe header + native metric lines.
  - RECORD tracking in agents/earth4d/records.json, and one upserted Ensue key per capability.
A failed probe (rc != 0) writes no trace and no record; the log is kept for diagnosis.

Usage:
  python -m deepearth.agents.earth4d.trace --metric family_from_spacetime \
      --probe "--forecast --n_shards 8" --tag forecast --device cuda:0 --ensue
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deepearth.autoresearch.programs.spacetime.probe_contract import (  # noqa: E402
    ContractError,
    ProbeResult,
)
from deepearth.autoresearch.programs.spacetime import probe_registry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]                 # .../deepearth
RECORDS = Path(__file__).resolve().parent / "records.json"  # the machine record (fill scorecard by breaking these)
DEFAULT_PROBE_MODULE = "deepearth.autoresearch.programs.spacetime.probe"
TRACE_AUTH_FD_ENV = "EARTH4D_TRACE_AUTH_FD"

# The encoder-probeable capabilities (scorecard.md Layer 2). The objective must be one of these; the
# probe MODE and the architecture are the agent's choice. This list and scorecard.md Layer 2/3 are one
# contract -- change both together.
CAPABILITIES = [
    "species_from_env", "species_from_spacetime", "family_from_env", "family_from_spacetime",
    "community_from_env", "calibration", "flowering_peak_month",
]

# Declared-and-refused, with the reason (scorecard.md Layer 3). These used to sit in CAPABILITIES with
# no PRIMARY_RE entry, so a run would fall through to the generic r"\bEarth4D\s+([\d.]+)" pattern and
# record whatever number matched first -- a legal --metric that measured nothing in particular. An
# explicit refusal is the honest version: the capability is real on the full-model board, it is simply
# not reachable through the encoder probe.
EXCLUDED_CAPABILITIES = {
    "family_from_vision": "borrowed frozen DINO/BioCLIP, and the stored record has no mode or shard "
                          "identity; it is not an Earth4D probe record",
    "lfmc_from_env": "non-encoder head: the capability lives in a downstream head",
    "mycorrhiza_from_env": "non-encoder head: the capability lives in a downstream head",
    "pollinator_from_env": "non-encoder head: the capability lives in a downstream head",
    "flowering_auc": "measured on the fusion model's flowering head, not the encoder",
    "flowering_fidelity": "measured on the fusion model's flowering head, not the encoder",
    "infer_clay": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_soil": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_climate": "env->env reconstruction runs through the field decoder, not the encoder probe",
    "infer_hydro": "env->env reconstruction runs through the field decoder, not the encoder probe",
}

# PROTOCOL VERSION. Bump this whenever a change alters what a run MEASURES rather than how well it does:
# a leak fix, a split change, a target/normalization change. Records carry the protocol they were set under,
# and a run under a different protocol RE-BASELINES the capability instead of "beating" it -- mode and shard
# count both match across such a change, so neither of those gates catches it.
#   v1-prefix     : everything up to 2026-07-29. Leaked in three ways (train mask admitted future-at-seen-place
#                   and past-at-held-place rows; time normalization fitted its span on test dates; env/vision
#                   standardization fitted mu/sd over test rows) and normalized time so the held-out future
#                   landed where the hash grid saturates.
#   v2-leakfix    : strict spatiotemporal split, train-only time normalization with horizon headroom,
#                   train-only feature standardization, deterministic seeding.
PROTOCOL = "v2-leakfix"
# Only explicitly identified, audited protocols may be migrated automatically.
# Absence of a protocol is not evidence that a hand-restored or pre-gate record
# belongs to the known v1 measurement regime.
REBASELINE_PROTOCOLS = frozenset({"v1-prefix"})

# Fair-baseline preference: Earth4D must beat a TRAINED generic PE, not just raw coords.
FAIR_ORDER = ["best-ctrl", "RFF", "mlp", "GAIN", "prop_acc", "best-coord", "raw"]


def _run(module: str, probe_args: str, device: str, log_path: str, result_path: str,
         capability: str) -> int:
    probe_argv = shlex.split(probe_args) + ["--device", device,
                                            "--result-json", result_path,
                                            "--capability", capability]
    cmd = [sys.executable, "-m", module] + probe_argv
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    read_fd, write_fd = os.pipe()
    try:
        authorization = (
            json.dumps(
                {"module": module, "argv": probe_argv},
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        os.write(write_fd, authorization)
        os.close(write_fd)
        write_fd = -1
        env[TRACE_AUTH_FD_ENV] = str(read_fd)
        print(f"[trace] $ {' '.join(cmd)}  (cwd={REPO})", flush=True)
        with open(log_path, "w") as lf:
            return subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(REPO),
                pass_fds=(read_fd,),
            ).returncode
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)










def _same_probe(probe, prev_probe):
    """Compare a migration command after shell-token normalization."""
    if not probe or not prev_probe:
        return False
    try:
        return shlex.split(probe) == shlex.split(prev_probe)
    except ValueError:
        return False


def _record_gate(
    key_val,
    prev,
    prev_proto,
    mode,
    prev_mode,
    shards,
    prev_shards,
    probe=None,
    prev_probe=None,
):
    """Return the like-for-like record decision and its component checks."""
    mode_ok = (prev is None) or (mode == prev_mode)
    shards_ok = (prev is None) or (shards == prev_shards)
    beats = key_val is not None and (prev is None or key_val > prev)
    rebaseline = (
        prev is not None
        and prev_proto in REBASELINE_PROTOCOLS
        and prev_proto != PROTOCOL
        and mode_ok
        and shards_ok
        and _same_probe(probe, prev_probe)
    )
    current_comparison = prev is None or prev_proto == PROTOCOL
    is_record = (
        beats and mode_ok and shards_ok and current_comparison
    ) or (rebaseline and key_val is not None)
    return is_record, rebaseline, beats, mode_ok, shards_ok


def _read_records(path=RECORDS):
    """Read one exact board snapshot for optimistic concurrency control."""
    raw = path.read_bytes() if path.exists() else b""
    return raw, json.loads(raw or b"{}")


def _commit_records_if_unchanged(expected_raw, records, path=RECORDS):
    """Atomically replace a board only if it has not changed since preflight."""
    lock_path = path.with_name("records.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current_raw = path.read_bytes() if path.exists() else b""
        if current_raw != expected_raw:
            return False
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w") as stream:
                stream.write(json.dumps(records, indent=2, sort_keys=True))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return True




def _bottleneck(fair, primary) -> str:
    """Diagnose which lever family the fair-gain points at (program.md, section 3 Diagnose).

    This string is written into records.json AND pushed to Ensue as the swarm's reason-to-move, so it
    has to agree with the program. It previously read a flat/negative fair-gain as ARCHITECTURE-LIMITED
    and told the agent to "swing bigger on the architecture" -- the exact inverse of the program, which
    reads a flat gain as the INPUT being signal-limited. Under the old string every flat-gain run
    advised the whole swarm to do the one thing the program forbids ("Don't default to architecture").
    """
    if fair is None:
        return "NO-FAIR-BASELINE (probe reported no vs-generic-PE gain — check output)"
    if fair <= 0:
        return ("INPUT-LIMITED: Earth4D does not beat a generic trained PE, so the coordinate/current "
                "channel lacks the signal → DATA lever, change the channel")
    if primary is not None and primary < 0.20:
        return ("ENCODER-LIMITED: the encoder beats the PE but the absolute score is low → ARCHITECTURE "
                "lever, change the mechanism")
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
        # A silent skip here means the swarm never learns this run happened, and the next agent pays to
        # rediscover the same dead-end. --ensue was explicitly requested, so a missing token is an error.
        sys.exit("[trace] --ensue was requested but no ENSUE_API_TOKEN is available (env or "
                 "/workspace/.env). The record was written locally; the swarm was NOT updated. "
                 "Export the token and re-publish rather than leaving the board stale.")
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
           f"THIS RUN '{trace['tag']}': primary={o['primary']} gain={o['fair_st_gain']} "
           f"decision={o.get('decision', 'legacy')} "
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
    ap = argparse.ArgumentParser(
        description="Earth4D legacy probe ledger — exact audited protocol migrations only"
    )
    ap.add_argument("--metric", required=True, help="objective capability (one of the scorecard capabilities)")
    ap.add_argument("--probe", required=True, help="probe flags = the architectural lever (quote the whole string)")
    ap.add_argument("--probe-module", default=DEFAULT_PROBE_MODULE)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ensue", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    if a.metric in EXCLUDED_CAPABILITIES:
        sys.exit("[trace] --metric %r is excluded: %s\n"
                 "        See agents/earth4d/scorecard.md Layer 3."
                 % (a.metric, EXCLUDED_CAPABILITIES[a.metric]))
    if a.metric not in CAPABILITIES:
        sys.exit("[trace] --metric %r is not an encoder-probeable capability. one of:\n  %s"
                 % (a.metric, "\n  ".join(CAPABILITIES)))
    modes = probe_registry.for_capability(a.metric)
    if not modes:
        sys.exit(f"[trace] no recording probe mode measures {a.metric!r}. "
                 f"See probe_registry --all.")
    print(f"[trace] {a.metric}: {len(modes)} mode(s) can set this record — "
          + ", ".join(m.mode for m in modes), flush=True)
    records_snapshot, preflight_records = _read_records()

    tag = a.tag or ("e4d_" + re.sub(r"\W+", "_", a.probe)[:24].strip("_"))
    log_path = a.log or str(Path(__file__).resolve().parent / "traces" / f"{tag}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[trace] OBJECTIVE={a.metric}  probe='{a.probe}'  tag={tag}", flush=True)
    result_path = str(Path(log_path).with_suffix(".result.json"))
    rc = _run(a.probe_module, a.probe, a.device, log_path, result_path, a.metric)
    text = Path(log_path).read_text(errors="ignore")
    if rc != 0:
        print(text[-1800:])
        sys.exit(f"[trace] probe FAILED (rc={rc}); see {log_path}")

    # The probe DECLARES what it measured. Nothing here parses stdout: a mode that does not emit a
    # contract cannot set a record, which is the point -- the old parser always produced *something*.
    try:
        result = ProbeResult.read(result_path)
    except (ContractError, OSError) as exc:
        sys.exit(f"[trace] probe emitted no usable result contract: {exc}\n"
                 f"        log preserved at {log_path}; no record was written")
    if result.diagnostic:
        sys.exit(f"[trace] {result.mode} is a DIAGNOSTIC and cannot set a record: "
                 f"{result.diagnostic_reason}\n        log preserved at {log_path}")
    if result.capability != a.metric:
        sys.exit(f"[trace] probe measured {result.capability!r} but --metric declared {a.metric!r}; "
                 f"refusing to record a different question's answer")

    known = {m.mode for m in modes}
    if result.mode not in known and not any(result.mode.startswith(k.split("(")[0]) for k in known):
        print(f"[trace] *** UNREGISTERED MODE {result.mode!r} for {a.metric}. Registered: "
              f"{sorted(known)}. Recording it, but add it to probe_registry so the next agent can "
              f"find it.", flush=True)

    primary = result.primary.value
    fair, fair_base = result.fair_gain(FAIR_ORDER)
    bottleneck = _bottleneck(fair, primary)
    mode = result.mode
    shards = result.n_shards
    header = result.render()
    gains = dict(result.gains)
    metrics = [f"{k} = {v}" for k, v in sorted(result.baselines.items())]

    # RECORD tracking + full run LEDGER (taxonomy: never lose a run's result; publish win OR dead-end w/ reason) --
    recs = preflight_records
    key_val = primary if primary is not None else fair
    cur = recs.get(a.metric, {})
    prev = cur.get("score")
    # RECORD GATE. A record used to fire on any parsed number that beat the stored one -- no check that the run
    # measured the SAME THING. That is how --pheno_disttarget peak_week (a different target, and a leaked one)
    # took flowering_peak_month 0.067 -> 0.683 and published it. A capability's record may only be beaten by a
    # run in the SAME probe mode; a different mode is a different target and gets flagged for review instead.
    prev_mode, prev_shards = cur.get("mode"), cur.get("n_shards")
    prev_proto = cur.get("protocol")
    # An UNSTAMPED record (pre-gate, or hand-restored) is treated as unknown-mode and does NOT auto-pass:
    # that is exactly how the leaked peak-week run slipped through on its second attempt.
    is_record, rebaseline, beats, mode_ok, shards_ok = _record_gate(
        key_val,
        prev,
        prev_proto,
        mode,
        prev_mode,
        shards,
        prev_shards,
        a.probe,
        cur.get("probe"),
    )
    migration_withheld = prev is not None and prev_proto != PROTOCOL and not rebaseline
    if rebaseline and key_val is not None:
        print(f"[trace] *** RE-BASELINE: record was set under protocol {prev_proto!r}, this run is {PROTOCOL!r}.\n"
              f"[trace]     {prev} and {key_val} measure different things, so this is not a comparison —\n"
              f"[trace]     the capability's baseline is being RESET to {key_val}. Prior record archived in the ledger.",
              flush=True)
    elif prev is not None and prev_proto != PROTOCOL and prev_proto not in REBASELINE_PROTOCOLS:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: prior protocol {prev_proto!r} is not an "
              f"explicitly audited migration source.\n"
              f"[trace]     The protected record remains unchanged; migrate it deliberately after provenance review.",
              flush=True)
    elif prev is not None and prev_proto in REBASELINE_PROTOCOLS and prev_proto != PROTOCOL and not rebaseline:
        print(f"[trace] *** PROTOCOL MIGRATION WITHHELD: the old probe command, mode, and shard count "
              f"must match exactly.\n"
              f"[trace]     old={cur.get('probe')!r} mode={prev_mode!r} shards={prev_shards!r}\n"
              f"[trace]     new={a.probe!r} mode={mode!r} shards={shards!r}",
              flush=True)
    if beats and not (mode_ok and shards_ok):
        why = ("mode %r != record mode %r" % (mode, prev_mode) if not mode_ok
               else "n_shards %r != record n_shards %r" % (shards, prev_shards))
        print(f"[trace] *** RECORD WITHHELD: {why}.\n"
              f"[trace]     {key_val} vs {prev} is not a like-for-like comparison. Match the record's protocol,\n"
              f"[trace]     or verify the new one is sound and set the record deliberately.", flush=True)
    ledger = cur.get("ledger", {"runs": 0, "records": [], "deadends": {}})
    ledger["runs"] = ledger.get("runs", 0) + 1
    if is_record:
        cur = {"score": key_val, "primary": primary, "fair_st_gain": fair,
               "fair_baseline": fair_base, "tag": tag, "probe": a.probe, "mode": mode, "n_shards": shards,
               "probe_module": a.probe_module, "protocol": PROTOCOL}
        ledger["records"] = (ledger.get("records", []) + [{"tag": tag, "score": key_val, "gain": fair,
                                                           "protocol": PROTOCOL,
                                                           "rebaseline_from": prev if rebaseline else None}])[-20:]
    elif migration_withheld and key_val is not None:
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val,
            "gain": fair,
            "why": (
                f"PROTOCOL MIGRATION WITHHELD (old protocol={prev_proto!r}, mode={prev_mode!r}, "
                f"n_shards={prev_shards!r}, probe={cur.get('probe')!r}; new mode={mode!r}, "
                f"n_shards={shards!r}, probe={a.probe!r})"
            ),
        }
    elif beats and not (mode_ok and shards_ok):
        ledger.setdefault("deadends", {})[tag] = {
            "score": key_val, "gain": fair,
            "why": (f"RECORD WITHHELD (mode {mode!r} vs {prev_mode!r}, n_shards {shards!r} vs {prev_shards!r}) "
                    f"-- not like-for-like; needs a deliberate check before it can count")}
    elif key_val is not None:
        # dead-end: a lever below record — kept WITH its reason, deduped by tag (no noise-floor spam)
        ledger.setdefault("deadends", {})[tag] = {"score": key_val, "gain": fair, "why": bottleneck}
        if len(ledger["deadends"]) > 40:
            ledger["deadends"] = dict(list(ledger["deadends"].items())[-40:])
    cur["ledger"] = ledger
    recs[a.metric] = cur
    if not _commit_records_if_unchanged(records_snapshot, recs):
        sys.exit(
            "[trace] WORKFLOW WITHHELD: records.json changed while the probe ran; "
            f"the probe log is preserved at {log_path}, but no record was written"
        )

    decision = (
        "rebaseline" if rebaseline
        else "record" if is_record
        else "migration_withheld" if migration_withheld
        else "no_record"
    )
    objective = {"primary": primary, "fair_st_gain": fair, "fair_baseline": fair_base,
                 "record": bool(is_record), "rebaseline": bool(rebaseline), "decision": decision,
                 "prev_record": prev, "record_value": key_val}
    trace = {"metric": a.metric, "tag": tag, "probe": a.probe, "probe_module": a.probe_module,
             "objective": objective, "gains": gains, "header": header, "metrics": metrics,
             "bottleneck": bottleneck, "rc": rc, "ledger": ledger}

    # one-screen consistent summary ---------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"OBJECTIVE  {a.metric}   probe='{a.probe}'")
    print(header or "(no '=== SPACETIME' header parsed — check the log)")
    print("-" * 76)
    print(f"  primary(score) = {primary}   fair_st_gain = {fair} (vs {fair_base})   all_gains = {gains}")
    record_text = (
        "RE-BASELINE (not a comparable win)" if rebaseline
        else "YES (new best!)" if is_record
        else "WITHHELD (protocol migration mismatch)" if migration_withheld
        else "no"
    )
    print(f"  RECORD = {record_text}   prev_record = {prev}")
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
