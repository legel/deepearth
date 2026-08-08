"""Ensue shared memory for the research loop.

Multiple agents, different GPUs, one goal: stronger human-interpretable capabilities. Results flow through a shared Ensue
org; git stays local. Ensue is the shared brain, and it is additive -- if it is unreachable the loop
continues solo.

    THINK -> CLAIM -> RUN -> PUBLISH

Keys are flat under the org:

    LOOP-deepearth-best            THE SCORECARD -- where the science stands, in one validated document
    LOOP-deepearth-<variable>      per-variable board: the best val_bpb for that variable
    LOOP-deepearth-runs/...        one record per experiment, win or loss
    LOOP-deepearth-claims/...      who is working on what (expires, so a dead agent does not block anyone)
    LOOP-deepearth-insights/...    a learning worth not rediscovering -- especially a dead end
    LOOP-deepearth-hypotheses/...  a proposed next experiment with its reasoning

`-best` is the one key read by things that are not this loop -- the front end, and the delivery skill.
Its shape is enforced by `scorecard()` below rather than described in prose, so a consumer can rely on
it: same inputs produce a byte-identical document, and a malformed one raises here instead of being
published as a silently wrong number.

A dead end published as an insight is worth as much as a win: it stops every other agent paying for
the same negative.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

API_URL = "https://api.ensue-network.ai/"
CLAIM_TTL_S = 30 * 60          # a screen run is ~2 min warm; 30 min covers a slow one without stranding work
KEY_FILE = Path(".autoresearch-key")

# The org already holds 2,640 memories, 568 under LOOP-. Keys are flat (no org arg): a per-variable
# board at LOOP-deepearth-<variable>, and one record per experiment under LOOP-deepearth-runs/<variable>/.
BOARD = "LOOP-deepearth-{variable}"
RUN = "LOOP-deepearth-runs/{variable}/{slug}-{stamp}-{hash}"
CLAIM = "LOOP-deepearth-claims/{slug}-{hash}"
INSIGHT = "LOOP-deepearth-insights/{slug}-{hash}"
HYPOTHESIS = "LOOP-deepearth-hypotheses/{slug}-{hash}"
BEST = "LOOP-deepearth-best"

SCHEMA = "deepearth.scorecard/2"   # v3 moves likelihood metrics out of the promotion headline
_ROUND = 6                          # every float is rounded here, so the same run twice is byte-identical
_UNSET = object()                   # "no CAS guard given", distinct from expected=None ("cold start")

try:
    from deepearth.autoresearch.scoring.objective import (
        QUARANTINED_BENCHMARKS, arithmetic as capability_arithmetic,
        capability_suite as observed_capability_suite, harmonic as capability_harmonic,
        is_diagnostic,
    )
except ModuleNotFoundError:  # direct execution from the repository root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scoring.objective import (
        QUARANTINED_BENCHMARKS, arithmetic as capability_arithmetic,
        capability_suite as observed_capability_suite, harmonic as capability_harmonic,
        is_diagnostic,
    )


_KEY_NAMES = ("ENSUE_API_KEY", "ENSUE_API_TOKEN")   # the old harness used _TOKEN; accept both


def _api_key() -> Optional[str]:
    for name in _KEY_NAMES:
        v = os.environ.get(name)
        if v:
            return v.strip()
    for p in (KEY_FILE, Path("/workspace/.autoresearch-key"), Path("/workspace/.env"),
              Path("autoresearch/.env"), Path("/workspace/deepearth/autoresearch/.env")):
        if not p.exists():
            continue
        txt = p.read_text()
        for name in _KEY_NAMES:
            m = re.search(rf"{name}\s*=\s*(\S+)", txt)
            if m:
                return m.group(1).strip().strip('"\'')
    return None


def _rpc(api_key: str, tool: str, arguments: dict) -> dict:
    import requests
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"jsonrpc": "2.0", "method": "tools/call",
              "params": {"name": tool, "arguments": arguments}, "id": 1},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("data: "):
        text = text[len("data: "):]
    data = json.loads(text)
    if "error" in data:
        raise RuntimeError(f"Ensue RPC error: {data['error']}")
    content = data.get("result", {}).get("content", [])
    if content and isinstance(content[0], dict) and "text" in content[0]:
        return json.loads(content[0]["text"])
    return data.get("result", {})


def _hardware() -> dict:
    """What the number was measured on. Wall-clock claims and step budgets do not transfer across
    GPU classes, so a scorecard without this cannot be compared to one from another box."""
    info: dict = {"gpus": None, "gpu": None, "torch": None, "cuda": None, "host": os.uname().nodename}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = torch.cuda.device_count()
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        # No torch on the box that publishes (the delivery skill runs from a laptop). nvidia-smi is
        # enough to name the hardware, and absence is recorded as null rather than guessed.
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            if out:
                info["gpu"], info["gpus"] = out[0].strip(), len(out)
        except Exception:
            pass
    return info


def architecture_graph(model: Any, max_depth: int = 3) -> dict:
    """The model as a graph the front end can draw: nodes are modules, edges are containment.

    Walked from the live `nn.Module` rather than declared in a config, so it cannot drift from what
    actually trained -- a declared architecture is another field that gets copied forward and is
    quietly wrong. Depth-limited because the full tree of a 796M model is thousands of nodes and
    nobody reads past the third level.

    Containment only. True dataflow edges would need `torch.fx`, which cannot trace this model: the
    hash-grid encoders are custom CUDA kernels and the forward signature takes dicts of variables.
    Pass `dataflow` to `scorecard(model=...)` if you want those edges, and say where they came from.
    """
    def _params(m: Any) -> int:
        return sum(p.numel() for p in m.parameters())

    nodes: list = []
    edges: list = []

    def walk(module: Any, path: str, depth: int) -> None:
        total = _params(module)
        nodes.append({"id": path or "model", "type": type(module).__name__,
                      "params": total, "params_pct": None, "depth": depth})
        if depth >= max_depth:
            return
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            edges.append({"from": path or "model", "to": child_path})
            walk(child, child_path, depth + 1)

    walk(model, "", 0)
    root = nodes[0]["params"] or 1
    for n in nodes:
        n["params_pct"] = round(100.0 * n["params"] / root, 2)   # where the capacity actually sits
    nodes.sort(key=lambda n: (n["depth"], n["id"]))
    edges.sort(key=lambda e: (e["from"], e["to"]))
    return {"nodes": nodes, "edges": edges, "edge_kind": "containment", "max_depth": max_depth}


def _num(value: Any, field: str, *, positive: bool = False) -> float:
    """A metric that is None, NaN, inf or a string is a crash upstream, not a score. Refuse it here."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"scorecard.{field}: expected a number, got {value!r}")
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"scorecard.{field}: not finite ({value!r})")
    if positive and f <= 0:
        raise ValueError(f"scorecard.{field}: must be > 0, got {f}")
    return round(f, _ROUND)


def scorecard(*, val_bpb: float, macro: float, decomposition: dict, revealed_dims: dict,
              benchmarks: dict, benchmark_protocol: str, capability_suite: Sequence[str],
              seeds: int, harmonic_floor: float, arithmetic_floor: float, noise_floor: float,
              retrieval_floors: Optional[dict] = None,
              params: int, steps: int, config: str, agent: str = "unknown",
              model: Optional[dict] = None, commit: Optional[str] = None, branch: Optional[str] = None,
              hardware: Optional[dict] = None, wall_clock_s: Optional[float] = None,
              delivery: Optional[dict] = None, previous: Optional[dict] = None) -> dict:
    """Build THE scorecard: the whole state of the science in one document the front end can render.

    Every field is required because a scorecard missing one is not a weaker scorecard, it is an
    unreadable one -- a consumer cannot tell "no benchmarks were run" from "benchmarks were forgotten".
    Raises rather than defaulting, so a broken run fails at publish time instead of overwriting a valid
    record with a plausible-looking wrong one.

    Determinism: floats are rounded to `_ROUND`, variables sort by share then name, benchmarks by name.
    The same run published twice yields the same document apart from `generated_at`.
    """
    if not decomposition:
        raise ValueError("scorecard.decomposition: empty -- val_bpb without its per-variable lens is a bare number")
    if not benchmarks:
        raise ValueError("scorecard.benchmarks: empty -- rule 32 requires 100% of the suite scored")
    if not benchmark_protocol:
        raise ValueError("scorecard.benchmark_protocol: missing")
    suite = tuple(sorted(capability_suite))
    observed = observed_capability_suite(benchmarks)
    if not suite or suite != observed:
        raise ValueError(f"scorecard.capability_suite: declared {suite}, observed {observed}")
    missing_quarantine = sorted(set(QUARANTINED_BENCHMARKS) - set(benchmarks))
    if missing_quarantine:
        raise ValueError(f"scorecard.benchmarks: quarantined measurements must still be reported: {missing_quarantine}")
    missing = sorted(set(decomposition) - set(revealed_dims))
    if missing:
        raise ValueError(f"scorecard.revealed_dims: missing {missing} -- shares cannot be computed")

    total_dims = sum(int(revealed_dims[v]) for v in decomposition)
    if total_dims <= 0:
        raise ValueError("scorecard.revealed_dims: total is zero")

    # Share of the likelihood aggregate, which is dimension-weighted. This explains why val_bpb is a
    # diagnostic rather than the capability-promotion gate.
    # `floor` is what a PERFECT predictor scores, in bits. Retrieval banks are drawn with replacement
    # and several variables are per-species, so identical rows split the softmax mass and the floor is
    # not zero -- phylo's is 2.11 nats (3.05 bits) of an 8.32-nat range. `headroom` is what is actually
    # left to win; reading `bits_per_dim` as headroom overstates phylo's by about a third.
    _floors = {k: _num(v, f"retrieval_floors.{k}") for k, v in (retrieval_floors or {}).items()}
    variables = []
    for name, bits in decomposition.items():
        b = _num(bits, f"decomposition.{name}")
        floor_bits = round(_floors[name] / math.log(2), _ROUND) if name in _floors else None
        variables.append({
            "name": name,
            "bits_per_dim": b,
            "revealed_dims": int(revealed_dims[name]),
            "share_pct": round(100.0 * int(revealed_dims[name]) / total_dims, 3),
            "floor": floor_bits,
            "headroom": None if floor_bits is None else round(b - floor_bits, _ROUND),
        })
    variables.sort(key=lambda v: (-v["share_pct"], v["name"]))

    harmonic = capability_harmonic(benchmarks, suite=suite)
    arithmetic = capability_arithmetic(benchmarks, suite=suite)
    if harmonic_floor < 0 or arithmetic_floor < 0:
        raise ValueError("scorecard.evidence.floors: measured spreads cannot be negative")

    benchmark_rows = []
    for name, value in benchmarks.items():
        row = {"name": name, "score": _num(value, f"benchmarks.{name}")}
        if name in QUARANTINED_BENCHMARKS:
            row.update(role="quarantined", reason=QUARANTINED_BENCHMARKS[name])
        elif is_diagnostic(name):
            row["role"] = "mechanism"
        else:
            row["role"] = "capability"
        benchmark_rows.append(row)

    doc = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "agent": agent,
        "headline": {
            "harmonic": _num(harmonic, "harmonic"),
            "arithmetic": _num(arithmetic, "arithmetic"),
        },
        "diagnostics": {
            "val_bpb": _num(val_bpb, "val_bpb", positive=True),
            "macro": _num(macro, "macro", positive=True),
        },
        # What was trained. The count alone cannot be reproduced from -- two very different models hit
        # 24M -- so the architecture knobs travel with it.
        "model": {
            "params": int(params),
            "params_m": round(int(params) / 1e6, 1),
            "steps": int(steps),
            "config": config,
            # Either the hyperparameters, or a graph from `architecture_graph(net)` -- which is derived
            # from the live module tree, so it cannot disagree with what trained.
            "architecture": dict(model) if model else {},
        },
        "variables": variables,
        "benchmarks": sorted(benchmark_rows, key=lambda b: b["name"]),
        "evidence": {
            "benchmark_protocol": benchmark_protocol,
            "capability_suite": list(suite),
            "seeds": int(seeds),
            "floors": {
                "harmonic": _num(harmonic_floor, "harmonic_floor"),
                "arithmetic": _num(arithmetic_floor, "arithmetic_floor"),
            },
            "likelihood_noise_floor": _num(noise_floor, "noise_floor"),
            "commit": commit if commit is not None else _git("rev-parse", "--short", "HEAD"),
            "branch": branch if branch is not None else _git("branch", "--show-current"),
            # Mirror and public evaluators differ by ~158 lines; the same config scores ~0.279 against
            # 0.332464. Stamping the source makes the two impossible to compare by accident downstream.
            "measured_on": "research-mirror",
            "total_revealed_dims": total_dims,
            # Detected, not declared -- a hand-typed GPU name is the field most likely to be copied
            # forward from the previous card and quietly wrong.
            "hardware": hardware if hardware is not None else _hardware(),
            "wall_clock_s": None if wall_clock_s is None else _num(wall_clock_s, "wall_clock_s"),
        },
        # Filled by ship-deepearth-improvement when the result reaches the product. Present and null
        # means "not shipped", which is a different statement from absent.
        "delivery": delivery or {"pr": None, "pr_url": None, "base_commit": None, "merged": False,
                                 "delivered_at": None},
        "previous": previous,
    }
    if int(seeds) < 2:
        raise ValueError(f"scorecard.seeds: {seeds} -- a single-seed number is not a result")
    return doc


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "experiment"


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Coordinator:
    """Ensue client. Every method degrades to a no-op when Ensue is unreachable."""

    def __init__(self, agent_id: str = "unnamed", api_key: Optional[str] = None):
        self.agent_id = agent_id
        self.api_key = api_key or _api_key()

    @property
    def connected(self) -> bool:
        return bool(self.api_key)

    def _call(self, tool: str, args: dict) -> Optional[Any]:
        if not self.connected:
            return None
        try:
            self._last_error = None
            return _rpc(self.api_key, tool, args)
        except Exception as e:                      # additive: never take the loop down with us
            self._last_error = e
            print(f"[ensue] {tool} failed: {e}", flush=True)
            return None

    # ---------------------------------------------------------------- primitives

    def get(self, *keys: str) -> dict:
        """Values by key name. Returns {key: parsed_value}."""
        r = self._call("get_memory", {"key_names": list(keys)}) or {}
        out = {}
        for rec in (r.get("results") or []):
            v = rec.get("value")
            try:
                v = json.loads(v) if isinstance(v, str) else v
            except (ValueError, TypeError):
                pass
            out[rec.get("key_name")] = v
        return out

    def put(self, key: str, value: Any, description: str = "", embed: bool = True) -> bool:
        """Upsert, returning whether the value actually landed.

        `create_memory` on an existing key fails with a duplicate-key violation and leaves the old
        value in place -- verified against the live API -- so the update fallback is required, not
        defensive. Returns False rather than None when both legs fail, because a caller that reports
        a published record which does not exist is worse than one that reports nothing.
        """
        payload = value if isinstance(value, str) else json.dumps(value)
        r = self._call("create_memory", {"items": [{"key_name": key, "value": payload,
                                                    "description": description[:200], "embed": embed}]})
        if r is not None and isinstance(r, dict) and not r.get("failed"):
            return True
        u = self._call("update_memory", {"key_name": key, "value": payload,
                                         "description": description[:200]})
        if u is None:
            print(f"[ensue] WRITE LOST for {key}: create and update both failed", flush=True)
            return False
        return True

    def keys(self, prefix: str, limit: int = 50) -> list:
        r = self._call("list_keys", {"prefix": prefix, "limit": limit}) or {}
        return r if isinstance(r, list) else (r.get("keys") or r.get("results") or [])

    def search(self, query: str, prefix: str = "LOOP-", limit: int = 10) -> list:
        """Semantic search over memories -- how you check whether an idea was already tried."""
        r = self._call("discover_memories", {"query": query, "prefix": prefix, "limit": limit}) or {}
        return r if isinstance(r, list) else (r.get("results") or r.get("keys") or [])

    # ---------------------------------------------------------------- connectivity

    def assert_connected(self, require: bool = True) -> dict:
        """Fail loudly rather than degrading to silence. Call once at loop start.

        Every way this client can break -- wrong key name, wrong tool name, wrong namespace -- returns
        "no results", which is indistinguishable from "nothing to find". A loop reading an empty world
        looks exactly like a loop with nothing to learn, so it has to be checked, not assumed.
        """
        if not self.api_key:
            msg = ("no Ensue key: set ENSUE_API_KEY or ENSUE_API_TOKEN, or provide .autoresearch-key "
                   "or /workspace/.env")
            if require:
                raise RuntimeError(msg)
            print(f"[ensue] {msg} -- running solo", flush=True)
            return {"connected": False, "total": 0}

        total = self._call("count_keys", {})
        if not isinstance(total, dict) or "count" not in total:
            err = str(getattr(self, "_last_error", "") or "")
            if "401" in err or "Unauthorized" in err:
                raise RuntimeError(f"Ensue rejected the key (401). It is present but not valid for this "
                                   f"org: {self.api_key[:6]}...")
            if "tool not found" in err:
                raise RuntimeError(f"Ensue has no `count_keys` tool: the API surface changed. Call "
                                   f"tools/list and fix the client before trusting any read. ({err[:80]})")
            raise RuntimeError(f"Ensue unreachable or count_keys returned {total!r}. Do not run a loop "
                               f"against an unverified world. ({err[:120]})")
        n = int(total["count"])
        loop = (self._call("count_keys", {"prefix": "LOOP-"}) or {}).get("count", 0)
        if n == 0:
            raise RuntimeError("Ensue returned 0 keys for the whole org. Either the key has no access or "
                               "the namespace is wrong -- do not run a loop against an empty world.")
        print(f"[ensue] connected as {self.agent_id}: {n} memories, {loop} under LOOP-", flush=True)
        return {"connected": True, "total": n, "loop": loop}

    # ---------------------------------------------------------------- THINK

    def state(self, variable: str = "aggregate") -> dict:
        """What to read before picking an experiment: the board, live claims, insights, open hypotheses.

        The org already holds prior campaigns under `LOOP-earth4d-*`. Read them -- a dead end published
        there is a dead end you do not have to pay for again.
        """
        board = self.get(BOARD.format(variable=variable))
        return {
            "board": board,
            "claims": self.keys("LOOP-deepearth-claims/", 50),
            "insights": self.keys("LOOP-deepearth-insights/", 30),
            "hypotheses": self.keys("LOOP-deepearth-hypotheses/", 30),
            "prior_campaigns": self.keys("LOOP-earth4d-", 30),
        }

    def already_tried(self, description: str) -> list:
        """Semantic check across every campaign, past and present, before spending a run."""
        return self.search(description, prefix="LOOP-", limit=8)

    # ---------------------------------------------------------------- CLAIM

    def claim(self, description: str) -> Optional[str]:
        """Reserve an experiment. None if someone holds a live claim on it."""
        key = CLAIM.format(slug=_slug(description), hash=_hash(description))
        existing = self.get(key).get(key)
        if isinstance(existing, dict) and (time.time() - existing.get("claimed_at", 0)) < CLAIM_TTL_S:
            print(f"[ensue] claimed by {existing.get('agent')}: {description}", flush=True)
            return None
        self.put(key, {"agent": self.agent_id, "description": description,
                       "claimed_at": time.time(), "at": _now()},
                 f"claim by {self.agent_id}: {description}")
        return key

    # ---------------------------------------------------------------- PUBLISH

    def publish_result(self, variable: str, description: str, val_bpb: float, decomposition: dict,
                       status: str, config: str, extra: Optional[dict] = None) -> str:
        """One experiment. `status` is `keep` or `discard` -- publish both."""
        if variable == "best":
            # BOARD.format(variable="best") IS the scorecard key. A board record there is shaped
            # nothing like scorecard() output, so it silently breaks every consumer that pins on the
            # schema. Reserve the name rather than trusting callers to avoid it.
            raise ValueError(
                "publish_result(variable='best') would overwrite LOOP-deepearth-best, the scorecard. "
                "Publish the champion with scorecard() + publish_best(); use variable='aggregate' for "
                "the campaign board.")
        key = RUN.format(variable=variable, slug=_slug(description),
                         stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                         hash=_hash(description))
        record = {"agent": self.agent_id, "variable": variable, "description": description,
                  "status": status, "val_bpb": val_bpb, "decomposition": decomposition,
                  "config": config, "commit": _git("rev-parse", "--short", "HEAD"),
                  "branch": _git("branch", "--show-current"), "at": _now(), **(extra or {})}
        self.put(key, record, f"deepearth {variable} {status}: val_bpb {val_bpb:.6f} -- {description}")
        if status == "keep":
            self._maybe_update_board(variable, val_bpb, record, config)
        return key

    def post_insight(self, insight: str, evidence: Optional[list] = None) -> None:
        """What you learned and WHY. Mandatory on a dead end -- it is what stops the next agent paying
        for the same negative."""
        self.put(INSIGHT.format(slug=_slug(insight), hash=_hash(insight)),
                 {"agent": self.agent_id, "insight": insight, "evidence": evidence or [], "at": _now()},
                 f"insight from {self.agent_id}: {insight[:120]}")

    def publish_hypothesis(self, title: str, reasoning: str, suggested: Optional[dict] = None,
                           evidence: Optional[list] = None) -> None:
        """The next experiment your result implies, so someone can run it instead of re-deriving it."""
        self.put(HYPOTHESIS.format(slug=_slug(title), hash=_hash(title)),
                 {"agent": self.agent_id, "title": title, "reasoning": reasoning,
                  "suggested": suggested or {}, "evidence": evidence or [], "at": _now()},
                 f"hypothesis from {self.agent_id}: {title[:120]}")

    # ---------------------------------------------------------------- best / the scorecard

    def best(self) -> dict:
        """The current scorecard, or {} if none is published. Read this in THINK: your baseline is the
        swarm's best, and anything already delivered is not a breakthrough left to find."""
        doc = self.get(BEST).get(BEST) or {}
        if doc and doc.get("schema") != SCHEMA:
            print(f"[ensue] scorecard schema is {doc.get('schema')}, this client speaks {SCHEMA}", flush=True)
        return doc if isinstance(doc, dict) else {}

    def publish_best(self, card: dict, *, expected: Any = _UNSET, force: bool = False) -> bool:
        """Promote a scorecard by compare-and-swap, mirroring Ensue's own `promote_best` contract.

        `expected` is the incumbent capability harmonic you OBSERVED when you decided to promote. The write
        lands only if the live incumbent still equals it; `None` means "I observed no incumbent"
        (cold start). Returns False when the incumbent moved -- a sibling won -- and the caller should
        re-read `best()`, re-decide against the new champion, and retry. Passing nothing re-reads and
        uses whatever is live, which is the single-agent case.

        Ensue exposes `promote_best`, a genuinely atomic per-project CAS built for this. We cannot use
        it: it is worker-only and requires an active claimed project (`pop_next_task`), and this org
        has none -- the loop runs as an agent with an API token, not a warm-pool worker. Generic memory
        keys have no conditional write, so this is OPTIMISTIC concurrency, not atomic. It closes the
        common race (two agents promoting against a stale incumbent) but not the narrow one between
        the check and the write. If the loop ever runs as a worker, swap the body for `promote_best`
        and this signature already matches.

        `force` skips the comparison entirely -- for delivery stamps and schema migrations, which edit
        the standing record without making a new scientific claim.
        """
        if card.get("schema") != SCHEMA:
            raise ValueError(f"publish_best: build it with scorecard(); got schema {card.get('schema')!r}")
        raw = {row["name"]: row["score"] for row in card.get("benchmarks", [])}
        suite = (card.get("evidence") or {}).get("capability_suite", [])
        expected_harmonic = capability_harmonic(raw, suite=suite)
        expected_arithmetic = capability_arithmetic(raw, suite=suite)
        if not math.isclose(card["headline"]["harmonic"], expected_harmonic, abs_tol=2e-6):
            raise ValueError("publish_best: headline harmonic does not match the reported capabilities")
        if not math.isclose(card["headline"]["arithmetic"], expected_arithmetic, abs_tol=2e-6):
            raise ValueError("publish_best: headline arithmetic does not match the reported capabilities")
        new = card["headline"]["harmonic"]
        new_arithmetic = card["headline"]["arithmetic"]
        cur = self.best()
        live = (cur.get("headline") or {}).get("harmonic") if cur else None

        if not force:
            if cur and cur.get("schema") != SCHEMA:
                print("[ensue] promotion frozen: standing scorecard predates v3; publish the fresh "
                      "two-seed v3 baseline with force=True before comparing candidates", flush=True)
                return False
            if expected is not _UNSET and live != expected:
                print(f"[ensue] NOT promoted: incumbent moved to {live} (you decided against "
                      f"{expected}). Re-read best(), re-decide, retry.", flush=True)
                return False
            if live is not None:
                evidence = card["evidence"]
                prior_evidence = cur.get("evidence") or {}
                if evidence["benchmark_protocol"] != prior_evidence.get("benchmark_protocol"):
                    print("[ensue] NOT promoted: benchmark protocols differ; establish a new baseline "
                          "instead of comparing incomparable scores", flush=True)
                    return False
                if evidence["capability_suite"] != prior_evidence.get("capability_suite"):
                    print("[ensue] NOT promoted: capability suites differ; re-run both with the same suite",
                          flush=True)
                    return False
                harmonic_floor = evidence["floors"]["harmonic"]
                gain = new - live
                if gain <= harmonic_floor:
                    print(f"[ensue] scorecard unchanged: harmonic gain {gain:+.6f} does not beat "
                          f"its {harmonic_floor:.6f} two-seed floor", flush=True)
                    return False
                live_arithmetic = cur["headline"]["arithmetic"]
                arithmetic_floor = evidence["floors"]["arithmetic"]
                regression = live_arithmetic - new_arithmetic
                if regression > arithmetic_floor:
                    print(f"[ensue] NOT promoted: arithmetic regressed {regression:.6f}, beyond its "
                          f"{arithmetic_floor:.6f} two-seed floor", flush=True)
                    return False
                card = {**card, "previous": {
                    "harmonic": live, "arithmetic": live_arithmetic,
                    "val_bpb": (cur.get("diagnostics") or {}).get("val_bpb"),
                    "agent": cur.get("agent"), "commit": prior_evidence.get("commit"),
                    "at": cur.get("generated_at"),
                }}

        ok = self.put(BEST, card,
                      f"deepearth scorecard: harmonic {new:.6f}, arithmetic {new_arithmetic:.6f}, "
                      f"val_bpb diagnostic {card['diagnostics']['val_bpb']:.6f} at {card['model']['params_m']}M params "
                      f"({card['evidence']['seeds']} seeds) by {card.get('agent')}")
        if not ok:
            raise RuntimeError(f"publish_best: the write to {BEST} was lost -- the scorecard is NOT "
                               f"published. Do not report this result as recorded.")
        print(f"[ensue] scorecard promoted: harmonic {new:.6f} (was {live})", flush=True)
        return True

    def stamp_delivery(self, *, pr: int, pr_url: str, base_commit: str, merged: bool = False) -> bool:
        """Record that the current best reached the product. Called by ship-deepearth-improvement, so
        the loop can tell what is already public from what is still only ours."""
        cur = self.best()
        if not cur:
            raise RuntimeError("stamp_delivery: no scorecard published yet")
        cur["delivery"] = {"pr": int(pr), "pr_url": pr_url, "base_commit": base_commit,
                           "merged": bool(merged), "delivered_at": _now()}
        return self.publish_best(cur, force=True)

    # ---------------------------------------------------------------- board

    def _maybe_update_board(self, variable: str, val_bpb: float, record: dict, config: str) -> bool:
        """Update a per-variable likelihood board. This is a diagnostic record, not champion promotion.

        The board ranks on `val_bpb`, which for the aggregate board is DIMENSION-WEIGHTED, while
        the likelihood screen uses `macro`. Those are different quantities and they can disagree -- one
        variable carries 95.3% of the aggregate, so an arm can win it while losing macro, or the
        reverse. When a record carries both and they disagree, refuse the update and say so rather
        than letting whichever one the board happens to compare decide the champion silently.
        """
        if not (val_bpb > 0) or val_bpb != val_bpb:            # <=0 or NaN is a crash, not a record
            print(f"[ensue] refusing bogus val_bpb {val_bpb}", flush=True)
            return False
        key = BOARD.format(variable=variable)
        cur = self.get(key).get(key) or {}
        prev = cur.get("val_bpb") if isinstance(cur, dict) else None
        if prev is not None and val_bpb >= prev:
            return False
        new_macro, old_macro = record.get("macro"), cur.get("macro") if isinstance(cur, dict) else None
        if new_macro is not None and old_macro is not None and new_macro >= old_macro:
            print(f"[ensue] REFUSING board update for {variable}: val_bpb improves "
                  f"({prev:.6f} -> {val_bpb:.6f}) but macro does not ({old_macro:.6f} -> {new_macro:.6f}). "
                  f"the likelihood screen uses macro; the aggregate is 95.3% one variable. Resolve which this "
                  f"board ranks before publishing.", flush=True)
            return False
        self.put(key, {**record, "config": config, "previous_val_bpb": prev,
                       "previous_by": cur.get("agent") if isinstance(cur, dict) else None,
                       "previous_description": cur.get("description") if isinstance(cur, dict) else None},
                 f"deepearth board {variable}: BEST {val_bpb:.6f} by {self.agent_id} -- {record['description'][:90]}")
        print(f"[ensue] new best for {variable}: {val_bpb:.6f} (was {prev})", flush=True)
        return True
