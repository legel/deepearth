"""Ensue shared memory for the research loop.

Multiple agents, different GPUs, one goal: the lowest `val_bpb`. Results flow through a shared Ensue
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

SCHEMA = "deepearth.scorecard/1"   # bump only on a breaking shape change; the front end pins on this
_ROUND = 6                          # every float is rounded here, so the same run twice is byte-identical


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
              benchmarks: dict, harmonic: float, arithmetic: float, seeds: int, noise_floor: float,
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
    missing = sorted(set(decomposition) - set(revealed_dims))
    if missing:
        raise ValueError(f"scorecard.revealed_dims: missing {missing} -- shares cannot be computed")

    total_dims = sum(int(revealed_dims[v]) for v in decomposition)
    if total_dims <= 0:
        raise ValueError("scorecard.revealed_dims: total is zero")

    # Share of the AGGREGATE, which is dimension-weighted -- this is the field that tells a reader why
    # one variable can move the gate on its own, and the reason `judge()` also requires coverage.
    variables = [
        {"name": name,
         "bits_per_dim": _num(bits, f"decomposition.{name}"),
         "revealed_dims": int(revealed_dims[name]),
         "share_pct": round(100.0 * int(revealed_dims[name]) / total_dims, 3)}
        for name, bits in decomposition.items()
    ]
    variables.sort(key=lambda v: (-v["share_pct"], v["name"]))

    doc = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "agent": agent,
        "headline": {
            "val_bpb": _num(val_bpb, "val_bpb", positive=True),   # the gate
            "macro": _num(macro, "macro", positive=True),          # coverage: every capability equal
            "harmonic": _num(harmonic, "harmonic"),                # the public-facing suite means
            "arithmetic": _num(arithmetic, "arithmetic"),
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
        "benchmarks": sorted(
            ({"name": k, "score": _num(v, f"benchmarks.{k}")} for k, v in benchmarks.items()),
            key=lambda b: b["name"],
        ),
        "evidence": {
            "seeds": int(seeds),
            "noise_floor": _num(noise_floor, "noise_floor"),
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

    def put(self, key: str, value: Any, description: str = "", embed: bool = True) -> None:
        """Upsert. create_memory first; update_memory if the key already exists."""
        payload = value if isinstance(value, str) else json.dumps(value)
        r = self._call("create_memory", {"items": [{"key_name": key, "value": payload,
                                                    "description": description[:200], "embed": embed}]})
        if r is None or (isinstance(r, dict) and r.get("failed")):
            self._call("update_memory", {"key_name": key, "value": payload,
                                         "description": description[:200]})

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

    def publish_best(self, card: dict, *, force: bool = False) -> bool:
        """Publish a scorecard built by `scorecard()`. Refuses one that does not beat the current
        `val_bpb`, so a concurrent agent cannot regress the record by publishing later.

        `force` is for delivery stamps and schema migrations -- an edit to the standing record that is
        not a new scientific claim.
        """
        if card.get("schema") != SCHEMA:
            raise ValueError(f"publish_best: build it with scorecard(); got schema {card.get('schema')!r}")
        new = card["headline"]["val_bpb"]
        cur = self.best()
        prev = (cur.get("headline") or {}).get("val_bpb") if cur else None
        if prev is not None and not force and new >= prev:
            print(f"[ensue] scorecard unchanged: {new:.6f} does not beat {prev:.6f}", flush=True)
            return False
        if prev is not None and not force:
            card = {**card, "previous": {"val_bpb": prev, "agent": cur.get("agent"),
                                         "commit": (cur.get("evidence") or {}).get("commit"),
                                         "at": cur.get("generated_at")}}
        self.put(BEST, card,
                 f"deepearth scorecard: val_bpb {new:.6f}, macro {card['headline']['macro']:.6f}, "
                 f"harmonic {card['headline']['harmonic']:.4f} at {card['model']['params_m']}M params "
                 f"({card['evidence']['seeds']} seeds) by {card.get('agent')}")
        print(f"[ensue] scorecard published: val_bpb {new:.6f} (was {prev})", flush=True)
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
        """Read-compare-write with sanity checks. val_bpb is a loss: lower wins.

        The board ranks on `val_bpb`, which for the aggregate board is DIMENSION-WEIGHTED, while
        `judge()` gates on `macro`. Those are different quantities and they can disagree -- one
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
                  f"judge() gates on macro; the aggregate is 95.3% one variable. Resolve which this "
                  f"board ranks before publishing.", flush=True)
            return False
        self.put(key, {**record, "config": config, "previous_val_bpb": prev,
                       "previous_by": cur.get("agent") if isinstance(cur, dict) else None,
                       "previous_description": cur.get("description") if isinstance(cur, dict) else None},
                 f"deepearth board {variable}: BEST {val_bpb:.6f} by {self.agent_id} -- {record['description'][:90]}")
        print(f"[ensue] new best for {variable}: {val_bpb:.6f} (was {prev})", flush=True)
        return True
