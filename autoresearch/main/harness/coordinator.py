"""Ensue shared memory for the research loop.

Multiple agents, different GPUs, one goal: the lowest `val_bpb`. Results flow through a shared Ensue
org; git stays local. Ensue is the shared brain, and it is additive -- if it is unreachable the loop
continues solo.

    THINK -> CLAIM -> RUN -> PUBLISH

Keys live under the org, namespaced, as ``<agent>--<slug>--<hash>``:

    results/<key>       a completed experiment: val_bpb, its decomposition, and the config that produced it
    claims/<key>        who is working on what (expires, so a dead agent does not block the swarm)
    hypotheses/<key>    a proposed next experiment with its reasoning
    insights/<key>      a learning worth not rediscovering -- especially a dead end
    best/config         the global best config
    best/metadata       its val_bpb, decomposition, and what it replaced

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
            return _rpc(self.api_key, tool, args)
        except Exception as e:                      # additive: never take the loop down with us
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

    # ---------------------------------------------------------------- board

    def _maybe_update_board(self, variable: str, val_bpb: float, record: dict, config: str) -> bool:
        """Read-compare-write with sanity checks. val_bpb is a loss: lower wins."""
        if not (val_bpb > 0) or val_bpb != val_bpb:            # <=0 or NaN is a crash, not a record
            print(f"[ensue] refusing bogus val_bpb {val_bpb}", flush=True)
            return False
        key = BOARD.format(variable=variable)
        cur = self.get(key).get(key) or {}
        prev = cur.get("val_bpb") if isinstance(cur, dict) else None
        if prev is not None and val_bpb >= prev:
            return False
        self.put(key, {**record, "config": config, "previous_val_bpb": prev,
                       "previous_by": cur.get("agent") if isinstance(cur, dict) else None,
                       "previous_description": cur.get("description") if isinstance(cur, dict) else None},
                 f"deepearth board {variable}: BEST {val_bpb:.6f} by {self.agent_id} -- {record['description'][:90]}")
        print(f"[ensue] new best for {variable}: {val_bpb:.6f} (was {prev})", flush=True)
        return True
