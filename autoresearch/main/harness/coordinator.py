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
ORG = "deepearth-autoresearch"
CLAIM_TTL_S = 30 * 60          # a screen run is ~2 min warm; 30 min covers a slow one without stranding work
KEY_FILE = Path(".autoresearch-key")


def _api_key() -> Optional[str]:
    key = os.environ.get("ENSUE_API_KEY")
    if key:
        return key.strip()
    for p in (KEY_FILE, Path("/workspace/.autoresearch-key"), Path("/workspace/.env")):
        if p.exists():
            txt = p.read_text()
            m = re.search(r"ENSUE_API_KEY\s*=\s*(\S+)", txt) or re.search(r"^(\S+)$", txt.strip())
            if m:
                return m.group(1).strip()
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


class Coordinator:
    """Ensue client. Every method degrades to a no-op when Ensue is unreachable."""

    def __init__(self, agent_id: str = "unnamed", api_key: Optional[str] = None):
        self.agent_id = agent_id
        self.api_key = api_key or _api_key()

    @property
    def connected(self) -> bool:
        return bool(self.api_key)

    def _call(self, tool: str, args: dict) -> Optional[dict]:
        if not self.connected:
            return None
        try:
            return _rpc(self.api_key, tool, args)
        except Exception as e:                      # additive: never take the loop down with us
            print(f"[ensue] {tool} failed: {e}", flush=True)
            return None

    def _key(self, ns: str, description: str) -> str:
        h = hashlib.sha256(description.encode()).hexdigest()[:6]
        return f"{ns}/{self.agent_id}--{_slug(description)}--{h}"

    # ---------------------------------------------------------------- THINK

    def state(self) -> dict:
        """Global best, live claims, and recent insights -- what to read before picking an experiment."""
        return {
            "best": self._call("memory_get", {"org": ORG, "key": "best/metadata"}) or {},
            "claims": (self._call("memory_list", {"org": ORG, "prefix": "claims/"}) or {}).get("items", []),
            "insights": (self._call("memory_list", {"org": ORG, "prefix": "insights/", "limit": 30}) or {}).get("items", []),
            "hypotheses": (self._call("memory_list", {"org": ORG, "prefix": "hypotheses/", "limit": 30}) or {}).get("items", []),
        }

    def pull_best(self) -> Optional[dict]:
        """The swarm's current best config. Your baseline is the global best, not your local one."""
        meta = self._call("memory_get", {"org": ORG, "key": "best/metadata"})
        cfg = self._call("memory_get", {"org": ORG, "key": "best/config"})
        return {"metadata": meta, "config": cfg} if meta else None

    # ---------------------------------------------------------------- CLAIM

    def claim(self, description: str) -> Optional[str]:
        """Reserve an experiment. Returns None if someone already holds a live claim on it."""
        key = self._key("claims", description)
        existing = self._call("memory_get", {"org": ORG, "key": key})
        if existing and (time.time() - existing.get("claimed_at", 0)) < CLAIM_TTL_S:
            print(f"[ensue] already claimed by {existing.get('agent')}: {description}", flush=True)
            return None
        self._call("memory_upsert", {"org": ORG, "key": key, "value": {
            "agent": self.agent_id, "description": description,
            "claimed_at": time.time(), "at": datetime.now(timezone.utc).isoformat()}})
        return key

    # ---------------------------------------------------------------- PUBLISH

    def publish_result(self, description: str, val_bpb: float, decomposition: dict,
                       status: str, config: str, extra: Optional[dict] = None) -> Optional[str]:
        """Publish a completed experiment. `status` is `keep` or `discard` -- publish both."""
        key = self._key("results", description)
        record = {
            "agent": self.agent_id, "description": description, "status": status,
            "val_bpb": val_bpb, "decomposition": decomposition, "config": config,
            "commit": _git("rev-parse", "--short", "HEAD"), "branch": _git("branch", "--show-current"),
            "at": datetime.now(timezone.utc).isoformat(), **(extra or {}),
        }
        self._call("memory_upsert", {"org": ORG, "key": key, "value": record})
        if status == "keep":
            self._maybe_update_best(val_bpb, record, config)
        return key

    def post_insight(self, insight: str, evidence: Optional[list] = None) -> None:
        """A learning worth not rediscovering. Post one every run, especially on a dead end -- explain
        WHY, not just what happened."""
        self._call("memory_upsert", {"org": ORG, "key": self._key("insights", insight), "value": {
            "agent": self.agent_id, "insight": insight, "evidence": evidence or [],
            "at": datetime.now(timezone.utc).isoformat()}})

    def publish_hypothesis(self, title: str, reasoning: str, suggested: Optional[dict] = None,
                           evidence: Optional[list] = None) -> None:
        """The next experiment your result implies. You already did the thinking; hand it on."""
        self._call("memory_upsert", {"org": ORG, "key": self._key("hypotheses", title), "value": {
            "agent": self.agent_id, "title": title, "reasoning": reasoning,
            "suggested": suggested or {}, "evidence": evidence or [],
            "at": datetime.now(timezone.utc).isoformat()}})

    # ---------------------------------------------------------------- best/

    def _maybe_update_best(self, val_bpb: float, record: dict, config: str) -> bool:
        """Read-compare-write with sanity checks. val_bpb is a loss: lower wins."""
        if not (val_bpb > 0) or val_bpb != val_bpb:                       # <=0 or NaN is a crash, not a record
            print(f"[ensue] refusing bogus val_bpb {val_bpb}", flush=True)
            return False
        cur = self._call("memory_get", {"org": ORG, "key": "best/metadata"}) or {}
        prev = cur.get("val_bpb")
        if prev is not None and val_bpb >= prev:
            return False
        self._call("memory_upsert", {"org": ORG, "key": "best/config", "value": {"config": config}})
        self._call("memory_upsert", {"org": ORG, "key": "best/metadata", "value": {
            **record, "previous_best_val_bpb": prev, "previous_best_by": cur.get("agent"),
            "previous_best_description": cur.get("description")}})
        print(f"[ensue] new global best: {val_bpb:.6f} (was {prev})", flush=True)
        return True
