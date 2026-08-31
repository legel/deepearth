"""Minimal Ensue memory client for the command console — the dashboard's one write path.

The console writes a scientist's directives to the same memory the DeepEarth research loop reads, so a
"word to the wise" reaches the agents without a PR round-trip:

    /build   <text>   ->  LOOP-deepearth-directives/*        (a hard override of weakest-first)
    /science <text>   ->  LOOP-deepearth-directives/*        (a new requirement -> science.md)
    <prose>           ->  LOOP-deepearth-customer-feedback/* (a soft input, read at THINK)

Self-contained REST (the dashboard is a standalone app), mirroring the deepearth.directive/1 schema the
loop consumes. Create-if-absent, so re-issuing never resets work the loop is mid-flight on. Key from
ENSUE_API_KEY (env or dashboard/.env), same convention as audit.py's model key.
"""
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://api.ensue-network.ai/"
SCHEMA_DIRECTIVE = "deepearth.directive/1"
DIRECTIVE = "LOOP-deepearth-directives/{kind}-{slug}-{hash}"
FEEDBACK = "LOOP-deepearth-customer-feedback/{slug}-{hash}"
DIRECTIVE_LIVE = ("open", "acknowledged", "in_progress", "needs-clarification", "blocked")
# /build or /science at the start of a line; body runs to the next command or the end of the message.
COMMAND = re.compile(r"(?ms)^[ \t]*/(build|science)\b[ \t]*(.*?)(?=^[ \t]*/(?:build|science)\b|\Z)")


def _key():
    k = os.environ.get("ENSUE_API_KEY") or os.environ.get("ENSUE_API_TOKEN")
    if k:
        return k.strip()
    env = Path(__file__).resolve().parent / ".env"
    if env.exists():
        m = re.search(r"ENSUE_API_(?:KEY|TOKEN)\s*=\s*(\S+)", env.read_text())
        if m:
            return m.group(1).strip().strip("'\"")
    return None


def _rpc(tool, args):
    key = _key()
    if not key:
        raise RuntimeError("no ENSUE_API_KEY (set it in the environment or dashboard/.env)")
    body = json.dumps({"jsonrpc": "2.0", "method": "tools/call",
                       "params": {"name": tool, "arguments": args}, "id": 1}).encode()
    req = urllib.request.Request(API_URL, data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = urllib.request.urlopen(req, timeout=30).read().decode().strip()
    if t.startswith("data: "):
        t = t[len("data: "):]
    d = json.loads(t)
    if "error" in d:
        raise RuntimeError(d["error"])
    c = d.get("result", {}).get("content", [])
    return json.loads(c[0]["text"]) if c and isinstance(c[0], dict) and "text" in c[0] else d.get("result", {})


def _slug(text, n=40):
    return (re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].rstrip("-") or "directive")


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _get(key):
    res = (_rpc("get_memory", {"key_names": [key]}).get("results") or [{}])[0]
    v = res.get("value") if res.get("status") == "success" else None
    return json.loads(v) if isinstance(v, str) else v


def _put(key, value, description):
    r = _rpc("create_memory", {"items": [{"key_name": key, "value": json.dumps(value),
                                          "description": description[:200], "embed": True}]})
    if isinstance(r, dict) and r.get("failed"):
        _rpc("update_memory", {"key_name": key, "value": json.dumps(value), "description": description[:200]})


def classify(text):
    """(kind, body) for each command in the message; prose yields one ('feedback', text)."""
    cmds = [(k, t.strip()) for k, t in COMMAND.findall(text) if t.strip()]
    return cmds if cmds else [("feedback", text.strip())]


def post(text, author="legel", source=None):
    """Write each command/feedback in one console message to memory. Returns [{kind, key, created}]."""
    source = source or {"kind": "console"}
    out = []
    for kind, body in classify(text):
        if kind == "feedback":
            key = FEEDBACK.format(slug=_slug(body),
                                  hash=_hash(f"{body}|{author}|{source.get('url', '')}"))
            created = not isinstance(_get(key), dict)
            if created:
                _put(key, {"author": author, "body": body, "source": source,
                           "needs_action": True, "at": _now()}, f"feedback from {author}: {body[:120]}")
        else:
            key = DIRECTIVE.format(kind=kind, slug=_slug(body), hash=_hash(f"{body}|{author}"))
            existing = _get(key)
            created = not (isinstance(existing, dict) and existing.get("status") in DIRECTIVE_LIVE)
            if created:  # create-if-absent: never reset a directive the loop is mid-flight on
                _put(key, {"schema": SCHEMA_DIRECTIVE, "kind": kind, "body": body, "author": author,
                           "status": "open", "priority": 0, "interpretation": None, "acceptance": [],
                           "science_pr": None, "deviations": [], "progress": [], "source": source,
                           "created_at": _now(), "updated_at": _now(), "closed_by": None},
                     f"/{kind} from {author}: {body[:120]}")
        out.append({"kind": kind, "key": key, "created": created})
    return out


def board():
    """The live directive board + feedback the console renders — statuses/progress come from the agents."""
    def _list(prefix):
        names = [k["key_name"] for k in (_rpc("list_keys", {"prefix": prefix, "limit": 100}).get("keys") or [])]
        if not names:
            return []
        out = []
        for res in _rpc("get_memory", {"key_names": names}).get("results", []):
            if res.get("status") == "success":
                v = res.get("value")
                v = json.loads(v) if isinstance(v, str) else v
                if isinstance(v, dict):
                    v["_key"] = res.get("key_name")
                    out.append(v)
        return out

    directives = sorted(_list("LOOP-deepearth-directives/"),
                        key=lambda d: d.get("created_at", ""), reverse=True)
    feedback = sorted(_list("LOOP-deepearth-customer-feedback/"),
                      key=lambda d: d.get("at", ""), reverse=True)
    return {"directives": directives, "feedback": feedback}
