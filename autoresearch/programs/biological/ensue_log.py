"""Durable Ensue + file logger for the pollitree pipeline. Import and call log_stage()."""
import json, sys, time, urllib.request
from pathlib import Path

LOGDIR = Path("/workspace/deepearth/autoresearch/logs/pollitree")
LOGDIR.mkdir(parents=True, exist_ok=True)
ENSUE_URL = "https://api.ensue-network.ai/"
TOKEN = "lmn_17ac7aae2b5c47afbe6f8e98221365dd"


def _post_ensue(key, value, desc):
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "create_memory",
                   "arguments": {"items": [{"key_name": key, "value": value, "description": desc}]}},
    }).encode()
    req = urllib.request.Request(ENSUE_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
        out = []
        for line in raw.splitlines():
            if line.startswith("data:"):
                out.append(line[5:].strip())
        return "OK " + (" ".join(out)[:200] if out else raw[:200])
    except Exception as e:
        return f"ENSUE_ERR {e}"


def log_stage(stage, value, desc):
    """Write a durable file log AND post to Ensue. stage e.g. 'assess'."""
    key = f"LOOP-biological-pollitree-{stage}"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    fp = LOGDIR / f"{stage}.log"
    with open(fp, "a") as f:
        f.write(f"\n===== {ts} | {key} =====\n{value}\n")
    res = _post_ensue(key, value, desc)
    with open(LOGDIR / "ensue_posts.log", "a") as f:
        f.write(f"{ts} {key} :: {res}\n")
    print(f"[logged {stage}] ensue={res}", flush=True)
    return res


if __name__ == "__main__":
    print(log_stage("smoketest", "pipeline logger online", "pollitree logger smoke test"))
