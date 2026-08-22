"""LLM audit: connectivity (code <-> rules <-> benchmarks) and system status via Gemini.

    python -m dashboard.audit               # audit changed files + refresh status
    python -m dashboard.audit --status-only
    python -m dashboard.audit --loop        # re-audit whenever repo HEAD changes

Outputs are schema-forced JSON with hard caps: notes <=90 chars, headlines <=90,
next-steps <=120. Inputs may be whole files; outputs never ramble.
"""
import argparse, hashlib, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STATE, CACHE = ROOT / "state", ROOT / "state" / "cache"


def _env(key, default=None):
    """Repo .env wins over ambient environment — the project controls which account pays."""
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return os.environ.get(key, default)


MODEL = _env("GEMINI_MODEL", "gemini-3.6-flash")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
PROMPT_V = "v1"                                       # bump to invalidate cache
SYSTEMS = ["earth4d", "phylo", "fusion", "method", "data"]
STATUSES = ["good", "warning", "serious", "critical", "unknown"]


def gemini(prompt: str, retries: int = 3) -> dict:
    key = _env("GEMINI_API_KEY")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"response_mime_type": "application/json",
                                            "temperature": 0.1}}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(f"{URL}?key={key}", body,
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


def cached(key: str, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{hashlib.sha256((PROMPT_V + key).encode()).hexdigest()[:24]}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = fn()
    p.write_text(json.dumps(out))
    return out


def _reg():
    return json.loads((STATE / "registry.json").read_text())


def _compact_rules(reg):
    return "\n".join(f"R{r['id']} [{r['system']}] {r['title']}: {r['summary']}"
                     for r in reg["rules"])


def _compact_benches(reg):
    return "\n".join(f"{b['id']} {b['measures']}" for b in reg["benchmarks"])


CONNECT = """You audit the DeepEarth repo. Map this file's blocks to the science rules and
benchmarks they IMPLEMENT or DIRECTLY SUPPORT. Be precise; most blocks connect to few or none.

RULES:
{rules}

BENCHMARKS:
{benches}

FILE {path} BLOCKS: {blocks}

FILE CONTENT:
{content}

Return JSON: {{"blocks": [{{"id": "<block id>", "rules": [{{"r": <int>, "s": <1 weak|2 clear|3 core>,
"note": "<=12 words why"}}], "benchmarks": [{{"b": "B<n>", "s": <1-3>, "note": "<=12 words"}}]}}]}}
Only blocks with at least one edge. Notes are evidence, not prose."""


def connect_file(reg, f):
    text = (REPO / f["path"]).read_text(errors="replace")
    if len(text) > 90_000:
        text = text[:60_000] + "\n...[truncated]...\n" + text[-25_000:]
    blocks = [b["id"] for b in reg["blocks"] if b["path"] == f["path"]]
    prompt = CONNECT.format(rules=_compact_rules(reg), benches=_compact_benches(reg),
                            path=f["path"], blocks=json.dumps(blocks), content=text)
    out = cached(f["path"] + f["sha"], lambda: gemini(prompt))
    edges = []
    ok = {b["id"] for b in reg["blocks"]}
    for blk in out.get("blocks", []):
        if blk.get("id") not in ok:
            continue
        for e in blk.get("rules", []):
            edges.append({"src": blk["id"], "dst": f"R{e['r']}", "s": e.get("s", 1),
                          "note": str(e.get("note", ""))[:90]})
        for e in blk.get("benchmarks", []):
            edges.append({"src": blk["id"], "dst": e.get("b", ""), "s": e.get("s", 1),
                          "note": str(e.get("note", ""))[:90]})
    return edges


def _latest_run():
    for p in sorted((ROOT / "runs").glob("*.jsonl"), reverse=True):
        events = [json.loads(l) for l in open(p) if l.strip()]
        if fin := next((e for e in events if e.get("t") == "final" and e.get("scores", {}).get("benchmarks")), None):
            return {"id": p.stem, "net_score": fin["scores"].get("net_score"),
                    "arithmetic": fin["scores"].get("arithmetic"), "benchmarks": fin["scores"]["benchmarks"]}
    return None


def _run_context(reg):
    run = _latest_run()
    if not run:
        return "none"
    champ = {b["key"]: b["current_score"] for b in reg["benchmarks"]}
    deltas = sorted(((k, v, v - champ[k]) for k, v in run["benchmarks"].items()
                     if champ.get(k) is not None), key=lambda x: x[2])
    movers = deltas[:6] + deltas[-6:]
    return (f"{run['id']}: H {run['net_score']} A {run['arithmetic']} (champion H 0.374 A 0.583; "
            f"local variant runs may lack optional channels — weigh deltas qualitatively)\n"
            + "\n".join(f"  {k} {v:.3f} ({d:+.3f} vs champion)" for k, v, d in movers))


REACH_ORD = ["live", "gated", "data-pipeline", "tests", "recipes", "tooling", "island"]


def _reachability(reg, graph):
    """Per rule: how its linked .py code actually reaches the champion path (from callgraph.json)."""
    p = STATE / "callgraph.json"
    if not p.exists():
        return {}
    defs = json.loads(p.read_text())["defs"]
    out = {}
    for r in reg["rules"]:
        blocks = {e["src"] for e in graph["edges"]
                  if e["dst"] == f"R{r['id']}" and e["src"].split(":")[0].endswith(".py")}
        cnt, gates, islands = {"live": 0, "gated": 0, "island": 0}, set(), set()
        seen = set()
        for b in blocks:                                 # every def contained in the linked code
            path, rng = b.rsplit(":", 1)
            s, e = map(int, rng.split("-"))
            for d in defs:
                if d["path"] == path and d["start"] <= e and d["end"] >= s and d["id"] not in seen:
                    seen.add(d["id"])
                    if d["reach"] in cnt:
                        cnt[d["reach"]] += 1
                        if d["reach"] == "gated":
                            gates.add(d.get("gate", "?"))
                        if d["reach"] == "island":
                            islands.add(d["id"].split("::")[1])
        out[r["id"]] = (cnt, sorted(gates), sorted(islands)[:6])
    return out


def _reach_lines(rules, reach):
    return "\n".join(
        f"R{r['id']}: {c['live']} live defs"
        + (f", gated ({','.join(g)})" if g else "")
        + (f", islands within linked code: {', '.join(i)}" if i else "")
        for r in rules if (v := reach.get(r["id"])) for c, g, i in [v]) or "no reachability data"


STATUS = """You audit DeepEarth's "{system}" system. Judge each rule's implementation status from
the evidence. Statuses: good (implemented + validated), warning (partial/untested),
serious (major gap), critical (absent/broken), unknown (insufficient evidence).

HARD CONSTRAINT — champion-path reachability (static call graph under the current config):
{reach}
"island" = never called; "gated" = behind a config flag that is off. Neither runs on the
champion path. If a listed island/gated def IS a mechanism this rule mandates (judge by name
and the rule text), the rule is at best warning — partially implemented — and your "next"
must name that dead/gated def. A rule whose only implementation is island => at best serious.

RULES OF THIS SYSTEM:
{rules}

CODE LINKED TO EACH RULE (block ids + connection notes):
{evidence}

BENCHMARK SCORES (current champion; null = inactive):
{scores}

LATEST TRACKED RUN (biggest movers vs champion):
{run}

OPERATIONAL FINDINGS (defects proven by actually running the system — weigh them):
{findings}

Return JSON: {{"rules": [{{"id": <int>, "status": "<status>", "headline": "<=15 words",
"evidence": ["<block or benchmark ids>"], "next": "<=18 words, concrete next step"}}],
"system": {{"status": "<worst-informed overall>", "headline": "<=15 words", "next": "<=18 words"}}}}
Judge from evidence only. Cite ids you actually saw."""


def status_system(reg, graph, system, reach):
    rules = [r for r in reg["rules"] if r["system"] == system]
    ev = {}
    for e in graph["edges"]:
        if e["dst"].startswith("R") and int(e["dst"][1:]) in {r["id"] for r in rules}:
            ev.setdefault(e["dst"], []).append(f"{e['src']} ({e['note']})")
    scores = {b["id"]: b["current_score"] for b in reg["benchmarks"]}
    fp = ROOT / "seed" / "findings.json"
    findings = "\n".join(f"{f['id']} [{f['sev']}] {f['title']}: {f['detail'][:220]}"
                         for f in json.loads(fp.read_text())["findings"]) if fp.exists() else "none"
    prompt = STATUS.format(
        system=system, reach=_reach_lines(rules, reach),
        rules="\n".join(f"R{r['id']} {r['title']}: {r['summary']}" for r in rules),
        evidence=json.dumps(ev, indent=0)[:40_000],
        scores=json.dumps(scores), run=_run_context(reg), findings=findings)
    key = system + graph["head"] + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    out = cached(key, lambda: gemini(prompt))
    for r in out.get("rules", []):
        r["status"] = r.get("status") if r.get("status") in STATUSES else "unknown"
        r["headline"] = str(r.get("headline", ""))[:90]
        r["next"] = str(r.get("next", ""))[:120]
    return out


def _apply_verification(edges):
    """Tier-2 verdicts outlive tier-1 rebuilds: re-add hunter edges, re-drop refuted ones."""
    p = STATE / "verification.json"
    if not p.exists():
        return edges
    v = json.loads(p.read_text())
    wrong = {(w["src"], w["dst"]) for h in v.get("hunts", []) for w in h.get("wrong", [])}
    have = {(e["src"], e["dst"]) for e in edges}
    out = [e for e in edges if (e["src"], e["dst"]) not in wrong]
    for h in v.get("hunts", []):
        for m in h.get("missed", []):
            if (m["block"], m["dst"]) not in have:
                out.append({"src": m["block"], "dst": m["dst"], "s": m["s"],
                            "note": ("✓✓ " + m["note"])[:90]})
    return out


def run(status_only=False, graph_only=False):
    reg = _reg()
    text_files = [f for f in reg["files"] if f["kind"] == "text"]
    graph_p = STATE / "graph.json"

    if not status_only:
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(lambda f: connect_file(reg, f), text_files))
        edges = _apply_verification([e for r in results for e in r])
        graph = {"head": reg["head"], "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "edges": edges}
        graph_p.write_text(json.dumps(graph) + "\n")
        print(f"graph: {len(edges)} edges over {len(text_files)} files")

    if not graph_only:
        graph = json.loads(graph_p.read_text())
        reach = _reachability(reg, graph)
        rules_out, sys_out = [], {}
        for s in SYSTEMS:
            out = status_system(reg, graph, s, reach)
            rules_out += out.get("rules", [])
            sys_out[s] = out.get("system", {})
        status = {"audited": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": reg["head"],
                  "rules": rules_out, "systems": sys_out}
        (STATE / "status.json").write_text(json.dumps(status) + "\n")
        print(f"status: {len(rules_out)} rules, {len(sys_out)} systems")


SEEDS = {
    "science": (["autoresearch/science.md"],
                """From science.md, return JSON {"rules": [{"id": <int>, "title": "<3-6 words>",
"summary": "<=20 words", "system": "<earth4d|phylo|fusion|method|data>",
"keywords": [<3-5 grep-able code terms>]}] for all 32 numbered rules,
"foundations": [{"name", "role": "<=10 words"}]}."""),
    "benchmarks": (["autoresearch/evaluate.py", "autoresearch/champion_scores.json"],
                   """From evaluate.py's BENCHMARKS registry and champion_scores.json, return JSON
{"benchmarks": [{"id": "B<n>", "key": "<exact score key>", "family": "<grouping>",
"measures": "<=12 words", "inputs": "<=8 words", "target": "<=8 words",
"current_score": <float|null>}], "scoring": {"net_score": {...}, "arithmetic_net": {...}},
"registry_location": "<where the registry lives>"}. Cover every benchmark."""),
    "tokens": (["core/fusion.py", "core/train.py"],
               """Trace the ACTUAL token structure of one training example through DeepEarth.encode/
forward. Return JSON {"tokens": [{"token_type", "count_per_example", "dim",
"composed_of", "origin"}], "context_window": {"formula_round0": "..."},
"masking": {...}, "champion_dims": {...}}. Precision over prose."""),
}


def refresh_seed(name):
    files, instr = SEEDS[name]
    content = "\n\n".join(f"=== {f} ===\n" + (REPO / f).read_text(errors="replace")[:150_000]
                          for f in files)
    out = gemini(f"{instr}\n\nSOURCE FILES:\n{content}")
    (ROOT / "seed" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"seed/{name}.json refreshed")


def triage_islands():
    """One batched call: every model-codebase island -> wire-in | delete | keep, with a reason."""
    cg = json.loads((STATE / "callgraph.json").read_text())
    graph = json.loads((STATE / "graph.json").read_text())
    islands = [d for d in cg["defs"] if d["reach"] == "island"
               and d["path"].startswith(("core/", "encoders/", "autoresearch/"))
               and "recipes/" not in d["path"]]
    rows = []
    for d in islands:
        rules = sorted({e["dst"] for e in graph["edges"]
                        if e["src"].rsplit(":", 1)[0] == d["path"] and e["dst"][0] == "R"
                        and (lambda s, t: s[0] <= d["end"] and s[1] >= d["start"])(
                            tuple(map(int, e["src"].rsplit(":", 1)[1].split("-"))), None)})
        rows.append(f"{d['id']} ({d['path']}:{d['start']}-{d['end']}) linked rules: {','.join(rules) or 'none'}")
    src = "\n".join(rows)
    out = gemini(f"""These defs in the DeepEarth model codebase are ISLANDS — never called from any
entry point under the champion config (static call graph). The project mandate: no code off the
critical path — every island must be wired in, deleted, or kept with an explicit reason.
Science rules R1-R32 are the charter; a def named in a rule's mechanism should usually be
wired-in, a convenience helper deleted, a documented external API kept.

ISLANDS:
{src}

Return JSON: {{"triage": [{{"id": "<def id>", "action": "wire-in|delete|keep",
"reason": "<=14 words"}}]}} — one entry per island, decisive.""")
    (STATE / "triage.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "triage": out.get("triage", [])}) + "\n")
    from collections import Counter
    print(f"triage: {len(out.get('triage', []))} islands -> "
          f"{dict(Counter(t['action'] for t in out.get('triage', [])))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--graph-only", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--refresh-seed", choices=[*SEEDS, "all"])
    ap.add_argument("--triage-islands", action="store_true")
    args = ap.parse_args()
    if args.triage_islands:
        return triage_islands()
    if args.refresh_seed:
        for n in (SEEDS if args.refresh_seed == "all" else [args.refresh_seed]):
            refresh_seed(n)
        return
    if not args.loop:
        return run(args.status_only, args.graph_only)
    last = None
    while True:                                        # after-every-PR autonomy
        import subprocess
        cur = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        if cur != last:
            subprocess.run(["python3", "-m", "dashboard.registry"], cwd=REPO)
            run()
            last = cur
        time.sleep(60)


if __name__ == "__main__":
    main()
