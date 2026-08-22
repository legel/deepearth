"""Deterministic extraction: repo tree + blocks + rules + benchmarks -> state/registry.json.

    python -m dashboard.registry
"""
import ast, hashlib, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STATE = ROOT / "state"
SEED = ROOT / "seed"
TEXT_EXT = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".html", ".css", ".js",
            ".cu", ".cpp", ".h", ".R", ".tex", ".cfg", ".toml", ".nwk"}
SKIP = ("dashboard/state/", "dashboard/runs/", "dashboard/seed/")


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def blocks_py(path, text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [{"name": path.name, "kind": "file", "start": 1, "end": text.count("\n") + 1}]
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append({"name": n.name, "kind": type(n).__name__.replace("Def", "").lower(),
                        "start": n.lineno, "end": n.end_lineno})
    covered = {l for b in out for l in range(b["start"], b["end"] + 1)}
    rest = [i for i in range(1, text.count("\n") + 2) if i not in covered]
    if rest:
        out.insert(0, {"name": "(module)", "kind": "module", "start": rest[0], "end": rest[-1]})
    return out


def blocks_md(path, text):
    lines = text.splitlines()
    heads = [(i + 1, l.lstrip("#").strip()) for i, l in enumerate(lines) if l.startswith("#")]
    if not heads:
        return [{"name": path.name, "kind": "doc", "start": 1, "end": len(lines)}]
    out = []
    for j, (start, name) in enumerate(heads):
        end = heads[j + 1][0] - 1 if j + 1 < len(heads) else len(lines)
        out.append({"name": name[:80], "kind": "section", "start": start, "end": end})
    return out


def build():
    files, blocks = [], []
    for rel in _git("ls-files").splitlines():
        if any(rel.startswith(s) for s in SKIP):
            continue
        p = REPO / rel
        ext = p.suffix
        if ext not in TEXT_EXT or not p.exists():
            files.append({"path": rel, "kind": "binary", "lines": 0})
            continue
        text = p.read_text(errors="replace")
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        files.append({"path": rel, "kind": "text", "lines": text.count("\n") + 1, "sha": sha})
        bs = blocks_py(p, text) if ext == ".py" else blocks_md(p, text) if ext == ".md" else \
            [{"name": p.name, "kind": "file", "start": 1, "end": text.count("\n") + 1}]
        for b in bs:
            blocks.append({"id": f"{rel}:{b['start']}-{b['end']}", "path": rel, **b})

    seed = {p.stem: json.loads(p.read_text()) for p in SEED.glob("*.json")}
    rules = seed.get("science", {}).get("rules", [])
    benches = seed.get("benchmarks", {}).get("benchmarks", [])
    champ = REPO / "autoresearch" / "champion_scores.json"
    record = json.loads(champ.read_text()) if champ.exists() else {}
    scores = record.get("scores", {})
    for b in benches:
        b["current_score"] = scores.get(b["key"])

    scoring = seed.get("benchmarks", {}).get("scoring", {})
    if "net_score" in scoring:
        scoring["net_score"]["value"] = record.get("harmonic")
    if "arithmetic_net" in scoring:
        scoring["arithmetic_net"]["value"] = record.get("arithmetic")

    head = _git("log", "-1", "--format=%H").strip()
    reg = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "head": head,
           "counts": {"files": len(files), "blocks": len(blocks), "rules": len(rules),
                      "benchmarks": len(benches)},
           "files": files, "blocks": blocks, "rules": rules, "benchmarks": benches,
           "scoring": scoring,
           "tokens": seed.get("tokens"), "data_schema": seed.get("data_schema"),
           "training": seed.get("training")}
    STATE.mkdir(exist_ok=True)
    (STATE / "registry.json").write_text(json.dumps(reg) + "\n")
    print(f"registry: {len(files)} files, {len(blocks)} blocks, {len(rules)} rules, "
          f"{len(benches)} benchmarks @ {head[:8]}")


if __name__ == "__main__":
    build()
