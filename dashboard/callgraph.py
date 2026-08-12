"""Static call-graph reachability under the ACTUAL config — proof of integration, not capability.

Crawls every def/class/method in the model codebase, builds the reference graph, and BFS-walks it
from the real entry points (train.py main). Config-gated branches are evaluated against the current
yaml: code behind a false gate is GATED, not live. Code no root reaches is an ISLAND.

Reach classes: live (on the champion train/eval path, gates evaluated) · gated (reachable only
through a branch the config turns off, with the gate key) · data-pipeline / recipes / tests /
tooling (reachable from those roots only) · island (nothing calls it).

    python -m dashboard.callgraph [--config autoresearch/deepcal.yaml]
"""
import argparse, ast, json, subprocess, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PKG = "deepearth"


def modpath(dotted):
    p = dotted.split(".")
    if p[0] == PKG:
        p = p[1:]
    return "/".join(p) + ".py" if p else None


def is_main_guard(n):
    return isinstance(n, ast.If) and isinstance(n.test, ast.Compare) \
        and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__"


class Module:
    def __init__(self, rel, tree):
        self.rel, self.tree = rel, tree
        self.imports = {}                                # local name -> "path.py" | "path.py::Name"
        self.defs = {}                                   # "Class.method"|"func"|"Class" -> node
        self.toplevel, self.cli = [], []                 # module-run statements | __main__-guard statements
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                self._imp(n)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[n.name] = n
            elif isinstance(n, ast.ClassDef):
                self.defs[n.name] = n
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.defs[f"{n.name}.{m.name}"] = m
            elif is_main_guard(n):
                self.cli += n.body
            else:
                self.toplevel.append(n)

    def _imp(self, n):
        if isinstance(n, ast.Import):
            for a in n.names:
                mp = modpath(a.name)
                if mp:
                    self.imports[a.asname or a.name.split(".")[0]] = mp
        else:
            if n.level:                                  # relative: from .x import y
                parts = self.rel.split("/")[:-1]
                parts = parts[:len(parts) - (n.level - 1)] if n.level > 1 else parts
                mod = (n.module or "").replace(".", "/")
                mp = "/".join(p for p in [*parts, mod] if p) + ".py"
            else:
                mp = modpath(n.module or "")
            if not mp:
                return
            for a in n.names:
                self.imports[a.asname or a.name] = f"{mp}::{a.name}"


class Analyzer:
    def __init__(self, config):
        self.cfg = {}                                    # flat config values
        for sect in config.values():
            if isinstance(sect, dict):
                self.cfg.update(sect)
        self.mods = {}                                   # rel -> Module
        for rel in subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"],
                                  capture_output=True, text=True).stdout.split():
            try:
                self.mods[rel] = Module(rel, ast.parse((REPO / rel).read_text(errors="replace")))
            except SyntaxError:
                pass
        self.ids = {}                                    # def id "rel::qual" -> node
        self.methods = {}                                # bare method/attr name -> [ids]
        for rel, m in self.mods.items():
            for q, n in m.defs.items():
                self.ids[f"{rel}::{q}"] = n
                self.methods.setdefault(q.split(".")[-1], []).append(f"{rel}::{q}")
        self.attrvals = self._attr_values()              # "_poll_phylo_weight" -> config value
        self.edges = {}                                  # id -> set of (target_id, gate|None)
        self.pseudo = {}                                 # "(module)"/"(cli)" pseudo-def statement lists

    def _attr_values(self):
        """model._x = m.get("k", d)  and  DeepEarth(k=m.get("k", d)) -> self.k = k  data flows."""
        out = {}
        tr = self.mods.get("autoresearch/train.py")
        if not tr:
            return out
        ctor_kwargs = {}
        for n in ast.walk(tr.tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute) \
                    and (v := self._get_value(n.value)) is not ...:
                out[n.targets[0].attr] = v
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "DeepEarth":
                for kw in n.keywords:
                    if kw.arg and (v := self._get_value(kw.value)) is not ...:
                        ctor_kwargs[kw.arg] = v
        fu = self.mods.get("core/fusion.py")
        init = fu and fu.defs.get("DeepEarth.__init__")
        if init:
            for n in ast.walk(init):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute) \
                        and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "self" \
                        and isinstance(n.value, ast.Name) and n.value.id in ctor_kwargs:
                    out[n.targets[0].attr] = ctor_kwargs[n.value.id]
        return out

    def _get_value(self, node):
        """Evaluate m.get("k", d) / config["s"]["k"]-style reads against the yaml; ... = unknown."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and node.args and isinstance(node.args[0], ast.Constant):
            k = node.args[0].value
            dflt = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
            return self.cfg.get(k, dflt)
        if isinstance(node, ast.Constant):
            return node.value
        return ...

    def _gate(self, test):
        """Evaluate an if-test against config. False -> (True, key). Unknown/true -> (False, None)."""
        neg = False
        while isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            neg, test = not neg, test.operand
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            lv, rv = self._probe(test.left), self._probe(test.comparators[0])
            if lv is not ... and rv is not ...:
                res = {ast.Gt: lambda: lv[1] > rv[1], ast.Lt: lambda: lv[1] < rv[1],
                       ast.GtE: lambda: lv[1] >= rv[1], ast.LtE: lambda: lv[1] <= rv[1],
                       ast.Eq: lambda: lv[1] == rv[1], ast.NotEq: lambda: lv[1] != rv[1]}.get(type(test.ops[0]))
                if res is not None:
                    try:
                        off = (not res()) != neg
                    except TypeError:
                        return False, None
                    return (off, lv[0] or rv[0]) if off else (False, None)
        pv = self._probe(test)
        if pv is not ...:
            off = not bool(pv[1])
            return (off != neg, pv[0]) if (off != neg) else (False, None)
        return False, None

    def _probe(self, node):
        """-> (config key or None, value) | ... if unknown."""
        if isinstance(node, ast.Constant):
            return None, node.value
        if isinstance(node, ast.Attribute) and node.attr in self.attrvals:
            return node.attr.lstrip("_"), self.attrvals[node.attr]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and node.args[1].value in self.attrvals:
            return node.args[1].value.lstrip("_"), self.attrvals[node.args[1].value]
        v = self._get_value(node)
        return (None, v) if v is not ... else ...

    def build(self):
        for did, node in self.ids.items():
            rel, q = did.split("::")
            cls = q.split(".")[0] if "." in q else None
            self.edges[did] = set()
            self._walk_body(node, rel, cls, did, None)
            if isinstance(node, ast.ClassDef):           # instantiation implies __init__ + forward run
                for m in ("__init__", "forward", "__call__", "__len__", "__getitem__"):
                    if f"{q}.{m}" in self.mods[rel].defs:
                        self._add(did, f"{rel}::{q}.{m}", None)
            if getattr(node, "decorator_list", None):    # decorator registration = framework-invoked
                self.edges.setdefault(f"{rel}::(module)", set()).add((did, None))
        for rel, m in self.mods.items():                 # module top-level + __main__ pseudo-defs
            for pq, stmts in (("(module)", m.toplevel), ("(cli)", m.cli)):
                pid = f"{rel}::{pq}"
                self.edges.setdefault(pid, set())
                for s in stmts:
                    self._refs(s, rel, None, pid, None)
                    self._walk_body(s, rel, None, pid, None)
        return self

    def _walk_body(self, node, rel, cls, src, gate):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                off, key = self._gate(child.test)
                g = key if off else gate
                for c in child.body:
                    self._walk_body(c, rel, cls, src, g)
                for c in child.orelse:
                    self._walk_body(c, rel, cls, src, gate)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not child:
                self._walk_body(child, rel, cls, src, gate)   # nested defs belong to enclosing def
                continue
            self._refs(child, rel, cls, src, gate)
            self._walk_body(child, rel, cls, src, gate)

    def resolve(self, mp, name, depth=0):
        """Find rel::name across package __init__ indirection, re-exports, and sys.path-bare imports."""
        cands = [mp, mp[:-3] + "/__init__.py"] if mp.endswith(".py") else [mp]
        cands += [r for r in self.mods if r.endswith("/" + mp)]     # sys.path-inserted bare imports
        for cand in dict.fromkeys(cands):
            if f"{cand}::{name}" in self.ids:
                return f"{cand}::{name}"
            m = self.mods.get(cand)
            if m and depth < 3 and name in m.imports:
                t = m.imports[name]
                if "::" in t:
                    return self.resolve(*t.split("::"), depth + 1)
        return None

    def _refs(self, node, rel, cls, src, gate):
        m = self.mods[rel]
        if isinstance(node, ast.Name) and isinstance(getattr(node, "ctx", None), ast.Load):
            n = node.id
            if n in m.defs:
                self._add(src, f"{rel}::{n}", gate)
            elif n in m.imports and "::" in m.imports[n]:
                if t := self.resolve(*m.imports[n].split("::")):
                    self._add(src, t, gate)
        if isinstance(node, ast.Attribute) and isinstance(getattr(node, "ctx", None), ast.Load):
            a = node.attr
            if isinstance(node.value, ast.Name) and node.value.id == "self" and cls \
                    and f"{cls}.{a}" in m.defs:
                self._add(src, f"{rel}::{cls}.{a}", gate)
            elif isinstance(node.value, ast.Name) and node.value.id in m.imports \
                    and "::" not in m.imports[node.value.id]:
                if t := self.resolve(m.imports[node.value.id], a):
                    self._add(src, t, gate)
            elif a in self.methods:                       # unresolved receiver: by-name, conservative
                for t in self.methods[a]:
                    self._add(src, t, gate)

    def _add(self, src, dst, gate):
        if dst != src:
            self.edges[src].add((dst, gate))

    def reach(self):
        roots = {"live": ["autoresearch/train.py::main", "autoresearch/train.py::train_and_evaluate",
                          "autoresearch/train.py::(cli)"],
                 "data-pipeline": [], "tests": [], "recipes": [], "tooling": []}
        for rel, m in self.mods.items():
            scope = ("data-pipeline" if rel.startswith("data/") else
                     "tests" if rel.startswith("tests/") else
                     "recipes" if rel.startswith("autoresearch/recipes/") else "tooling")
            if rel != "autoresearch/train.py" and m.cli:
                roots[scope].append(f"{rel}::(cli)")
            elif rel != "autoresearch/train.py" and scope != "tooling" and m.toplevel:
                roots[scope].append(f"{rel}::(module)")   # guard-less scripts run at import

            for q in m.defs:
                if q.startswith("cmd_") and rel.startswith("data/"):
                    roots["data-pipeline"].append(f"{rel}::{q}")
                elif q.startswith("test_") and rel.startswith("tests/"):
                    roots["tests"].append(f"{rel}::{q}")
                elif q == "main" and rel.startswith("dashboard/"):
                    roots["tooling"].append(f"{rel}::{q}")

        label, gatekey = {}, {}

        def bfs(seeds, name, skip_gated):
            todo = [s for s in seeds if (s in self.ids or s in self.edges) and s not in label]
            for s in todo:
                label[s] = name
            while todo:
                cur = todo.pop()
                for dst, g in self.edges.get(cur, ()):
                    if g and skip_gated:
                        continue
                    if dst not in label:
                        label[dst] = name
                        if g:
                            gatekey.setdefault(dst, g)
                        todo.append(dst)
                mod = cur.split("::")[0] + "::(module)"  # reaching a def means its module was imported
                if mod not in label and mod in self.edges:
                    label[mod] = name
                    todo.append(mod)

        bfs(roots["live"], "live", True)
        todo = [d for d, l in list(label.items()) if l == "live"]   # gated expansion off the live set
        while todo:
            cur = todo.pop()
            for dst, g in self.edges.get(cur, ()):
                if dst not in label:
                    label[dst] = "gated"
                    gatekey.setdefault(dst, g or gatekey.get(cur, "?"))
                    todo.append(dst)
        for name in ("data-pipeline", "tests", "recipes", "tooling"):
            bfs(roots[name], name, False)

        out = []
        for did, node in sorted(self.ids.items()):
            rel, q = did.split("::")
            kind = "class" if isinstance(node, ast.ClassDef) else "method" if "." in q else "function"
            r = label.get(did, "island")
            if r == "island" and "." in q and q.split(".")[-1].startswith("__"):
                r = label.get(f"{rel}::{q.split('.')[0]}", "island")   # dunders run implicitly with the class
            d = {"id": did, "path": rel, "start": node.lineno, "end": node.end_lineno,
                 "kind": kind, "reach": r}
            if did in gatekey:
                d["gate"] = gatekey[did]
            out.append(d)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "autoresearch" / "deepcal.yaml"))
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    a = Analyzer(config).build()
    defs = a.reach()
    from collections import Counter
    stats = Counter(d["reach"] for d in defs)
    (ROOT / "state").mkdir(exist_ok=True)
    (ROOT / "state" / "callgraph.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": args.config,
         "stats": dict(stats), "defs": defs}) + "\n")
    print(f"callgraph: {len(defs)} defs -> {dict(stats)}")
    for d in defs:
        if d["reach"] == "island" and d["path"].startswith(("core/", "encoders/", "autoresearch/")):
            print(f"  ISLAND {d['id']}  ({d['path']}:{d['start']}-{d['end']})")
    for d in defs:
        if d["reach"] == "gated":
            print(f"  GATED  {d['id']}  [{d.get('gate')}]")


if __name__ == "__main__":
    main()
