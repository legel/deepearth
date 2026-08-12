"""Zero-code-change run tracking. Wraps train.py, tees stdout, emits runs/<id>.jsonl.

    python -m dashboard.tracker [train.py args...]        # e.g. autoresearch/deepcal.yaml --tag exp1

train.py is untouched: its stdout is already parseable by design (science.md 19).
Every line passes through unchanged; matched lines also become logger events.
"""
import os, re, subprocess, sys
from pathlib import Path
from dashboard.logger import RunLogger

REPO = Path(__file__).resolve().parent.parent

PATTERNS = [
    ("step", re.compile(r"^\s*step (\d+) loss ([-\d.eE]+) \[(\d+)s\]"),
     lambda m: {"step": int(m[1]), "loss": float(m[2]), "elapsed": int(m[3])}),
    ("startup", re.compile(r"^(\S[^:]*): (\d+) observations, ([\d.]+)M parameters, train (\d+) / held-out regions (\d+)"),
     lambda m: {"name": m[1], "observations": int(m[2]), "params_m": float(m[3]),
                "train": int(m[4]), "test": int(m[5])}),
    ("trained", re.compile(r"^trained (\d+) steps in (\d+)s"),
     lambda m: {"steps": int(m[1]), "seconds": int(m[2])}),
    ("transfer", re.compile(r"^held-out regions \(conditioning on (.*?)\): (.+)"),
     lambda m: {"given": m[1], "scores": {k: float(v) for k, v in
                (p.rsplit(" ", 1) for p in m[2].split(" | "))}}),
    ("bench", re.compile(r"^\s{2}(B\d\S*)\s+([\d.]+)\s*$"),
     lambda m: {"key": m[1], "score": float(m[2])}),
    ("net", re.compile(r"^NET SCORE.*: ([\d.]+)"), lambda m: {"net_score": float(m[1])}),
    ("arith", re.compile(r"^\s*\(arithmetic mean: ([\d.]+)\)"), lambda m: {"arithmetic": float(m[1])}),
    ("tag", re.compile(r"^tag:\s+(\S+)"), lambda m: {"tag": m[1]}),
    ("vram", re.compile(r"^peak_vram_mb:\s+([\d.]+)"), lambda m: {"peak_vram_mb": float(m[1])}),
]


def parse(line):
    for t, rx, fn in PATTERNS:
        if m := rx.match(line):
            return t, fn(m)
    return None, None


def main():
    args = sys.argv[1:]
    tag = args[args.index("--tag") + 1] if "--tag" in args else \
        next((Path(a).stem for a in args if a.endswith((".yaml", ".yml"))), "run")
    log = RunLogger(tag, config={"argv": args})
    env = {**os.environ, "PYTHONPATH": f"{REPO.parent}:{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.Popen([sys.executable, "autoresearch/train.py", *args], cwd=REPO, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    bench, final = {}, {}
    for line in proc.stdout:
        print(line, end="", flush=True)                    # unchanged passthrough
        t, d = parse(line)
        if t == "bench":
            bench[d["key"]] = d["score"]
        elif t in ("net", "arith", "tag", "vram"):
            final.update(d)
        elif t:
            log.event(t, **d)
    proc.wait()
    log.final({"benchmarks": bench, **final}, exit_code=proc.returncode)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
