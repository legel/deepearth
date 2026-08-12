"""One command, coherent state: registry -> callgraph -> flow -> audit, in dependency order.

Prevents artifact skew (a callgraph at one head joined against a registry at another).
observations/trace/reconstruct are heavier and data/GPU-bound; run them explicitly.

    python -m dashboard.refresh [--no-audit]
"""
import argparse, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audit", action="store_true", help="skip the Gemini audit entirely")
    ap.add_argument("--graph-only", action="store_true",
                    help="audit connectivity only (cached; cheap per commit), skip status")
    args = ap.parse_args()
    steps = [("registry", []), ("callgraph", []), ("flow", [])]
    if not args.no_audit:
        steps.append(("audit", ["--graph-only"] if args.graph_only else []))
    for s, extra in steps:
        print(f"== dashboard.{s}", flush=True)
        r = subprocess.run([sys.executable, "-m", f"dashboard.{s}", *extra], cwd=REPO)
        if r.returncode:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
