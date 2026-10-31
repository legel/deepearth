"""The component study: one run per entry of recipe.COMPONENTS against one control, at one size and step count, same seed,
same data order. The control's second seed sets the noise threshold.

    python ablate.py plan --groups add --gpu 1 > study.sh          the commands
    python ablate.py report --step 8000                            the table, percent change of validation CRPS

A component in the groups input, backbone and head is removed from the recipe; one in add is added to it. A negative number
means lower CRPS with the change. Values within the threshold are not told apart from the control.
"""

import argparse
import json
import sys
from pathlib import Path

from recipe import COMPONENTS, FINAL

OUT = Path("/home/photon/4tb/fluxfill/model")
SIZE = "--d 192 --depth 6 --heads 6 --batch 4 --accum 2 --steps 24000 --stop-at 8000 --eval-every 2000 --workers 1"
CONTROL = "abl2-ctl"
KEYS = ("val_crps", "val_crps_NEE", "val_crps_LE", "val_crps_H", "val_crps_FCH4", "val_cov95")


def flags_of(c) -> str:
    """FINAL minus c.remove (value-carrying flags are removed with their value) plus c.add, then the size."""
    toks, rm = FINAL.split(), c.remove.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] in rm:
            i += 2 if i + 1 < len(toks) and not toks[i + 1].startswith("--") else 1
        else:
            out.append(toks[i]); i += 1
    size = SIZE.split()
    add = c.add.split()
    if "--depth" in add:                  # the schedule's own depth replaces the study's
        j = size.index("--depth"); del size[j:j + 2]
    return " ".join(out + size + add)


def last_eval(name, step):
    p = OUT / name / "log.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in open(p) if '"eval"' in x]
    rows = [r for r in rows if r["step"] == step]
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plan", "report"])
    ap.add_argument("--groups", nargs="+", default=["add", "input", "backbone", "head"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=8000)
    ap.add_argument("--with-control", action="store_true")
    a = ap.parse_args()
    if a.what == "plan":
        py = sys.executable
        print(f"cd {Path(__file__).resolve().parent}\nexport OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES={a.gpu}")
        run = lambda n, f: print(f'[ -f {OUT}/{n}/log.jsonl ] && grep -q \'"step": 8000, "event": "eval"\' {OUT}/{n}/log.jsonl || '
                                 f'{{ taskset -c 6-11 {py} train.py --name {n} {f} --seed {a.seed} > {OUT}/{n}.train.log 2>&1; '
                                 f'echo "== $(date \'+%F %T\') {n} rc=$?"; rm -f {OUT}/{n}/*.pt; }}')
        if a.with_control:
            base = FINAL + " " + SIZE
            run(CONTROL, base)
            print(f'taskset -c 6-11 {py} train.py --name {CONTROL}-s1 {base} --seed 1 > {OUT}/{CONTROL}-s1.train.log 2>&1; rm -f {OUT}/{CONTROL}-s1/*.pt')
        for c in COMPONENTS:
            if c.group in a.groups:
                run(f"abl2-{c.key}", flags_of(c))
        return
    ctl, ctl2 = last_eval(CONTROL, a.step), last_eval(CONTROL + "-s1", a.step)
    assert ctl, "no control at this step"
    print("| Component | Group | " + " | ".join(k.replace("val_", "") for k in KEYS) + " |")
    print("|---|---|" + "---|" * len(KEYS))
    pct = lambda r, k: f"{100 * (r[k] / ctl[k] - 1):+.1f}%" if k in r and k in ctl else ""
    print(f"| control | | " + " | ".join(f"{ctl[k]:.4f}" for k in KEYS) + " |")
    if ctl2:
        print("| second seed (noise) | | " + " | ".join(pct(ctl2, k) for k in KEYS) + " |")
    for c in COMPONENTS:
        r = last_eval(f"abl2-{c.key}", a.step)
        if r:
            print(f"| {c.label} | {c.group} | " + " | ".join(pct(r, k) for k in KEYS) + " |")


if __name__ == "__main__":
    main()
