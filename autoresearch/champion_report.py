"""champion_report.py — the STANDARD way to save champion benchmark scores and format a git-commit report.

Every time a benchmark run becomes the new champion, run this. It (1) reads the run's scores, (2) diffs them
against the previous champion record (autoresearch/champion_scores.json), (3) prints a ready-to-paste commit
message — a headline with the net score BEFORE -> AFTER, then an enumerated per-benchmark BEFORE -> AFTER list —
and (4) with --save, promotes the run to the new champion record (the "before" for the next upgrade).

This makes every champion commit read consistently (see science.md rule 30). Example headline:
    "Latent Clade Attention doubled in size (harmonic 0.0204 -> 0.0231, arith 0.446 -> 0.461)"

Usage:
    # after a benchmark run whose stdout/eval was captured to run.log:
    python -m deepearth.autoresearch.champion_report --log run.log --desc "widen d_model 256->384" --save
    # prints the commit message and promotes run.log's scores to the champion record.
    # omit --save to preview the report without changing the record (e.g. a candidate that did not win).
"""
import re
import json
import argparse
from pathlib import Path

RECORD = Path(__file__).with_name("champion_scores.json")   # the current champion (before for the next run)

try:                                                        # canonical order so EVERY benchmark is listed (inactive ones marked, never silently missing)
    from deepearth.autoresearch.evaluate import BENCHMARKS as _CANON
except Exception:
    _CANON = []


def parse_run(log_path: str) -> dict:
    """Extract {Bxx_name: score}, harmonic (net_score) and arithmetic mean from a train/eval run log."""
    txt = Path(log_path).read_text()
    scores = {}
    # score may be followed by trailing text on the diagnostic lines, e.g. "B24_geo_information_gain 0.593 (net
    # contrib 0.997)" -- match the score after the name, not requiring end-of-line, so B24/B56-B62 are captured.
    for m in re.finditer(r"^\s*(B\d+_\w+)\s+(-?[0-9.]+)(?:\s|$)", txt, re.M):
        scores[m.group(1)] = float(m.group(2))                 # last occurrence wins (final eval)
    try:                                                       # RECOMPUTE the net from scores with the live logic, so
        from deepearth.autoresearch.evaluate import net_score, arithmetic_net   # every champion record is comparable
        return {"scores": scores, "harmonic": float(net_score(scores)), "arithmetic": float(arithmetic_net(scores))}
    except Exception:                                          # fallback: parse whatever the log printed
        h = re.search(r"net_score:\s+([0-9.]+)", txt)
        a = re.search(r"arithmetic mean:\s+([0-9.]+)", txt)
        return {"scores": scores, "harmonic": float(h.group(1)) if h else None,
                "arithmetic": float(a.group(1)) if a else None}


def _n(x):                                                     # benchmark sort key: B<number>
    return int(re.match(r"B(\d+)", x).group(1))


def _f(v):
    return f"{v:.3f}" if v is not None else "  -  "


def format_commit(new: dict, old: dict | None, desc: str, config: str = "") -> str:
    ns = new["scores"]
    os_ = (old or {}).get("scores", {})
    oh, nh = (old or {}).get("harmonic"), new["harmonic"]
    oa, na = (old or {}).get("arithmetic"), new["arithmetic"]
    if old is None:                                            # first record -> a baseline report (no "before")
        head = f"{desc} (BASELINE: harmonic {_f(nh)}, arith {_f(na)})"
    else:
        head = f"{desc} (harmonic {_f(oh)} -> {_f(nh)}, arith {_f(oa)} -> {_f(na)})"
    lines = [head, ""]
    if config:
        lines += [config, ""]
    lines.append("All benchmarks (before -> after):" if old is not None else "All benchmarks (baseline):")
    allb = sorted(set(_CANON) | set(ns) | set(os_), key=_n)   # EVERY declared benchmark + anything seen; none missing
    for i, name in enumerate(allb, 1):
        after = ns.get(name)
        before = os_.get(name)
        if after is None:                                     # not produced by THIS run's holdout (e.g. B25/B31 forecast need a temporal run)
            if before is not None:                            # carry the champion-record value (from its proper holdout) rather than blank it
                lines.append(f"{i:>2}. {name}: {_f(before)} (carried; its holdout not re-run here)")
            else:
                lines.append(f"{i:>2}. {name}: inactive (needs its holdout: e.g. B25/B31 = temporal)")
        elif old is None:
            lines.append(f"{i:>2}. {name}: {_f(after)}")
        else:
            d = f"{after - before:+.3f}" if before is not None else "  new  "
            flag = "" if before is None else ("  ^" if after > before + 1e-9 else ("  v" if after < before - 1e-9 else "  ="))
            lines.append(f"{i:>2}. {name}: {_f(before)} -> {_f(after)} ({d}){flag}")
    if old is not None:                                        # regression guard summary (science.md: NO metric regressing)
        reg = [f"{n} ({os_[n]:.3f}->{ns[n]:.3f})" for n in sorted(ns, key=_n)
               if n in os_ and ns[n] < os_[n] - 0.005]
        lines += ["", f"REGRESSIONS (>0.005): {', '.join(reg) if reg else 'none'}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="run log (train + eval stdout) to score")
    ap.add_argument("--desc", required=True, help="one-line result description for the commit headline")
    ap.add_argument("--config", default="", help="optional config/hardware note line")
    ap.add_argument("--save", action="store_true", help="promote this run to the champion record")
    a = ap.parse_args()
    new = parse_run(a.log)
    if not new["scores"]:
        raise SystemExit(f"no Bxx scores found in {a.log}")
    old = json.loads(RECORD.read_text()) if RECORD.exists() else None
    print(format_commit(new, old, a.desc, a.config))
    if a.save:
        RECORD.write_text(json.dumps({"label": a.desc, "config": a.config, "harmonic": new["harmonic"],
                                      "arithmetic": new["arithmetic"], "scores": new["scores"]}, indent=2))
        print(f"\n[champion_scores.json updated: {len(new['scores'])} benchmarks, "
              f"harmonic {new['harmonic']}, arith {new['arithmetic']}]")


if __name__ == "__main__":
    main()
