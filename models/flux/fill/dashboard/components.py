"""Component study of FluxFill as dashboard data: every entry of model/recipe.py COMPONENTS with a finished run at step
8,000 against the control of the same seed (model/ablate.py names the runs), and the size probes at matched steps.
Results tab and report Table 4 read this; both update with each build."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = [("val_crps", "crps"), ("val_crps_NEE", "NEE"), ("val_crps_LE", "LE"), ("val_crps_H", "H"),
        ("val_crps_FCH4", "FCH4"), ("val_cov95", "cov95")]
STEP = 8000
SIZES = [("M", "width 320, depth 10", "v4-d320-48k"), ("XL", "width 640, depth 16", "scale-XL-lr3e-4"),
         ("XL", "width 640, depth 16", "scale-XL-lr1.5e-4"), ("XL", "width 640, depth 16", "scale-XL-lr6e-4")]
SIZE_STEPS = (2000, 4000, 6000)
# The decision on a component, by the rule of the study: adopted only when it holds on a second seed.
DECISIONS = {"var_mean": "Adopted", "gate": "Adopted", "mot_all": "Combination decides; FCH4 differs by seed",
             "mot_ffn": "No gain", "rope_half": "No gain", "tttv": "No gain", "asinh": "Rejected", "fgn": "Rejected: coverage",
             "swiglu": "Second seed queued", "quantile": "Second seed queued", "rmsnorm": "No gain", "plr": "No gain",
             "time_prithvi": "No gain", "time_sinr": "No gain"}


def recipe():
    spec = importlib.util.spec_from_file_location("recipe", HERE.parent / "model" / "recipe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recipe"] = mod
    spec.loader.exec_module(mod)
    return mod


def evals(run_dir: Path) -> dict:
    p = run_dir / "log.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(x) for x in p.read_text().splitlines() if '"eval"' in x]
    return {r["step"]: r for r in rows}


def collect(model_root: Path) -> dict:
    ctl = [evals(model_root / n).get(STEP) for n in ("abl2-ctl", "abl2-ctl-s1")]
    pct = lambda r, c, k: 100 * (r[k] / c[k] - 1) if k in r and k in c else None
    out = {"step": STEP, "control": None, "noise": None, "rows": [], "total": 0, "sizes": [], "size_steps": list(SIZE_STEPS)}
    comps = recipe().COMPONENTS
    out["total"] = len(comps)
    if ctl[0]:
        out["control"] = {n: ctl[0].get(k) for k, n in KEYS}
    if all(ctl):
        out["noise"] = {n: pct(ctl[1], ctl[0], k) for k, n in KEYS}
    for c in comps:
        seeds = [(evals(model_root / f"abl2-{c.key}").get(STEP), ctl[0]), (evals(model_root / f"abl2-{c.key}-s1").get(STEP), ctl[1])]
        seeds = [(r, b) for r, b in seeds if r and b]
        if not seeds:
            continue
        row = {"key": c.key, "label": c.label, "group": c.group, "seeds": len(seeds), "decision": DECISIONS.get(c.key, "")}
        for k, n in KEYS:
            v = [pct(r, b, k) for r, b in seeds]
            v = [x for x in v if x is not None]
            row[n] = sum(v) / len(v) if v else None
        out["rows"].append(row)
    for size, shape, run in SIZES:
        cfg, ev = model_root / run / "config.json", evals(model_root / run)
        if not cfg.exists() or not ev:
            continue
        lr = json.loads(cfg.read_text())["lr"]
        points = {s: {"crps": ev[s]["val_crps"], "flux": ev[s]["val_crps_flux"]} for s in SIZE_STEPS if s in ev}
        out["sizes"].append({"size": size, "shape": shape, "lr": lr, "points": points})
    return out


def table_html(d: dict) -> str:
    """Report Table 4."""
    f = lambda v: "–" if v is None else "0.0%" if abs(v) < 0.05 else f"{v:+.1f}%"
    h = ['<div class="table-wrap"><table class="rep-table"><thead><tr><th>Added to the control</th><th class="num">Seeds</th>'
         '<th class="num">CRPS, all</th><th class="num">NEE</th><th class="num">LE</th><th class="num">H</th><th class="num">FCH4</th>'
         '<th class="num">95% cov.</th><th>Decision</th></tr></thead><tbody>']
    names = ("crps", "NEE", "LE", "H", "FCH4", "cov95")
    if d["noise"]:
        h.append('<tr class="ours"><td>Second control seed (noise)</td><td class="num"></td>' + "".join(f'<td class="num">{f(d["noise"][n])}</td>' for n in names) + "<td></td></tr>")
    for r in d["rows"]:
        h.append(f'<tr><td>{r["label"]}</td><td class="num">{r["seeds"]}</td>' + "".join(f'<td class="num">{f(r[n])}</td>' for n in names) + f'<td>{r["decision"]}</td></tr>')
    return "".join(h) + "</tbody></table></div>"
