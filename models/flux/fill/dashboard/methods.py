"""Data of the dashboard's Methods and Report tabs, generated from the code, the run directories and the docs.

- Baseline cards: the method's description, citation, code source and pinned version, and the hyperparameters read
  from the module-level constants of baselines/*.py (by parsing the source, so a change in the code changes the card).
- FluxFill experiments: one row per run directory in the model root, from its config.json and log.jsonl, the
  parameter counts of param_counts.py, and the experiment id and purpose of model/NOTES.md (E1 to E20; the registry
  below records the run of each id, and unknown runs fall back to the first NOTES entry that names them).
- References: docs/REFERENCES.md, numbered in order of appearance in that file.
- Report: dashboard/report_draft.html with {{name}} values from the dataset files and [[Key]] citations.
Standard library only (build.py runs under the system Python).
"""

import ast
import datetime as dt
import html
import json
import re
import unicodedata
from pathlib import Path

FILL = Path(__file__).resolve().parent.parent
BASE = FILL / "baselines"


# ------------------------------------------------------------------------------------------- constants from code

def _eval(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(x, env) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, env)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return {ast.Mult: lambda: a * b, ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Div: lambda: a / b,
                ast.Pow: lambda: a ** b}[type(node.op)]()
    if isinstance(node, ast.Call):
        f = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if f == "dict":
            return {k.arg: _eval(k.value, env) for k in node.keywords}
        if f in ("array", "asarray", "tuple", "list") and node.args:
            return _eval(node.args[0], env)
    raise ValueError("not a literal")


def constants(path: Path) -> dict:
    """Module-level NAME = literal assignments (dicts, dict(...), lists, numbers, strings, simple arithmetic)."""
    env = {}
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        for t in targets:
            if isinstance(t, ast.Name):
                try:
                    env[t.id] = _eval(node.value, env)
                except (ValueError, KeyError, TypeError):
                    pass
            elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for n, v in zip(t.elts, node.value.elts):
                    try:
                        env[n.id] = _eval(v, env)
                    except (ValueError, KeyError, TypeError, AttributeError):
                        pass
    return env


def kv(d: dict, keys=None) -> list:
    return [[k, fmt(v)] for k, v in d.items() if keys is None or k in keys]


def fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v)
    return str(v)


# ------------------------------------------------------------------------------------------- baselines

def baselines(counts: dict) -> list:
    clim = constants(BASE / "climatology.py")
    irv = constants(BASE / "irvin_rf.py")
    vek = constants(BASE / "vekuri_de.py")
    liu = constants(BASE / "liu_xgb.py")
    pp = constants(BASE / "pypots_imputers.py")
    chr2 = constants(BASE / "chronos2.py")
    bc = counts.get("baselines", {})
    hp = vek["HPARAMS"]
    d_in = len(vek["DRIVERS"]) + 5
    h = hp["hidden"]
    vek_params = (d_in * h + h) + (h * h + h) + (h * 2 + 2)

    def n(k):
        return f"{bc[k]['n_params']:,}" if k in bc else "not counted"

    sa, cs = bc.get("saits", {}), bc.get("csdi", {})
    out = [
        dict(id="climatology", name="Climatology", ref=[], code="baselines/climatology.py",
             version="this repository", cls="Empirical distribution; no model is fitted.",
             hyper=[["pool", f"same half-hour of day, within ±{clim['HALF_WIDTH_DAYS']} days of the same day of year"],
                    ["years", "every other calendar year of the site; the window's own years excluded"],
                    ["samples", "64, drawn with replacement from the pool"]],
             inputs="The variable's own measured values at the site. No drivers.",
             training="The site's other calendar years.",
             uncertainty="The empirical spread of the pool; samples independent per half-hour.",
             limits="Fills nothing at a site without other years. Does not respond to the weather of the window. "
                    "Sums assume independent half-hours."),
        dict(id="mds", name="Marginal distribution sampling (MDS)", ref=["Reichstein2005", "Wutzler2018"],
             code="REddyProc 1.3.2 in R 4.3 (REddyProc needs R below 4.4); baselines/mds.R, baselines/mds.py",
             version="REddyProc 1.3.2", cls="Look-up of similar meteorological conditions within moving time windows.",
             hyper=[["call", "sMDSGapFill(variable, FillAll = TRUE), REddyProc defaults"],
                    ["drivers", "Rg = SW_IN (clipped at 0), Tair = TA, VPD; ERA5 where the tower value is hidden or "
                                "missing"],
                    ["minimum", "3 visible days of the variable in the window, else left unfilled"],
                    ["variables", fmt(constants(BASE / "mds.py")["FILL"])]],
             inputs="The visible part of the 180-day window only, as MDS runs on a site-year.",
             training="None: the look-up uses the window's visible values.",
             uncertainty="Gaussian with REddyProc's _fsd (the standard deviation of the values averaged), "
                         "independent per half-hour.",
             limits="Leaves gaps of 90 and 180 days unfilled. Fills fluxes only."),
        dict(id="irvin_rf", name="Random forest ensemble", ref=["Irvin2021"],
             code="github.com/stanfordmlgroup/methane-gapfill-ml; baselines/irvin_rf.py",
             version="commit 473db3e", cls="scikit-learn RandomForestRegressor, an ensemble of forests.",
             hyper=kv(irv["RF_PARAMS"]) + [["members", str(irv["N_MEMBERS"])],
                                          ["member data", "bootstrap resample of whole days"],
                                          ["predictors", "yearly sin, yearly cos, days since 2000-01-01, " +
                                           fmt(irv["DRIVERS"])],
                                          ["missing predictors", "training mean"]],
             inputs="Time features and the drivers " + fmt(irv["DRIVERS"]) + "; ERA5 where the tower driver is "
                    "hidden or missing.",
             training="The site's other calendar years plus the visible part of the window.",
             uncertainty="Laplace with location the member mean and scale the member standard deviation / 2, times a "
                         "Platt factor fitted on held-out 7-day blocks holding 10% of the context (never on the "
                         "window).",
             limits="One fixed configuration from the released search space (the release runs RandomizedSearchCV "
                    "over 20 configurations). The released time feature `delta` is corrected. Fluxes only; samples "
                    "independent per half-hour."),
        dict(id="vekuri_de", name="Deep ensemble, Laplace likelihood", ref=["Vekuri2025"],
             code="github.com/hvekuri/nn_gapfilling_unc (Keras); PyTorch port in baselines/vekuri_de.py",
             version="commit 7edd07a",
             cls=f"{hp['members']} multilayer perceptrons, {d_in} inputs, 2 hidden layers of {h} ReLU units, 2 "
                 f"outputs (location, log scale); {vek_params:,} parameters per member.",
             hyper=[["learning rate", fmt(hp["lr"]) + " (Adam, Keras update rule)"], ["batch", str(hp["batch"])],
                    ["L2", fmt(hp["l2"])], ["epochs", f"at most {hp['epochs']}, early stopping patience "
                                                       f"{hp['patience']}"],
                    ["validation split", f"{int(100 * hp['val_frac'])}% random per member"],
                    ["members", str(hp["members"])]],
             inputs=fmt(vek["DRIVERS"]) + ", DOY, Month_sin, Month_cos, Hour_sin, Hour_cos, standardised.",
             training="The site's other calendar years plus the visible part of the window.",
             uncertainty="Mixture of the members' Laplace distributions: aleatoric (Laplace scale) plus epistemic "
                         "(spread of member means). Each sample follows one member's mean trajectory, so sums carry "
                         "the between-member spread.",
             limits="Fluxes only. Aleatoric noise independent per half-hour."),
        dict(id="liu_xgb", name="XGBoost with vegetation state", ref=["Liu2025"],
             code="github.com/YujieLiu666/fluxgapfill_XGB_vs_MDS; baselines/liu_xgb.py",
             version="commit a9a6eeb", cls="XGBRegressor, gbtree, hist, multi-quantile objective.",
             hyper=kv(liu["XGB_PARAMS"]) + [["objective", "reg:quantileerror"],
                                           ["quantile levels", fmt(liu["LEVELS"])]],
             inputs="Days since 2000-01-01, DOY sin and cos, " + fmt(liu["DRIVERS"]) + "; the first "
                    f"{liu['N_PC']} principal components of the site's Clay embeddings of clear chips (clear "
                    f"fraction ≥ {liu['CLEAR_MIN']}, interpolated over at most {liu['MAX_INTERP_DAYS']} days) in place "
                    "of PhenoCam GCC where the site has a Clay store.",
             training="The site's other calendar years plus the visible part of the window.",
             uncertainty="Quantiles from multi-quantile regression (the release has none), fitted in-sample; "
                         "samples by inverse-CDF draws, independent per half-hour.",
             limits="One fixed configuration inside the released search space (the release runs BayesSearchCV with "
                    "up to 5,100 trials). SW_IN stands in for PPFD. Fluxes only."),
        dict(id="saits", name="SAITS", ref=["Du2023"], code="PyPOTS 1.5 _SAITS core; baselines/pypots_imputers.py",
             version="PyPOTS 1.5",
             cls=f"Self-attention imputation transformer; {n('saits')} parameters.",
             hyper=kv(pp["SAITS_CFG"]) + [["steps", f"{sa.get('steps', '?')}, best at {sa.get('best_step', '?')}"],
                                         ["tokens", f"{pp['WINDOW_DAYS']} day-tokens of {sa.get('n_variables', 22)} "
                                                    f"variables × 48 half-hours + 2 day-of-year features"]],
             inputs="Every benchmark variable of the window (tower and ERA5), standardised with training statistics.",
             training=f"{sa.get('n_train_sites', '?')} training sites of the frozen split; training masks: all tower "
                      "rows over a block (p 0.5), one variable over a block (p 0.2), 20% at random (p 0.3).",
             uncertainty="Gaussian with the root mean square residual per variable and gap class on calibration "
                         "sites (ours, not part of SAITS); independent per half-hour.",
             limits="Point imputer; its spread is added afterwards."),
        dict(id="csdi", name="CSDI", ref=["Tashiro2021", "Du2023"], code="PyPOTS 1.5 _CSDI core; "
             "baselines/pypots_imputers.py", version="PyPOTS 1.5",
             cls=f"Conditional score-based diffusion imputer; {n('csdi')} parameters.",
             hyper=kv(pp["CSDI_CFG"]) + [["steps", f"{cs.get('steps', '?')}, best at {cs.get('best_step', '?')}"],
                                        ["chunk", f"{pp['CHUNK']} half-hours (6 days)"], ["samples", "64"]],
             inputs="Every benchmark variable plus hour of day and day of year (sine, cosine), half-hourly.",
             training=f"{cs.get('n_train_sites', '?')} training sites of the frozen split; the same training masks "
                      "as SAITS.",
             uncertainty="64 diffusion samples, joint within each 6-day chunk, independent between chunks; sampling "
                         "under float16 autocast.",
             limits="Attention over the full 8,640 half-hours does not fit the 2 GB GPU budget, so windows are "
                    "split into 6-day chunks."),
        dict(id="chronos2", name="Chronos-2, zero-shot", ref=["Ansari2025"],
             code="chronos-forecasting 2.3.2; weights amazon/chronos-2; baselines/chronos2.py",
             version="snapshot " + bc.get("chronos2", {}).get("snapshot", "29ec376")[:7],
             cls=f"Pretrained time-series foundation model; {n('chronos2')} parameters; no fine-tuning.",
             hyper=[["context", f"{chr2['CONTEXT_STEPS']} half-hours before the gap"],
                    ["covariates", fmt(chr2["ERA5"]) + ", hour and day of year (sine, cosine)"],
                    ["targets", "fluxes jointly as one group; meteorology per variable"],
                    ["precision", "bfloat16"], ["quantiles", "21, from 0.01 to 0.99"]],
             inputs="The history before each hidden block (or after it, time-reversed, when there is no history), "
                    "with the ERA5 drivers as known covariates.",
             training="None (pretrained by its authors).",
             uncertainty="21 predicted quantiles, sampled by inverse-CDF draws, independent per half-hour.",
             limits="Gap filling posed as forecasting. Blocks with no day of observations on either side are left "
                    "unfilled, which is most 180-day gaps."),
    ]
    return out


# ------------------------------------------------------------------------------------------- experiments

REGISTRY = {
    # run directory: (experiment ids, purpose, outcome); from model/NOTES.md
    "overfit2": ("E1", "Overfit 2 training windows with fixed masks to find bugs before long runs.",
                 "Loss fell monotonically; hidden-value MAE 0.19 to 0.11 standard deviations."),
    "overfit4": ("E1", "Overfit 4 training windows for 4,000 steps.",
                 "Hidden-value CRPS 0.054 standard deviations; 95% coverage 0.95."),
    "v0a": ("E2", "Student-t head only: control arm of E2.",
            "Stopped after the matched-step comparison at step 1,000 (CRPS 0.2739)."),
    "v0b-flow": ("E2, E3", "Student-t head plus flow head on one encoder: does the flow head change the Student-t "
                           "head's validation CRPS?",
                 "No degradation (0.2720 at step 1,000); over-confidence at unseen sites found by step 10,000 (E3)."),
    "v0c-reg": ("E4", "Dropout 0.1 and the site location hidden in half of the windows, for over-confidence at "
                      "unseen sites.", "Stopped at step 2,200; site memory (E6) followed."),
    "v0e-mem": ("E6", "Site memory: mean, spread and presence of the site's other years at each half-hour, for the "
                      "long-gap deficit of E5.",
                "Adopted: CRPS skill against climatology +0.17 against +0.03 without memory (18 windows)."),
    "v1-frozen.restart1": ("E8", "First attempt of E8.", "Restarted at step 2,000 with a capped memory cache."),
    "v1-frozen": ("E8, E10", "First model on the frozen split: memory, flow head, land drivers, site weights. Arm A "
                             "of the E10 comparison.", "Stopped at step 6,000 as planned; arm B had lower CRPS at every matched step."),
    "v1-frozen-memres": ("E10, E11", "Heads predict relative to the site-memory mean, for the NEE shrinkage "
                                     "diagnosed in E10. Arm B of E10.",
                         "Adopted. Step 22,000 is the checkpoint scored as FluxFill v0 and v1."),
    "v1b-imgnll": ("E13", "Imagery head trained on a Gaussian negative log-likelihood in place of MSE.",
                   "Tower CRPS unchanged (0.1711 against 0.1710); imagery cosine 0.419 to 0.710."),
    "v1c-mix3": ("E17, E19", "Mixture of 3 Student-t per half-hour, for the shape of the long-gap NEE marginal "
                             "(E17).", "Not adopted: closes 19% of the long-gap NEE deficit; 95% coverage 0.879."),
    "v1d-local": ("E18", "Local mean diurnal course of the visible neighbouring days as the residual base, for NEE "
                         "at short gaps.", "Not adopted as is: NEE 1-day CRPS 13% lower, H and LE worse."),
    "v1e-gate": ("E20", "Learned gate between the local and the memory base, inside a 2-component Student-t "
                        "mixture.", ""),
}
EVALUATIONS = [
    ("E5", "Validation scoring of v0b-flow step 5,000 against climatology.",
     "Skill 0.00 overall; the deficit grows with gap length for site-specific quantities. Led to site memory."),
    ("E7", "Joint samples by masked generation (v1) against marginals (v0).",
     "Window-sum 95% coverage 0.07-0.33 (v0) to 0.40-0.53 (v1)."),
    ("E9", "First frozen-split validation scores, step 2,000.", "CRPS skill +0.066 against climatology."),
    ("E12", "Imagery removed at inference, 8 windows at 2 sites.", "CRPS ratio 1.002; 7 training sites had imagery, "
                                                                   "so this is not yet a statement about imagery."),
    ("E14", "FCH4 bias diagnosis.", "The bias comes from 2 sites whose only FCH4 years are the target window."),
    ("E15", "Head-to-head with the 8 baselines on identical windows.", "The tables on the Results tab."),
    ("E16", "Calibration of the flow head.", "Under-dispersed before feedback; feedback over 16 rounds adds NEE "
                                              "error."),
    ("E17", "What climatology captures on NEE at long gaps.", "The deficit is the form of the marginal: location "
                                                              "drift, mean against median, and shape."),
]


def _notes_fallback(run: str, notes: str):
    pat = re.compile(r"(?<![\w.-])" + re.escape(run) + r"(?![\w-])")
    for line in notes.splitlines():
        if pat.search(line):
            m = re.search(r"\bE(\d+)\b", line)
            if m:
                return f"E{m.group(1)}", re.sub(r"\s+", " ", line.strip("- ")).split(". ")[0][:200] + ".", ""
    return "", "", ""


def experiments(model_root: Path, counts: dict, notes_path: Path) -> dict:
    notes = notes_path.read_text() if notes_path.exists() else ""
    rows = []
    now = dt.datetime.now().timestamp()
    for d in sorted(model_root.iterdir(), key=lambda p: p.stat().st_mtime if (p / "config.json").exists() else 0):
        cfg_p, log_p = d / "config.json", d / "log.jsonl"
        if not cfg_p.exists():
            continue
        c = json.loads(cfg_p.read_text())
        last, best, nparams_log = 0, None, None
        if log_p.exists():
            for line in log_p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = max(last, int(r.get("step", 0)))
                nparams_log = r.get("n_params", nparams_log)
                if r.get("event") == "eval" and r.get("val_crps") is not None:
                    if best is None or r["val_crps"] < best[0]:
                        best = (r["val_crps"], int(r["step"]), r.get("val_cov95"), r.get("val_crps_flux"))
        target = c.get("stop_at") or c.get("steps")
        age = now - log_p.stat().st_mtime if log_p.exists() else 1e9
        status = "complete" if last >= target else ("running" if age < 1200 else "stopped")
        heads = "Student-t"
        if c.get("gated_base"):
            heads = "2-component gated Student-t"
        elif c.get("components", 1) > 1:
            heads = f"{c['components']}-component Student-t mixture"
        if c.get("flow"):
            heads += " + flow matching"
        flags = []
        for k, lab in (("memory", "site memory"), ("mem_residual", "memory residual"), ("local_base", "local base"),
                       ("gated_base", "gated base"), ("img_nll", "imagery NLL"), ("no_imagery", "no imagery")):
            if c.get(k):
                flags.append(lab)
        if c.get("dropout"):
            flags.append(f"dropout {c['dropout']:g}")
        if c.get("site_drop"):
            flags.append(f"site drop {c['site_drop']:g}")
        if c.get("imagery_sites") and c["imagery_sites"] != "all":
            flags.append(f"imagery at {len(c['imagery_sites'].split(','))} sites")
        if c.get("mode") == "overfit":
            flags.append(f"overfit {c.get('n_overfit')} windows")
        eid, purpose, outcome = REGISTRY.get(d.name, None) or _notes_fallback(d.name, notes) or ("", "", "")
        n_sites = c.get("n_train_sites") or len(c.get("train_sites", []))
        npar = nparams_log or counts.get("runs", {}).get(d.name, {}).get("n_params")
        rows.append(dict(run=d.name, exp=eid, purpose=purpose, outcome=outcome, n_params=npar, d=c.get("d"),
                         depth=c.get("depth"), heads_n=c.get("heads"), head=heads, flags=flags,
                         split=c.get("split"), train_sites=n_sites, val_sites=len(c.get("val_sites", [])),
                         steps=last, target=target, batch=c.get("batch", 0) * c.get("accum", 1), lr=c.get("lr"),
                         best_crps=None if best is None else best[0], best_step=None if best is None else best[1],
                         best_cov95=None if best is None else best[2], best_crps_flux=None if best is None else best[3],
                         status=status, mode=c.get("mode")))
    scored = counts.get("scored")
    return {"runs": rows, "evaluations": [dict(exp=e, purpose=p, outcome=o) for e, p, o in EVALUATIONS],
            "scored": scored}


# ------------------------------------------------------------------------------------------- references

def references(path: Path) -> list:
    """Bullets of REFERENCES.md that start with authors and a (year); key = first surname + year (+ a, b ...)."""
    text = path.read_text()
    bullets, cur = [], None
    for line in text.splitlines():
        if line.startswith("- "):
            if cur:
                bullets.append(cur)
            cur = line[2:].strip()
        elif cur is not None and line.startswith("  ") and line.strip():
            cur += " " + line.strip()
        else:
            if cur:
                bullets.append(cur)
            cur = None
    if cur:
        bullets.append(cur)
    out, seen = [], {}
    for b in bullets:
        m = re.match(r"^(.+?)\((\d{4})(?:-\d{4})?\)\.\s*(.*)$", b)
        if not m:
            continue
        sur = re.split(r"[,\s]", m.group(1).strip())[0]
        sur = "".join(ch for ch in unicodedata.normalize("NFKD", sur) if not unicodedata.combining(ch))
        key = re.sub(r"[^A-Za-z]", "", sur) + m.group(2)
        if key in seen:
            seen[key] += 1
            key += "abcdefgh"[seen[key] - 1]
        else:
            seen[key] = 0
        body = re.sub(r"\*(.+?)\*", r"<i>\1</i>", html.escape(b, quote=False))
        body = re.sub(r"doi:(\S+?)(\.?)(?=\s|$)", lambda mm: f'<a href="https://doi.org/{mm.group(1)}">doi:{mm.group(1)}'
                      f'</a>{mm.group(2)}', body)
        body = re.sub(r"arXiv:(\d{4}\.\d{4,5})", r'<a href="https://arxiv.org/abs/\1">arXiv:\1</a>', body)
        out.append({"key": key, "n": len(out) + 1, "html": body})
    return out


def cite(text: str, refs: list) -> str:
    """[[Key]] or [[Key1, Key2]] -> numbered links [n] to the reference list."""
    idx = {r["key"]: r["n"] for r in refs}

    def rep(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        missing = [k for k in keys if k not in idx]
        if missing:
            raise KeyError(f"citation keys not in REFERENCES.md: {missing}")
        return "[" + ", ".join(f'<a class="cite" href="#ref-{idx[k]}">{idx[k]}</a>' for k in keys) + "]"
    return re.sub(r"\[\[([A-Za-z0-9, ]+)\]\]", rep, text)


def render(template: str, values: dict, refs: list) -> str:
    def rep(m):
        k = m.group(1)
        if k not in values:
            raise KeyError(f"report value {k} not provided")
        v = values[k]
        return f"{v:,}" if isinstance(v, int) else str(v)
    return cite(re.sub(r"\{\{([a-z0-9_]+)\}\}", rep, template), refs)


def _default(text: str, name: str):
    m = re.search(name + r": int = (\d+)", text)
    return int(m.group(1)) if m else None


def fluxfill_card(model_root: Path, counts: dict) -> dict:
    """The scored FluxFill checkpoint as a method card, from its config.json and the model code."""
    sc = counts.get("scored") or {}
    cp = model_root / sc.get("run", "") / "config.json"
    if not sc or not cp.exists():
        return None
    c = json.loads(cp.read_text())
    net = (FILL / "model" / "net.py").read_text()
    train = (FILL / "model" / "train.py").read_text()
    sb = (FILL / "model" / "score_bench.py").read_text()
    meth = (FILL / "model" / "method.py").read_text()
    opt = re.search(r"AdamW\(net\.parameters\(\), lr=a\.lr, weight_decay=([\d.]+), betas=\(([\d.]+), ([\d.]+)\)\)", train)
    clip = re.search(r"clip_grad_norm_\(net\.parameters\(\), ([\d.]+)\)", train)
    rounds = re.search(r'"--steps", type=int, default=(\d+)', sb)
    samples = re.search(r'"--samples", type=int, default=(\d+)', sb)
    flow_steps = re.search(r"flow_steps: int = (\d+)", meth)
    rows = len(c.get("stats", {}).get("vars", []))
    hyper = [["width d", str(c["d"])], ["block pairs (time + variable)", str(c["depth"])], ["heads", str(c["heads"])],
             ["MLP ratio", str(_default(net, "mlp"))], ["variable rows", f"{rows} + 1 imagery row"],
             ["flow head", f"adaLN MLP, width {_default(net, 'flow_width')}, {_default(net, 'flow_depth')} blocks"],
             ["site memory", "mean, spread, presence of other years at the half-hour, ±15 days"],
             ["optimiser", f"AdamW, weight decay {opt.group(1)}, betas ({opt.group(2)}, {opt.group(3)})" if opt else "AdamW"],
             ["learning rate", f"{c['lr']:g}, {c['warmup']} warm-up steps, cosine to 10% of the peak"],
             ["gradient clip", clip.group(1) if clip else "?"], ["batch", f"{c['batch']} × {c.get('accum', 1)} windows"],
             ["steps", f"{c['steps']:,}; scored at step {sc['step']:,}"],
             ["v1 sampling", f"{rounds.group(1) if rounds else '?'} rounds, {flow_steps.group(1) if flow_steps else '?'} ODE "
                             f"steps per token, {samples.group(1) if samples else '?'} samples"]]
    return dict(id="fluxfill", name="FluxFill v0 and v1", ref=["Li2024", "Lipman2023", "Chang2022"],
                code="models/flux/fill/model (net.py, train.py, method.py)", version=f"run {sc['run']}, step {sc['step']:,}",
                cls=f"Transformer with factorised time and variable attention over day-tokens; {sc['n_params']:,} "
                    "parameters. v0: Student-t per half-hour. v1: joint samples by masked generation with a "
                    "flow-matching head.",
                hyper=hyper,
                inputs="Every tower variable, ERA5 and ERA5-Land drivers, Sentinel-2 Clay CLS embeddings, the site "
                       "location and site memory; any of them may be hidden.",
                training=f"{len(c.get('train_sites', []))} training sites of the frozen split, held-out years excluded; "
                         "selection on validation sites.",
                uncertainty="v0: Student-t marginals, samples independent per half-hour. v1: joint samples over the "
                            "window, so sums carry the dependence between half-hours.",
                limits="v1 under-covers half-hourly values and feedback adds NEE error (E16); NEE at 1-day and at 90 to "
                       "180-day gaps and FCH4 at sites without FCH4 history are open (E14, E17, E18).")


def build(model_root: Path, counts: dict) -> dict:
    refs = references(FILL / "docs" / "REFERENCES.md")
    bl = baselines(counts)
    ff = fluxfill_card(model_root, counts)
    for b in bl + ([ff] if ff else []):
        b["cite"] = cite("[[" + ", ".join(b["ref"]) + "]]", refs) if b["ref"] else ""
    return {"baselines": bl, "fluxfill": ff, "experiments": experiments(model_root, counts, FILL / "model" / "NOTES.md"),
            "references": refs}
