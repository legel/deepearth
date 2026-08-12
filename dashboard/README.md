# DeepEarth Dashboard

Agentic audit and monitoring for DeepEarth. Traces 100% of the codebase to the
32 science.md principles and the benchmark suite. Shows all training data, live
runs, and LLM-inferred system status — every claim auditable.

## Run

```bash
export GEMINI_API_KEY=...            # required for audit.py only
python -m dashboard.registry         # extract repo -> state/registry.json  (free, fast)
python -m dashboard.audit            # infer graph + status via Gemini      (cached, concise)
python -m dashboard.server           # http://localhost:8321
```

`registry` and `server` need no key and no network. `audit --loop` watches repo
HEAD and re-audits changed files only.

## Live training

```bash
python -m dashboard.tracker autoresearch/deepcal.yaml   # wraps train.py, zero code change
```

train.py output passes through unchanged; parsed events (steps, losses, eval,
final benchmark suite) stream to `dashboard/runs/<id>.jsonl` and the Runs view
tails them live — loss curve, held-out transfer, per-benchmark champion deltas.
For custom scripts, `dashboard.logger.RunLogger` emits the same events directly.

## Layout

See `ARCHITECTURE.md`. Pipeline: `registry.py` (deterministic) -> `audit.py`
(Gemini, cached) -> `state/*.json` -> `server.py` (thin Flask reader) ->
`static/` (one-page app: Status, Code, Science, Benchmarks, Data).

## Operating guide

| artifact | command | when |
|---|---|---|
| `state/registry.json` | `python -m dashboard.registry` | after any commit |
| `state/graph.json`, `state/status.json` | `python -m dashboard.audit` (`--loop` to daemonize) | after any merge; cached per file hash |
| `state/observations.npz` | `python -m dashboard.observations` | after data cache changes |
| `runs/<id>.jsonl` | `python -m dashboard.tracker <config> [--tag t]` | every training run |
| `state/reconstructions.json` | `python -m dashboard.reconstruct <ckpt>` | after a run worth inspecting |
| `state/callgraph.json` | `python -m dashboard.callgraph` | after code/config changes — reachability truth |
| `state/flow.json` | `python -m dashboard.flow` | after callgraph or data cache changes |
| `state/verification.json` | periodic adversarial agent fleet (see ARCHITECTURE.md) | before decisions |

`seed/*.json` are curated registries (benchmark semantics, rule structure, token
architecture, operational findings). `python -m dashboard.audit --refresh-seed
<science|benchmarks|tokens|all>` regenerates them from source via Gemini when
evaluate.py / science.md / fusion.py change shape; review the diff before
committing. `findings.json` and `data_schema.json` stay hand-curated — append
findings as they are discovered.

## Handoff notes

- Python 3.10+, Flask, numpy, torch (reconstruct/tracker only). LLM calls are raw REST.
- `GEMINI_API_KEY` required for audit only; `GEMINI_MODEL` overrides the default model.
- Known setup order for a fresh clone: parent dir named `deepearth` on sys.path,
  `bash encoders/spacetime/install.sh` (the committed .so is ABI-stale — see F2),
  AlphaEarth recipe before champion configs (F1). Details: Status wall findings.
- All generated state is gitignored; regenerate with the commands above.
- Do not edit `state/` by hand — it is overwritten on every audit.
