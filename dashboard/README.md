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

## Handoff notes

- Python 3.10+, Flask, numpy. No other dependencies. LLM calls are raw REST.
- `GEMINI_MODEL` env overrides the default model.
- All generated state is gitignored; regenerate with the three commands above.
- Do not edit `state/` by hand — it is overwritten on every audit.
