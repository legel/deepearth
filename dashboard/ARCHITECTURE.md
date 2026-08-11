# DeepEarth Dashboard — Architecture

Continuous agentic audit of DeepEarth. Every code block traces to science.md
principles and benchmarks. Every status claim is auditable back to its evidence.
All training data is visible, down to single observations on a satellite map.

## Pipeline

```
registry.py   deterministic extraction   repo tree, science.md rules, evaluate.py
                                         benchmarks, champion scores
                                            -> state/registry.json
audit.py      LLM inference (Gemini)     connectivity: code block <-> rule <-> benchmark
                                         status: per rule + per system, concise, cited
                                            -> state/graph.json, state/status.json
logger.py     training capture           RunLogger JSONL events, flushed live
                                            -> runs/<run_id>.jsonl
server.py     Flask localhost            serves state/, runs/, repo file content, static/
static/       one-page app               Status | Code | Science | Benchmarks | Data
```

`registry.py` is deterministic and free. `audit.py` spends LLM tokens; it caches
by content hash and only re-audits changed blocks. `server.py` is a thin reader:
it never computes, it serves what the pipeline wrote.

## State artifacts (gitignored)

- `state/registry.json` — files, blocks, rules, benchmarks, scores. Ground truth
  extracted from the repo, no LLM.
- `state/graph.json` — edges `{code block <-> rule}`, `{code block <-> benchmark}`,
  `{rule <-> benchmark}`, each `{src, dst, strength, note<=140ch}`.
- `state/status.json` — per rule and per system: `{status, headline<=90ch,
  evidence: [block/benchmark ids], next<=120ch}`. Status one of
  `good | warning | serious | critical | unknown`.
- `state/cache/` — LLM response cache keyed by sha256 of (prompt schema, content).
- `runs/*.jsonl` — training events: `config`, `step`, `eval`, `final`.

## LLM contract (efficiency is a requirement)

Inputs may be large (whole files, score tables). Outputs are strict JSON,
schema-forced, with hard character caps per field. Two inference types:

1. **Connectivity** — batch: one call per source file, returns edges for all its
   blocks. Re-run only for files whose hash changed since last audit.
2. **Status** — one call per system (earth4d, phylo, fusion, method, data) +
   one rollup. Inputs: the system's rules, its linked blocks (signatures, not
   bodies), benchmark scores and deltas. Output: status + headline + evidence ids.

Model: `GEMINI_MODEL` (default `gemini-3.6-flash`) via REST
(`generativelanguage.googleapis.com`), key from `GEMINI_API_KEY`. No SDK
dependency. `audit.py --loop` re-audits on repo HEAD change — run it after
every PR merge.

## Views

- **Status** — the gallery wall: four system tiles + 32 rule tiles, color-coded,
  sorted worst-first. Click any tile -> the claim, its evidence blocks and
  benchmarks, the LLM note. Nothing without a citation.
- **Code** — file tree with coverage; file view annotates each block with its
  rules/benchmarks. Uncovered blocks are visibly bare.
- **Science** — each rule: linked blocks, linked benchmarks, status history.
- **Benchmarks** — all ~60 with current scores, champion deltas, linked rules.
- **Data** — (x,y,z,t) observations on a satellite map; per-observation
  modalities; train/test split; token structure of a training example;
  reconstruction vs ground truth. Live runs stream from `runs/`.

Traversal is the point: every entity links to every connected entity.

## Extension points

- New view: add a section to `static/app.js` + an endpoint reading state.
- New audit dimension: add a schema + prompt to `audit.py`; write a new state file.
- New logger events: emit any `{"t": <type>, ...}` dict; views ignore unknown types.
- Hosting: any WSGI host; state is plain JSON on disk.
