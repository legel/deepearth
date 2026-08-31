# DeepEarth Dashboard — Architecture

Continuous agentic audit of DeepEarth. Every code block traces to science.md
principles and benchmarks. Every status claim is auditable back to its evidence.
All training data is visible, down to single observations on a satellite map.

## Pipeline

```
registry.py      deterministic         repo tree/blocks, rules, benchmarks, scores -> state/registry.json
callgraph.py     deterministic         static reachability under the ACTUAL config: every def
                                       live | gated(key) | island | pipeline/tests/tooling -> state/callgraph.json
flow.py          deterministic         modality census from cache-file headers + gbifID joins;
                                       architecture dims from the yaml -> state/flow.json
audit.py         Gemini (cached)       connectivity edges + per-rule status; consumes reachability,
                                       run movers, and findings -> state/graph.json, state/status.json
tracker.py       zero-code-change      wraps train.py, parses its stdout -> runs/<id>.jsonl (live)
trace.py         GPU, per checkpoint   forward hooks on every nn.Module, real batch -> state/trace.json
reconstruct.py   GPU, per checkpoint   masked posteriors vs ground truth, 64 held-out obs
                                            -> state/reconstructions.json
observations.py  deterministic         343k-obs map index, exact holdout replica -> state/observations.npz
refresh.py       orchestrator          registry -> callgraph -> flow -> audit, in order (the
                                       post-commit hook runs it with --graph-only)
server.py        thin Flask reader     serves state/, runs/, code content, static/
static/          one-page app          Status | Graph | Flow | Code | Science | Benchmarks | Data | Runs
```

Deterministic layers are free and always fresh (post-commit hook). `audit.py` spends LLM
tokens behind a content-hash cache. GPU layers run per checkpoint. `server.py` never
computes. `/api/meta` reports each artifact's build head; the header warns on skew.

## State artifacts (gitignored)

- `state/registry.json` — files, blocks, rules, benchmarks, scores. No LLM.
- `state/callgraph.json` — per-def reachability verdicts + gates. The proof-of-integration
  layer: capability never counts as implementation.
- `state/flow.json` — the dataset as it exists on disk (real shapes/dtypes/coverage) +
  config-derived architecture dims.
- `state/graph.json` — edges `{code block <-> rule/benchmark}` `{src, dst, s, note<=90ch}`;
  tier-2 verified hunt edges (✓✓) survive every rebuild.
- `state/status.json` — per rule + system `{status, headline, evidence, next}`.
- `state/verification.json` — adversarial fleet verdicts; they override tier-1 in every view.
- `state/trace.json` — the executed model: module events with real shapes/values.
- `state/reconstructions.json` — real masked posteriors vs ground truth per observation.
- `state/cache/` — LLM response cache (content-hash keyed).
- `runs/*.jsonl` — training events: `config`, `startup`, `step`, `transfer`, `final`.
- `seed/findings.json` — committed, curated: defects proven by running the system (F1–F7),
  each with receipts.

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

## Two-tier audit operations

Tier 1 runs continuously and cheaply: `audit.py --loop` (Gemini) re-maps changed
files and refreshes status after every merge. Tier 2 runs periodically or before
decisions: a fleet of stronger agents adversarially re-reads every non-good
claim (refute-first, exhaustive search before accepting any absence claim) and
hunts missed edges in core files. Its output is `state/verification.json`
(`{verifications: [{id, claimed, verdict, status, note, key_evidence}], hunts}`);
verified verdicts override tier-1 statuses in every view (✓✓ badge) and hunter
edges merge into the graph (notes prefixed ✓✓). First run: tier 2 overturned a
false critical (R24), cleared two false alarms (R16, R32), and hardened one real
failure (R27). Operate both; trust neither alone.

`seed/findings.json` records operational findings — defects discovered by
actually running the system (missing data dependencies, stale binaries) that no
static audit can see. They render on the Status wall. Append; don't rewrite.

## The one write path

The dashboard is a read-only viewer over `state/` and `runs/` with a single exception: the **Console**
(`#/console`). It is command-and-control — a scientist types `/build …` / `/science …` / prose, and
`POST /api/directive` writes it to the Ensue memory network (`ensue.py`, schema `deepearth.directive/1`),
the same memory the research loop reads. The board (`GET /api/directives`) is a live view of those
directives + the agents' status/progress written back. Nothing touches `state/`; Ensue is the source of
truth. Needs `ENSUE_API_KEY` (env or `dashboard/.env`); absent it, the board degrades to empty.

## Extension points

- New view: add a section to `static/app.js` + an endpoint reading state.
- New audit dimension: add a schema + prompt to `audit.py`; write a new state file.
- New logger events: emit any `{"t": <type>, ...}` dict; views ignore unknown types.
- Hosting: any WSGI host; state is plain JSON on disk.
