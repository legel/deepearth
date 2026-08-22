---
name: dashboard
description: Operate the DeepEarth dashboard — refresh audit state, run and compare training experiments, trace checkpoints, triage dead code, and verify behavior. Use when working in dashboard/, auditing code↔science↔benchmark connections, launching training runs, investigating findings, or validating that a change is real end-to-end.
---

# DeepEarth Dashboard operations

The dashboard (`dashboard/` at the repo root) ties every line of code to the science.md
principles and the benchmark suite, shows the dataset and model as they actually exist, and
audits the repo continuously. Server: `python -m dashboard.server` → http://localhost:8321.
Full tour: `dashboard/README.md` and `dashboard/ARCHITECTURE.md`.

**Always run commands from the repo root.** The package imports as `dashboard.*`, configs use
repo-relative paths, and the shell's cwd is the number-one cause of phantom failures.

## The one command that usually suffices

```bash
python -m dashboard.refresh            # registry -> callgraph -> flow -> audit, coherent
python -m dashboard.refresh --graph-only   # what the post-commit hook runs (cheap)
```

State artifacts live in `dashboard/state/` (gitignored, regenerable — never edit by hand).
`/api/meta` reports each artifact's build head; the UI header shows ▲ state skew if they
diverge — the fix is always `refresh`, never manual patching.

## Layered truth — respect the tiers

1. **Deterministic** (free): `registry`, `callgraph` (reachability under the ACTUAL config —
   live / gated(key) / island), `flow` (dataset census from cache-file headers),
   `observations`. These are ground truth; LLM tiers consume them.
2. **Gemini** (cached, `GEMINI_API_KEY` read from repo `.env` first, then env): `audit`
   connectivity + status. Outputs are schema-forced and concise. `--triage-islands`
   dispositions every off-path def. `--refresh-seed` regenerates curated registries.
3. **Adversarial verification** (`state/verification.json`): fleet verdicts override tier-1
   in every view and survive graph rebuilds. Only replace entries with new verified verdicts.

Never claim a principle is implemented because code exists: the callgraph decides whether it
runs. Capability ≠ implementation.

## Experiments (rule-20 protocol)

```bash
CUDA_VISIBLE_DEVICES=0 python -m dashboard.tracker <config.yaml> --tag <experiment-id>
```

- The tracker wraps `autoresearch/train.py` untouched; runs stream live to the Runs tab.
- Local runs on this box need the AlphaEarth-dependent flags off (`task_niche_prior`,
  `family_alphaearth_expert`, `alphaearth_geo`, `orthogonal_blank_hidden: 0`) — see finding
  F1. Tag such variants clearly (`*-noae`); they are not champion-comparable.
- Compare any two finished runs in the Runs tab (A vs B, every benchmark), and record
  conclusions as receipts appended to `dashboard/seed/findings.json` (append-only ledger).

Per checkpoint, on a free GPU:

```bash
python -m dashboard.trace <ckpt> --config <yaml>        # executed-model Flow diagram
python -m dashboard.reconstruct <ckpt> --config <yaml>  # posteriors vs truth + R23 invariant
```

## Pitfalls that have actually bitten

- **Prepared-cache tags**: the trainer absolutizes `cache_dir` before hashing its cache tag.
  Any tool touching `train_and_evaluate` must use `dashboard/_shared.py`
  (`normalize_config`, `prepared_path`) — never recompute the tag by hand.
- `autoresearch/evaluate.py` and `prepare.py` are immutable ground truth. Do not edit them.
- The package must be importable as `deepearth` (parent dir on sys.path); the tracker
  handles this for subprocesses.
- After a torch upgrade or on a fresh clone, rebuild the CUDA kernel:
  `bash encoders/spacetime/install.sh` (finding F2 — the committed .so goes stale).
- Deleting "dead" code: the callgraph misses some framework registrations; verify each island
  with a repo-wide grep AND read the def before deleting (a `@register_fake` kernel once
  looked dead and was not).

## Verify, don't assume

```bash
node dashboard/tests/sweep.js          # 21 behavioral assertions, prints ALL CLEAN
```

Needs `puppeteer-core` (drives system Chrome) and the server on :8321. Extend the sweep with
an assertion for every UI feature you add — every regression ever fixed is guarded there.
For visual checks, headless Chrome screenshots work, but scrolled-viewport captures are
unreliable — verify scroll behavior through the DOM instead.
