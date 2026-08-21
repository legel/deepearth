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
python -m dashboard.refresh --cache /path/to/deepcal --no-audit  # include data census
```

State artifacts live in `dashboard/state/` (gitignored, regenerable — never edit by hand).
`/api/meta` reports each artifact's build head; the UI header shows ▲ state skew if they
diverge — the fix is always `refresh`, never manual patching.

## Layered truth — respect the tiers

1. **Deterministic** (free): `registry`, `callgraph` (reachability from the production entrypoint —
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
CUDA_VISIBLE_DEVICES=0 python -m dashboard.tracker \
  --cache /path/to/deepcal --device cuda --steps 2291 --seed 1337 --tag <experiment-id>
```

- The tracker wraps `deepearth.core.train`; runs stream live to the Runs tab.
- Public comparisons use 2,291 steps and seeds 1337 and 1338.
- Compare finished runs in the Runs tab and retain the full benchmark receipt.

Per checkpoint, on a free GPU:

```bash
python -m dashboard.trace <ckpt> --cache /path/to/deepcal
python -m dashboard.reconstruct <ckpt> --cache /path/to/deepcal
```

## Pitfalls that have actually bitten

- Dashboard checkpoint tools use the strict production loader in `dashboard/_shared.py`.
- `autoresearch/evaluate.py` is fixed public measurement; architecture belongs in `core/`.
- Install the package with `pip install -e .` before invoking module entrypoints.
- After a torch upgrade or on a fresh clone, build the CUDA kernel with
  `bash encoders/spacetime/install.sh` in the active environment.
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
