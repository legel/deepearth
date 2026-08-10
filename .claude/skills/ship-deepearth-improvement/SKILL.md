---
name: ship-deepearth-improvement
description: Convert a confirmed private DeepEarth research improvement into a minimal production pull request against legel/deepearth main, and own the delivery-side repairs the research loop hands off. Use when selecting, porting, validating, documenting, auditing, pushing, or opening a customer-facing DeepEarth PR from private Ensue research, or when the /research loop defers a broken record -- a crash, missing method, unbuildable config, stale or non-reproducing baseline, or an instrument that disagrees with itself. Enforces fresh-main ancestry, production-only scope, baseline comparisons, and zero autoresearch leakage.
---

# Ship a DeepEarth Improvement

Turn one proven research result into one reviewable production change. Treat the private repository as evidence and the public repository as the product.

Read [references/repository-contract.md](references/repository-contract.md) before changing branches or files. Read [assets/pull-request-template.md](assets/pull-request-template.md) before drafting the PR.

For public base `3c45b99`, read [references/main-baseline-3c45b99.md](references/main-baseline-3c45b99.md) before evaluating a candidate. For any other base SHA, establish and add a new SHA-keyed baseline reference first.

## 1. Select a shippable result

Require all of the following:

- A named production behavior or capability improves.
- Candidate and baseline used the same data, split, seed policy, budget, hardware class, and scoring protocol.
- The named human capability rises and the current-protocol `judge()` passes at both the 1k screen and
  8k confirmation.
- The `val_bpb` decomposition and mechanism diagnostics support the claimed causal story. A gain that
  lands somewhere else is an initialization
  re-roll, not a result -- adding parameters shifts the RNG stream and re-initializes the whole model.
- The mechanism can live in production without the harness.
- The evidence is attributable to this mechanism, not a scoring, data, or infrastructure change.

Stop without preparing a PR if any condition is missing. Do not turn promising, provisional, mechanical, or isolated records into customer claims.

Read `coord.best()` first. It is the standing scorecard — the aggregate and macro `val_bpb`, the per-variable decomposition, both suite means, the seeds and measured floor behind the claim, and the delivery fields. If `delivery.pr` is already set for this result, it has shipped; there is nothing to select. Selecting against anything else risks preparing a PR for a result the swarm has already superseded.

### Compact model replacement

Use this screen when the proposal is to replace the public 800M default, not add an isolated mechanism. It supersedes
the private 1k/8k and `val_bpb` gates above, but not the evidence standard:

- Use the registered 800M public-main scorecard as the frozen acceptance bar; do not rerun the large model merely to
  qualify a compact candidate. Change only the production model/config needed to express the compact candidate.
- Run two compact-model seeds through Lance's unchanged public evaluator with the registered data, split and fixed-time
  budget. Private scorecards establish provenance but never decide this delivery.
- Pass only when both compact seeds beat the registered 800M harmonic and meet or exceed its arithmetic mean. If the
  registered baseline lacks an exact protocol identity, stop and repair the record rather than rerunning the model.
- Report Lance's complete public scorecard plus parameters, peak VRAM, steps completed and wall time. No private
  aggregate, encoder score or mechanism diagnostic may substitute for harmonic and arithmetic.
- Treat the 100M floor in public `science.md` rule 5 as a falsifiable design prior. A smaller replacement may revise
  that rule only when the matched public evidence above passes and the reduction in parameters or memory is material.
- Expose the compact model's exact hash capacities and total parameter count. Do not present hash collisions as free:
  explain them as parameter sharing and inspect the full scorecard for lost spatial detail.

If the compact candidate does not pass, do not open a compromise PR. Keep the 800M default and return to research.

## 2. Start from public main

Fetch `legel/deepearth` immediately before beginning. Record the exact `origin/main` SHA and its registered private baseline.

Create a fresh `delivery/<short-slug>` branch and separate worktree directly from `origin/main`. Never branch from, merge, or cherry-pick the private autoresearch branch. Manually transplant the smallest production implementation needed for the confirmed mechanism.

If public `main` advances, rebase onto the new SHA and repeat the affected validation before opening the PR.

## 3. Keep the product diff narrow

Ship one improvement per PR.

- Change production code and ordinary production tests only.
- Preserve public APIs unless the improvement explicitly requires an API change.
- Avoid unrelated cleanup, formatting, renaming, dependency churn, or documentation rewrites.
- Do not copy private commit messages, experiment scaffolding, rejected variants, scorecards, prompts, programs, or agent commentary.
- Do not add `autoresearch/`, `.agents/`, `.claude/`, campaign files, checkpoints, raw results, or run logs.
- If the mechanism currently exists only in a harness file, extract the minimal runtime behavior into the appropriate production module and test it through the public API.

Match the surrounding repository style. Prefer direct code and short comments that explain invariant or intent, not research history.

### DeepSeek code style

Use DeepSeek's production-code style as the default for every delivered diff:

- Let names and control flow explain ordinary behavior.
- Comment only invariants, tensor shapes, numerical constraints, or non-obvious safety boundaries.
- Prefer one-line comments. Keep multi-line comments rare and tightly local.
- Keep docstrings to the public contract; do not turn them into design notes.
- Never put experiment history, scoring rationale, citations, campaign rules, or rejected alternatives in production code.
- Move scientific motivation and validation evidence to the commit and PR description.

Do not restyle unrelated code. Apply this standard strictly to every line the delivery changes.

## 4. Validate against the frozen baseline

Run validation from a clean checkout of the candidate and the exact registered `main` baseline.

1. Run focused correctness tests for the changed functionality.
2. Run relevant production smoke tests and import/build checks.
3. Run the canonical evaluation with the same protocol as the baseline. For a compact replacement, score only the
   candidate with the unchanged public evaluator and compare it to the registered 800M scorecard.
4. Compare the full scorecard, not only the target metric.
5. Investigate any material regression before proceeding.
6. Save logs and scorecards privately; summarize only decision-relevant evidence publicly.

Do not modify scoring, harness, data definitions, or the baseline to make a candidate pass.

## 5. Audit the delivery branch

Commit the complete production change, then run:

```bash
python .claude/skills/ship-deepearth-improvement/scripts/audit_delivery.py \
  --repo /path/to/delivery/worktree \
  --base origin/main
```

Run the copy of the script from the private research checkout; the delivery worktree must not contain this skill. Resolve every failure before pushing.

Deliver one coherent commit by default. Before pushing, squash fixups, reversions, formatting follow-ups, and experimental mistakes. Use multiple commits only when each is independently necessary and reviewable; never expose the search process as commit history.

Use an imperative production-oriented subject. Put the measured before/after result and validation protocol in the commit body. Do not mention internal agent names or describe a rejected-search history.

## 6. Draft and open the PR

Use the bundled PR template. Keep the public description concise and factual:

- **What:** the production behavior changed.
- **Why:** the user/scientific capability it improves.
- **Science:** cite the exact `autoresearch/science.md` rule number(s), explain how the mechanism realizes them, and name the capability expected to improve.
- **How:** the implementation at a useful abstraction level.
- **Evidence:** exact baseline and candidate results, protocol, seeds, and focused tests.
- **Scope:** explicit non-goals and compatibility statement.

Do not open a scientific PR whose description cannot connect the production mechanism to a named
`science.md` rule, a measured human-capability improvement, the full scorecard, and its supporting
diagnostics.

There is no longer a separate encoder measurement to discount: probe and fusion optimize one objective,
so an encoder result is a term in the fusion number rather than a proxy for it. State the harmonic and
arithmetic benchmark means as well -- they are the promotion gate and the language the public
repository is reviewed in. Report `val_bpb` as supporting likelihood evidence.

For a compact-model replacement, omit private `val_bpb` from the customer-facing claim. Report both compact public
scorecards against the registered 800M harmonic/arithmetic means, plus efficiency measurements.

Never compare a mirror run to a public-main run: the evaluators differ by ~158 lines and the same config
scores ~0.279 on the mirror against 0.332464 on public main. Baselines must come from the same tree.

Push only the delivery branch. Target `legel/deepearth:main`. Open the PR only when the user has authorized external publication or explicitly asked to create the PR.

## 7. Stamp the delivery in Ensue

The moment the PR exists, record it against the standing scorecard so the loop and the front end can
tell what is public from what is still only ours:

```python
import sys; sys.path.insert(0, "autoresearch/main/harness")
from coordinator import Coordinator
Coordinator("ship").stamp_delivery(pr=36, pr_url="https://github.com/legel/deepearth/pull/36",
                                   base_commit="4d6cb44")           # merged=True once it lands
```

If the shipped result is not yet the published best, build its scorecard with `coordinator.scorecard()`
and `publish_best()` it before stamping. `scorecard()` validates and raises — a card missing its
benchmarks, its seed count, or its decomposition is rejected rather than published as a partial record
the front end would render as fact.

## 8. Hand off

Return:

- PR link or ready-to-push branch name.
- Base and candidate SHAs.
- Changed production files.
- Before/after target metric and full-scorecard disposition.
- Tests run and their outcomes.
- Any known limitation.

Keep private evidence in the Ensue repository. Never recreate an autoresearch branch in Lance's repository.
