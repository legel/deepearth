---
name: ship-deepearth-improvement
description: Convert a confirmed private DeepEarth research improvement into a minimal production pull request against legel/deepearth main. Use when Codex needs to select, port, validate, document, audit, push, or open a customer-facing DeepEarth PR from private Ensue research while enforcing fresh-main ancestry, production-only scope, baseline comparisons, DeepSeek-style delivery, and zero autoresearch leakage.
---

# Ship a DeepEarth Improvement

Turn one proven research result into one reviewable production change. Treat the private repository as evidence and the public repository as the product.

Read [references/repository-contract.md](references/repository-contract.md) before changing branches or files. Read [assets/pull-request-template.md](assets/pull-request-template.md) before drafting the PR.

For public base `3c45b99`, read [references/main-baseline-3c45b99.md](references/main-baseline-3c45b99.md) before evaluating a candidate. For any other base SHA, establish and add a new SHA-keyed baseline reference first.

## 1. Select a shippable result

Require all of the following:

- A named production behavior or capability improves.
- Candidate and baseline used the same data, split, seed policy, budget, hardware class, and scoring protocol.
- The result passed its research confirmation gate, including matched controls and two seeds when required.
- The mechanism can live in production without the harness.
- The evidence is attributable to this mechanism, not a scoring, data, or infrastructure change.

Stop without preparing a PR if any condition is missing. Do not turn promising, provisional, mechanical, or isolated records into customer claims.

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

## 4. Validate against the frozen baseline

Run validation from a clean checkout of the candidate and the exact registered `main` baseline.

1. Run focused correctness tests for the changed functionality.
2. Run relevant production smoke tests and import/build checks.
3. Run the private canonical evaluation with the same protocol as the baseline.
4. Compare the full scorecard, not only the target metric.
5. Investigate any material regression before proceeding.
6. Save logs and scorecards privately; summarize only decision-relevant evidence publicly.

Do not modify scoring, harness, data definitions, or the baseline to make a candidate pass.

## 5. Audit the delivery branch

Commit the complete production change, then run:

```bash
python .agents/skills/ship-deepearth-improvement/scripts/audit_delivery.py \
  --repo /path/to/delivery/worktree \
  --base origin/main
```

Run the copy of the script from the private research checkout; the delivery worktree must not contain this skill. Resolve every failure before pushing.

Prefer one commit. Use an imperative production-oriented subject. Put the measured before/after result and validation protocol in the commit body. Do not mention internal agent names or describe a rejected-search history.

## 6. Draft and open the PR

Use the bundled PR template. Keep the public description concise and factual:

- **What:** the production behavior changed.
- **Why:** the user/scientific capability it improves.
- **How:** the implementation at a useful abstraction level.
- **Evidence:** exact baseline and candidate results, protocol, seeds, and focused tests.
- **Scope:** explicit non-goals and compatibility statement.

Push only the delivery branch. Target `legel/deepearth:main`. Open the PR only when the user has authorized external publication or explicitly asked to create the PR.

## 7. Hand off

Return:

- PR link or ready-to-push branch name.
- Base and candidate SHAs.
- Changed production files.
- Before/after target metric and full-scorecard disposition.
- Tests run and their outcomes.
- Any known limitation.

Keep private evidence in the Ensue repository. Never recreate an autoresearch branch in Lance's repository.
