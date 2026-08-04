# DeepEarth delivery contract

## Repository boundary

| Role | Repository | Branch |
|---|---|---|
| Public product and PR target | `git@github.com:legel/deepearth.git` | `main` |
| Private research, harness, evidence, and skills | `git@github.com:mutable-state-inc/deepearth-research.git` | `deepcal-ensue-autoresearch` |

Use local remote names:

- `origin`: Lance's public repository.
- `research`: Ensue's private standalone mirror.

The private repository is not a GitHub fork. Delivery branches that become PRs must be fresh descendants of the current public `origin/main` and pushed to a location from which `legel/deepearth:main` can receive the PR.

## Customer-facing conventions

- Send systemic, production-ready improvements rather than research history.
- Explain what changed, why it matters, how it works, and how it was tested.
- State which functionality improves and provide exact before/after evidence.
- Keep each PR independently reviewable and mergeable.
- Default to one coherent commit; squash fixups and experimental mistakes before publication.
- Use concise, direct DeepSeek-style prose and code consistent with the surrounding repository.
- Keep production comments local and terse: invariants, shapes, and safety constraints only. Put research rationale and evidence in the PR.
- Test against the current public `main` so Lance never has to merge an intermediate research branch.

## Forbidden public content

Reject a delivery diff containing any of these:

- `autoresearch/`
- `.agents/` or `.claude/`
- `program.md`, scorecards, experiment ledgers, campaign notes, or agent prompts
- model checkpoints, prepared caches, generated results, or run logs
- private repository URLs, credentials, tokens, internal hostnames, or machine paths

Ordinary focused production tests are allowed and encouraged.

## Baseline identity

Resolve the live base before every delivery:

```bash
git fetch origin refs/heads/main:refs/remotes/origin/main
git rev-parse origin/main
```

Look up the matching private baseline artifact by exact SHA. Never compare a candidate to a score from a different Git tree or scoring protocol. If no matching baseline exists, establish it before implementing or publishing the PR.

## Publication boundary

Research evidence may be summarized publicly, but raw private artifacts remain private. Pushing a branch or opening a PR is an external publication action; perform it only when explicitly authorized by the user.
