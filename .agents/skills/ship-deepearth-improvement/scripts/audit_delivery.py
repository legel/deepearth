#!/usr/bin/env python3
"""Fail closed when a DeepEarth delivery branch violates the public PR contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


# Both repositories have a directory called `autoresearch/` and they mean opposite things: privately it is the
# research harness, publicly it is the shipped training/eval/config/data package. Block the private subtrees by
# their real paths -- none of them exist on public main -- not the shared parent, which is the product itself.
FORBIDDEN_PREFIXES = (
    "autoresearch/probes/",
    "autoresearch/main/",
    "autoresearch/scoring/",
    "autoresearch/tests/",
    "autoresearch/data/",
    "autoresearch/scorecard.md",
    ".agents/",
    ".claude/",
    "customer_feedback/",
)
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "scorecards",
    "campaigns",
}
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".log",
    ".pt",
    ".pth",
)


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--max-commits", type=int, default=1)
    parser.add_argument("--max-file-bytes", type=int, default=5 * 1024 * 1024)
    args = parser.parse_args()

    repo = args.repo.resolve()
    failures: list[str] = []
    notes: list[str] = []

    if git(repo, "status", "--porcelain"):
        failures.append("worktree is dirty")

    branch = git(repo, "branch", "--show-current")
    if not branch.startswith("delivery/"):
        failures.append(f"branch must start with delivery/ (found {branch or 'detached HEAD'})")

    if subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", args.base, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        failures.append(f"{args.base} is not an ancestor of HEAD")

    count_text = git(repo, "rev-list", "--count", f"{args.base}..HEAD")
    commit_count = int(count_text or "0")
    if commit_count < 1:
        failures.append("delivery branch has no commit beyond the base")
    if commit_count > args.max_commits:
        failures.append(f"delivery has {commit_count} commits; maximum is {args.max_commits}")

    merges = git(repo, "rev-list", "--merges", f"{args.base}..HEAD")
    if merges:
        failures.append("delivery history contains merge commits")

    changed = [p for p in git(repo, "diff", "--name-only", f"{args.base}...HEAD").splitlines() if p]
    if not changed:
        failures.append("delivery has no changed files")

    for rel in changed:
        path = Path(rel)
        if rel.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"forbidden research path: {rel}")
        if FORBIDDEN_PARTS.intersection(path.parts):
            failures.append(f"forbidden generated/research path: {rel}")
        if rel.endswith(FORBIDDEN_SUFFIXES):
            failures.append(f"forbidden artifact type: {rel}")
        full = repo / path
        if full.is_file() and full.stat().st_size > args.max_file_bytes:
            failures.append(f"file exceeds {args.max_file_bytes} bytes: {rel}")

    subjects = git(repo, "log", "--format=%s", f"{args.base}..HEAD").splitlines()
    for subject in subjects:
        lowered = subject.lower()
        if "autoresearch" in lowered or "agent" in lowered:
            failures.append(f"internal research language in commit subject: {subject}")

    numstat = git(repo, "diff", "--numstat", f"{args.base}...HEAD")
    additions = deletions = 0
    for line in numstat.splitlines():
        added, removed, _ = line.split("\t", 2)
        if added.isdigit():
            additions += int(added)
        if removed.isdigit():
            deletions += int(removed)
    notes.append(f"branch={branch}")
    notes.append(f"base={git(repo, 'rev-parse', args.base)}")
    notes.append(f"head={git(repo, 'rev-parse', 'HEAD')}")
    notes.append(f"commits={commit_count}")
    notes.append(f"files={len(changed)} additions={additions} deletions={deletions}")

    for note in notes:
        print(f"[audit] {note}")
    for rel in changed:
        print(f"[audit] changed={rel}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    print("[PASS] delivery branch satisfies the public PR scope contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
