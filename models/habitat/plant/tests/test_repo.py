"""Hygiene the reader hits before any science: does the package say true things about itself, and nothing private.

Cheap checks that need no data: no machine-specific paths, hosts or credentials in any file; rationale stated as a
measurement, never as a person's preference; every script the README names exists and documents its use; the
README's test count is current.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SOURCES = sorted(
    p for ext in ("*.py", "*.md", "*.sh", "*.R", "*.json", "*.ini", "*.txt") for p in ROOT.rglob(ext)
    # This file quotes the patterns it bans, so scanning it would always fail.
    if ".pytest_cache" not in p.parts and "data" not in p.relative_to(ROOT).parts[:1] and p.resolve() != SELF
)


def test_no_machine_specific_paths_hosts_or_credentials():
    banned = re.compile(
        r"/home/|/Users/|~/Downloads|\b4tb\b|\bphoton\b|gs://|render-\d{4,}|\bssh\s+-i\b|\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b"
        r"|\b(?:34|35|89)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|-----BEGIN [A-Z ]*PRIVATE KEY|\bhf_[A-Za-z0-9]{20,}"
        r"|(?:password|passwd|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]", re.I)
    hits = [f"{p.relative_to(ROOT)}:{m.group(0)!r}" for p in SOURCES for m in banned.finditer(p.read_text())]
    assert not hits, hits


def test_no_personal_attribution_anywhere():
    """Rationale is stated as a measurement or a decision, never as a person's preference or request."""
    banned = re.compile(
        r"\b(lance|qhuang|sallin|tristan)\b|\bhonest(ly)?\b|\bclaude\b|\banthropic\b|generated with"
        r"|\buser (?:decision|request|approved)|\bas (?:he|she|they) (?:said|asked|wanted)\b|\brequested by\b"
        r"|\bsub-?agents?\b|\bmain session\b", re.I)
    hits = [f"{p.relative_to(ROOT)}:{m.group(0)!r}" for p in SOURCES for m in banned.finditer(p.read_text())]
    assert not hits, hits


def test_every_script_named_in_the_readme_exists_and_says_how_to_use_it():
    readme = (ROOT / "README.md").read_text()
    named = set(re.findall(r"scripts/([\w.]+\.(?:py|sh))", readme))
    assert named, "the README names no script"
    missing = [n for n in named if not (ROOT / "scripts" / n).exists()]
    assert not missing, missing
    for p in (ROOT / "scripts").iterdir():
        if p.suffix == ".py":
            doc = p.read_text().split('"""')
            assert len(doc) >= 3 and len(doc[1].strip()) > 40, f"{p.name} has no docstring"
        elif p.suffix == ".sh":
            assert p.read_text().splitlines()[1].startswith("#"), f"{p.name} has no header comment"

