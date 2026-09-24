"""Hygiene the reader hits before any physics: does the package say true things about itself."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SOURCES = sorted(
    p for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md"))
    # This file quotes the patterns it bans, so scanning it would always fail.
    if ".pytest_cache" not in p.parts and "cache" not in p.parts and p.resolve() != SELF
)


def test_no_source_recommends_a_command_that_does_not_exist():
    bad = [f"{p.relative_to(ROOT)}" for p in SOURCES if "-m wind" in p.read_text()]
    assert not bad, f"`python3 -m wind` is not runnable; use `python3 cli.py` ({bad})"


def test_every_cli_stage_named_in_the_readme_exists():
    import cli

    readme = (ROOT / "README.md").read_text()
    parser_stages = set()
    for line in (ROOT / "cli.py").read_text().splitlines():
        if line.strip().startswith(('p = add("', 'add("')):
            parser_stages.add(line.split('"')[1])
    # A stage named inside an inline code span carries the closing backtick with it.
    named = {line.split("cli.py ")[1].split()[0].strip("`.,")
             for line in readme.splitlines() if "python3 cli.py " in line}
    assert named <= parser_stages, (
        f"README names stages cli.py does not define: {named - parser_stages}")
    assert hasattr(cli, "main")


def test_no_personal_attribution_anywhere():
    """Rationale is stated as a measurement, never as a person's preference or request."""
    banned = re.compile(
        r"\b(lance|qhuang|asu\.edu)\b"
        r"|\bas (?:he|she|they) (?:said|asked|wanted)\b"
        r"|\bper (?:his|her|their) request\b"
        r"|\brequested by\b|\breviewer (?:said|asked)\b|\bthe owner\b",
        re.I)
    hits = []
    for p in SOURCES:
        for m in banned.finditer(p.read_text()):
            hits.append(f"{p.relative_to(ROOT)}:{m.group(0)!r}")
    assert not hits, hits


def test_no_gpl_solver_is_vendored():
    """The numerics are original; nothing from a GPL wind model may appear by name in code."""
    banned = re.compile(r"urock|udales|windninja|quic[-_ ]?urb|qes[-_ ]?winds", re.I)
    hits = [f"{p.relative_to(ROOT)}" for p in SOURCES if p.suffix == ".py"
            and banned.search(p.read_text())]
    assert not hits, hits


def test_no_fetched_file_is_a_symlink():
    """Fetched data must be fetched, not borrowed from another checkout."""
    links = [str(p.relative_to(ROOT)) for d in ("data", "outputs")
             for p in (ROOT / d).rglob("*") if p.is_symlink()]
    assert not links, f"symlinked into this tree rather than fetched: {links}"


def test_the_readme_test_count_is_current():
    """A stale count is the cheapest possible signal that the README was not re-read."""
    readme = (ROOT / "README.md").read_text()
    match = re.search(r"#\s*(\d+) tests", readme)
    assert match, "README no longer states a test count"
    claimed = int(match.group(1))
    collected = pytest.collected_count  # set by conftest
    assert claimed == collected, f"README says {claimed} tests, the suite collects {collected}"


def test_receipts_named_in_the_readme_exist():
    readme = (ROOT / "README.md").read_text()
    named = set(re.findall(r"docs/[\w.\-]+\.json", readme))
    missing = [n for n in named if not (ROOT / n).exists()]
    assert not missing, missing
