"""Hygiene the reader hits before any physics: does the package say true things about itself.

Every one of these pins a defect that shipped. They are cheap, they need no data, and each
failure is something a first-time reader would hit within a minute of cloning.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SOURCES = sorted(
    p for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.js")) + list(ROOT.rglob("*.md"))
    # This file quotes the patterns it bans, so scanning it would always fail.
    if ".pytest_cache" not in p.parts and "cache" not in p.parts and p.resolve() != SELF
)


def test_no_source_recommends_a_command_that_does_not_exist():
    """Eight asserts told the user to run `python3 -m hydro ...`. There is no such module."""
    bad = [f"{p.relative_to(ROOT)}" for p in SOURCES if "-m hydro" in p.read_text()]
    assert not bad, f"`python3 -m hydro` is not runnable; use `python3 cli.py` ({bad})"


def test_every_cli_stage_named_in_the_readme_exists():
    import cli

    readme = (ROOT / "README.md").read_text()
    parser_stages = set()
    for line in (ROOT / "cli.py").read_text().splitlines():
        if line.strip().startswith(('p = add("', 'add("')):
            parser_stages.add(line.split('"')[1])
    named = {line.split("cli.py ")[1].split()[0]
             for line in readme.splitlines() if "python3 cli.py " in line}
    assert named <= parser_stages, (
        f"README names stages cli.py does not define: {named - parser_stages}")
    assert hasattr(cli, "main")


def test_no_personal_attribution_anywhere():
    """Rationale is stated as a measurement, never as a person's preference or request.

    Word boundaries matter here: a plain substring scan for a name flags "mass balance" in four
    files and trains the reader to ignore the check.
    """
    import re

    banned = re.compile(
        r"\b(lance|qhuang|asu\.edu)\b"
        r"|\bas (?:he|she|they) (?:said|asked|wanted)\b"
        r"|\bper (?:his|her|their) request\b"
        r"|\brequested by\b|\breviewer (?:said|asked)\b",
        re.I)
    hits = []
    for p in SOURCES:
        for m in banned.finditer(p.read_text()):
            hits.append(f"{p.relative_to(ROOT)}:{m.group(0)!r}")
    assert not hits, hits


def test_no_fetched_file_is_a_symlink():
    """Fetched data must be fetched, not borrowed from another checkout.

    Vacuous on a clone, where `data/` does not exist yet -- and that is the point: it fires only
    on the machine where it can be wrong. This tree spent its whole life with ten of them, so
    `soil`, `nlcd`, `naip`, `roads_and_buildings`, `asos` and `discharge` had never once run
    while every result was reported as reproduced from a coordinate. The give-away was that the
    consumed ASOS file carried columns (`datetime_utc`, `prcp_mm`) that `fetch.asos` does not
    write; `forcing`'s fuzzy column matcher read it anyway.
    """
    links = [str(p.relative_to(ROOT)) for d in ("data", "outputs")
             for p in (ROOT / d).rglob("*") if p.is_symlink()]
    assert not links, (
        f"symlinked into this tree rather than fetched: {links}. Nothing here is reproducible "
        f"from a coordinate while these exist -- delete them and run `python3 cli.py fetch`.")


def test_the_readme_test_count_is_current():
    """A stale count is the cheapest possible signal that the README was not re-read."""
    import re

    readme = (ROOT / "README.md").read_text()
    match = re.search(r"#\s*(\d+) tests", readme)
    assert match, "README no longer states a test count"
    claimed = int(match.group(1))
    collected = pytest.collected_count  # set by conftest
    assert claimed == collected, f"README says {claimed} tests, the suite collects {collected}"
