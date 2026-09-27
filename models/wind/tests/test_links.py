"""Every code link in the docs lands on the line that defines what it names.

A link such as [`solver.py` `project`](solver.py#L714) goes stale silently when code above it
moves: GitHub still opens the file, just on the wrong line.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", *ROOT.glob("*_simulation.md")])
# [`file.py` `name`](file.py#L12) or [`name`](file.py#L12); a bare [`file.py`](file.py#L12) names no symbol.
LINK = re.compile(r"\[(?:`(?P<file>[\w./]+)` )?`(?P<name>[\w.]+)`\]\((?P<target>[\w./]+\.py)#L(?P<line>\d+)(?:-L\d+)?\)")


def defines(line: str, name: str) -> bool:
    return bool(re.match(rf"\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", line)
                or re.match(rf"\s*{re.escape(name)}\s*(?::[^=]*)?=(?!=)", line))


def links():
    for doc in DOCS:
        for m in LINK.finditer(doc.read_text()):
            if m["name"] == m["target"]:
                continue
            yield doc.name, m["target"], m["name"], int(m["line"])


def test_the_docs_carry_code_links():
    assert len(list(links())) >= 5, "no code links found; the pattern no longer matches the docs"


def test_every_code_link_lands_on_its_definition():
    bad = []
    for doc, target, name, n in links():
        src = (ROOT / target).read_text().splitlines()
        at = [i + 1 for i, line in enumerate(src) if defines(line, name)]
        if n > len(src) or not defines(src[n - 1], name):
            bad.append(f"{doc}: {target}#L{n} for `{name}`, defined at {at or 'no line'}")
    assert not bad, "\n".join(bad)
