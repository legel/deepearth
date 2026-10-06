"""Every code link in the docs lands on the line that defines what it names.

A link such as [`store.py` `build_store`](ranges/joint/store.py#L314) goes stale silently when code above it moves:
GitHub still opens the file, just on the wrong line.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
# [`file.py` `name`](file.py#L12) or [`name`](file.py#L12); a bare [`file.py`](file.py#L12) names no symbol.
LINK = re.compile(r"\[(?:`(?P<file>[\w./]+)` )?`(?P<name>[\w.]+)`\]\((?P<target>[\w./]+\.(?:py|R|sh))#L(?P<line>\d+)(?:-L\d+)?\)")
# any relative link to a file of this model
FILE = re.compile(r"\]\((?P<target>(?!https?:)[\w./-]+?)(?:#[\w-]+)?\)")


def defines(line: str, name: str) -> bool:
    return bool(re.match(rf"\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", line)
                or re.match(rf"\s*{re.escape(name)}\s*(?::[^=]*)?=(?!=)", line)
                or re.match(rf"\s*(?:{re.escape(name)}\s*<-|{re.escape(name)}\(\))", line))


def links():
    for doc in DOCS:
        for m in LINK.finditer(doc.read_text()):
            if m["name"] == m["target"]:
                continue
            yield doc, (doc.parent / m["target"]).resolve(), m["name"], int(m["line"])


def test_the_docs_carry_code_links():
    assert len(list(links())) >= 10, "no code links found; the pattern no longer matches the docs"


def test_every_code_link_lands_on_its_definition():
    bad = []
    for doc, target, name, n in links():
        src = target.read_text().splitlines()
        at = [i + 1 for i, line in enumerate(src) if defines(line, name)]
        if n > len(src) or not defines(src[n - 1], name):
            bad.append(f"{doc.name}: {target.name}#L{n} for `{name}`, defined at {at or 'no line'}")
    assert not bad, "\n".join(bad)


def test_every_relative_link_exists():
    missing = []
    for doc in DOCS:
        for m in FILE.finditer(doc.read_text()):
            t = m["target"]
            if t.startswith(("mailto:", "#")):
                continue
            if not (doc.parent / t).exists():
                missing.append(f"{doc.name}: {t}")
    assert not missing, missing
