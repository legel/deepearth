"""Every code link in the dataset card lands on the line that defines what it names."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[(?:`(?P<file>[\w./]+)` )?`(?P<name>[\w.]+)`\]\((?P<target>[\w./]+\.py)#L(?P<line>\d+)\)")


def defines(line: str, name: str) -> bool:
    return bool(re.match(rf"\s*(?:def|class)\s+{re.escape(name)}\b", line)
                or re.match(rf"\s*{re.escape(name)}\s*(?::[^=]*)?=(?!=)", line))


def test_every_code_link_lands_on_its_definition():
    links = list(LINK.finditer((ROOT / "README.md").read_text()))
    assert len(links) >= 10
    bad = []
    for m in links:
        src = (ROOT / m["target"]).read_text().splitlines()
        n = int(m["line"])
        if n > len(src) or not defines(src[n - 1], m["name"]):
            bad.append(f"{m['target']}#L{n} for `{m['name']}`")
    assert not bad, "\n".join(bad)
