"""Markdown code fences in the documentation open and close in pairs.

A fence longer than three backticks is closed only by a fence at least as
long, so one lengthened opening fence turns the rest of a page into a code
block without any Sphinx warning (0.3.0 rendered most of DESIGN.md that way).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(
    [ROOT / "README.md", ROOT / "CHANGELOG.md", *(ROOT / "docs").rglob("*.md")]
)
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_fences_are_three_backticks_and_paired(page):
    fences = []
    for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            fences.append((number, match.group(2), match.group(3).strip()))
    assert all(mark == "```" for _, mark, _ in fences), [
        (n, mark) for n, mark, _ in fences if mark != "```"
    ]
    assert len(fences) % 2 == 0, f"unpaired fence in {page.name}: {fences}"
    closing = fences[1::2]
    assert all(info == "" for _, _, info in closing), closing
