"""Markdown table rows have as many cells as their header.

MyST (markdown-it) splits a table row at every unescaped ``|``, also inside a
code span, and drops the cells beyond the header's count without a warning.
A row such as ``| file | 3 | the bound `|x| <= 1` |`` therefore renders
truncated at ``the bound `` (0.3.0 lost the ends of nine rows of DESIGN.md
Sections 12.2 and 12.10 that way). A pipe inside a cell is written ``\\|``.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(
    [ROOT / "README.md", ROOT / "CHANGELOG.md", *(ROOT / "docs").rglob("*.md")]
)
PAGES = [p for p in PAGES if "_build" not in p.relative_to(ROOT).parts]
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
DELIMITER = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


def _cells(row):
    body = row.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return len(UNESCAPED_PIPE.split(body))


def _tables(text):
    """Yield (line number, header cells, [(line number, cells), ...])."""
    lines = text.splitlines()
    in_fence = False
    for n, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        following = lines[n + 1].strip() if n + 1 < len(lines) else ""
        if in_fence or "|" not in line or not DELIMITER.match(following):
            continue
        rows, m = [], n + 2
        while m < len(lines) and lines[m].strip().startswith("|"):
            rows.append((m + 1, _cells(lines[m])))
            m += 1
        yield n + 1, _cells(line), rows


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_table_rows_have_the_header_cell_count(page):
    bad = [
        (row_line, cells, header)
        for _, header, rows in _tables(page.read_text(encoding="utf-8"))
        for row_line, cells in rows
        if cells != header
    ]
    assert not bad, f"{page.name}: (line, cells, header cells) {bad}"


def test_the_check_detects_a_pipe_in_a_code_span():
    text = "| a | b |\n|---|---|\n| x `|M|_1` | y |\n| x `\\|M\\|_1` | y |\n"
    (_, header, rows), *_ = _tables(text)
    assert header == 2
    assert [cells for _, cells in rows] == [4, 2]
