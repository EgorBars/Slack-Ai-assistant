"""Normalize Notion markdown for reliable header-based chunking."""

from __future__ import annotations

import re

# Standalone section titles in flat Notion pages (no markdown headers)
_SECTION_TITLE_PATTERNS = [
    re.compile(r"^Постановка на .+$", re.I),
    re.compile(r"^Результаты приемки$", re.I),
    re.compile(r"^Тестирование требований$", re.I),
    re.compile(r"^Note:$", re.I),
]

# Bold-only lines: **Section title** or **1) Something**
_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*\s*$")
# Bold prefix: **Goal:** rest of line
_BOLD_PREFIX = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
# Numbered subsection without markdown header: 1.1. Title
_NUMBERED_SECTION = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")


def _is_section_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.startswith("#"):
        return False
    return any(p.match(stripped) for p in _SECTION_TITLE_PATTERNS)


def _promote_line_to_header(line: str, level: int = 2) -> str:
    prefix = "#" * level
    stripped = line.strip()
    if stripped.startswith("#"):
        return stripped
    return f"{prefix} {stripped}"


def normalize_markdown(content: str, title: str) -> str:
    """
    Convert Notion markdown into structured markdown for chunking.

    - Adds document title as H1
    - Promotes bold section lines to headers
    - Promotes flat section titles (e.g. «Постановка на дизайн») to H2
    - Normalizes tab-indented list items to markdown nested lists
    """
    lines = content.replace("\r\n", "\n").split("\n")
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # Already a markdown header
        if stripped.startswith("#"):
            result.append(stripped)
            continue

        # **Bold title** on its own line → H2
        bold_match = _BOLD_LINE.match(stripped)
        if bold_match:
            result.append(f"## {bold_match.group(1).strip()}")
            continue

        # **Label:** value → H3 + content
        bold_prefix = _BOLD_PREFIX.match(stripped)
        if bold_prefix:
            label, rest = bold_prefix.group(1).strip(), bold_prefix.group(2).strip()
            result.append(f"## {label}")
            if rest:
                result.append(rest)
            continue

        # 1.1. Subsection title (short, no bullet) → H3
        num_match = _NUMBERED_SECTION.match(stripped)
        if num_match and len(num_match.group(2)) < 100 and not stripped.startswith("- "):
            num, text = num_match.group(1), num_match.group(2)
            # Только подпункты вида 1.1., 3.2. — не обычные «1. пункт»
            if re.match(r"^\d+\.\d", num):
                result.append(f"### {num} {text}")
                continue

        # Flat section titles → H2
        if _is_section_title(stripped):
            result.append(f"## {stripped}")
            continue

        # Tab-indented numbered/bullet items → markdown nested list
        if line.startswith("\t"):
            depth = len(line) - len(line.lstrip("\t"))
            inner = line.lstrip("\t")
            indent = "  " * (depth - 1)
            if re.match(r"^\d+\.\s", inner):
                result.append(f"{indent}{inner}")
            elif inner.startswith("- "):
                result.append(f"{indent}{inner}")
            else:
                result.append(f"{indent}- {inner}")
            continue

        result.append(stripped)

    body = "\n".join(result).strip()

    # Collapse excessive blank lines
    body = re.sub(r"\n{3,}", "\n\n", body)

    if not body.startswith("# "):
        body = f"# {title}\n\n{body}"

    return body
