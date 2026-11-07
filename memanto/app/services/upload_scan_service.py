"""
Prescan text-class uploads for directive payloads before they reach Moorcheh.

Files uploaded to a memory namespace are embedded as document text and later
returned into LLM prompts on recall/answer/daily summary. A .md/.txt/.csv/
.json carrying a directive payload would therefore re-introduce the trojan-
horse vector that INJECTION_PATTERNS already blocks on every JSON write
path. Binary formats (pdf/docx/xlsx) are extracted server-side and are not
text-scannable at the route layer, so they remain a documented follow-up.
"""

from __future__ import annotations

import re
from pathlib import Path

from memanto.app.constants import INJECTION_PATTERNS

# Pre-compiled once; matching against lowercased text uses IGNORECASE in
# case the raw (non-lowercased) text is ever scanned directly.
_INJECTION_RULES = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
)

# Uploads whose bytes are plain text and can be scanned at the route layer.
# Everything else (pdf, docx, xlsx) is extracted server-side by Moorcheh.
TEXT_UPLOAD_SUFFIXES = frozenset({".txt", ".csv", ".json", ".md"})

# Directive payloads are short; reading more than this costs time without
# gaining coverage. Bounded keeps a 5 GB .txt from being read line-by-line
# to no additional effect.
_TEXT_SCAN_MAX_BYTES = 1_048_576  # 1 MB


def scan_upload_for_injections(path: Path) -> list[str]:
    """Return INJECTION_PATTERNS rules matched in a text-class upload.

    Empty list means the file is clean (or is a binary format this function
    does not scan). Callers reject the upload when the returned list is
    non-empty; the matched rule names are included so the 4xx response can
    tell the user what to fix.
    """
    suffix = path.suffix.lower()
    if suffix not in TEXT_UPLOAD_SUFFIXES:
        return []

    matched: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(_TEXT_SCAN_MAX_BYTES)

    lowered = text.lower()
    for pattern, compiled in zip(INJECTION_PATTERNS, _INJECTION_RULES):
        if compiled.search(lowered):
            matched.append(pattern)
    return matched
