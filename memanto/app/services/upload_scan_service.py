"""
Prescan text-class uploads for directive payloads before they reach Moorcheh.

Files uploaded to a memory namespace are embedded as document text and later
returned into LLM prompts on recall/answer/daily summary. A .md/.txt/.csv/
.json carrying a directive payload would therefore re-introduce the trojan-
horse vector that INJECTION_PATTERNS already blocks on every JSON write
path. Binary formats (pdf/docx/xlsx) are extracted server-side and are not
text-scannable at the route layer, so they remain a documented follow-up.

Scanning is chunked and streaming: the file is read in fixed-size chunks
with a carry-over overlap so a payload straddling a chunk boundary still
matches, and memory is bounded no matter how large the upload is. Failing
closed on any match is intentional: for a memory store the safe direction
on failure is "reject the write", not "write the poison".
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

# Chunked streaming scan, bounded memory: the whole file is walked in
# _CHUNK_SIZE-byte reads, so a multi-GB .txt does not allocate a whole-file
# object.
_CHUNK_SIZE = 1024 * 1024  # 1 MB
# A payload can straddle a chunk boundary. Carry the tail of each window so
# such a match is found on the next iteration. 255 bytes covers the literal
# form of every rule here (longest literal is 28 bytes). It is a practical
# bound, not a guarantee: rules using unbounded quantifiers (`\s+`, `.*`) can
# produce a match longer than the overlap, so a payload straddling a boundary
# with more than 255 bytes of intervening whitespace is not detected. Accepted:
# these rules guard against verbatim payloads, and any paraphrase evades them.
_OVERLAP = 255


def _windows(lowered_chunks):
    """Yield each lowercased chunk together with the carried tail of the
    previous window, so a pattern that spans a chunk boundary is still
    inside one of the windows we test."""
    carry = ""
    for chunk in lowered_chunks:
        yield carry + chunk
        carry = chunk[-_OVERLAP:]


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
    seen = set()
    with open(path, "rb") as f:
        for window in _windows(_chunk_stream(f, _CHUNK_SIZE)):
            for pattern, compiled in zip(INJECTION_PATTERNS, _INJECTION_RULES):
                if pattern in seen:
                    continue
                if compiled.search(window):
                    # Dedupe: the same rule can re-match on the carried
                    # overlap across consecutive windows.
                    seen.add(pattern)
                    matched.append(pattern)
    return matched


def _chunk_stream(f, chunk_size: int):
    """Read fixed-size chunks from an open binary stream, lowercased."""
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        yield chunk.decode("utf-8", errors="replace").lower()
