#!/usr/bin/env python3
"""Deterministic Unicode-aware word counting for translated Markdown text.

The LLM judge used to be asked to report ``evaluation_word_count`` itself,
but LLMs are not reliable exact word counters. This module provides a
deterministic, Python-side replacement: ``count_translation_words()``.

The counter must work correctly across all evaluation languages, including
(but not limited to) French, German, Italian, Spanish, Portuguese, Albanian,
Turkish (``ç ğ ı İ ö ş ü``) and Serbo-Croatian (``č ć š ž đ``). It therefore
relies on Unicode-aware regex character classes (``\\w`` with the ``re.UNICODE``
flag, which is implicit in Python 3 ``str`` patterns) rather than an
ASCII-only ``[A-Za-z]`` pattern.
"""

from __future__ import annotations

import re

# --- Cleanup patterns --------------------------------------------------------

# HTML page-break divider inserted between pages by the translation pipeline.
_PAGE_BREAK_RE = re.compile(
    r'<div\s+style=["\']page-break-after:\s*always;?["\']\s*></div>',
    re.IGNORECASE,
)

# Any remaining HTML tags (defensive: strip generically, not just page breaks).
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Markdown formatting markers: headings/emphasis (#, *, _, ~) and code fences
# / inline code (`). These are stripped as standalone punctuation but must
# not eat the underscores/asterisks used inside actual words (rare, but we
# only strip runs of these symbols, not word-internal single characters).
_MD_MARKERS_RE = re.compile(r"[#*_~`]+")

# A "word" is a run of Unicode letters/digits/marks, optionally joined by a
# single internal apostrophe (') or hyphen (-) so that "l'entreprise" and
# "state-of-the-art" each count as one word, while stray leading/trailing
# apostrophes or hyphens are not required.
#
# \w in a Python `str` pattern is Unicode-aware by default (matches letters
# from any script, including accented Latin, Turkish ı/İ/ğ/ş/ç/ö/ü and
# Serbo-Croatian č/ć/š/ž/đ), so no explicit ASCII ranges are needed.
_WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)


def count_translation_words(text: str) -> int:
    """Return a deterministic count of words in translated Markdown ``text``.

    Steps:
        1. Strip the HTML page-break divider(s) and any other HTML tags.
        2. Strip Markdown formatting markers (``# * _ ~ ` ``).
        3. Count runs of Unicode word characters, treating a single internal
           apostrophe or hyphen as part of the same word (so ``l'entreprise``
           and ``state-of-the-art`` each count as one word).

    Works for Latin-script languages with diacritics (French, German,
    Italian, Spanish, Portuguese, Albanian, Turkish) as well as
    Serbo-Croatian diacritics (č, ć, š, ž, đ), since it relies on Python's
    Unicode-aware ``\\w`` rather than an ASCII-only character class.

    Args:
        text: Raw translated Markdown for a single page (untruncated).

    Returns:
        The number of words found; 0 for empty/whitespace-only input.
    """
    if not text:
        return 0

    cleaned = _PAGE_BREAK_RE.sub(" ", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _MD_MARKERS_RE.sub(" ", cleaned)

    return len(_WORD_RE.findall(cleaned))
