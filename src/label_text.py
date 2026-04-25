"""Normalize messy MusicCaps-style label strings for prompts and display."""

from __future__ import annotations

import re


def sanitize_vocab_label(raw: str) -> str:
    """
    Strip artifacts from labels that were split from Python-list strings, e.g.
    ``'instrumental'`` → ``instrumental``, ``'amateur recording']`` → ``amateur recording``.
    """
    t = str(raw).strip().lower()
    for _ in range(4):
        if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
            t = t[1:-1].strip().lower()
        elif len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1].strip().lower()
        else:
            break
    t = re.sub(r"^[\[\('\"]+", "", t)
    t = re.sub(r"[\]\)'\"]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or str(raw).strip()
