"""Loader for the MADE multilingual diagnostic query set.

Reads ``data/queryset/queries_<lang>.jsonl`` (54 expert-authored diagnostic
queries, each available in 15 languages) and returns query dicts ready for
``run_made_pipeline``.

Each query record on disk has the fields:

    {
        "id":       "Q01",                       # stable id, shared across languages
        "language": "en",
        "query":    "...",                       # the query text in this language
        "level":    "Dataset"|"Instance"|"Iteration",
        "category": "Task/Lang"|"Capability"|"Compliance"|"Behavior"|"Culture"|"Improvement",
        "template": one of the six query templates
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QUERYSET_DIR = Path(__file__).resolve().parent.parent / "data" / "queryset"

LANGUAGES = [
    "zh", "en", "ar", "de", "es", "fr", "it",
    "ja", "ko", "ms", "pl", "pt", "ru", "th", "tr",
]

RESPONSE_LANGUAGE = {
    "zh": "Chinese", "en": "English", "ar": "Arabic", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "th": "Thai", "tr": "Turkish",
}


def load_queries(lang: str = "zh", queryset_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the 54 diagnostic queries in the requested language.

    The returned dicts are ready to pass to ``run_made_pipeline``: the query
    text is exposed under both ``query`` and ``text``, and ``response_language``
    tells the Reporter which language to write the report in.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language '{lang}'. Choose one of: {LANGUAGES}")
    base = Path(queryset_dir or QUERYSET_DIR)
    path = base / f"queries_{lang}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Query set not found: {path}")

    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        q["text"] = q["query"]
        q["lang"] = lang
        q["response_language"] = RESPONSE_LANGUAGE.get(lang, "English")
        out.append(q)
    return out


def get_query(qid: str, lang: str = "zh", queryset_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a single query by id (e.g. ``"Q01"``) in the requested language."""
    for q in load_queries(lang, queryset_dir):
        if q["id"] == qid:
            return q
    raise KeyError(f"Query {qid!r} not found in language {lang!r}")
