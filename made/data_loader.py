"""Evaluation-record loader for MADE.

MADE diagnoses a model-evaluation *substrate*: a flat collection of
per-(model, sample) records, where each record is one model's answer to one
benchmark item together with its correctness and metadata.

Records are read from newline-delimited JSON files (``*.jsonl``) under a data
root. Point MADE at your data with the ``MADE_DATA_ROOT`` environment
variable, or pass ``data_root=`` explicitly. Every ``*.jsonl`` file found
recursively under the root is loaded; each non-empty line must be a JSON
object following the Record schema documented in the README:

    {
        "sample_id":          "unique id of this (model, item) record",
        "model":              "model name",
        "source_dataset":     "benchmark name",
        "language":           "language code or name",
        "country_or_culture": "country / culture label (optional)",
        "prompt":             "the input shown to the model",
        "response_raw":       "the model's raw output",
        "response_final":     "the parsed / final answer (optional)",
        "gold":               "the reference answer (optional)",
        "correct":            true | false | null,   # objective accuracy; null = no binary verdict
        "tag_category":       ["fine-grained capability tag", ...],
        "meta":               { ... arbitrary extra fields ... }
    }

Correctness is not limited to binary accuracy:
  - objective benchmarks: ``correct`` is true / false / null (null is excluded
    from accuracy denominators).
  - subjective / pairwise-judged benchmarks: set ``correct`` to null and put the
    verdict in ``meta.eval_result`` ∈ {"win", "tie", "lose"}; tools report
    win_rate = (win + 0.5*tie) / known instead of accuracy.
  - translation / generation-quality benchmarks: leave ``correct`` null and keep
    quality signals under ``meta``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger("made.data_loader")

Record = dict[str, Any]

MODEL_DISPLAY: dict[str, str] = {}

_EMPTY_MARKERS = {"", "none", "null", "n/a", "na", "nan"}


def _is_response_empty(text: Any) -> bool:
    return str(text or "").strip().lower() in _EMPTY_MARKERS


def _coerce_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "correct"}:
        return True
    if s in {"false", "0", "no", "wrong", "incorrect"}:
        return False
    return None


def _normalize(rec: dict[str, Any]) -> Record:
    response_raw = rec.get("response_raw", rec.get("response", ""))
    response_final = rec.get("response_final", response_raw)
    sample_id = str(rec.get("sample_id") or rec.get("id") or "")
    tags = rec.get("tag_category") or []
    if isinstance(tags, str):
        tags = [tags] if tags else []
    return {
        "sample_id": sample_id,
        "question_id": str(rec.get("question_id") or sample_id),
        "model": rec.get("model", ""),
        "language": rec.get("language", ""),
        "country_or_culture": rec.get("country_or_culture", ""),
        "prompt": rec.get("prompt", ""),
        "response_raw": response_raw,
        "response_final": response_final,
        "response_empty": _is_response_empty(response_raw),
        "choices": rec.get("choices", ""),
        "gold": str(rec.get("gold", "")),
        "correct": _coerce_correct(rec.get("correct")),
        "source_dataset": rec.get("source_dataset", ""),
        "tag_category": list(tags),
        "meta": rec.get("meta", {}) or {},
    }


def load_all(data_root: str | Path | None = None) -> list[Record]:
    """Load every ``*.jsonl`` evaluation record under the data root."""
    root = Path(data_root or os.environ.get("MADE_DATA_ROOT") or "data/demo")
    if not root.exists():
        raise FileNotFoundError(
            f"Data root {str(root)!r} not found. Set MADE_DATA_ROOT to a "
            f"directory of *.jsonl evaluation records, or pass data_root=."
        )
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found under {str(root)!r}.")
    records: list[Record] = []
    for f in files:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed line %d in %s", ln, f.name)
                continue
            records.append(_normalize(rec))
    MODEL_DISPLAY.update({r["model"]: r["model"] for r in records if r["model"]})
    log.info("loaded %d records from %d file(s) under %s", len(records), len(files), root)
    return records


def data_summary(records: list[Record]) -> dict[str, Any]:
    """Quick summary stats useful for prompts and sanity checks."""
    if not records:
        return {
            "total_records": 0,
            "models": [],
            "benchmarks": [],
            "languages": [],
            "countries": [],
            "num_models": 0,
            "num_benchmarks": 0,
            "num_languages": 0,
            "num_countries": 0,
            "records_per_model": {},
            "records_per_benchmark": {},
            "model_display_names": {},
            "records_per_model_display": {},
        }
    models = sorted({r["model"] for r in records})
    benchmarks = sorted({r["source_dataset"] for r in records})
    languages = sorted({r["language"] for r in records if r.get("language")})
    countries = sorted({r["country_or_culture"] for r in records if r.get("country_or_culture")})
    rpm: dict[str, int] = defaultdict(int)
    rpb: dict[str, int] = defaultdict(int)
    for r in records:
        rpm[r["model"]] += 1
        rpb[r["source_dataset"]] += 1
    model_display_names = {m: MODEL_DISPLAY.get(m, m) for m in models}
    return {
        "total_records": len(records),
        "models": models,
        "benchmarks": benchmarks,
        "languages": languages,
        "countries": countries,
        "num_models": len(models),
        "num_benchmarks": len(benchmarks),
        "num_languages": len(languages),
        "num_countries": len(countries),
        "records_per_model": dict(rpm),
        "records_per_benchmark": dict(rpb),
        "model_display_names": model_display_names,
        "records_per_model_display": {model_display_names[m]: rpm[m] for m in models},
    }


def slice_records_for_query(
    records: list[Record],
    benchmark_scope: str | None = None,
    languages: list[str] | None = None,
    target_models: list[str] | None = None,
) -> list[Record]:
    """Filter records by benchmark, language, and model.

    ``benchmark_scope`` may name a single benchmark or several joined by
    ``+`` (e.g. ``"MMMLU+BELEBELE"``). ``languages`` and ``target_models``
    are matched exactly against the record fields.
    """
    out = records
    if benchmark_scope:
        wanted = {t.strip() for t in str(benchmark_scope).split("+") if t.strip()}
        if wanted:
            out = [r for r in out if r["source_dataset"] in wanted]
    if languages:
        lang_set = set(languages)
        out = [r for r in out if r["language"] in lang_set]
    if target_models:
        model_set = set(target_models)
        out = [r for r in out if r["model"] in model_set]
    return out
