"""Deterministic analysis tools used by the MADE agents.

Conventions:
- Three-valued correctness everywhere: ``correct`` is True / False / None.
  None records are excluded from accuracy denominators; tools return
  ``known_n`` / ``unknown_n`` / ``coverage`` so reports never compute fake
  accuracy on translation- or format-only slices.
- ``group_stats`` supports schema-aware axes: country_or_culture / model /
  language / source_dataset / tag_category / round_number / failure_type.
- Tag tools support sort_by = count | error_rate | wrong_count.
- Case-pack output keys are standardized: sample_id / source_dataset /
  language / country / model / tag_category / failure_type / why_selected.
"""

import os
import re
from collections import Counter, defaultdict
from typing import Any
from made.data_loader import Record


def _default_min_cell_n() -> int:
    """Minimum samples per (model, benchmark, language) cell for it to be
    reported in the per-language dashboard breakdown. Defaults to 20 (a
    statistical-robustness floor); override with MADE_MIN_CELL_N or by
    passing ``min_cell_n=`` explicitly. Lower it for small datasets so the
    per-language breakdown is not filtered away."""
    try:
        return int(os.environ.get("MADE_MIN_CELL_N", "20"))
    except ValueError:
        return 20

def _record_correct(rec: Record):
    """Return rec['correct'] as True / False / None.

    Anything not exactly True or False is treated as None (unknown), which
    keeps the field robust to upstream str/int leakage.
    """
    c = rec.get("correct")
    if c is True:
        return True
    if c is False:
        return False
    return None

def _split_by_correct(records: list[Record]):
    """Return (known_records, unknown_records).

    known: correct in {True, False}. unknown: correct is None.
    """
    known, unknown = [], []
    for r in records:
        if _record_correct(r) is None:
            unknown.append(r)
        else:
            known.append(r)
    return known, unknown

def _accuracy(records: list[Record]) -> float:
    """Accuracy over records with KNOWN binary correctness.

    Returns 0.0 when no known records — but callers should consult
    `known_n` to decide whether the value is meaningful.
    """
    known, _ = _split_by_correct(records)
    if not known:
        return 0.0
    return sum(1 for r in known if _record_correct(r) is True) / len(known)

def _response_rate(records: list[Record]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if not r.get("response_empty")) / len(records)

def _coverage(records: list[Record]) -> dict:
    known, unknown = _split_by_correct(records)
    return {
        "known_n": len(known),
        "unknown_n": len(unknown),
        "coverage": round(len(known) / len(records), 4) if records else 0.0,
    }

SUBJECTIVE_BENCHMARKS = {"S-AlpacaEval", "S-MT-Bench"}

def _is_subjective_record(rec: Record) -> bool:
    return rec.get("source_dataset") in SUBJECTIVE_BENCHMARKS

def _subjective_eval(rec: Record) -> str:
    """Map a subjective record's eval_result to a normalised label.

    Returns 'win' | 'tie' | 'lose' | 'unknown'. Lives in record.meta.eval_result
    """
    meta = rec.get("meta") or {}
    ev = meta.get("eval_result")
    if ev in ("win", "tie", "lose"):
        return ev
    return "unknown"

def _subjective_breakdown(records: list[Record]) -> dict:
    """Compute win/tie/lose/unknown counts + two win_rate variants.

    win_rate      = (win_n + 0.5 * tie_n) / known_n     ← paper main metric
    only_win_rate = win_n / known_n                      ← debug / strict fact pack
    known_n       = win_n + tie_n + lose_n

    For non-subjective records this returns is_subjective=False with zero
    counts; the caller should ignore win_rate fields in that case.
    """
    win_n = tie_n = lose_n = unknown_n = 0
    has_subjective = False
    for r in records:
        if not _is_subjective_record(r):
            continue
        has_subjective = True
        ev = _subjective_eval(r)
        if ev == "win":
            win_n += 1
        elif ev == "tie":
            tie_n += 1
        elif ev == "lose":
            lose_n += 1
        else:
            unknown_n += 1
    known_n = win_n + tie_n + lose_n
    return {
        "is_subjective": has_subjective,
        "win_n": win_n,
        "tie_n": tie_n,
        "lose_n": lose_n,
        "unknown_n": unknown_n,
        "subjective_known_n": known_n,
        "win_rate": round((win_n + 0.5 * tie_n) / known_n, 4) if known_n else None,
        "only_win_rate": round(win_n / known_n, 4) if known_n else None,
    }

def slice_filter(
    records: list[Record],
    model: str | None = None,
    country: str | None = None,
    correct: bool | None = None,
    response_empty: bool | None = None,
    question_id: str | None = None,
    language: str | None = None,
    source_dataset: str | None = None,
    tag: str | None = None,
) -> list[Record]:
    """Filter records by various criteria.

    `correct` is matched STRICTLY (True/False); to match unknown, pass
    `correct=None` is reserved for "no filter on correctness". Use
    `correct=False` and `correct=True` for binary filters; if you need
    only `correct is None` records, filter the result explicitly.
    """
    out = records
    if model is not None:
        out = [r for r in out if r.get("model") == model]
    if country is not None:
        out = [r for r in out if r.get("country_or_culture") == country]
    if correct is True:
        out = [r for r in out if _record_correct(r) is True]
    elif correct is False:
        out = [r for r in out if _record_correct(r) is False]
    if response_empty is not None:
        out = [r for r in out if r.get("response_empty") == response_empty]
    if question_id is not None:
        out = [r for r in out if r.get("question_id") == question_id]
    if language is not None:
        out = [r for r in out if r.get("language") == language]
    if source_dataset is not None:
        out = [r for r in out if r.get("source_dataset") == source_dataset]
    if tag is not None:
        out = [r for r in out if tag in _record_tags(r)]
    return out

GROUP_AXES = {
    "country_or_culture",
    "model",
    "language",
    "source_dataset",
    "tag_category",
    "round_number",
    "failure_type",
    "question_id",
}

def _group_key(rec: Record, axis: str, all_records: list[Record] | None = None):
    """Extract group key for a record on a given axis.

    For tag_category the record may have multiple tags — we return a list of
    tags so the caller can fan out (one record can land in multiple tag groups).
    For round_number the value lives under rec['meta']['round_number'].

    when classifying by `failure_type`, callers can pass
    `all_records` so cross-model ambiguity / factual_error checks work
    (without it the classifier degrades to 3-class and never emits
    `ambiguity` / `factual_error`).
    """
    if axis == "tag_category":
        return _record_tags(rec)
    if axis == "round_number":
        v = (rec.get("meta") or {}).get("round_number")
        return [v] if v is not None else []
    if axis == "failure_type":
        if _record_correct(rec) is True:
            return ["correct"]
        if _record_correct(rec) is None:
            return ["unknown"]
        ft = classify_failure(rec, all_records)
        return [ft]
    v = rec.get(axis)
    return [v] if v else []

def group_stats(
    records: list[Record],
    group_by: str = "country_or_culture",
) -> list[dict[str, Any]]:
    """Compute accuracy / response rate / known_n / coverage by an axis.

    Supports the schema-aware axes listed in GROUP_AXES. For
    `tag_category` and `round_number`, one record may contribute to
    multiple groups (tag fan-out) or none (missing round_number).
    """
    if group_by not in GROUP_AXES:
        groups: dict[Any, list[Record]] = defaultdict(list)
        for r in records:
            v = r.get(group_by)
            if v:
                groups[v].append(r)
    else:
        groups = defaultdict(list)
        all_ref = records if group_by == "failure_type" else None
        for r in records:
            for k in _group_key(r, group_by, all_ref):
                groups[k].append(r)
    results = []
    for key in sorted(groups, key=lambda x: str(x)):
        g = groups[key]
        cov = _coverage(g)
        row = {
            "group": key,
            "axis": group_by,
            "count": len(g),
            "known_n": cov["known_n"],
            "unknown_n": cov["unknown_n"],
            "coverage": cov["coverage"],
            "accuracy": round(_accuracy(g), 4),
            "response_rate": round(_response_rate(g), 4),
            "correct_count": sum(1 for r in g if _record_correct(r) is True),
            "wrong_count": sum(1 for r in g if _record_correct(r) is False),
            "empty_count": sum(1 for r in g if r.get("response_empty")),
        }
        subj = _subjective_breakdown(g)
        if subj["is_subjective"]:
            row["subjective"] = {k: v for k, v in subj.items() if k != "is_subjective"}
            row["metric_semantics"] = "win_rate"
        results.append(row)
    return results

def overall_stats(records: list[Record]) -> dict[str, Any]:
    cov = _coverage(records)
    out = {
        "count": len(records),
        "known_n": cov["known_n"],
        "unknown_n": cov["unknown_n"],
        "coverage": cov["coverage"],
        "accuracy": round(_accuracy(records), 4),
        "response_rate": round(_response_rate(records), 4),
        "correct_count": sum(1 for r in records if _record_correct(r) is True),
        "wrong_count": sum(1 for r in records if _record_correct(r) is False),
        "empty_count": sum(1 for r in records if r.get("response_empty")),
    }
    subj = _subjective_breakdown(records)
    if subj["is_subjective"]:
        out["subjective"] = {k: v for k, v in subj.items() if k != "is_subjective"}
        out["metric_semantics"] = "win_rate"
    return out

def flores_quality_guard(records: list[Record]) -> dict[str, Any]:
    """Report whether a FLORES-101 slice has reliable
    translation-quality signal.

    Some FLORES-style data expose `judge_score` (0/1 if any) but in practice
    almost all rows are `correct=None` — translation quality is BLEU/COMET-
    style, not binary. Without a quality field we cannot claim "Model X is
    better at translation"; we can only report structural signals
    (length_ratio, response_rate, degenerate_flag).

    The reporter SHOULD quote `recommended_caveat` verbatim whenever the
    slice's main benchmark is Flores-101 and `has_quality_score=False`.
    """
    flores = [r for r in records if r.get("source_dataset") == "Flores-101"]
    is_flores = len(flores) > 0
    if not is_flores:
        return {
            "is_flores": False,
            "has_quality_score": True,
            "flores_record_n": 0,
            "recommended_caveat": "",
        }
    def _has_q(r: dict[str, Any]) -> bool:
        if r.get("judge_score") is not None:
            return True
        m = r.get("meta") or {}
        for k in ("judge_score", "comet", "bleu", "chrf", "translation_quality_score"):
            if m.get(k) is not None:
                return True
        return False
    has_quality = any(_has_q(r) for r in flores)
    caveat = (
        ""
        if has_quality
        else (
            "Translation quality unavailable: this slice has no per-sample "
            "quality score (no COMET / BLEU / judge_score in the underlying "
            "FLORES-101 data). Do NOT claim that one model is better or "
            "worse at translation. Only structural signals "
            "(length_ratio / response_rate / degenerate_flag) may be cited."
        )
    )
    return {
        "is_flores": True,
        "has_quality_score": has_quality,
        "flores_record_n": len(flores),
        "recommended_caveat": caveat,
    }

def subjective_breakdown(
    records: list[Record],
    group_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Explicit subjective win/tie/lose tool.

    Use this for S-AlpacaEval / S-MT-Bench slices instead of `group_stats`'s
    `accuracy` field. When `group_by` is None, returns a single dict over
    all subjective records. When set (e.g. 'language', 'model'), returns a
    list per-group with `win_rate` semantics, suitable for the evidence
    table's `metric_name='win_rate'` rows.
    """
    if group_by is None:
        subj = _subjective_breakdown(records)
        return {
            "metric_semantics": "win_rate",
            **{k: v for k, v in subj.items()},
        }
    groups: dict[Any, list[Record]] = defaultdict(list)
    for r in records:
        if not _is_subjective_record(r):
            continue
        for k in _group_key(r, group_by, None):
            groups[k].append(r)
    out = []
    for key in sorted(groups, key=lambda x: str(x)):
        g = groups[key]
        b = _subjective_breakdown(g)
        out.append({
            "group": key,
            "axis": group_by,
            "count": len(g),
            "metric_semantics": "win_rate",
            **{k: v for k, v in b.items() if k != "is_subjective"},
        })
    return out

def top_bottom_slices(
    stats_list: list[dict[str, Any]],
    metric: str = "accuracy",
    n: int = 5,
    min_count: int = 5,
    use_known_n: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Return top-N and bottom-N slices.

    defaults to known_n for the min-count filter (so translation-only
    slices with all known_n=0 don't dominate the bottom).
    """
    cnt_key = "known_n" if use_known_n else "count"
    filtered = [s for s in stats_list if s.get(cnt_key, s.get("count", 0)) >= min_count]
    by_metric = sorted(filtered, key=lambda s: s.get(metric, 0))
    return {
        "bottom": by_metric[:n],
        "top": by_metric[-n:][::-1],
    }

def compare_models(
    records: list[Record],
    model_a: str,
    model_b: str,
    group_by: str = "country_or_culture",
) -> list[dict[str, Any]]:
    """Compare two models slice by slice along an axis."""
    recs_a = slice_filter(records, model=model_a)
    recs_b = slice_filter(records, model=model_b)
    stats_a = {s["group"]: s for s in group_stats(recs_a, group_by)}
    stats_b = {s["group"]: s for s in group_stats(recs_b, group_by)}
    all_groups = sorted(set(stats_a) | set(stats_b), key=lambda x: str(x))
    results = []
    for g in all_groups:
        sa = stats_a.get(g, {"accuracy": 0, "response_rate": 0, "count": 0, "known_n": 0})
        sb = stats_b.get(g, {"accuracy": 0, "response_rate": 0, "count": 0, "known_n": 0})
        results.append({
            "group": g,
            "axis": group_by,
            "model_a": model_a,
            "model_b": model_b,
            "accuracy_a": sa.get("accuracy", 0),
            "accuracy_b": sb.get("accuracy", 0),
            "accuracy_delta": round(sa.get("accuracy", 0) - sb.get("accuracy", 0), 4),
            "response_rate_a": sa.get("response_rate", 0),
            "response_rate_b": sb.get("response_rate", 0),
            "known_n_a": sa.get("known_n", 0),
            "known_n_b": sb.get("known_n", 0),
            "count_a": sa.get("count", 0),
            "count_b": sb.get("count", 0),
        })
    return results

def compare_overall(records: list[Record]) -> list[dict[str, Any]]:
    """Overall stats for each model."""
    models = sorted({r["model"] for r in records})
    results = []
    for m in models:
        s = overall_stats(slice_filter(records, model=m))
        s["model"] = m
        results.append(s)
    return results

def retrieve_error_cases(
    records: list[Record],
    model: str | None = None,
    country: str | None = None,
    limit: int = 10,
    language: str | None = None,
    source_dataset: str | None = None,
) -> list[Record]:
    """Retrieve representative error cases (correct=False)."""
    filtered = slice_filter(
        records, model=model, country=country, correct=False,
        language=language, source_dataset=source_dataset,
    )
    filtered.sort(key=lambda r: r.get("sample_id", ""))
    return filtered[:limit]

def retrieve_disagreement_cases(
    records: list[Record],
    model_correct: str,
    model_wrong: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find cases where model_correct got it right but model_wrong got it wrong.

    cross-model alignment is by sample_id (UUID). Only known/known
    pairs participate (skip if either model's record is correct=None).
    """
    by_sample: dict[str, dict[str, Record]] = defaultdict(dict)
    for r in records:
        if r.get("model") in (model_correct, model_wrong):
            by_sample[r["sample_id"]][r["model"]] = r

    cases = []
    for sid, models in sorted(by_sample.items()):
        if model_correct in models and model_wrong in models:
            rc = models[model_correct]
            rw = models[model_wrong]
            rc_c = _record_correct(rc)
            rw_c = _record_correct(rw)
            if rc_c is True and rw_c is False:
                cases.append({
                    "sample_id": sid,
                    "question_id": rc.get("question_id"),
                    "source_dataset": rc.get("source_dataset"),
                    "language": rc.get("language"),
                    "country": rc.get("country_or_culture"),
                    "tag_category": _record_tags(rc),
                    "prompt": rc.get("prompt", ""),
                    "choices": rc.get("choices", ""),
                    "gold": rc.get("gold", ""),
                    "correct_model": model_correct,
                    "correct_response": rc.get("response_raw", "")[:500],
                    "correct_answer": rc.get("response_final", ""),
                    "wrong_model": model_wrong,
                    "wrong_response": rw.get("response_raw", "")[:500],
                    "wrong_answer": rw.get("response_final", ""),
                })
    return cases[:limit]

def inspect_response_patterns(
    records: list[Record],
    model: str | None = None,
) -> dict[str, Any]:
    """Analyze response patterns for a set of records.

    cognitive_failure_rate / completion_failure_rate are computed only
    over known records to avoid translation tasks producing fake numbers.
    """
    recs = slice_filter(records, model=model) if model else records
    total = len(recs)
    if total == 0:
        return {"total": 0, "known_n": 0, "unknown_n": 0, "coverage": 0.0}

    cov = _coverage(recs)
    known_n = cov["known_n"]
    empty_count = sum(1 for r in recs if r.get("response_empty"))
    answer_dist = Counter(r.get("response_final", "") for r in recs if r.get("response_final"))
    no_answer = sum(1 for r in recs if not r.get("response_final"))

    if answer_dist:
        most_common_letter, most_common_count = answer_dist.most_common(1)[0]
        expected_pct = 1.0 / max(len(answer_dist), 1)
        actual_pct = most_common_count / sum(answer_dist.values())
        answer_bias = actual_pct > expected_pct * 1.5
    else:
        most_common_letter, answer_bias = "", False

    correct_count = sum(1 for r in recs if _record_correct(r) is True)
    wrong_answered = sum(1 for r in recs if _record_correct(r) is False and r.get("response_final"))
    wrong_empty = sum(1 for r in recs if _record_correct(r) is False and not r.get("response_final"))

    return {
        "total": total,
        "known_n": known_n,
        "unknown_n": cov["unknown_n"],
        "coverage": cov["coverage"],
        "empty_responses": empty_count,
        "empty_rate": round(empty_count / total, 4),
        "no_extractable_answer": no_answer,
        "no_answer_rate": round(no_answer / total, 4),
        "answer_distribution": dict(answer_dist),
        "most_common_answer": most_common_letter,
        "answer_bias_detected": answer_bias,
        "correct": correct_count,
        "wrong_answered": wrong_answered,
        "wrong_empty": wrong_empty,
        "accuracy": round(_accuracy(recs), 4) if known_n else None,
        "cognitive_failure_rate": round(wrong_answered / known_n, 4) if known_n else None,
        "completion_failure_rate": round(wrong_empty / known_n, 4) if known_n else None,
    }

def detect_degenerate_repetition(
    records: list[Record],
    model: str | None = None,
) -> dict[str, Any]:
    """Detect degenerate/repetitive output patterns in model responses."""
    recs = slice_filter(records, model=model) if model else records
    total = len(recs)
    if total == 0:
        return {"total": 0, "degenerate_count": 0}

    degenerate_cases = []
    for r in recs:
        raw = r.get("response_raw", "")
        if not raw or len(raw) < 20:
            continue
        score, pattern_type = _compute_repetition_score(raw)
        if score > 0.4:
            degenerate_cases.append({
                "sample_id": r["sample_id"],
                "source_dataset": r.get("source_dataset"),
                "language": r.get("language"),
                "country": r.get("country_or_culture"),
                "model": r.get("model"),
                "repetition_score": round(score, 3),
                "pattern_type": pattern_type,
                "excerpt": raw[:200],
                "response_length": len(raw),
            })

    degenerate_cases.sort(key=lambda x: x["repetition_score"], reverse=True)
    return {
        "total": total,
        "degenerate_count": len(degenerate_cases),
        "degenerate_rate": round(len(degenerate_cases) / total, 4) if total else 0,
        "worst_cases": degenerate_cases[:10],
        "by_country": _degenerate_by_country(degenerate_cases),
    }

def _compute_repetition_score(text: str) -> tuple[float, str]:
    """Compute a repetition score for a response. Returns (score, pattern_type)."""
    if len(text) < 20:
        return 0.0, "none"
    text_len = len(text)

    loop_match = re.search(r'(.{10,}?)\1{2,}', text)
    if loop_match:
        repeated_span = loop_match.end() - loop_match.start()
        if repeated_span / text_len > 0.2:
            return 0.9, "phrase_loop"

    char_runs = re.findall(r'(.)\1{4,}', text)
    if char_runs:
        run_total = sum(len(m) for m in re.findall(r'(.)\1{4,}', text))
        if run_total / text_len > 0.2:
            return 0.8, "char_repetition"

    sentences = [s.strip() for s in re.split(r'[。.!！?？\n]+', text) if len(s.strip()) > 5]
    if len(sentences) >= 4:
        sent_counts = Counter(sentences)
        repeated_sents = sum(c - 1 for c in sent_counts.values() if c > 1)
        sent_repeat_rate = repeated_sents / len(sentences)
        if sent_repeat_rate > 0.4:
            return min(0.6 + sent_repeat_rate * 0.3, 1.0), "sentence_repeat"

    struct_matches = re.findall(r'(\{[^{}]{5,50}\})\s*\1', text)
    if len(struct_matches) >= 2:
        return 0.7, "structure_collapse"

    words = text.split()
    if len(words) >= 10:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            repeated = sum(c - 1 for c in trigram_counts.values() if c > 1)
            trigram_score = repeated / len(trigrams)

            if trigram_score > 0.5:
                return min(trigram_score, 1.0), "phrase_loop"

            unique_words = len(set(words))
            vocab_ratio = unique_words / len(words)
            if vocab_ratio < 0.15 and len(words) > 50:
                return 0.7, "template_collapse"

            if trigram_score > 0.4:
                return round(trigram_score, 3), "mild_repetition"

    return 0.0, "none"

def _degenerate_by_country(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for c in cases:
        if c.get("country"):
            counts[c["country"]] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

FAILURE_TYPES = [
    "cognitive_failure",
    "output_failure",
    "degenerate_output",
    "factual_error",
    "ambiguity",
]

def detect_cross_model_ambiguity(
    all_records: list[Record],
    sample_id: str,
) -> bool:
    """Cross-model ambiguity check: True if ≥2 distinct non-empty answers AND
    fewer than half of responding models match the gold."""
    answers: list[str] = []
    gold = None
    for r in all_records:
        if r.get("sample_id") != sample_id:
            continue
        gold = r.get("gold", "")
        final = r.get("response_final", "")
        if final:
            answers.append(final)
    if len(set(answers)) < 2:
        return False
    correct_count = sum(1 for a in answers if a == gold)
    return correct_count < len(answers) / 2

def _nominate_factual_error(
    record: Record,
    all_records: list[Record],
) -> bool:
    """Factual-error candidate: ≥2 other models got this sample right
    AND this model has a substantive non-empty response."""
    raw = record.get("response_raw", "")
    if len(raw) < 50:
        return False
    sid = record["sample_id"]
    model = record["model"]
    correct_others = 0
    for r in all_records:
        if r.get("sample_id") == sid and r.get("model") != model and _record_correct(r) is True:
            correct_others += 1
    return correct_others >= 2

def classify_failure(
    record: Record,
    all_records: list[Record] | None = None,
) -> str:
    """Classify a single record's failure type.

    Three-valued aware: returns "unknown" when correct is None (no binary GT)
    and "correct" when correct is True. Otherwise picks one of the 5 failure
    types using the cascade.
    """
    c = _record_correct(record)
    if c is True:
        return "correct"
    if c is None:
        return "unknown"

    final = record.get("response_final", "")
    raw = record.get("response_raw", "")

    if record.get("response_empty") or not final:
        return "output_failure"

    score, _ = _compute_repetition_score(raw)
    if score > 0.4:
        return "degenerate_output"

    if all_records is not None:
        sid = record["sample_id"]
        if detect_cross_model_ambiguity(all_records, sid):
            return "ambiguity"
        if record.get("_factual_verdict") == "factual_error":
            return "factual_error"
        if _nominate_factual_error(record, all_records):
            return "factual_error"

    if final:
        return "cognitive_failure"
    return "output_failure"

def retrieve_cases_by_failure_type(
    records: list[Record],
    model: str | None = None,
    failure_type: str | None = None,
    country: str | None = None,
    limit: int = 8,
    all_records: list[Record] | None = None,
    language: str | None = None,
    source_dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve cases classified by failure type with case-pack fields."""
    recs = slice_filter(
        records, model=model, country=country, correct=False,
        language=language, source_dataset=source_dataset,
    )
    cross_ref = all_records if all_records is not None else records

    classified = []
    for r in recs:
        ft = classify_failure(r, cross_ref)
        if failure_type and ft != failure_type:
            continue
        classified.append(_to_case_pack(r, ft, why=f"failure_type={ft}"))

    classified.sort(key=lambda x: (x.get("country") or "", x.get("sample_id") or ""))
    return classified[:limit]

def failure_type_stats(
    records: list[Record],
    model: str | None = None,
    all_records: list[Record] | None = None,
) -> dict[str, Any]:
    """Compute failure type distribution.

    distribution is over `wrong_n` only (correct=False), and we report
    `correct_n`, `wrong_n`, `unknown_n` separately.
    """
    recs = slice_filter(records, model=model)
    correct_n = sum(1 for r in recs if _record_correct(r) is True)
    wrong_recs = [r for r in recs if _record_correct(r) is False]
    unknown_n = sum(1 for r in recs if _record_correct(r) is None)
    total = len(recs)
    wrong_n = len(wrong_recs)

    cross_ref = all_records if all_records is not None else records
    counts: dict[str, int] = defaultdict(int)
    for r in wrong_recs:
        ft = classify_failure(r, cross_ref)
        counts[ft] += 1

    dist = {
        ft: {"count": c, "rate": round(c / wrong_n, 4) if wrong_n else 0.0}
        for ft, c in sorted(counts.items(), key=lambda x: -x[1])
    }
    return {
        "total": total,
        "correct_n": correct_n,
        "wrong_n": wrong_n,
        "unknown_n": unknown_n,
        "total_wrong": wrong_n,
        "distribution": dist,
    }

def compare_groups_full(
    records: list[Record],
    models: list[str] | None = None,
    group_by: str = "country_or_culture",
) -> list[dict[str, Any]]:
    """Full group-level comparison table across all models."""
    if models is None:
        models = sorted({r["model"] for r in records})

    model_group_stats: dict[str, dict[str, dict]] = {}
    for m in models:
        gs = group_stats(slice_filter(records, model=m), group_by)
        model_group_stats[m] = {s["group"]: s for s in gs}

    all_groups = sorted({g for mgs in model_group_stats.values() for g in mgs}, key=lambda x: str(x))
    results = []
    for g in all_groups:
        row: dict[str, Any] = {"group": g, "axis": group_by}
        for m in models:
            s = model_group_stats.get(m, {}).get(g, {})
            row[f"{m}_accuracy"] = s.get("accuracy", 0)
            row[f"{m}_known_n"] = s.get("known_n", 0)
            row[f"{m}_count"] = s.get("count", 0)
            row[f"{m}_empty_count"] = s.get("empty_count", 0)
            row[f"{m}_response_rate"] = s.get("response_rate", 0)
        accs = [row.get(f"{m}_accuracy", 0) for m in models]
        row["mean_accuracy"] = round(sum(accs) / len(accs), 4) if accs else 0
        row["max_delta"] = round(max(accs) - min(accs), 4) if accs else 0
        row["min_known_n"] = min((row.get(f"{m}_known_n", 0) for m in models), default=0)
        row["min_count"] = min((row.get(f"{m}_count", 0) for m in models), default=0)
        results.append(row)

    return results

def estimate_group_support(
    records: list[Record],
    model: str | None = None,
    group: str | None = None,
    axis: str = "country_or_culture",
) -> dict[str, Any]:
    """Estimate how well-supported conclusions about a group are.

    support level depends on known_n, not raw count.
    """
    recs = records
    if model:
        recs = slice_filter(recs, model=model)
    if group:
        if axis == "country_or_culture":
            recs = slice_filter(recs, country=group)
        elif axis == "language":
            recs = slice_filter(recs, language=group)
        elif axis == "source_dataset":
            recs = slice_filter(recs, source_dataset=group)
        elif axis == "tag_category":
            recs = slice_filter(recs, tag=group)

    n = len(recs)
    cov = _coverage(recs)
    known_n = cov["known_n"]
    if n == 0:
        return {"count": 0, "known_n": 0, "support_level": "none",
                "warning": "No data for this group"}

    acc = _accuracy(recs)
    variance_proxy = round(acc * (1 - acc) / known_n, 6) if known_n else None

    if known_n >= 100:
        level = "high"
        warning = ""
    elif known_n >= 50:
        level = "medium"
        warning = f"known_n {known_n} is moderate; interpret with caution"
    elif known_n >= 20:
        level = "low"
        warning = f"known_n {known_n} is small; conclusions are tentative"
    elif known_n > 0:
        level = "very_low"
        warning = f"known_n {known_n} is too small for reliable conclusions"
    else:
        level = "no_known"
        warning = "No records with binary correctness; cannot compute accuracy"

    return {
        "count": n,
        "known_n": known_n,
        "unknown_n": cov["unknown_n"],
        "coverage": cov["coverage"],
        "accuracy": round(acc, 4) if known_n else None,
        "support_level": level,
        "variance_proxy": variance_proxy,
        "warning": warning,
    }

def find_high_disagreement_cases_by_group(
    records: list[Record],
    model_a: str,
    model_b: str,
    group: str,
    limit: int = 5,
):
    """Find disagreement cases within a specific group."""
    group_recs = slice_filter(records, country=group)
    a_right = retrieve_disagreement_cases(group_recs, model_a, model_b, limit=limit)
    b_right = retrieve_disagreement_cases(group_recs, model_b, model_a, limit=limit)

    return {
        "group": group,
        f"{model_a}_correct_{model_b}_wrong": a_right,
        f"{model_b}_correct_{model_a}_wrong": b_right,
        "total_disagreements": len(a_right) + len(b_right),
    }

def _record_tags(rec: Record) -> list[str]:
    """Normalize a record's tag list."""
    tags = rec.get("tag_category") or []
    if isinstance(tags, str):
        tags = [tags]
    return [t for t in tags if t]

def tag_stats(
    records: list[Record],
    model: str | None = None,
    top_k: int = 30,
    sort_by: str = "count",
    min_count: int = 0,
) -> list[dict[str, Any]]:
    """Per-tag accuracy + sample count + error_rate + wrong_count.

    sort_by: "count" (default), "error_rate", "wrong_count", "accuracy",
    or "delta" (delta vs cross-model mean — only meaningful when model is set).
    """
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        if model and r.get("model") != model:
            continue
        for t in _record_tags(r):
            d = out.setdefault(t, {"tag": t, "count": 0, "correct": 0, "wrong": 0, "unknown": 0})
            d["count"] += 1
            c = _record_correct(r)
            if c is True:
                d["correct"] += 1
            elif c is False:
                d["wrong"] += 1
            else:
                d["unknown"] += 1
    rows = []
    for d in out.values():
        n = d["count"]
        known = d["correct"] + d["wrong"]
        wrong = d["wrong"]
        rows.append({
            "tag": d["tag"],
            "count": n,
            "known_n": known,
            "unknown_n": d["unknown"],
            "coverage": round(known / n, 4) if n else 0.0,
            "accuracy": round(d["correct"] / known, 4) if known else None,
            "wrong_count": wrong,
            "error_rate": round(wrong / known, 4) if known else None,
        })

    if model is not None and sort_by == "delta":
        all_tag = {row["tag"]: row for row in tag_stats(records, model=None, top_k=10**6)}
        for row in rows:
            other_acc = all_tag.get(row["tag"], {}).get("accuracy")
            if row.get("accuracy") is not None and other_acc is not None:
                row["delta_vs_all"] = round(row["accuracy"] - other_acc, 4)
            else:
                row["delta_vs_all"] = None

    rows = [r for r in rows if r["count"] >= min_count]

    if sort_by == "error_rate":
        rows.sort(key=lambda x: -(x.get("error_rate") or -1))
    elif sort_by == "wrong_count":
        rows.sort(key=lambda x: -x["wrong_count"])
    elif sort_by == "accuracy":
        rows.sort(key=lambda x: (x.get("accuracy") if x.get("accuracy") is not None else -1))
    elif sort_by == "delta":
        rows.sort(key=lambda x: (x.get("delta_vs_all") if x.get("delta_vs_all") is not None else 0))
    else:
        rows.sort(key=lambda x: -x["count"])
    return rows[:top_k]

def failure_type_stats_by_tag(
    records: list[Record],
    model: str,
    tag: str,
) -> dict[str, Any]:
    """Failure-type distribution for one (model, tag) slice.

    cross-model reference uses ALL records (full slice the caller
    passed), not just the (model, tag) slice. This was an earlier issue
    where ambiguity/factual classification couldn't see other models'
    answers when the slice was tag-only.
    """
    sliced = [r for r in records if r.get("model") == model and tag in _record_tags(r)]
    return failure_type_stats(sliced, model=model, all_records=records)

def retrieve_cases_by_tag(
    records: list[Record],
    model: str,
    tag: str,
    correct: bool | None = False,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve representative cases for a (model, tag) combination."""
    out = []
    for r in records:
        if r.get("model") != model:
            continue
        if tag not in _record_tags(r):
            continue
        if correct is True and _record_correct(r) is not True:
            continue
        if correct is False and _record_correct(r) is not False:
            continue
        out.append(_to_case_pack(r, classify_failure(r, records),
                                  why=f"tag={tag}; correct={correct}"))
        if len(out) >= limit:
            break
    return out

def tag_language_matrix(
    records: list[Record],
    model: str | None = None,
    top_tags: int = 10,
    top_langs: int = 10,
) -> dict[str, Any]:
    """tag × language interaction weakness matrix.

    Returns top tags × top languages cells with accuracy / known_n.
    Top axis is selected by total known_n descending so we focus on
    well-supported intersections.
    """
    recs = slice_filter(records, model=model) if model else records
    ts = tag_stats(recs, model=None, top_k=10**6, sort_by="count")
    tags_pick = [t["tag"] for t in ts[:top_tags] if t["tag"]]
    lang_groups = group_stats(recs, "language")
    lang_groups_sorted = sorted(lang_groups, key=lambda x: -(x.get("known_n", 0)))
    langs_pick = [g["group"] for g in lang_groups_sorted[:top_langs] if g.get("group")]

    cells = []
    for tag in tags_pick:
        for lang in langs_pick:
            slice_recs = [r for r in recs if r.get("language") == lang and tag in _record_tags(r)]
            if not slice_recs:
                continue
            cov = _coverage(slice_recs)
            cells.append({
                "tag": tag, "language": lang,
                "count": len(slice_recs),
                "known_n": cov["known_n"],
                "unknown_n": cov["unknown_n"],
                "accuracy": round(_accuracy(slice_recs), 4) if cov["known_n"] else None,
            })
    cells.sort(key=lambda x: (x.get("accuracy") if x.get("accuracy") is not None else 1, -x["known_n"]))
    return {
        "model": model,
        "tags": tags_pick,
        "languages": langs_pick,
        "cells": cells,
        "weakest_cells": [c for c in cells if c.get("accuracy") is not None and c["known_n"] >= 20][:15],
    }

def model_tag_matrix(
    records: list[Record],
    target_models: list[str] | None = None,
    top_tags: int = 10,
) -> dict[str, Any]:
    """model × tag matrix; helps surface model-specific tag weaknesses."""
    if target_models is None:
        target_models = sorted({r["model"] for r in records})
    ts = tag_stats(records, model=None, top_k=10**6, sort_by="count")
    tags_pick = [t["tag"] for t in ts[:top_tags] if t["tag"]]
    rows = []
    for tag in tags_pick:
        row = {"tag": tag}
        accs = []
        for m in target_models:
            slice_recs = [r for r in records if r.get("model") == m and tag in _record_tags(r)]
            cov = _coverage(slice_recs)
            acc = round(_accuracy(slice_recs), 4) if cov["known_n"] else None
            row[f"{m}_accuracy"] = acc
            row[f"{m}_known_n"] = cov["known_n"]
            if acc is not None:
                accs.append(acc)
        row["max_delta"] = round(max(accs) - min(accs), 4) if accs else None
        rows.append(row)
    rows.sort(key=lambda x: -(x["max_delta"] or 0))
    return {
        "models": target_models,
        "tags": tags_pick,
        "rows": rows,
    }

def benchmark_dashboard(
    records: list[Record],
    target_models: list[str] | None = None,
    min_cell_n: int | None = None,
) -> dict[str, Any]:
    """Wide model × benchmark × language dashboard.

    Used for queries that ask "compare X across all benchmarks" without
    forcing the analyst to call group_stats N times.

    ``min_cell_n`` is the minimum known-correctness sample count for a
    (model, benchmark, language) cell to appear in the per-language
    breakdown; it defaults to ``MADE_MIN_CELL_N`` (20). Lower it on small
    datasets so the per-language breakdown is not filtered away.
    """
    if min_cell_n is None:
        min_cell_n = _default_min_cell_n()
    if target_models is None:
        target_models = sorted({r["model"] for r in records})
    benchmarks = sorted({r.get("source_dataset") for r in records if r.get("source_dataset")})
    rows = []
    for b in benchmarks:
        b_recs = slice_filter(records, source_dataset=b)
        row = {"benchmark": b, "total": len(b_recs)}
        for m in target_models:
            mb = slice_filter(b_recs, model=m)
            cov = _coverage(mb)
            row[f"{m}_accuracy"] = round(_accuracy(mb), 4) if cov["known_n"] else None
            row[f"{m}_known_n"] = cov["known_n"]
            row[f"{m}_unknown_n"] = cov["unknown_n"]
            row[f"{m}_response_rate"] = round(_response_rate(mb), 4)
        rows.append(row)
    weak_cells = []
    for r in records[:0]:
        pass
    cells: dict[tuple, dict] = {}
    for r in records:
        m = r.get("model")
        b = r.get("source_dataset")
        lang = r.get("language")
        if m not in target_models or not b or not lang:
            continue
        k = (m, b, lang)
        c = cells.setdefault(k, {"correct": 0, "wrong": 0, "unknown": 0, "total": 0})
        c["total"] += 1
        cc = _record_correct(r)
        if cc is True:
            c["correct"] += 1
        elif cc is False:
            c["wrong"] += 1
        else:
            c["unknown"] += 1
    flat = []
    for (m, b, lang), c in cells.items():
        known = c["correct"] + c["wrong"]
        if known < min_cell_n:
            continue
        flat.append({
            "model": m, "benchmark": b, "language": lang,
            "known_n": known, "unknown_n": c["unknown"],
            "accuracy": round(c["correct"] / known, 4),
        })
    flat.sort(key=lambda x: x["accuracy"])
    return {
        "models": target_models,
        "benchmarks": benchmarks,
        "rows": rows,
        "weakest_model_benchmark_language_cells": flat[:25],
    }

def representative_case_search(
    records: list[Record],
    model: str | None = None,
    diversify_by: str = "language",
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Pick representative diagnostic cases.

    Heuristic: among incorrect samples, prefer those where the failure
    type is NOT the most common (rare-failure samples teach more), and
    spread across the diversify_by axis (language by default).
    """
    recs = slice_filter(records, model=model, correct=False) if model else \
        [r for r in records if _record_correct(r) is False]
    classified = [(r, classify_failure(r, records)) for r in recs]
    ft_counter = Counter(ft for _, ft in classified)
    if ft_counter:
        most_common_ft, _ = ft_counter.most_common(1)[0]
    else:
        most_common_ft = None
    by_axis: dict[str, list[tuple]] = defaultdict(list)
    for rec, ft in classified:
        if diversify_by == "tag_category":
            tags = _record_tags(rec)
            if not tags:
                by_axis["_no_tag"].append((rec, ft))
            else:
                for t in tags:
                    by_axis[t].append((rec, ft))
        else:
            key = rec.get(diversify_by) or "_unknown"
            by_axis[key].append((rec, ft))
    out = []
    cycled_axes = list(by_axis.keys())
    for tries in range(20):
        added = 0
        for ax in cycled_axes:
            bucket = by_axis.get(ax) or []
            if not bucket:
                continue
            rare = [b for b in bucket if b[1] != most_common_ft]
            pick_pool = rare if rare else bucket
            rec, ft = pick_pool[0]
            bucket.remove((rec, ft))
            out.append(_to_case_pack(rec, ft,
                                     why=f"diverse-{diversify_by}={ax}; rare={ft != most_common_ft}"))
            added += 1
            if len(out) >= limit:
                return out
        if not added:
            break
    return out

def factual_or_logic_issue_cases(
    records: list[Record],
    model: str | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """heuristic CANDIDATE filter for factual / logic-error cases.

    Returns SUSPECTED factual/logic-error cases — the Case-Analyst LLM
    step is responsible for the final accept/reject decision. Do not
    cite the output of this function as 'confirmed factual errors' in
    the Reporter; mark it as 'candidates pending review'.

    A candidate is a record where:
    - this model is wrong on a sample,
    - the response_raw has substantial content (>=80 chars), and
    - ≥2 OTHER models in the same dataset answered the same sample correctly.
    """
    recs = slice_filter(records, model=model, correct=False) if model else \
        [r for r in records if _record_correct(r) is False]
    out = []
    for r in recs:
        raw = r.get("response_raw", "")
        if len(raw) < 80:
            continue
        if not _nominate_factual_error(r, records):
            continue
        if detect_cross_model_ambiguity(records, r["sample_id"]):
            continue
        out.append(_to_case_pack(r, "factual_error",
                                  why="≥2 other models answered correctly; substantive wrong response"))
        if len(out) >= limit:
            break
    return out

def degeneration_cases(
    records: list[Record],
    model: str | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Surface degeneration / repetition / format collapse cases."""
    out = []
    base = slice_filter(records, model=model) if model else records
    for r in base:
        raw = r.get("response_raw", "")
        if not raw:
            continue
        score, kind = _compute_repetition_score(raw)
        if score <= 0.4:
            continue
        cp = _to_case_pack(r, "degenerate_output",
                           why=f"repetition_score={score:.2f}; pattern={kind}")
        cp["repetition_score"] = round(score, 3)
        cp["pattern_type"] = kind
        out.append(cp)
        if len(out) >= limit:
            break
    return out

def iteration_delta(
    records: list[Record],
    model_old: str,
    model_new: str,
    group_by: str = "tag_category",
) -> dict[str, Any]:
    """Iteration / version delta tool.

    Bucket samples by axis; for each bucket compare model_old vs model_new
    accuracy on KNOWN samples and surface fixes (old wrong, new right) and
    regressions (old right, new wrong) and stable wrongs.
    """
    cmp = compare_models(records, model_new, model_old, group_by=group_by)
    cmp.sort(key=lambda x: -x.get("accuracy_delta", 0))

    by_sample: dict[str, dict[str, Record]] = defaultdict(dict)
    for r in records:
        if r.get("model") in (model_old, model_new):
            by_sample[r["sample_id"]][r["model"]] = r

    fixes, regressions, stable_wrong = [], [], []
    for sid, mp in by_sample.items():
        ro = mp.get(model_old)
        rn = mp.get(model_new)
        if not ro or not rn:
            continue
        co = _record_correct(ro)
        cn = _record_correct(rn)
        if co is False and cn is True:
            fixes.append(_to_case_pack(rn, "fixed", why=f"{model_old}→{model_new}: wrong→correct"))
        elif co is True and cn is False:
            regressions.append(_to_case_pack(rn, "regressed", why=f"{model_old}→{model_new}: correct→wrong"))
        elif co is False and cn is False:
            stable_wrong.append(_to_case_pack(rn, "stable_wrong", why="both models wrong"))

    return {
        "model_old": model_old,
        "model_new": model_new,
        "axis": group_by,
        "per_axis_delta_top": cmp[:10],
        "per_axis_delta_bottom": cmp[-10:][::-1],
        "fixes_count": len(fixes),
        "regressions_count": len(regressions),
        "stable_wrong_count": len(stable_wrong),
        "fixes_sample": fixes[:5],
        "regressions_sample": regressions[:5],
        "stable_wrong_sample": stable_wrong[:5],
    }

def _to_case_pack(rec: Record, failure_type: str, why: str = "") -> dict[str, Any]:
    """Standardized case-pack record. The fields here are exactly what
    Reporter / Judge / case_pack.json all key off.
    """
    return {
        "sample_id": rec.get("sample_id"),
        "question_id": rec.get("question_id"),
        "source_dataset": rec.get("source_dataset"),
        "language": rec.get("language"),
        "country": rec.get("country_or_culture"),
        "tag_category": _record_tags(rec),
        "model": rec.get("model"),
        "failure_type": failure_type,
        "prompt_excerpt": (rec.get("prompt") or "")[:300],
        "gold": rec.get("gold"),
        "response_final": rec.get("response_final"),
        "response_excerpt": (rec.get("response_raw") or "")[:400],
        "response_empty": rec.get("response_empty"),
        "correct": _record_correct(rec),
        "why_selected": why,
    }

def bind_report(
    query: str,
    planner_output: dict[str, Any],
    evidence_findings: dict[str, Any],
    case_findings: dict[str, Any],
    reflector_opinions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind all agent outputs into a structured package for the reporter."""
    return {
        "query": query,
        "plan": planner_output,
        "evidence": evidence_findings,
        "cases": case_findings,
        "reflections": reflector_opinions,
    }
