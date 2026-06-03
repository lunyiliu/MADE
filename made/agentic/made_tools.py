"""MADE analysis functions wrapped as agentic Tool objects.

The Analysts call these tools in a bounded ReAct-style loop. A
``ToolCallLedger`` captures every (tool_name, args, result_summary) so the
Reporter can cite numbers with provenance; the ledger is attached to the
registry and updated through a ledger-aware ``execute`` wrapper at
registry-build time.
"""

from typing import Any
import json

from made.agentic.base import Tool, ToolRegistry
from made.data_loader import Record, data_summary
from made.tools import (
    GROUP_AXES,
    tag_stats, failure_type_stats_by_tag, retrieve_cases_by_tag,
    slice_filter, group_stats, overall_stats, top_bottom_slices,
    compare_models, compare_overall, compare_groups_full,
    retrieve_error_cases, retrieve_disagreement_cases,
    retrieve_cases_by_failure_type, failure_type_stats,
    inspect_response_patterns, detect_degenerate_repetition,
    estimate_group_support, detect_cross_model_ambiguity,
    benchmark_dashboard as _benchmark_dashboard,
    tag_language_matrix as _tag_language_matrix,
    model_tag_matrix as _model_tag_matrix,
    representative_case_search as _representative_case_search,
    factual_or_logic_issue_cases as _factual_or_logic_issue_cases,
    degeneration_cases as _degeneration_cases,
    iteration_delta as _iteration_delta,
)

class ToolCallLedger:
    """Captures every tool call's name, args, and a compact result for
    later use in the Reporter's evidence ledger.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, name: str, args: dict, result: Any) -> None:
        try:
            r_str = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            r_str = str(result)
        if len(r_str) > 4000:
            r_str = r_str[:4000] + "... [truncated]"
        self.entries.append({
            "tool": name,
            "args": args,
            "result_preview": r_str,
        })

    def to_summary(self, max_chars: int = 8000) -> str:
        out = []
        size = 0
        for i, e in enumerate(self.entries):
            line = f"[{i}] {e['tool']}({json.dumps(e['args'], ensure_ascii=False, default=str)}) -> {e['result_preview']}"
            if size + len(line) > max_chars:
                out.append(f"... ({len(self.entries) - i} more entries truncated)")
                break
            out.append(line)
            size += len(line)
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {"n_calls": len(self.entries), "entries": self.entries}

def attach_ledger(registry: ToolRegistry, ledger: ToolCallLedger) -> ToolRegistry:
    """Wrap registry.execute so every call is appended to the ledger.

    Idempotent: if registry.execute is already wrapped, no-op. Returns the
    same registry instance for chaining.
    """
    if getattr(registry, "_ledger_attached", False):
        return registry
    orig_execute = registry.execute

    def wrapped(name, params):
        result = orig_execute(name, params)
        try:
            ledger.record(name, params, result)
        except Exception:
            pass
        return result

    registry.execute = wrapped  # type: ignore[assignment]
    registry._ledger_attached = True  # type: ignore[attr-defined]
    return registry

class DataOverviewTool(Tool):
    """High-level dataset overview with per-model accuracy."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "data_overview"

    @property
    def description(self) -> str:
        return (
            "Get a high-level overview of the (already-sliced-for-this-query) "
            "dataset: total records, models, languages, source datasets, "
            "tag count, three-valued correctness coverage, and per-model "
            "overall stats (accuracy is over known_n only). Call this first."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kw) -> Any:
        summary = data_summary(self._records)
        model_stats = compare_overall(self._records)
        return {
            "total_records": summary["total_records"],
            "models": summary["models"],
            "model_display_names": summary["model_display_names"],
            "benchmarks": summary["benchmarks"],
            "languages": summary["languages"],
            "countries": summary["countries"],
            "num_languages": summary["num_languages"],
            "num_countries": summary["num_countries"],
            "records_per_model": summary["records_per_model"],
            "records_per_benchmark": summary["records_per_benchmark"],
            "model_overall_stats": model_stats,
        }

class GroupStatsTool(Tool):
    """Per-group accuracy and response rate, schema-aware axes."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "group_stats"

    @property
    def description(self) -> str:
        return (
            "Group records by an axis and compute three-valued correctness "
            "stats per group: count, known_n, unknown_n, coverage, accuracy, "
            "wrong_count, response_rate. Choose group_by carefully: "
            "country_or_culture for culture-oriented benchmarks; "
            "language for cross-language analysis (BELEBELE/MMMLU/INCLUDE); "
            "source_dataset for cross-benchmark; tag_category for "
            "fine-grained capability/topic; round_number for S-MT-Bench "
            "multi-turn; failure_type for error-type breakdown. "
            "DO NOT use country_or_culture for non-cultural benchmarks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Filter by model name. Omit for all models combined.",
                },
                "group_by": {
                    "type": "string",
                    "description": "Axis to group by.",
                    "enum": sorted(GROUP_AXES),
                },
            },
        }

    def execute(self, model=None, group_by="country_or_culture", **kw) -> Any:
        recs = slice_filter(self._records, model=model) if model else self._records
        return group_stats(recs, group_by)

class TopBottomSlicesTool(Tool):
    """Find best/worst performing groups along any axis."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "top_bottom_slices"

    @property
    def description(self) -> str:
        return (
            "Find the top-N and bottom-N performing groups for a model "
            "by accuracy or response_rate, along any supported axis. Groups with "
            "known_n < min_count are excluded so translation-only slices "
            "don't dominate the bottom."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "metric": {
                    "type": "string",
                    "enum": ["accuracy", "response_rate"],
                },
                "n": {"type": "integer", "minimum": 1, "maximum": 20},
                "group_by": {
                    "type": "string",
                    "enum": sorted(GROUP_AXES),
                },
                "min_count": {"type": "integer", "minimum": 0},
            },
        }

    def execute(self, model=None, metric="accuracy", n=5,
                group_by="country_or_culture", min_count=5, **kw) -> Any:
        recs = slice_filter(self._records, model=model) if model else self._records
        stats = group_stats(recs, group_by)
        return top_bottom_slices(stats, metric=metric, n=n, min_count=min_count)

class CompareModelsTool(Tool):
    """Pairwise model comparison along any axis."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "compare_models"

    @property
    def description(self) -> str:
        return (
            "Compare two models slice by slice, showing accuracy delta per "
            "group. Pick group_by to match the query: language, country, "
            "tag_category, source_dataset, etc. Sorted by |delta|."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_a": {"type": "string"},
                "model_b": {"type": "string"},
                "group_by": {
                    "type": "string",
                    "enum": sorted(GROUP_AXES),
                },
            },
            "required": ["model_a", "model_b"],
        }

    def execute(self, model_a: str, model_b: str,
                group_by: str = "country_or_culture", **kw) -> Any:
        cmp = compare_models(self._records, model_a, model_b, group_by=group_by)
        return sorted(cmp, key=lambda x: abs(x.get("accuracy_delta", 0)), reverse=True)

class CompareOverallTool(Tool):
    """Overall stats for each model."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "compare_overall"

    @property
    def description(self) -> str:
        return ("Get overall accuracy/known_n/coverage/response_rate for each "
                "model on the current sliced dataset.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kw) -> Any:
        return compare_overall(self._records)

class CompareGroupsFullTool(Tool):
    """Full cross-model × cross-group comparison table."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "compare_groups_full"

    @property
    def description(self) -> str:
        return (
            "Full cross-model comparison table along an axis. Rows are "
            "groups, columns include each model's accuracy/known_n. "
            "Includes mean_accuracy and max_delta per group."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": sorted(GROUP_AXES),
                },
            },
        }

    def execute(self, group_by: str = "country_or_culture", **kw) -> Any:
        return compare_groups_full(self._records, group_by=group_by)

class BenchmarkDashboardTool(Tool):
    """model × benchmark × language wide dashboard."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "benchmark_dashboard"

    @property
    def description(self) -> str:
        return (
            "Wide diagnostic dashboard: rows = benchmarks, columns = "
            "per-model accuracy/known_n; also surfaces the bottom 25 "
            "(model, benchmark, language) cells with known_n >= 20. Use "
            "this when the query asks 'compare X across all benchmarks' "
            "or 'wide weakness scan'. Avoids forcing the analyst to call "
            "group_stats N times."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Models to include; omit for all.",
                },
            },
        }

    def execute(self, target_models: list[str] | None = None, **kw) -> Any:
        return _benchmark_dashboard(self._records, target_models=target_models)

class ResponsePatternsTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "response_patterns"

    @property
    def description(self) -> str:
        return (
            "Analyze response patterns for a model: empty rates, answer "
            "distribution, answer bias, cognitive vs completion failure rates. "
            "Failure rates are over known_n only (three-valued semantics)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"model": {"type": "string"}},
            "required": ["model"],
        }

    def execute(self, model: str, **kw) -> Any:
        return inspect_response_patterns(self._records, model=model)

class DegenerateDetectionTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "degenerate_detection"

    @property
    def description(self) -> str:
        return (
            "Detect degenerate output patterns for a model: phrase loops, "
            "CJK char repetition, sentence repeats, structure collapse. "
            "Returns count, rate, and worst cases."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"model": {"type": "string"}},
            "required": ["model"],
        }

    def execute(self, model: str, **kw) -> Any:
        return detect_degenerate_repetition(self._records, model=model)

class FailureTypeStatsTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "failure_type_stats"

    @property
    def description(self) -> str:
        return (
            "Failure-type distribution over wrong_n (correct=False only). "
            "Types: cognitive_failure, output_failure, degenerate_output, "
            "factual_error, ambiguity. Returns counts and rates per type."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"model": {"type": "string"}},
            "required": ["model"],
        }

    def execute(self, model: str, **kw) -> Any:
        return failure_type_stats(self._records, model=model, all_records=self._records)

class SupportEstimateTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "support_estimate"

    @property
    def description(self) -> str:
        return (
            "Check sample-support level (high/medium/low/very_low/no_known) "
            "for a model-group combination on any supported axis. Use this to "
            "verify whether conclusions about a slice are statistically "
            "reliable. Support level uses known_n, not raw count."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "group": {"type": "string"},
                "axis": {
                    "type": "string",
                    "enum": ["country_or_culture", "language",
                            "source_dataset", "tag_category"],
                },
            },
        }

    def execute(self, model=None, group=None,
                axis: str = "country_or_culture", **kw) -> Any:
        return estimate_group_support(self._records, model=model,
                                       group=group, axis=axis)

class TagStatsTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "tag_stats"

    @property
    def description(self) -> str:
        return (
            "Per-tag accuracy + sample count + error_rate + "
            "wrong_count for a model. CALL THIS for fine-grained "
            "capabilities / sub-skills / topic categories. sort_by lets "
            "you rank by error_rate (find weakest), wrong_count (most "
            "errors), accuracy (lowest), or delta (vs cross-model average — "
            "model-specific weakness). Top 30 by default."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "top_k": {"type": "integer", "default": 30},
                "sort_by": {
                    "type": "string",
                    "enum": ["count", "error_rate", "wrong_count", "accuracy", "delta"],
                },
                "min_count": {"type": "integer", "default": 0,
                               "description": "Drop tags with fewer than this many records (use 20+ for stable rankings)"},
            },
        }

    def execute(self, model: str | None = None, top_k: int = 30,
                sort_by: str = "count", min_count: int = 0, **kw) -> Any:
        return tag_stats(self._records, model=model, top_k=top_k,
                         sort_by=sort_by, min_count=min_count)

class FailureTypeByTagTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "failure_type_stats_by_tag"

    @property
    def description(self) -> str:
        return (
            "Drill down: given a model and a specific tag, show what "
            "failure types dominate. Use AFTER tag_stats identifies a weak "
            "tag, to ask 'why does the model fail on this tag — knowledge "
            "gap, format error, hallucination?'. Cross-model reference now "
            "uses the full slice."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "tag": {"type": "string"},
            },
            "required": ["model", "tag"],
        }

    def execute(self, model: str, tag: str, **kw) -> Any:
        return failure_type_stats_by_tag(self._records, model=model, tag=tag)

class CasesByTagTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "retrieve_cases_by_tag"

    @property
    def description(self) -> str:
        return (
            "Get representative sample records for a (model, tag) "
            "combination, in case-pack format (sample_id, "
            "source_dataset, language, country, tag, model, failure_type, "
            "why_selected). Defaults to wrong cases."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "tag": {"type": "string"},
                "correct": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["model", "tag"],
        }

    def execute(self, model: str, tag: str,
                correct: bool | None = False, limit: int = 5, **kw) -> Any:
        return retrieve_cases_by_tag(self._records, model=model, tag=tag,
                                       correct=correct, limit=limit)

class TagLanguageMatrixTool(Tool):
    """tag × language interaction weakness matrix."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "tag_language_matrix"

    @property
    def description(self) -> str:
        return (
            "Find tag × language cells where the model is weakest. Useful "
            "for queries like 'on which (tag, language) combinations is "
            "the model worst?'. Returns the bottom 15 cells with known_n "
            "≥ 20."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "top_tags": {"type": "integer", "default": 10},
                "top_langs": {"type": "integer", "default": 10},
            },
        }

    def execute(self, model: str | None = None,
                top_tags: int = 10, top_langs: int = 10, **kw) -> Any:
        return _tag_language_matrix(self._records, model=model,
                                      top_tags=top_tags, top_langs=top_langs)

class ModelTagMatrixTool(Tool):
    """model × tag matrix."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "model_tag_matrix"

    @property
    def description(self) -> str:
        return (
            "Compare models across the dominant tags. Surfaces tags with "
            "the largest cross-model accuracy delta, telling you which "
            "fine-grained capabilities discriminate between models the most."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target_models": {"type": "array", "items": {"type": "string"}},
                "top_tags": {"type": "integer", "default": 10},
            },
        }

    def execute(self, target_models: list[str] | None = None,
                top_tags: int = 10, **kw) -> Any:
        return _model_tag_matrix(self._records,
                                  target_models=target_models, top_tags=top_tags)

class ErrorCasesTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "error_cases"

    @property
    def description(self) -> str:
        return (
            "Retrieve representative wrong-answer cases (correct=False) for "
            "a model, optionally filtered by country/language/source_dataset. "
            "Excludes correct=None translation samples."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "country": {"type": "string"},
                "language": {"type": "string"},
                "source_dataset": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
        }

    def execute(self, model=None, country=None, language=None,
                source_dataset=None, limit=8, **kw) -> Any:
        cases = retrieve_error_cases(
            self._records, model=model, country=country,
            language=language, source_dataset=source_dataset, limit=limit,
        )
        from made.tools import _to_case_pack, classify_failure
        return [_to_case_pack(c, classify_failure(c, self._records),
                                 why="error_cases retrieval") for c in cases]

class DisagreementCasesTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "disagreement_cases"

    @property
    def description(self) -> str:
        return (
            "Find cases where model_correct answered right but model_wrong "
            "answered wrong on the same sample. only includes "
            "known/known pairs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_correct": {"type": "string"},
                "model_wrong": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15},
            },
            "required": ["model_correct", "model_wrong"],
        }

    def execute(self, model_correct: str, model_wrong: str, limit=6, **kw) -> Any:
        return retrieve_disagreement_cases(
            self._records, model_correct, model_wrong, limit=limit,
        )

class FailureTypeCasesTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "failure_type_cases"

    @property
    def description(self) -> str:
        return (
            "Retrieve error cases classified by failure type. case-pack "
            "fields are returned (sample_id, source_dataset, language, "
            "country, tag_category, model, failure_type, why_selected)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "failure_type": {
                    "type": "string",
                    "enum": [
                        "cognitive_failure", "output_failure",
                        "degenerate_output", "factual_error", "ambiguity",
                    ],
                },
                "country": {"type": "string"},
                "language": {"type": "string"},
                "source_dataset": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 15},
            },
        }

    def execute(self, model=None, failure_type=None, country=None,
                language=None, source_dataset=None, limit=6, **kw) -> Any:
        return retrieve_cases_by_failure_type(
            self._records, model=model, failure_type=failure_type,
            country=country, language=language,
            source_dataset=source_dataset,
            limit=limit, all_records=self._records,
        )

class AmbiguityCheckTool(Tool):
    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "ambiguity_check"

    @property
    def description(self) -> str:
        return (
            "Check if a specific sample's gold label is culturally "
            "contested (≥2 distinct non-empty answers and majority disagree "
            "with gold)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"sample_id": {"type": "string"}},
            "required": ["sample_id"],
        }

    def execute(self, sample_id: str, **kw) -> Any:
        return {"sample_id": sample_id,
                "is_ambiguous": detect_cross_model_ambiguity(self._records, sample_id)}

class RepresentativeCaseSearchTool(Tool):
    """pick high-diagnostic-value cases."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "representative_case_search"

    @property
    def description(self) -> str:
        return (
            "Pick representative diagnostic cases for a model: prefers "
            "rare-failure-type cases and spreads across the diversify_by "
            "axis (language by default; alternatives: country, "
            "source_dataset, tag_category). Output is case-pack format."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "diversify_by": {
                    "type": "string",
                    "enum": ["language", "country_or_culture",
                             "source_dataset", "tag_category"],
                },
                "limit": {"type": "integer", "default": 8},
            },
        }

    def execute(self, model=None, diversify_by: str = "language",
                limit: int = 8, **kw) -> Any:
        return _representative_case_search(self._records, model=model,
                                             diversify_by=diversify_by, limit=limit)

class FactualOrLogicIssueCasesTool(Tool):
    """heuristic CANDIDATE retrieval for factual / logic-error cases.

    Note: this is a candidate filter, NOT a model-self-judgment. The
    actual factual_error / cognitive_failure decision happens in the
    Case Analyst LLM step (which can use `preclassify_failure_types`)
    and the Reporter ledger. Treat this output as 'these samples merit
    closer inspection', not as 'these are confirmed factual errors'.
    """

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "factual_or_logic_issue_cases"

    @property
    def description(self) -> str:
        return (
            "Surface CANDIDATE factual / logic-error cases (not confirmed): "
            "this model is wrong, response is substantive (>=80 chars), AND "
            "≥2 other models on the same sample answered correctly. "
            "Excludes ambiguity candidates. The Analyst should still confirm "
            "or reject each candidate via the Case-Analyst LLM step."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "limit": {"type": "integer", "default": 6},
            },
        }

    def execute(self, model=None, limit: int = 6, **kw) -> Any:
        return _factual_or_logic_issue_cases(self._records, model=model, limit=limit)

class DegenerationCasesTool(Tool):
    """degeneration / format-collapse retrieval."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "degeneration_cases"

    @property
    def description(self) -> str:
        return (
            "Surface degeneration cases (repetition_score > 0.4): phrase "
            "loops, char repetition, sentence repeat, structure collapse, "
            "template collapse. case-pack fields plus repetition_score."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "limit": {"type": "integer", "default": 6},
            },
        }

    def execute(self, model=None, limit: int = 6, **kw) -> Any:
        return _degeneration_cases(self._records, model=model, limit=limit)

class IterationDeltaTool(Tool):
    """iteration / version delta."""

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "iteration_delta"

    @property
    def description(self) -> str:
        return (
            "Iteration / version delta between an old and a new model. "
            "Returns per-axis accuracy delta (top/bottom 10) plus sample-"
            "level fixes / regressions / stable_wrong counts and examples. "
            "Use this for queries that compare two models as 'before vs "
            "after' (e.g. Qwen3-235B vs Qwen3-14B)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_old": {"type": "string"},
                "model_new": {"type": "string"},
                "group_by": {
                    "type": "string",
                    "enum": sorted(GROUP_AXES),
                },
            },
            "required": ["model_old", "model_new"],
        }

    def execute(self, model_old: str, model_new: str,
                group_by: str = "tag_category", **kw) -> Any:
        return _iteration_delta(self._records, model_old=model_old,
                                  model_new=model_new, group_by=group_by)

EVIDENCE_TOOL_CLASSES = [
    DataOverviewTool, GroupStatsTool, TopBottomSlicesTool,
    CompareModelsTool, CompareOverallTool, CompareGroupsFullTool,
    BenchmarkDashboardTool,
    ResponsePatternsTool, DegenerateDetectionTool,
    FailureTypeStatsTool, SupportEstimateTool,
    TagStatsTool, FailureTypeByTagTool,
    TagLanguageMatrixTool, ModelTagMatrixTool,
    IterationDeltaTool,
]

CASE_TOOL_CLASSES = [
    ErrorCasesTool, DisagreementCasesTool, FailureTypeCasesTool,
    AmbiguityCheckTool, ResponsePatternsTool, FailureTypeStatsTool,
    GroupStatsTool, DataOverviewTool,
    TagStatsTool, FailureTypeByTagTool, CasesByTagTool,
    RepresentativeCaseSearchTool, FactualOrLogicIssueCasesTool,
    DegenerationCasesTool,
]

ALL_TOOL_CLASSES = list({c for c in EVIDENCE_TOOL_CLASSES + CASE_TOOL_CLASSES})

def create_evidence_tools(records: list[Record],
                           drop: set[str] | None = None) -> ToolRegistry:
    """Create tool registry for the Evidence Analyst.

    `drop` is a set of tool names to disable (e.g. when a caller turns off
    the tag tools).
    """
    drop = drop or set()
    registry = ToolRegistry()
    for cls in EVIDENCE_TOOL_CLASSES:
        t = cls(records)
        if t.name in drop:
            continue
        registry.register(t)
    return registry

def create_case_tools(records: list[Record],
                       drop: set[str] | None = None) -> ToolRegistry:
    drop = drop or set()
    registry = ToolRegistry()
    for cls in CASE_TOOL_CLASSES:
        t = cls(records)
        if t.name in drop:
            continue
        registry.register(t)
    return registry

def create_all_tools(records: list[Record],
                      drop: set[str] | None = None) -> ToolRegistry:
    drop = drop or set()
    registry = ToolRegistry()
    for cls in ALL_TOOL_CLASSES:
        t = cls(records)
        if t.name in drop:
            continue
        registry.register(t)
    return registry

TAG_TOOL_NAMES = {
    "tag_stats", "failure_type_stats_by_tag", "retrieve_cases_by_tag",
    "tag_language_matrix", "model_tag_matrix",
}
ITERATION_TOOL_NAMES = {"iteration_delta"}
