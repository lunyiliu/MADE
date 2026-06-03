"""MADE agent implementations: Planner, Evidence Analyst, Case Analyst,
Language Reflector and Reporter.

Each agent receives a runtime ``data_profile`` derived from the sliced
evaluation records, so prompts name the exact benchmarks, languages, models
and three-valued correctness coverage present in the slice rather than any fixed prior. The Planner routes the query to the analysts it needs;
the Analysts run bounded tool loops whose calls are captured in a
``ToolCallLedger`` that backs the Reporter's evidence citations; the Language
Reflector intervenes three times (pre-plan, mid-analysis, pre-report).
"""

import json
import logging
import re
from typing import Any

from made.data_loader import Record, data_summary, MODEL_DISPLAY
from made.tools import (
    slice_filter, group_stats, overall_stats, top_bottom_slices,
    compare_models, compare_overall, compare_groups_full,
    retrieve_error_cases, retrieve_disagreement_cases,
    retrieve_cases_by_failure_type, failure_type_stats,
    inspect_response_patterns, detect_degenerate_repetition,
    estimate_group_support,
    detect_cross_model_ambiguity, classify_failure,
    bind_report,
    benchmark_dashboard, tag_stats, tag_language_matrix,
    representative_case_search, factual_or_logic_issue_cases,
    degeneration_cases, _to_case_pack,
)
from made.prompts import (
    render,
    PLANNER_SYSTEM, EVIDENCE_ANALYST_SYSTEM, CASE_ANALYST_SYSTEM,
    REFLECTOR_SYSTEM, REPORTER_SYSTEM,
    REFLECTOR_POST_REPORT_SYSTEM, REPORTER_REVISION_SYSTEM,
    EVIDENCE_ANALYST_AGENTIC_SYSTEM, CASE_ANALYST_AGENTIC_SYSTEM,
)
from made.agentic import AgentLoop
from made.agentic.made_tools import (
    create_evidence_tools, create_case_tools,
    ToolCallLedger, attach_ledger,
    TAG_TOOL_NAMES, ITERATION_TOOL_NAMES,
)

log = logging.getLogger("made.agents")

def build_data_profile(records: list[Record], plan_query: dict[str, Any] | None = None) -> str:
    """Build a runtime data profile string for prompt injection.

    Output is a compact, machine-and-human-readable description that
    accurately reflects THIS slice rather than any fixed prior.
    """
    summary = data_summary(records)
    n_total = len(records)
    n_known = sum(1 for r in records
                   if r.get("correct") is True or r.get("correct") is False)
    n_unknown = n_total - n_known
    coverage = n_known / n_total if n_total else 0.0

    n_tags = len({t for r in records for t in (r.get("tag_category") or []) if t})

    parts = [
        f"slice_size: {n_total} records",
        f"benchmarks_present: {summary['benchmarks']}",
        f"models_present ({summary['num_models']}): {summary['models']}",
        f"languages_present ({summary['num_languages']}): {summary['languages']}",
        f"countries_present ({summary['num_countries']}): {summary['countries'][:30]}{'…' if summary['num_countries'] > 30 else ''}",
        f"tag_count: {n_tags} distinct fine-grained capability tags in this slice",
        f"correctness_coverage: known_n={n_known}, unknown_n={n_unknown}, coverage={coverage:.3f}",
        ("note: when coverage is low (e.g. FLORES translation slice has "
         "no binary correctness), accuracy is unreliable — use empty_rate / "
         "length_ratio / format_compliance instead."),
    ]
    if plan_query and plan_query.get("benchmark_scope"):
        parts.append(f"query_benchmark_scope: {plan_query['benchmark_scope']}")
    if plan_query and plan_query.get("languages"):
        parts.append(f"query_languages: {plan_query['languages']}")
    return "\n".join(parts)

class LLMCallError(RuntimeError):
    """Raised when an LLM call fails (non-2xx HTTP or transport error).

    Carries the error type and HTTP status so a caller's outer retry can
    report the failure precisely.
    """

    def __init__(self, message: str, *,
                 error_type: str | None = None, status=None):
        super().__init__(message)
        self.error_type = error_type
        self.status = status

def _check_call(result: dict, *, allow_empty_text: bool = False) -> None:
    """Raise `LLMCallError` if the call is unusable.

    A call is unusable when the HTTP status is not 2xx, or an error_type was
    set (transport / request error). Otherwise returns silently.
    `allow_empty_text` lets the planner-retry path tolerate empty text.
    """
    status = result.get("status")
    if status is not None and not (200 <= int(status) < 300):
        raise LLMCallError(
            f"HTTP {status} ({result.get('error_type')}): "
            f"{result.get('error') or 'non-2xx'}",
            error_type=result.get("error_type"),
            status=status,
        )
    if result.get("error_type"):
        raise LLMCallError(
            f"{result.get('error_type')}: {result.get('error')}",
            error_type=result.get("error_type"),
            status=status,
        )

def _call_llm(client, system: str, user: str, max_tokens: int = 4096,
               allow_empty_text: bool = False) -> str:
    """Call the LLM via the client and return its text.

    Raises LLMCallError on a non-2xx HTTP status or transport error.
    Callers that have their own retry can catch it.
    """
    old_max = client.config.max_tokens
    client.config.max_tokens = max_tokens
    result = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        sample_count=1,
    )
    client.config.max_tokens = old_max
    _check_call(result, allow_empty_text=allow_empty_text)
    return result.get("text", "") or ""

def _parse_json(text: str) -> dict:
    """Best-effort JSON parse from LLM output."""
    text = text.strip()
    text = re.sub(r'<think(?:ing)?>\s*.*?\s*</think(?:ing)?>', '', text, flags=re.DOTALL)
    if "```" in text:
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    log.warning(f"Failed to parse JSON from LLM output ({len(text)} chars), returning raw text")
    return {"raw_text": text[:2000], "_parse_failed": True}

def _llm_judge_factual_error(client, record: Record, all_records: list[Record]) -> str:
    sid = record["sample_id"]
    other_responses = []
    for r in all_records:
        if r.get("sample_id") == sid and r.get("model") != record.get("model"):
            other_responses.append({
                "model": r.get("model"),
                "answer": r.get("response_final", ""),
                "correct": r.get("correct"),
                "excerpt": r.get("response_raw", "")[:300],
            })

    prompt_text = (
        f"Question: {record.get('prompt', '')[:500]}\n"
        f"Gold answer: {record.get('gold', '')}\n"
        f"This model ({record.get('model','?')}) answered: {record.get('response_final', '')}\n"
        f"Response excerpt: {record.get('response_raw', '')[:500]}\n\n"
        f"Other models' answers:\n{json.dumps(other_responses, ensure_ascii=False, default=str)}\n\n"
        f"Does this model's response contain a factual error in its reasoning "
        f"(e.g., wrong facts, incorrect premises, flawed logic based on false claims), "
        f"or is it a cognitive failure (understood the question but chose wrong)?\n"
        f"Reply with exactly one word: factual_error or cognitive_failure"
    )
    raw = _call_llm(client, "You are a concise classifier.", prompt_text, max_tokens=64)
    raw = raw.strip().lower()
    if "factual_error" in raw:
        return "factual_error"
    return "cognitive_failure"

def preclassify_failure_types(client, records: list[Record], max_llm_calls: int = 30) -> int:
    """Pre-classify factual_error candidates via LLM."""
    from made.tools import _nominate_factual_error

    candidates = [
        r for r in records
        if r.get("correct") is False and r.get("response_final")
        and _nominate_factual_error(r, records)
        and not detect_cross_model_ambiguity(records, r["sample_id"])
    ]
    candidates = candidates[:max_llm_calls]
    factual_count = 0
    for r in candidates:
        verdict = _llm_judge_factual_error(client, r, records)
        r["_factual_verdict"] = verdict
        if verdict == "factual_error":
            factual_count += 1
    log.info(f"Pre-classified {len(candidates)} candidates: {factual_count} factual_error")
    return factual_count

ROUTING_DEFAULTS = {
    "task_type": "hybrid",
    "need_evidence": True,
    "need_case": True,
    "need_tag_tools": False,
    "need_iteration_tools": False,
    "primary_focus": "weakness",
    "target_models": [],
    "target_benchmarks": [],
    "target_languages": [],
    "target_groups": [],
    "evidence_analyst_tasks": [],
    "case_analyst_tasks": [],
    "planned_tool_groups": [],
    "questions_to_answer": [],
    "risks": [],
}

def _coerce_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in {"true", "yes", "1"}:
            return True
        if v.lower() in {"false", "no", "0"}:
            return False
    return default

def _expand_all_target_models(plan: dict, records: list[Record]) -> dict:
    """expand `target_models=["all"]` (or any list containing "all" / "*" /
    "any" / empty) into the actual model list present in the sliced
    `records`. The Planner prompt allows ["all"] when the query is ambiguous;
    downstream tools cannot filter by the string "all", so we resolve here.
    """
    cur = plan.get("target_models") or []
    if not isinstance(cur, list):
        cur = [cur]
    flagged = any(
        (isinstance(m, str) and m.strip().lower() in {"all", "*", "any"})
        for m in cur
    )
    if cur and not flagged:
        return plan
    actual = sorted({r.get("model") for r in records if r.get("model")})
    plan["target_models"] = actual
    plan["_target_models_expanded_from_all"] = True
    return plan

def _normalize_plan(plan: dict, query_text: str,
                     records: list[Record] | None = None) -> dict:
    """Coerce planner output into the routing schema. We do NOT default
    need_evidence/need_case to True if the LLM gave explicit False; we only
    fill missing keys.

    when `records` is provided AND plan has `["all"]`
    target_models, we expand to the actual sliced models so downstream
    tools don't filter against the literal string "all".
    """
    out = dict(ROUTING_DEFAULTS)
    for k, v in plan.items():
        if k in out:
            out[k] = v
        else:
            out[k] = v
    for k in ("need_evidence", "need_case", "need_tag_tools", "need_iteration_tools"):
        out[k] = _coerce_bool(out.get(k, ROUTING_DEFAULTS[k]),
                              default=ROUTING_DEFAULTS[k])
    if not out.get("task_type"):
        out["task_type"] = "hybrid"
    if not out["need_evidence"] and not out["need_case"]:
        log.warning("Planner returned need_evidence=False AND need_case=False; "
                    "forcing need_evidence=True for safety.")
        out["need_evidence"] = True
    if records is not None:
        out = _expand_all_target_models(out, records)
    return out

def run_planner(client, query: dict[str, Any], records: list[Record],
                 max_attempts: int = 2) -> dict[str, Any]:
    """Run the Planner agent. data_profile from records, parse-retry,
    no setdefault(True)."""
    profile = build_data_profile(records, plan_query=query)
    system = render(PLANNER_SYSTEM, data_profile=profile)
    user_msg = (
        f"User query: {query['text']}\n\n"
        f"Analyze this query and produce a structured analysis plan in JSON. "
        f"Apply the routing rules and few-shot examples carefully — do not "
        f"default to need_evidence=true && need_case=true unless the query is "
        f"truly hybrid."
    )
    plan: dict = {}
    for attempt in range(1, max_attempts + 1):
        raw = _call_llm(client, system, user_msg, max_tokens=2048)
        plan = _parse_json(raw)
        if plan.get("_parse_failed"):
            log.warning(f"Planner JSON parse failed (attempt {attempt}/{max_attempts})")
            if attempt == max_attempts:
                plan = dict(ROUTING_DEFAULTS)
            continue
        if "need_evidence" not in plan and "need_case" not in plan:
            log.warning(
                f"Planner missing routing fields (attempt {attempt}/{max_attempts}); "
                f"keys: {list(plan.keys())[:8]}"
            )
            if attempt == max_attempts:
                pass
            else:
                continue
        break

    plan = _normalize_plan(plan, query["text"], records=records)
    return plan

EVIDENCE_MIN_FINDINGS = 3
EVIDENCE_MIN_CHARS = 500
ANALYST_MAX_RETRIES = 2

def _pre_call_evidence_mandatory(
    records: list[Record],
    plan: dict,
    ledger: "ToolCallLedger | None" = None,
) -> str:
    """Pre-call schema-aware mandatory tools.

    every call is also recorded in the shared
    `ToolCallLedger` so the Reporter / Judge see provenance for these
    pre-call results — not only the agentic-loop calls.
    """
    target_models = plan.get("target_models", []) or []
    parts = []

    def _record(name: str, args: dict, result):
        if ledger is not None:
            try:
                ledger.record(name, args, result)
            except Exception:
                pass

    overview = compare_overall(records)
    _record("compare_overall", {}, overview)
    parts.append(f"[data_overview] {json.dumps(overview, ensure_ascii=False, default=str)}")

    benchmarks_present = sorted({r.get("source_dataset") for r in records
                                  if r.get("source_dataset")})
    if len(benchmarks_present) > 1 or plan.get("task_type") == "dataset":
        dash = benchmark_dashboard(records, target_models=target_models or None)
        _record("benchmark_dashboard",
                 {"target_models": target_models or None}, dash)
        parts.append(f"[benchmark_dashboard] {json.dumps(dash, ensure_ascii=False, default=str)[:5000]}")

    primary_axis = "country_or_culture"
    if plan.get("need_tag_tools"):
        primary_axis = "tag_category"
    elif "language" in str(plan.get("planned_tool_groups", "")).lower():
        primary_axis = "language"

    if target_models:
        for m in target_models[:3]:
            gs = group_stats(slice_filter(records, model=m), primary_axis)
            _record("group_stats",
                     {"model": m, "group_by": primary_axis}, gs)
            parts.append(
                f"[group_stats({m}, {primary_axis})] "
                f"{json.dumps(gs, ensure_ascii=False, default=str)[:3000]}"
            )

    full_groups = compare_groups_full(records, target_models, group_by=primary_axis)
    _record("compare_groups_full", {"group_by": primary_axis}, full_groups)
    parts.append(
        f"[compare_groups_full({primary_axis})] "
        f"{json.dumps(full_groups, ensure_ascii=False, default=str)[:4000]}"
    )

    if plan.get("need_tag_tools"):
        for m in target_models[:2]:
            ts = tag_stats(records, model=m, sort_by="error_rate", min_count=20, top_k=10)
            _record("tag_stats",
                     {"model": m, "sort_by": "error_rate", "min_count": 20, "top_k": 10}, ts)
            parts.append(
                f"[tag_stats({m}, sort_by=error_rate, min_count=20)] "
                f"{json.dumps(ts, ensure_ascii=False, default=str)}"
            )

    for m in target_models[:2]:
        ft = failure_type_stats(records, model=m, all_records=records)
        _record("failure_type_stats", {"model": m}, ft)
        parts.append(f"[failure_type_stats({m})] {json.dumps(ft, ensure_ascii=False, default=str)}")

    return "\n\n".join(parts)

def run_evidence_analyst(client, plan: dict[str, Any], records: list[Record]) -> dict[str, Any]:
    """non-agentic Evidence Analyst (kept for fallback / smoke)."""
    profile = build_data_profile(records)
    tool_results = _pre_call_evidence_mandatory(records, plan)
    tasks = json.dumps(plan.get("evidence_analyst_tasks", []), ensure_ascii=False)
    system = render(
        EVIDENCE_ANALYST_SYSTEM,
        data_profile=profile,
        tool_results=tool_results,
        tasks=tasks,
    )
    raw = _call_llm(client, system, "Perform your analysis and output structured findings.", max_tokens=4096)
    return _parse_json(raw)

def run_evidence_analyst_agentic(
    client, plan: dict[str, Any], records: list[Record],
    ledger: ToolCallLedger | None = None,
    tools_drop: set[str] | None = None,
) -> dict[str, Any]:
    """Run the Evidence Analyst as an agentic tool-calling loop."""
    profile = build_data_profile(records)
    tools = create_evidence_tools(records, drop=tools_drop)
    if ledger is not None:
        attach_ledger(tools, ledger)
    loop = AgentLoop(client, tools, max_iterations=15)

    tasks_str = json.dumps(plan.get("evidence_analyst_tasks", []), ensure_ascii=False)
    mandatory_data = _pre_call_evidence_mandatory(records, plan, ledger=ledger)

    system = render(
        EVIDENCE_ANALYST_AGENTIC_SYSTEM,
        data_profile=profile,
        plan=json.dumps(plan, indent=2, ensure_ascii=False, default=str)[:3000],
        tasks=tasks_str,
    )
    user_msg = (
        "The following mandatory tool results have already been collected for you. "
        "Use them as your data foundation, then call ADDITIONAL tools as needed "
        "(e.g., top_bottom_slices, compare_models, support_estimate, "
        "tag_stats(sort_by=error_rate), benchmark_dashboard) to deepen your "
        "analysis.\n\n"
        f"{mandatory_data}\n\n"
        "After gathering sufficient evidence, output your findings as a DETAILED "
        "JSON object. Each finding must cite specific numbers and known_n. "
        "Aim for at least 5 aggregate_findings with full evidence."
    )

    findings: dict = {}
    result: dict = {}
    for attempt in range(1, ANALYST_MAX_RETRIES + 1):
        result = loop.run(system, user_msg, max_tokens=4096)
        if result.get("stop_reason") == "error":
            raise LLMCallError(
                f"Evidence Analyst agent loop failed: {result.get('error')}",
                error_type="AgentLoopError",
                status=None,
            )
        findings = _parse_json(result.get("content", ""))
        content_str = json.dumps(findings, ensure_ascii=False, default=str)
        n_findings = len(findings.get("aggregate_findings", []))
        if n_findings >= EVIDENCE_MIN_FINDINGS and len(content_str) >= EVIDENCE_MIN_CHARS:
            break
        if attempt < ANALYST_MAX_RETRIES:
            log.warning(
                "Evidence Analyst output too sparse (findings=%d, chars=%d), retry %d/%d",
                n_findings, len(content_str), attempt, ANALYST_MAX_RETRIES,
            )

    findings["_agentic_meta"] = {
        "iterations": result.get("iterations"),
        "tools_used": result.get("tools_used"),
        "tool_events": result.get("tool_events"),
        "stop_reason": result.get("stop_reason"),
        "total_usage": result.get("total_usage"),
    }
    findings["_mandatory_data_chars"] = len(mandatory_data)
    log.info(
        "Evidence Analyst (agentic): %d iterations, %d tool calls, stop=%s",
        result.get("iterations", 0), len(result.get("tools_used", [])),
        result.get("stop_reason"),
    )
    return findings

CASE_MIN_CASES = 3
CASE_MIN_CHARS = 400

def _pre_call_case_mandatory(
    records: list[Record],
    plan: dict,
    ledger: "ToolCallLedger | None" = None,
) -> str:
    """mandatory case-side tool calls also recorded in ledger."""
    target_models = plan.get("target_models", []) or []
    parts = []

    def _record(name: str, args: dict, result):
        if ledger is not None:
            try:
                ledger.record(name, args, result)
            except Exception:
                pass

    for m in target_models[:2]:
        ft = failure_type_stats(records, model=m, all_records=records)
        _record("failure_type_stats", {"model": m}, ft)
        parts.append(f"[failure_type_stats({m})] {json.dumps(ft, ensure_ascii=False, default=str)}")

        rep = representative_case_search(records, model=m, diversify_by="language", limit=5)
        _record("representative_case_search",
                 {"model": m, "diversify_by": "language", "limit": 5}, rep)
        parts.append(f"[representative_case_search({m})] {json.dumps(rep, ensure_ascii=False, default=str)[:3000]}")

        errors = retrieve_error_cases(records, model=m, limit=5)
        cps = [_to_case_pack(e, classify_failure(e, records),
                              why="error_cases retrieval") for e in errors]
        _record("error_cases", {"model": m, "limit": 5}, cps)
        parts.append(f"[error_cases({m})] {json.dumps(cps, ensure_ascii=False, default=str)[:3000]}")

    if len(target_models) >= 2:
        disagree = retrieve_disagreement_cases(
            records, model_correct=target_models[0], model_wrong=target_models[1], limit=5,
        )
        _record("disagreement_cases",
                 {"model_correct": target_models[0],
                  "model_wrong": target_models[1], "limit": 5}, disagree)
        parts.append(f"[disagreement_cases({target_models[0]} right, {target_models[1]} wrong)] "
                      f"{json.dumps(disagree, ensure_ascii=False, default=str)[:3000]}")

    return "\n\n".join(parts)

def run_case_analyst(client, plan: dict[str, Any], records: list[Record]) -> dict[str, Any]:
    """non-agentic Case Analyst."""
    profile = build_data_profile(records)
    tool_results = _pre_call_case_mandatory(records, plan)
    tasks = json.dumps(plan.get("case_analyst_tasks", []), ensure_ascii=False)
    system = render(
        CASE_ANALYST_SYSTEM,
        data_profile=profile,
        tool_results=tool_results,
        tasks=tasks,
    )
    raw = _call_llm(client, system, "Analyze the cases and output structured findings.", max_tokens=4096)
    return _parse_json(raw)

def run_case_analyst_agentic(
    client, plan: dict[str, Any], records: list[Record],
    ledger: ToolCallLedger | None = None,
    tools_drop: set[str] | None = None,
) -> dict[str, Any]:
    """Run the Case Analyst as an agentic tool-calling loop."""
    profile = build_data_profile(records)
    tools = create_case_tools(records, drop=tools_drop)
    if ledger is not None:
        attach_ledger(tools, ledger)
    loop = AgentLoop(client, tools, max_iterations=15)

    tasks_str = json.dumps(plan.get("case_analyst_tasks", []), ensure_ascii=False)
    mandatory_data = _pre_call_case_mandatory(records, plan, ledger=ledger)

    system = render(
        CASE_ANALYST_AGENTIC_SYSTEM,
        data_profile=profile,
        plan=json.dumps(plan, indent=2, ensure_ascii=False, default=str)[:3000],
        tasks=tasks_str,
    )
    user_msg = (
        "The following mandatory tool results have already been collected for you. "
        "Use them as your data foundation, then call ADDITIONAL tools as needed "
        "(failure_type_cases, ambiguity_check, representative_case_search, "
        "factual_or_logic_issue_cases, degeneration_cases, retrieve_cases_by_tag) "
        "to deepen your analysis.\n\n"
        f"{mandatory_data}\n\n"
        "After your investigation, output findings as a DETAILED JSON object. "
        "Include at least 5 representative_cases with full diagnosis. "
        "Quote specific prompt/response text as evidence."
    )

    findings: dict = {}
    result: dict = {}
    for attempt in range(1, ANALYST_MAX_RETRIES + 1):
        result = loop.run(system, user_msg, max_tokens=4096)
        if result.get("stop_reason") == "error":
            raise LLMCallError(
                f"Case Analyst agent loop failed: {result.get('error')}",
                error_type="AgentLoopError",
                status=None,
            )
        findings = _parse_json(result.get("content", ""))
        content_str = json.dumps(findings, ensure_ascii=False, default=str)
        n_cases = len(findings.get("representative_cases", []))
        if n_cases >= CASE_MIN_CASES and len(content_str) >= CASE_MIN_CHARS:
            break
        if attempt < ANALYST_MAX_RETRIES:
            log.warning(
                "Case Analyst output too sparse (cases=%d, chars=%d), retry %d/%d",
                n_cases, len(content_str), attempt, ANALYST_MAX_RETRIES,
            )

    findings["_agentic_meta"] = {
        "iterations": result.get("iterations"),
        "tools_used": result.get("tools_used"),
        "tool_events": result.get("tool_events"),
        "stop_reason": result.get("stop_reason"),
        "total_usage": result.get("total_usage"),
    }
    findings["_mandatory_data_chars"] = len(mandatory_data)
    log.info(
        "Case Analyst (agentic): %d iterations, %d tool calls, stop=%s",
        result.get("iterations", 0), len(result.get("tools_used", [])),
        result.get("stop_reason"),
    )
    return findings

def run_reflector(
    client, stage: str, context: dict[str, Any],
    records: list[Record] | None = None,
    evidence_ledger_summary: str = "",
) -> dict[str, Any]:
    """Run the Language Reflector. full intensity only, data_profile injected."""
    profile = build_data_profile(records or [])

    def _ctx_str(d, max_len=4000):
        s = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        return s[:max_len] if len(s) > max_len else s

    if stage == "pre_plan":
        stage_context = (
            "The Planner has produced an initial plan. Review on BOTH axes:\n\n"
            "Cultural: missing cultural/linguistic considerations? Does the plan "
            "adequately cover the language/culture axes that matter for THIS query?\n\n"
            "Non-cultural: query correctly understood? right tools planned? "
            "evidence/case routing appropriate? grounding risks?\n\n"
            "If routing is wrong (e.g. query is dataset-only but plan opens case path; "
            "query is iteration but iteration tools aren't planned), recommend "
            "action=replan in required_actions.\n\n"
            f"Plan:\n{_ctx_str(context)}"
        )
    elif stage == "mid_analysis":
        stage_context = (
            "Evidence Analyst and Case Analyst have produced intermediate findings.\n\n"
            "Cultural review:\n"
            "- Cultural bias or English-centric framing?\n"
            "- Over-generalization from limited (low known_n) data?\n"
            "- Confusion between model capability and language/cultural artifacts?\n\n"
            "Non-cultural review:\n"
            "- Findings grounded in actual tool output?\n"
            "- Small-known_n conclusions flagged?\n"
            "- Analysis drift from the query?\n"
            "- Failure-type classifications consistent?\n"
            "- Three-valued correctness respected (no fake accuracy on FLORES-style slices)?\n\n"
            f"Evidence findings:\n{_ctx_str(context.get('evidence', {}))}\n\n"
            f"Case findings:\n{_ctx_str(context.get('cases', {}))}"
        )
    elif stage == "post_report":
        system = render(
            REFLECTOR_POST_REPORT_SYSTEM,
            report_draft=context.get("report", "")[:6000],
            evidence_pool=_ctx_str(context.get("evidence", {}), 4000),
            case_pool=_ctx_str(context.get("cases", {}), 3000),
            evidence_ledger=evidence_ledger_summary[:4000] if evidence_ledger_summary else "(none)",
            heuristic_warnings=json.dumps(
                context.get("heuristic_warnings", []), ensure_ascii=False, default=str,
            ),
        )
        user_msg = (
            "Review the actual report against the evidence pool, case pool, "
            "and evidence ledger. Output ONLY the JSON object."
        )
        raw = _call_llm(client, system, user_msg, max_tokens=4096)
        return _parse_json(raw)
    else:
        stage_context = _ctx_str(context)

    system = render(
        REFLECTOR_SYSTEM,
        stage=stage,
        stage_context=stage_context,
        data_profile=profile,
    )
    user_msg = (
        f"Perform your {stage} reflection on both cultural and non-cultural axes. "
        f"Output ONLY the JSON object — no markdown, no explanation."
    )
    raw = _call_llm(client, system, user_msg, max_tokens=4096)
    return _parse_json(raw)

def _extract_bottom_groups(evidence: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(evidence, dict):
        for key, val in evidence.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if "bottom" in subkey.lower() or "weak" in subkey.lower():
                        if isinstance(subval, list):
                            groups = []
                            for item in subval:
                                if isinstance(item, dict) and "group" in item:
                                    groups.append(item["group"])
                                elif isinstance(item, str):
                                    groups.append(item)
                            if groups:
                                result[key] = groups
    return result

def _extract_model_accuracies(evidence: dict[str, Any]) -> dict[str, float]:
    accs: dict[str, float] = {}
    if isinstance(evidence, dict):
        for key, val in evidence.items():
            if isinstance(val, dict) and "accuracy" in val:
                accs[key] = val["accuracy"]
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if "accuracy" in subkey and isinstance(subval, (int, float)):
                        accs[key] = subval
    return accs

def check_report_grounding(
    report: str, evidence: dict[str, Any], cases: dict[str, Any],
) -> dict[str, Any]:
    """Heuristic grounding check on the draft report."""
    warnings = []
    report_sample_ids = set(re.findall(
        r'[A-Z]{2}-[a-z]{2}-\d+_\d+|MCQID_\d+|[A-Z]{2}_\d+_\d+',
        report,
    ))
    evidence_sample_ids = set()
    if isinstance(cases, dict):
        for case_list_key in ["representative_cases", "model_disagreements",
                               "failure_cases", "error_cases"]:
            for c in cases.get(case_list_key, []):
                if isinstance(c, dict) and "sample_id" in c:
                    evidence_sample_ids.add(c["sample_id"])

    if report_sample_ids:
        unknown_ids = report_sample_ids - evidence_sample_ids
        if unknown_ids and evidence_sample_ids:
            warnings.append(
                f"Report mentions {len(unknown_ids)} sample IDs not found in case analyst output: "
                f"{list(unknown_ids)[:5]}"
            )

    report_pcts = re.findall(r'(\d+\.\d{2,})%', report)
    evidence_str = json.dumps(evidence, ensure_ascii=False, default=str)
    cases_str = json.dumps(cases, ensure_ascii=False, default=str) if cases else ""
    all_evidence_str = evidence_str + cases_str

    for pct in report_pcts:
        pct_val = float(pct)
        found = False
        if pct in all_evidence_str:
            found = True
        if not found:
            rounded = str(round(pct_val, 1))
            if rounded in all_evidence_str:
                found = True
        if not found:
            decimal_form = str(round(pct_val / 100, 4))
            if decimal_form in all_evidence_str:
                found = True
        if not found:
            decimal_rounded = str(round(pct_val / 100, 3))
            if decimal_rounded in all_evidence_str:
                found = True
        if not found:
            if pct_val != round(pct_val):
                warnings.append(f"Report cites {pct}% which doesn't appear in evidence")

    bottom_groups = _extract_bottom_groups(evidence)
    worst_claims = re.findall(
        r'(?:worst|weakest|lowest|poorest)\s+(\w[\w\s]{2,20})',
        report, re.IGNORECASE,
    )
    if worst_claims and bottom_groups:
        all_bottoms = set()
        for groups in bottom_groups.values():
            all_bottoms.update(g.lower() for g in groups)
        for claim in worst_claims[:3]:
            claim_clean = claim.strip().lower()
            if claim_clean and not any(claim_clean in b for b in all_bottoms):
                if not any(kw in claim_clean for kw in ["gpt", "qwen", "gemini", "model"]):
                    warnings.append(
                        f"Report claims '{claim.strip()}' is worst, "
                        f"but evidence bottom groups are: {list(all_bottoms)[:5]}"
                    )

    model_accs = _extract_model_accuracies(evidence)
    better_claims = re.findall(
        r'(?:outperforms?|better than|stronger than|beats)\s*(\S+)',
        report, re.IGNORECASE,
    )
    if better_claims and model_accs:
        for claim_model in better_claims[:3]:
            claim_clean = claim_model.strip().lower().rstrip('，。,.')
            matched = [k for k in model_accs if claim_clean in k.lower() or k.lower() in claim_clean]
            if not matched and len(claim_clean) > 2:
                warnings.append(
                    f"Report claims superiority over '{claim_model.strip()}' "
                    f"— not found in evidence model keys: {list(model_accs.keys())}"
                )

    cite_tag_count = len(re.findall(r'\[N=\d+', report))
    quant_claims = len(re.findall(r'(\d+(?:\.\d+)?)%|accuracy', report))
    if quant_claims > 5 and cite_tag_count == 0:
        warnings.append("Report has many quantitative claims but no [N=..., metric=...] citations")

    return {
        "grounding_warnings": warnings[:10],
        "report_sample_ids": len(report_sample_ids),
        "evidence_sample_ids": len(evidence_sample_ids),
        "potentially_ungrounded": len(warnings) > 0,
        "citation_tag_count": cite_tag_count,
        "quantitative_claim_count": quant_claims,
    }

def _build_raw_data_context(
    records: list[Record],
    plan: dict[str, Any],
    *,
    include_aggregate: bool = True,
    include_cases: bool = True,
    include_tag: bool = True,
    scrub_tag_in_cases: bool = False,
) -> str:
    """Build the raw-data context for the Reporter (schema-aware).

    The context is partitioned into three independently-toggleable classes
    of content:

    - `include_aggregate=False`: drops overall stats, full group comparison,
      benchmark_dashboard, per-model top/bottom + response_patterns +
      failure_types, and the model-vs-model compare table.
    - `include_cases=False`: drops sample errors and disagreement cases.
    - `include_tag=False` (also when the query doesn't need the tag axis):
      drops tag_stats and forces the primary axis off `tag_category` so the
      full group table doesn't back-channel tag info.
    """
    target_models = plan.get("target_models", []) or []
    parts = []

    primary_axis = "country_or_culture"
    if plan.get("need_tag_tools") and include_tag:
        primary_axis = "tag_category"
    elif "language" in str(plan.get("planned_tool_groups", "")).lower():
        primary_axis = "language"

    if include_aggregate:
        parts.append(
            f"## Overall model stats\n"
            f"{json.dumps(compare_overall(records), indent=2, ensure_ascii=False, default=str)}"
        )
        full_groups = compare_groups_full(records, target_models, group_by=primary_axis)
        parts.append(
            f"## Full group comparison ({primary_axis})\n"
            f"{json.dumps(full_groups, indent=2, ensure_ascii=False, default=str)[:5000]}"
        )
        benchmarks_present = sorted({r.get("source_dataset") for r in records if r.get("source_dataset")})
        if len(benchmarks_present) > 1:
            dash = benchmark_dashboard(records, target_models=target_models or None)
            parts.append(
                f"## Benchmark dashboard\n"
                f"{json.dumps(dash, indent=2, ensure_ascii=False, default=str)[:4000]}"
            )
        for m in target_models:
            recs = slice_filter(records, model=m)
            by_axis = group_stats(recs, primary_axis)
            tb = top_bottom_slices(by_axis)
            parts.append(
                f"## {MODEL_DISPLAY.get(m, m)} by {primary_axis}\n"
                f"Top: {json.dumps(tb['top'], ensure_ascii=False, default=str)}\n"
                f"Bottom: {json.dumps(tb['bottom'], ensure_ascii=False, default=str)}"
            )
            rp = inspect_response_patterns(records, model=m)
            ft = failure_type_stats(records, model=m, all_records=records)
            parts.append(
                f"## Response patterns for {MODEL_DISPLAY.get(m, m)}\n"
                f"{json.dumps(rp, indent=2, ensure_ascii=False, default=str)}\n"
                f"## Failure types for {MODEL_DISPLAY.get(m, m)}\n"
                f"{json.dumps(ft, indent=2, ensure_ascii=False, default=str)}"
            )
        if len(target_models) >= 2:
            cmp = compare_models(records, target_models[0], target_models[1], group_by=primary_axis)
            cmp_sorted = sorted(cmp, key=lambda x: abs(x.get("accuracy_delta", 0)), reverse=True)[:10]
            parts.append(
                f"## Comparison {target_models[0]} vs {target_models[1]} ({primary_axis})\n"
                f"{json.dumps(cmp_sorted, indent=2, ensure_ascii=False, default=str)}"
            )

    if plan.get("need_tag_tools") and include_tag and include_aggregate:
        for m in target_models[:2]:
            ts = tag_stats(records, model=m, sort_by="error_rate", min_count=20, top_k=10)
            parts.append(
                f"## tag_stats({m}, sort_by=error_rate, min_count=20)\n"
                f"{json.dumps(ts, indent=2, ensure_ascii=False, default=str)}"
            )

    if include_cases:
        def _maybe_scrub(cp):
            if scrub_tag_in_cases and isinstance(cp, dict):
                cp = dict(cp)
                cp.pop("tag_category", None)
            return cp

        for m in target_models[:2]:
            errors = retrieve_error_cases(records, model=m, limit=5)
            cps = [_maybe_scrub(_to_case_pack(e, classify_failure(e, records),
                                                why="raw_data_context")) for e in errors]
            parts.append(
                f"## Sample errors for {MODEL_DISPLAY.get(m, m)}\n"
                f"{json.dumps(cps, indent=2, ensure_ascii=False, default=str)}"
            )
        if len(target_models) >= 2:
            disagree = retrieve_disagreement_cases(records, target_models[1], target_models[0], limit=3)
            disagree_scrubbed = [_maybe_scrub(d) for d in disagree[:3]]
            parts.append(
                f"## Disagreement: {target_models[1]} correct, {target_models[0]} wrong\n"
                f"{json.dumps(disagree_scrubbed, indent=2, ensure_ascii=False, default=str)}"
            )

    return "\n\n".join(parts)

def run_reporter(
    client, query_text: str, bound_package: dict[str, Any],
    raw_data_context: str = "",
    evidence_ledger: str = "",
    data_profile: str = "",
    response_language: str = "Chinese",
) -> str:
    """Run the Synthesizer/Reporter agent.

    `response_language` is interpolated into REPORTER_SYSTEM via the
    `$response_language$` placeholder. Default `"Chinese"`; multilingual runs pass a different
    value (e.g. `"Japanese"`).
    """
    def _summarize(d: Any, max_len: int = 6000) -> str:
        s = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        return s[:max_len] if len(s) > max_len else s

    plan = bound_package.get("plan", {})
    low_conf = json.dumps(plan.get("low_confidence_claims", []), ensure_ascii=False, default=str)
    dropped = json.dumps(plan.get("dropped_claims", []), ensure_ascii=False, default=str)
    rewrites = json.dumps(plan.get("rewrite_instructions", []), ensure_ascii=False, default=str)

    system = render(
        REPORTER_SYSTEM,
        query=query_text,
        data_profile=data_profile,
        plan_summary=_summarize(plan),
        evidence_summary=_summarize(bound_package.get("evidence", {})),
        case_summary=_summarize(bound_package.get("cases", {})),
        reflector_summary=_summarize(bound_package.get("reflections", [])),
        raw_data_context=raw_data_context,
        evidence_ledger=evidence_ledger[:8000] if evidence_ledger else "(no ledger)",
        low_confidence_claims=low_conf,
        dropped_claims=dropped,
        rewrite_instructions=rewrites,
        response_language=response_language,
    )
    raw = _call_llm(
        client, system,
        f"Generate the final diagnostic report for query: {query_text}\n\n"
        "CRITICAL: Only use numbers and cases from the input above. Do NOT invent data. "
        "Every quantitative claim MUST end with [N=..., metric=...] citation.",
        max_tokens=32768,
    )
    return raw

def run_reporter_revision(
    client, query_text: str, draft: str,
    reflector_post: dict[str, Any], evidence: dict[str, Any],
    raw_data_context: str = "",
    evidence_ledger: str = "",
    response_language: str = "Chinese",
) -> str:
    """Run a targeted revision of the report.

    `response_language` follows the same contract as `run_reporter()`:
    default `"Chinese"` is the default behaviour; multilingual callers
    pass the audited query's display language.
    """
    def _summarize(d: Any, max_len: int = 5000) -> str:
        s = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        return s[:max_len] if len(s) > max_len else s

    is_empty_draft = len(draft.strip()) < 500
    if is_empty_draft:
        user_msg = (
            f"The previous draft was empty or too short. Write a COMPLETE diagnostic "
            f"report from scratch for query: {query_text}\n\n"
            "Use the evidence pool, ledger and raw data context below. Follow the "
            "9-section structure. Include tables, specific numbers, and "
            "[N=..., metric=...] citations on every quantitative claim."
        )
    else:
        user_msg = (
            "Revise the report to fix the listed issues. Keep changes minimal and "
            "targeted. Ensure every quantitative claim has a [N=..., metric=...] "
            "citation backed by the evidence pool / ledger."
        )

    system = render(
        REPORTER_REVISION_SYSTEM,
        query=query_text,
        draft_report=draft[:6000] if not is_empty_draft else "(Draft was empty — write from scratch)",
        cultural_issues=json.dumps(reflector_post.get("cultural_issues", []), ensure_ascii=False, default=str),
        grounding_issues=json.dumps(reflector_post.get("grounding_issues", []), ensure_ascii=False, default=str),
        overclaims=json.dumps(reflector_post.get("overclaims", []), ensure_ascii=False, default=str),
        rewrite_instructions=json.dumps(reflector_post.get("rewrite_instructions", []), ensure_ascii=False, default=str),
        evidence_pool=_summarize(evidence),
        evidence_ledger=evidence_ledger[:6000] if evidence_ledger else "(no ledger)",
        raw_data_context=raw_data_context,
        response_language=response_language,
    )
    raw = _call_llm(client, system, user_msg, max_tokens=32768)
    return raw

CLAIM_TAG_RE = re.compile(r'\[N=(\d+)(?:,\s*metric=([^,\]]+)=([\d\.\-]+))?\]')

def extract_report_claims(report_text: str) -> list[dict[str, Any]]:
    """Pull `[N=..., metric=...]`-tagged claims out of the report.

    Returns a list of {claim_id, citation, n, metric, value, tail_text}.
    """
    claims = []
    cid = 0
    for m in CLAIM_TAG_RE.finditer(report_text):
        cid += 1
        n = int(m.group(1))
        metric = (m.group(2) or "").strip()
        val_raw = (m.group(3) or "").strip()
        try:
            val = float(val_raw) if val_raw else None
        except ValueError:
            val = None
        start = max(0, m.start() - 200)
        snippet = report_text[start:m.start()].rsplit("\n", 1)[-1].strip()
        claims.append({
            "claim_id": f"C{cid:03d}",
            "citation": m.group(0),
            "n": n,
            "metric": metric,
            "value": val,
            "claim_text_tail": snippet[:200],
        })
    return claims
