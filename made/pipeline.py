"""MADE pipeline: orchestrates the five agents for a single query.

- ``PipelineConfig`` consolidates optional component toggles
  (evidence / case / reflector / tag tools / revision loop / claim ledger /
  raw-data context / iteration tools); the default enables everything.
- ``ToolCallLedger`` captures every Analyst tool call, used to back the
  Reporter's evidence ledger and the post-report grounding check.
- The Reflector's ``replan`` action is a closed loop: it may trigger one
  extra Planner call and the routing fields propagate.
- Routing gates suppress leakage: when ``need_evidence`` is False the
  raw-data context is suppressed; when ``need_case`` is False the case pool
  is empty; a disabled agent is skipped and its output is removed from the
  Reporter's input package.
- Standardized ``intermediate`` keys for downstream analysis:
  stages_timing_sec, evidence_ledger_n_calls, routing_outcome,
  claims_extracted.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from made.data_loader import Record
from made.tools import bind_report
from made.agents import (
    run_planner,
    run_evidence_analyst, run_evidence_analyst_agentic,
    run_case_analyst, run_case_analyst_agentic,
    run_reflector,
    run_reporter,
    run_reporter_revision,
    check_report_grounding,
    preclassify_failure_types,
    _build_raw_data_context,
    build_data_profile,
    extract_report_claims,
    ROUTING_DEFAULTS,
)
from made.agentic.state import SharedState
from made.agentic.made_tools import (
    ToolCallLedger,
    TAG_TOOL_NAMES, ITERATION_TOOL_NAMES,
)

log = logging.getLogger("made.pipeline")

@dataclass
class PipelineConfig:
    """Optional component toggles for the pipeline.

    Each flag disables one part of the pipeline (an agent, the revision
    loop, the claim ledger, the raw-data context, or a tool family). The
    default — everything enabled — is the full MADE system; the toggles let
    callers run reduced configurations for analysis.
    """
    disable_planner: bool = False
    disable_evidence: bool = False
    disable_case: bool = False
    disable_reflector: bool = False
    disable_reporter: bool = False
    disable_revision_loop: bool = False
    disable_claim_ledger: bool = False
    disable_raw_data_context: bool = False
    disable_tag_tools: bool = False
    disable_iteration_tools: bool = False
    scrub_tag_in_cases: bool = False

    def tools_drop(self) -> set[str]:
        out: set[str] = set()
        if self.disable_tag_tools:
            out |= TAG_TOOL_NAMES
        if self.disable_iteration_tools:
            out |= ITERATION_TOOL_NAMES
        return out

    def to_dict(self) -> dict[str, bool]:
        return {
            "disable_planner": self.disable_planner,
            "disable_evidence": self.disable_evidence,
            "disable_case": self.disable_case,
            "disable_reflector": self.disable_reflector,
            "disable_reporter": self.disable_reporter,
            "disable_revision_loop": self.disable_revision_loop,
            "disable_claim_ledger": self.disable_claim_ledger,
            "disable_raw_data_context": self.disable_raw_data_context,
            "disable_tag_tools": self.disable_tag_tools,
            "disable_iteration_tools": self.disable_iteration_tools,
            "scrub_tag_in_cases": self.scrub_tag_in_cases,
        }

def run_made_pipeline(
    client,
    query: dict[str, Any],
    records: list[Record],
    verbose: bool = True,
    agentic: bool = True,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the full MADE pipeline for a single query."""
    cfg = config or PipelineConfig()
    qid = query["id"]
    qtext = query["text"]
    response_language = str(query.get("response_language") or "Chinese")
    intermediate: dict[str, Any] = {
        "agentic_mode": agentic,
        "config": cfg.to_dict(),
        "made_version": "1.0",
        "stages_timing_sec": {},
        "query_lang": query.get("lang"),
        "response_language": response_language,
    }
    state = SharedState(records, query)
    timings = intermediate["stages_timing_sec"]

    data_profile = build_data_profile(records, plan_query=query)
    intermediate["data_profile"] = data_profile

    if verbose:
        log.info(f"[{qid}] Starting MADE pipeline (agentic={agentic}, "
                 f"cfg={cfg.to_dict()}) for: {qtext[:60]}...")

    if cfg.disable_planner:
        from made.agents import _expand_all_target_models
        plan = {
            "task_type": "hybrid",
            "need_evidence": True,
            "need_case": True,
            "need_tag_tools": False,
            "need_iteration_tools": False,
            "primary_focus": "weakness",
            "target_models": ["all"],
            "target_benchmarks": ["all"],
            "target_languages": [],
            "target_groups": [],
            "evidence_analyst_tasks": [
                "Compute broad aggregate performance patterns for the sliced data.",
                "Identify weak model/language/benchmark groups when "
                "directly visible from generic statistics.",
            ],
            "case_analyst_tasks": [
                "Retrieve representative generic error cases.",
                "Summarize recurring failure patterns without relying on "
                "query-specific Planner routing.",
            ],
            "planned_tool_groups": [
                "compare_overall", "group_stats", "error_cases",
            ],
            "questions_to_answer": [],
            "risks": [
                "Planner is disabled; routing is intentionally generic and "
                "may miss tag-heavy or iteration-heavy structure."
            ],
            "_skipped": "planner disabled: static generic hybrid plan applied",
        }
        plan = _expand_all_target_models(plan, records)
        timings["planner"] = 0.0
        intermediate["plan"] = plan
    else:
        t0 = time.time()
        client.config.stage = "planner"
        plan = run_planner(client, query, records)
        client.config.stage = "pipeline"
        timings["planner"] = round(time.time() - t0, 2)
        intermediate["plan"] = plan
    intermediate["routing_initial"] = {
        "task_type": plan.get("task_type"),
        "need_evidence": plan.get("need_evidence"),
        "need_case": plan.get("need_case"),
        "need_tag_tools": plan.get("need_tag_tools"),
        "need_iteration_tools": plan.get("need_iteration_tools"),
    }
    need_evidence = plan.get("need_evidence", True)
    need_case = plan.get("need_case", True)
    if cfg.disable_evidence:
        need_evidence = False
    if cfg.disable_case:
        need_case = False

    if verbose:
        log.info(f"[{qid}] Planner decided: evidence={need_evidence}, case={need_case}, "
                 f"tag={plan.get('need_tag_tools')}, iter={plan.get('need_iteration_tools')}")

    if cfg.disable_reflector:
        reflector_pre = {"reflection_stage": "pre_plan", "_skipped": True}
        intermediate["reflector_pre_plan"] = reflector_pre
        replanned = False
    else:
        t0 = time.time()
        client.config.stage = "reflector_pre_plan"
        reflector_pre = run_reflector(client, "pre_plan", plan, records=records)
        client.config.stage = "pipeline"
        timings["reflector_pre_plan"] = round(time.time() - t0, 2)
        intermediate["reflector_pre_plan"] = reflector_pre

        wants_replan = any(
            isinstance(a, dict) and a.get("action") == "replan"
            for a in reflector_pre.get("required_actions", [])
        )
        replanned = False
        if wants_replan:
            if cfg.disable_planner:
                log.info(
                    f"[{qid}] Reflector requested replan, but "
                    f"disable_planner=True; keeping static no_planner plan."
                )
                intermediate["replan_skipped_due_to_disable_planner"] = True
                replanned = False
            else:
                log.info(f"[{qid}] Reflector requested replan; running Planner again...")
                replan_query = dict(query)
                concerns = (reflector_pre.get("non_cultural_warnings", []) +
                            reflector_pre.get("cultural_warnings", []))
                replan_query["text"] = (
                    f"{query['text']}\n\n[Replan note: previous routing was flagged. "
                    f"Reflector concerns: {concerns[:5]}]"
                )
                t0 = time.time()
                client.config.stage = "planner_replan"
                plan = run_planner(client, replan_query, records)
                client.config.stage = "pipeline"
                timings["planner_replan"] = round(time.time() - t0, 2)
                intermediate["plan_after_replan"] = plan
                replanned = True
                need_evidence = plan.get("need_evidence", need_evidence)
                need_case = plan.get("need_case", need_case)
                if cfg.disable_evidence:
                    need_evidence = False
                if cfg.disable_case:
                    need_case = False

        _process_reflector_actions(plan, reflector_pre, "pre_plan")

    intermediate["routing_final"] = {
        "task_type": plan.get("task_type"),
        "need_evidence": need_evidence,
        "need_case": need_case,
        "need_tag_tools": plan.get("need_tag_tools"),
        "need_iteration_tools": plan.get("need_iteration_tools"),
        "replanned": replanned,
    }

    if need_case and not cfg.disable_case:
        t0 = time.time()
        client.config.stage = "preclassify"
        n_factual = preclassify_failure_types(client, records, max_llm_calls=30)
        client.config.stage = "pipeline"
        timings["preclassify"] = round(time.time() - t0, 2)
        intermediate["factual_error_count"] = n_factual

    ledger = ToolCallLedger() if not cfg.disable_claim_ledger else None
    tools_drop = set(cfg.tools_drop())
    if cfg.disable_planner:
        tools_drop |= TAG_TOOL_NAMES | ITERATION_TOOL_NAMES

    evidence: dict = {}
    if need_evidence:
        t0 = time.time()
        client.config.stage = "evidence_analyst"
        if agentic:
            evidence = run_evidence_analyst_agentic(
                client, plan, records,
                ledger=ledger, tools_drop=tools_drop,
            )
        else:
            evidence = run_evidence_analyst(client, plan, records)
        client.config.stage = "pipeline"
        timings["evidence"] = round(time.time() - t0, 2)
        state.evidence_pool = evidence
    else:
        if verbose:
            log.info(f"[{qid}] Step 3: Skipping Evidence Analyst (need_evidence=False)")
    intermediate["evidence"] = evidence

    cases: dict = {}
    if need_case:
        t0 = time.time()
        client.config.stage = "case_analyst"
        if agentic:
            cases = run_case_analyst_agentic(
                client, plan, records,
                ledger=ledger, tools_drop=tools_drop,
            )
        else:
            cases = run_case_analyst(client, plan, records)
        client.config.stage = "pipeline"
        timings["case"] = round(time.time() - t0, 2)
        state.case_pool = cases
    else:
        if verbose:
            log.info(f"[{qid}] Step 4: Skipping Case Analyst (need_case=False)")
    intermediate["cases"] = cases

    if not cfg.disable_reflector and (need_evidence or need_case):
        t0 = time.time()
        client.config.stage = "reflector_mid"
        reflector_mid = run_reflector(
            client, "mid_analysis",
            {"evidence": evidence, "cases": cases},
            records=records,
        )
        client.config.stage = "pipeline"
        timings["reflector_mid"] = round(time.time() - t0, 2)
        intermediate["reflector_mid_analysis"] = reflector_mid
        _process_reflector_actions(plan, reflector_mid, "mid_analysis")
    else:
        reflector_mid = {"_skipped": cfg.disable_reflector}
        intermediate["reflector_mid_analysis"] = reflector_mid

    bound = bind_report(
        query=qtext,
        planner_output=plan,
        evidence_findings=evidence,
        case_findings=cases,
        reflector_opinions=[reflector_pre, reflector_mid],
    )

    raw_data_context = ""
    if not cfg.disable_raw_data_context:
        t0 = time.time()
        include_aggregate = (not cfg.disable_evidence) and need_evidence
        include_cases = (not cfg.disable_case) and need_case
        include_tag = (not cfg.disable_tag_tools)
        raw_data_context = _build_raw_data_context(
            records, plan,
            include_aggregate=include_aggregate,
            include_cases=include_cases,
            include_tag=include_tag,
            scrub_tag_in_cases=cfg.scrub_tag_in_cases,
        )
        timings["raw_data_context"] = round(time.time() - t0, 2)
        intermediate["raw_data_context_chars"] = len(raw_data_context)
        intermediate["raw_data_context_gating"] = {
            "include_aggregate": include_aggregate,
            "include_cases": include_cases,
            "include_tag": include_tag,
            "scrub_tag_in_cases": cfg.scrub_tag_in_cases,
        }
        if ledger is not None:
            ledger.record(
                "_build_raw_data_context",
                {
                    "include_aggregate": include_aggregate,
                    "include_cases": include_cases,
                    "include_tag": include_tag,
                },
                {"chars": len(raw_data_context),
                 "preview": raw_data_context[:600]},
            )
    else:
        intermediate["raw_data_context_chars"] = 0
        intermediate["raw_data_context_gating"] = {
            "include_aggregate": False,
            "include_cases": False,
            "include_tag": False,
        }

    evidence_ledger_summary = ""
    if ledger is not None:
        evidence_ledger_summary = ledger.to_summary()
        intermediate["evidence_ledger_n_calls"] = len(ledger.entries)
        intermediate["evidence_ledger_chars"] = len(evidence_ledger_summary)
    else:
        intermediate["evidence_ledger_n_calls"] = 0
        intermediate["evidence_ledger_chars"] = 0

    if cfg.disable_reporter:
        report_draft = _build_no_reporter_report(
            qtext, evidence, cases,
            [reflector_pre, reflector_mid],
            evidence_ledger_summary,
            data_profile,
        )
        timings["reporter_draft"] = 0.0
        intermediate["report_draft_chars"] = len(report_draft)
        intermediate["reporter_skipped"] = True
    else:
        t0 = time.time()
        client.config.stage = "reporter_draft"
        report_draft = run_reporter(
            client, qtext, bound,
            raw_data_context=raw_data_context,
            evidence_ledger=evidence_ledger_summary,
            data_profile=data_profile,
            response_language=response_language,
        )
        client.config.stage = "pipeline"
        timings["reporter_draft"] = round(time.time() - t0, 2)
        intermediate["report_draft_chars"] = len(report_draft)
        if len(report_draft) < 4000:
            intermediate["report_draft"] = report_draft

    grounding = check_report_grounding(report_draft, evidence, cases)
    intermediate["grounding_heuristic"] = grounding

    if not cfg.disable_reflector:
        t0 = time.time()
        client.config.stage = "reflector_post"
        reflector_post = run_reflector(
            client, "post_report",
            {
                "report": report_draft,
                "evidence": evidence,
                "cases": cases,
                "heuristic_warnings": grounding.get("grounding_warnings", []),
            },
            records=records,
            evidence_ledger_summary=evidence_ledger_summary,
        )
        client.config.stage = "pipeline"
        timings["reflector_post"] = round(time.time() - t0, 2)
        intermediate["reflector_post_report"] = reflector_post
    else:
        reflector_post = {"pass": True, "_skipped": True}
        intermediate["reflector_post_report"] = reflector_post

    draft_too_short = len(report_draft.strip()) < 500
    revision_triggered = False
    if not cfg.disable_revision_loop and not cfg.disable_reporter:
        revision_triggered = draft_too_short or not reflector_post.get("pass", False)
    intermediate["revision_triggered"] = revision_triggered
    intermediate["draft_too_short"] = draft_too_short

    if revision_triggered:
        reason = "draft too short" if draft_too_short else "reflector flagged issues"
        if verbose:
            log.info(f"[{qid}] Step 9: Revision triggered ({reason})...")
        t0 = time.time()
        client.config.stage = "reporter_revise"
        report = run_reporter_revision(
            client, qtext, report_draft, reflector_post, evidence,
            raw_data_context=raw_data_context,
            evidence_ledger=evidence_ledger_summary,
            response_language=response_language,
        )
        client.config.stage = "pipeline"
        timings["reporter_revise"] = round(time.time() - t0, 2)
        intermediate["report_revised_chars"] = len(report)
        _draft_len = len(report_draft.strip())
        _rev_len = len(report.strip())

        def _structural_failure(text: str) -> str | None:
            """cheap structural check on revision
            output. Returns a reason string when revision fails the
            check, else None. Three signals are required; failing any
            triggers fallback to draft.

            1. has at least one markdown heading (`#` line)
            2. has >= 2 numeric evidence markers (`[N=` / `metric=` /
               `win_rate` / `accuracy` / percentages)
            3. has >= 2 paragraph signals (bullet, numbered item, or
               blank-line-separated paragraph)
            """
            if not text or len(text.strip()) < 200:
                return "too_short_or_empty"
            if not re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
                return "no_heading"
            ev_n = (
                len(re.findall(r"\[N=", text))
                + len(re.findall(r"metric\s*=", text))
                + len(re.findall(r"\bwin_rate\b", text))
                + text.lower().count("accuracy")
                + len(re.findall(r"\b\d+(?:\.\d+)?%\b", text))
            )
            if ev_n < 2:
                return f"too_few_evidence_markers({ev_n})"
            para_n = (
                len(re.findall(r"^\s*[-*]\s+", text, re.MULTILINE))
                + len(re.findall(r"^\s*\d+\.\s+", text, re.MULTILINE))
                + text.count("\n\n")
            )
            if para_n < 2:
                return "no_paragraph_structure"
            return None

        length_too_short = _rev_len < max(200, _draft_len // 4)
        struct_reason = None if length_too_short else _structural_failure(report)
        if length_too_short or struct_reason is not None:
            fb_reason = "length_below_threshold" if length_too_short else struct_reason
            if verbose:
                log.warning(
                    f"[{qid}] revision returned {_rev_len} chars / "
                    f"reason={fb_reason} (draft was {_draft_len}); "
                    f"falling back to draft"
                )
            intermediate["revision_fallback_to_draft"] = True
            intermediate["revision_fallback_reason"] = fb_reason
            intermediate["revision_fallback_revised_chars"] = _rev_len
            intermediate["draft_chars"] = _draft_len
            intermediate["revision_chars"] = _rev_len
            intermediate["report_source"] = "draft_fallback"
            report = report_draft
            intermediate["report_revised_chars"] = len(report)
        else:
            intermediate["revision_fallback_to_draft"] = False
            intermediate["draft_chars"] = _draft_len
            intermediate["revision_chars"] = _rev_len
            intermediate["report_source"] = "revised"
        grounding = check_report_grounding(report, evidence, cases)
        intermediate["grounding_post_revision"] = grounding
    else:
        report = report_draft
        intermediate["report_source"] = "draft_unrevised"
        intermediate["draft_chars"] = len(report_draft.strip())

    claims = extract_report_claims(report)
    intermediate["claims_extracted"] = claims
    intermediate["claims_count"] = len(claims)

    if agentic:
        tool_audit: dict[str, Any] = {}
        if isinstance(evidence, dict) and evidence.get("_agentic_meta"):
            tool_audit["evidence_analyst"] = evidence["_agentic_meta"]
        if isinstance(cases, dict) and cases.get("_agentic_meta"):
            tool_audit["case_analyst"] = cases["_agentic_meta"]
        intermediate["tool_audit"] = tool_audit

    if ledger is not None:
        intermediate["evidence_ledger"] = ledger.to_dict()

    if verbose:
        log.info(
            f"[{qid}] Pipeline complete. Report length: {len(report)} chars; "
            f"revision={'yes' if revision_triggered else 'no'}; "
            f"ledger_calls={intermediate['evidence_ledger_n_calls']}; "
            f"claims={intermediate['claims_count']}"
        )

    return {
        "query_id": qid,
        "query_text": qtext,
        "report": report,
        "intermediate": intermediate,
        "grounding": grounding,
    }

def _process_reflector_actions(
    plan: dict[str, Any],
    reflector_output: dict[str, Any],
    stage: str,
) -> None:
    """Process reflector's required_actions and integrate feedback into plan.

    replan is handled outside this fn (in the main pipeline). Other
    actions still fold into plan.{risks, supplementary_tools,
    low_confidence_claims, dropped_claims, rewrite_instructions}.
    """
    cultural_warnings = reflector_output.get("cultural_warnings", [])
    non_cultural_warnings = reflector_output.get("non_cultural_warnings", [])
    required_revisions = reflector_output.get("required_revisions", [])
    all_warnings = cultural_warnings + non_cultural_warnings + required_revisions
    if all_warnings:
        plan.setdefault("risks", []).extend(all_warnings)

    actions = reflector_output.get("required_actions", [])
    for action in actions:
        if not isinstance(action, dict):
            continue
        act_type = action.get("action", "")
        target = action.get("target", "")
        reason = action.get("reason", "")
        if act_type == "replan":
            continue
        elif act_type == "add_tool":
            log.info(f"Reflector requests additional tool: {target} ({reason})")
            plan.setdefault("supplementary_tools", []).append({
                "tool": target, "reason": reason, "stage": stage,
            })
        elif act_type == "lower_confidence":
            plan.setdefault("low_confidence_claims", []).append(target)
        elif act_type == "drop_claim":
            plan.setdefault("dropped_claims", []).append(target)
        elif act_type == "rewrite_section":
            log.info(f"Reflector requests section rewrite: {target} ({reason})")
            plan.setdefault("rewrite_instructions", []).append({
                "section": target, "reason": reason, "stage": stage,
            })

def _build_no_reporter_report(
    qtext: str,
    evidence: dict[str, Any],
    cases: dict[str, Any],
    reflector_opinions: list[dict[str, Any]],
    evidence_ledger_summary: str,
    data_profile: str,
) -> str:
    """Build a report shell without invoking the Reporter LLM.

    Used when ``disable_reporter`` is set. Sections come from analyst
    findings + reflector opinions concatenated in fixed order, with no
    synthesis or claim-evidence binding.
    """
    parts = [
        f"# Diagnostic findings (no-synthesis mode)\n",
        f"## Query\n{qtext}\n",
        f"## Data profile\n```\n{data_profile}\n```\n",
    ]
    if isinstance(evidence, dict) and evidence:
        parts.append("## Evidence Analyst findings")
        for k, v in evidence.items():
            if k.startswith("_"):
                continue
            parts.append(f"### {k}\n```json\n{json.dumps(v, ensure_ascii=False, indent=2, default=str)[:3000]}\n```")
    if isinstance(cases, dict) and cases:
        parts.append("## Case Analyst findings")
        for k, v in cases.items():
            if k.startswith("_"):
                continue
            parts.append(f"### {k}\n```json\n{json.dumps(v, ensure_ascii=False, indent=2, default=str)[:3000]}\n```")
    if reflector_opinions:
        parts.append("## Reflector opinions")
        for i, r in enumerate(reflector_opinions):
            if not r or r.get("_skipped"):
                continue
            parts.append(f"### Opinion {i + 1}\n```json\n{json.dumps(r, ensure_ascii=False, indent=2, default=str)[:2000]}\n```")
    if evidence_ledger_summary:
        parts.append("## Evidence ledger\n```\n" + evidence_ledger_summary[:3000] + "\n```")
    return "\n\n".join(parts)
