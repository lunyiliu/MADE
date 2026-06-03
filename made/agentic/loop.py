"""Synchronous agent loop: LLM ↔ tool iteration.

A simplified ReAct-style runner: it sends messages + tool definitions to the
LLM; if the LLM responds with tool calls they are executed and the results
fed back; the loop ends when the LLM returns text with no tool calls or the
iteration budget is reached. Tool results are truncated to bound context and
every tool call is recorded for the audit trail. Failed LLM calls are retried
a few times; if they keep failing the loop returns ``stop_reason="error"``.
"""

import json
import logging
import time
from typing import Any

from made.agentic.base import ToolRegistry

log = logging.getLogger("made.agentic.loop")

TOOL_RESULT_MAX_CHARS = 12_000
MAX_LLM_RETRIES = 3

class AgentLoop:
    """Runs an LLM ↔ tool iteration loop for a single MADE agent.

    The loop sends messages + tool definitions to the LLM. If the LLM
    responds with tool_calls, they are executed and results fed back.
    The loop terminates when the LLM responds with text only (no tool
    calls) or max_iterations is reached.
    """

    def __init__(
        self,
        client,
        tools: ToolRegistry,
        max_iterations: int = 15,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_iterations = max_iterations

    def run(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Run the agent loop.

        Returns:
            dict with keys: content, tools_used, tool_events,
            iterations, stop_reason, total_usage
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tool_defs = self.tools.get_definitions()
        tools_used: list[str] = []
        tool_events: list[dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        text_content = ""

        for iteration in range(self.max_iterations):
            log.info("  agentic iter %d/%d, calling LLM...", iteration + 1, self.max_iterations)

            old_max = self.client.config.max_tokens
            self.client.config.max_tokens = max_tokens

            result = None
            for attempt in range(1, MAX_LLM_RETRIES + 1):
                result = self.client.chat(
                    messages=messages,
                    sample_count=1,
                    tools=tool_defs,
                )
                if not result.get("error"):
                    break
                log.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt, MAX_LLM_RETRIES, result["error"],
                )
                if attempt < MAX_LLM_RETRIES:
                    time.sleep(5 * attempt)

            self.client.config.max_tokens = old_max

            usage = result.get("usage") or {}
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

            if result.get("error"):
                log.error("LLM error in agent loop after %d retries: %s", MAX_LLM_RETRIES, result["error"])
                return {
                    "content": result.get("text", ""),
                    "tools_used": tools_used,
                    "tool_events": tool_events,
                    "iterations": iteration + 1,
                    "stop_reason": "error",
                    "error": result["error"],
                    "total_usage": total_usage,
                }

            tool_calls = result.get("tool_calls") or []
            text_content = result.get("text") or ""

            if not tool_calls:
                log.info(
                    "Agent loop completed: %d iterations, %d tool calls",
                    iteration + 1, len(tools_used),
                )
                return {
                    "content": text_content,
                    "tools_used": tools_used,
                    "tool_events": tool_events,
                    "iterations": iteration + 1,
                    "stop_reason": "completed",
                    "total_usage": total_usage,
                }

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                tc_id = tc.get("id", f"call_{iteration}_{len(tools_used)}")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")
                raw_args = func.get("arguments", "{}")

                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args or {}

                log.info("  → tool: %s(%s)", tool_name,
                         json.dumps(args, ensure_ascii=False)[:200])

                tool_result = self.tools.execute(tool_name, args)

                try:
                    result_str = json.dumps(
                        tool_result, ensure_ascii=False, default=str,
                    )
                except (TypeError, ValueError):
                    result_str = str(tool_result)

                truncated = False
                if len(result_str) > TOOL_RESULT_MAX_CHARS:
                    result_str = (
                        result_str[:TOOL_RESULT_MAX_CHARS]
                        + "\n...[truncated]"
                    )
                    truncated = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

                is_error = isinstance(tool_result, dict) and "error" in tool_result
                tools_used.append(tool_name)
                tool_events.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "args_summary": json.dumps(args, ensure_ascii=False)[:120],
                    "result_chars": len(result_str),
                    "truncated": truncated,
                    "status": "error" if is_error else "ok",
                })

        log.warning("Agent loop hit max iterations (%d)", self.max_iterations)
        return {
            "content": text_content,
            "tools_used": tools_used,
            "tool_events": tool_events,
            "iterations": self.max_iterations,
            "stop_reason": "max_iterations",
            "total_usage": total_usage,
        }
