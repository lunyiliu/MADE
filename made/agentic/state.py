"""Shared state for a single MADE pipeline run.

Provides a mutable evidence/case pool that agents read and write,
enabling data sharing across the pipeline without re-running tools.
"""

from typing import Any
from made.data_loader import Record

class SharedState:
    """Mutable state shared across all agents in one pipeline run."""

    def __init__(self, records: list[Record], query: dict[str, Any]) -> None:
        self.records = records
        self.query = query
        self.plan: dict[str, Any] = {}
        self.evidence_pool: dict[str, Any] = {}
        self.case_pool: dict[str, Any] = {}
        self.reflector_actions: list[dict[str, Any]] = []
        self.tool_audit: list[dict[str, Any]] = []

    def add_evidence(self, key: str, value: Any) -> None:
        self.evidence_pool[key] = value

    def add_cases(self, key: str, value: Any) -> None:
        self.case_pool[key] = value

    def add_tool_audit(self, agent: str, events: list[dict]) -> None:
        self.tool_audit.append({"agent": agent, "events": events})
