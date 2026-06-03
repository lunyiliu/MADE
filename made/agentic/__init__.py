"""MADE agentic framework: Tool registry, agent loop, shared state.

Core patterns:
- Tool ABC with JSON Schema parameters
- ToolRegistry for name-based lookup + OpenAI function format
- AgentLoop for synchronous LLM ↔ tool iteration
- SharedState for cross-agent evidence sharing
"""

from made.agentic.base import Tool, ToolRegistry
from made.agentic.loop import AgentLoop
from made.agentic.state import SharedState

__all__ = ["Tool", "ToolRegistry", "AgentLoop", "SharedState"]
