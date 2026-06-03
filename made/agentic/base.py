"""Tool ABC and ToolRegistry — a minimal tool-registry pattern.

Tool: abstract base defining name/description/parameters/execute interface.
ToolRegistry: name-based dict registry with OpenAI function format output,
parameter validation, and guarded execution.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

log = logging.getLogger("made.agentic")

class Tool(ABC):
    """Abstract base class for a callable analysis tool.

    Subclasses must define name, description, parameters (JSON Schema),
    and a synchronous execute(**kwargs) method.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(self, **kwargs) -> Any: ...

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

class ToolRegistry:
    """Name-based registry for Tool instances.

    Provides registration, lookup, batch schema export (OpenAI format),
    and guarded execution with error wrapping.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, params: dict[str, Any]) -> Any:
        """Look up and execute a tool, returning its result or an error dict."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Tool '{name}' not found. Available: {list(self._tools.keys())}"}
        try:
            return tool.execute(**params)
        except TypeError as e:
            return {"error": f"Invalid parameters for {name}: {e}"}
        except Exception as e:
            log.warning("Tool %s raised %s: %s", name, type(e).__name__, e)
            return {"error": f"Error executing {name}: {e}"}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({self.tool_names})"
