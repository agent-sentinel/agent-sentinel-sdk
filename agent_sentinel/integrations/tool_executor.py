"""
Base tool executor for AgentSentinel.

Provides SentinelToolExecutor base class that wraps tool functions
with policy enforcement and returns structured ToolResult objects
instead of raising exceptions — enabling LLM self-repair loops.
"""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..guard import guarded_action
from ..errors import AgentSentinelError, EvidenceViolationError

logger = logging.getLogger("agent_sentinel.integrations.tool_executor")


@dataclass
class ToolResult:
    """Result of a tool execution through SentinelToolExecutor."""
    tool_call_id: str
    tool_name: str
    output: Any = None
    blocked: bool = False
    error: Optional[str] = None
    remediation: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "blocked": self.blocked,
            "error": self.error,
            "remediation": self.remediation,
        }


ToolConfig = Dict[str, Any]  # {produces_evidence, is_commit, requires, ...}
ToolSpec = Tuple[Callable[..., Any], ToolConfig]


class SentinelToolExecutor:
    """
    Base tool executor that wraps tool functions with policy enforcement.

    Tools are registered with their enforcement config, wrapped with
    guarded_action, and executed through execute_tool() which catches
    policy violations and returns ToolResult instead of raising.

    Usage:
        executor = SentinelToolExecutor({
            "lookup_order": (lookup_order_fn, {"produces_evidence": True}),
            "issue_refund": (issue_refund_fn, {"is_commit": True, "requires": ["lookup_order"]}),
        })

        result = executor.execute_tool("lookup_order", {"order_id": "123"}, tool_call_id="tc_1")
    """

    def __init__(self, tools: Optional[Dict[str, ToolSpec]] = None):
        self._tools: Dict[str, Callable] = {}
        self._configs: Dict[str, ToolConfig] = {}
        self._originals: Dict[str, Callable] = {}

        if tools:
            for name, (func, config) in tools.items():
                self.register(name, func, config)

    def register(self, name: str, func: Callable, config: Optional[ToolConfig] = None) -> None:
        """Register a tool function with enforcement config."""
        config = config or {}
        guarded_config = {
            "name": name,
            **{k: v for k, v in config.items() if v is not None},
        }
        wrapped = guarded_action(**guarded_config)(func)
        self._tools[name] = wrapped
        self._configs[name] = config
        self._originals[name] = func

    def execute_tool(
        self,
        name: str,
        kwargs: Dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        """
        Execute a tool with policy enforcement.

        Returns ToolResult instead of raising on policy violations,
        enabling LLM self-repair loops.
        """
        if name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=f"Unknown tool: {name}",
            )

        func = self._tools[name]

        if inspect.iscoroutinefunction(func):
            raise TypeError(
                f"Tool '{name}' is async. Use async_execute_tool() instead."
            )

        try:
            output = func(**kwargs)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                output=output,
            )
        except EvidenceViolationError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=e.remediation.to_dict(),
            )
        except AgentSentinelError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=e.to_dict() if hasattr(e, 'to_dict') else None,
            )

    async def async_execute_tool(
        self,
        name: str,
        kwargs: Dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        """Async variant of execute_tool."""
        if name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=f"Unknown tool: {name}",
            )

        func = self._tools[name]

        try:
            if inspect.iscoroutinefunction(func):
                output = await func(**kwargs)
            else:
                output = func(**kwargs)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                output=output,
            )
        except EvidenceViolationError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=e.remediation.to_dict(),
            )
        except AgentSentinelError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=e.to_dict() if hasattr(e, 'to_dict') else None,
            )

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_tool_configs(self) -> Dict[str, ToolConfig]:
        """Get registered tool configurations (for auto-registration)."""
        return dict(self._configs)
