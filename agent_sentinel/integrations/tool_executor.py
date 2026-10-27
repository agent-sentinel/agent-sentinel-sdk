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
from ..errors import AgentSentinelError

logger = logging.getLogger("agent_sentinel.integrations.tool_executor")


def _remediation_of(error: AgentSentinelError) -> Optional[dict]:
    """
    Build the blocked-action payload handed back to the model.

    The published release had two different shapes here, and both have callers:

      - evidence violations returned the bare RemediationPayload dict, so
        callers read ``["reason_code"]`` and ``["retry_guidance"]``
      - every other error returned ``error.to_dict()``, so callers read
        ``["message"]``, ``["error_code"]``, ``["details"]``, ``["recoverable"]``

    Picking either one alone breaks the other group. The two key sets are
    disjoint, so this flattens them into a single dict that satisfies both:

        {error_type, error_code, message, details, recoverable,
         reason_code, retry_guidance, safe_alternatives, ...}

    A nested ``remediation`` key is also included so new code has one obvious
    place to look without relying on the flattening.
    """
    if not hasattr(error, "to_dict"):
        return None

    payload = error.to_dict()

    remediation = getattr(error, "remediation", None)
    if remediation is None or not hasattr(remediation, "to_dict"):
        return payload

    guidance = remediation.to_dict()

    # Guard the disjointness assumption rather than trusting it forever: if a
    # future payload field ever collides with an error field, the error field
    # wins at top level and the guidance stays reachable under "remediation".
    for key, value in guidance.items():
        if key not in payload:
            payload[key] = value

    payload["remediation"] = guidance
    return payload


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
        except AgentSentinelError as e:
            # Every violation type that carries a RemediationPayload surfaces it
            # here — evidence, deny list, allowlist, rate limit, budget. Errors
            # without one (network, replay divergence) fall back to their error
            # dict, which is diagnostic rather than actionable.
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=_remediation_of(e),
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
        except AgentSentinelError as e:
            # Every violation type that carries a RemediationPayload surfaces it
            # here — evidence, deny list, allowlist, rate limit, budget. Errors
            # without one (network, replay divergence) fall back to their error
            # dict, which is diagnostic rather than actionable.
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                blocked=True,
                error=str(e),
                remediation=_remediation_of(e),
            )

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def get_tool_configs(self) -> Dict[str, ToolConfig]:
        """Get registered tool configurations (for auto-registration)."""
        return dict(self._configs)
