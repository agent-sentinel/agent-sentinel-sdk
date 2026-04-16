"""
OpenAI tool-use integration for AgentSentinel.

Provides OpenAISentinelTools that wraps tool functions and dispatches
OpenAI tool_call objects through policy enforcement.

Usage:
    from agent_sentinel.integrations.openai import OpenAISentinelTools

    tools = OpenAISentinelTools({
        "lookup_order": (lookup_order, {"produces_evidence": True}),
        "issue_refund": (issue_refund, {"is_commit": True, "requires": ["lookup_order"]}),
    })

    # In tool-call loop:
    for tool_call in response.choices[0].message.tool_calls:
        result = tools.execute(tool_call)
        messages.append(tools.to_tool_message(result))
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .tool_executor import SentinelToolExecutor, ToolResult

logger = logging.getLogger("agent_sentinel.integrations.openai")


class OpenAISentinelTools(SentinelToolExecutor):
    """
    OpenAI-specific tool executor.

    Understands OpenAI's tool_call format:
        tool_call.id -> str
        tool_call.function.name -> str
        tool_call.function.arguments -> str (JSON)
    """

    def execute(self, tool_call: Any) -> ToolResult:
        """
        Execute an OpenAI tool_call with policy enforcement.

        Args:
            tool_call: OpenAI tool_call object from response.choices[0].message.tool_calls

        Returns:
            ToolResult with output or remediation payload
        """
        name = tool_call.function.name
        try:
            kwargs = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            kwargs = {}

        return self.execute_tool(name, kwargs, tool_call_id=tool_call.id)

    async def async_execute(self, tool_call: Any) -> ToolResult:
        """Async variant of execute."""
        name = tool_call.function.name
        try:
            kwargs = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            kwargs = {}

        return await self.async_execute_tool(name, kwargs, tool_call_id=tool_call.id)

    @staticmethod
    def to_tool_message(result: ToolResult) -> dict:
        """
        Format ToolResult as an OpenAI tool message for the conversation.

        Returns:
            Dict compatible with OpenAI messages format:
            {"role": "tool", "tool_call_id": "...", "content": "..."}
        """
        if result.blocked:
            content = json.dumps({
                "error": result.error,
                "blocked": True,
                "remediation": result.remediation,
            })
        else:
            content = json.dumps(result.output) if not isinstance(result.output, str) else result.output

        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": content,
        }
