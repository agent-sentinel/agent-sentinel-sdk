"""
Anthropic tool-use integration for AgentSentinel.

Provides AnthropicSentinelTools that wraps tool functions and dispatches
Anthropic tool_use content blocks through policy enforcement.

Usage:
    from agent_sentinel.integrations.anthropic_tools import AnthropicSentinelTools

    tools = AnthropicSentinelTools({
        "lookup_order": (lookup_order, {"produces_evidence": True}),
        "issue_refund": (issue_refund, {"is_commit": True, "requires": ["lookup_order"]}),
    })

    # In tool-use loop:
    for block in response.content:
        if block.type == "tool_use":
            result = tools.execute(block)
            tool_results.append(tools.to_tool_result_block(result))
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .tool_executor import SentinelToolExecutor, ToolResult

logger = logging.getLogger("agent_sentinel.integrations.anthropic_tools")


class AnthropicSentinelTools(SentinelToolExecutor):
    """
    Anthropic-specific tool executor.

    Understands Anthropic's tool_use content block format:
        block.id -> str
        block.name -> str
        block.input -> dict
    """

    def execute(self, block: Any) -> ToolResult:
        """
        Execute an Anthropic tool_use block with policy enforcement.

        Args:
            block: Anthropic content block where block.type == "tool_use"

        Returns:
            ToolResult with output or remediation payload
        """
        name = block.name
        kwargs = block.input if isinstance(block.input, dict) else {}

        return self.execute_tool(name, kwargs, tool_call_id=block.id)

    async def async_execute(self, block: Any) -> ToolResult:
        """Async variant of execute."""
        name = block.name
        kwargs = block.input if isinstance(block.input, dict) else {}

        return await self.async_execute_tool(name, kwargs, tool_call_id=block.id)

    @staticmethod
    def to_tool_result_block(result: ToolResult) -> dict:
        """
        Format ToolResult as an Anthropic tool_result block.

        Returns:
            Dict compatible with Anthropic tool_result format:
            {"type": "tool_result", "tool_use_id": "...", "content": "..."}
        """
        if result.blocked:
            content = json.dumps({
                "error": result.error,
                "blocked": True,
                "remediation": result.remediation,
            })
            return {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": content,
                "is_error": True,
            }
        else:
            content = json.dumps(result.output) if not isinstance(result.output, str) else result.output
            return {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": content,
            }
