"""
LangGraph integration for AgentSentinel.

Provides SentinelToolNode that wraps LangGraph tool execution with
policy enforcement, evidence tracking, and structured block feedback.

Usage:
    from agent_sentinel.integrations.langgraph import SentinelToolNode

    # Define tools with evidence config
    tools = SentinelToolNode(
        tools={
            "lookup_order": (lookup_order_fn, {"produces_evidence": True}),
            "issue_refund": (issue_refund_fn, {"is_commit": True, "requires": ["lookup_order"]}),
        }
    )

    # Use in LangGraph
    graph = StateGraph(State)
    graph.add_node("tools", tools)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .tool_executor import SentinelToolExecutor, ToolResult

logger = logging.getLogger("agent_sentinel.integrations.langgraph")

# Try importing LangGraph/LangChain types
try:
    from langchain_core.messages import AIMessage, ToolMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    AIMessage = Any  # type: ignore
    ToolMessage = Any  # type: ignore


ToolSpec = Tuple[Callable[..., Any], Dict[str, Any]]


class SentinelToolNode:
    """
    LangGraph-compatible tool node with AgentSentinel policy enforcement.

    Replaces or wraps a standard LangGraph ToolNode. When the graph routes
    to this node, it processes tool_calls from the last AIMessage, enforces
    policies (evidence requirements, argument constraints, groundedness),
    and returns ToolMessages.

    Blocked tools return remediation payloads as the tool message content,
    enabling the LLM to self-repair by calling prerequisite tools first.

    Args:
        tools: Dict of tool_name -> (callable, config_dict) where config_dict
               supports produces_evidence, is_commit, requires, grounding_rules, etc.
    """

    def __init__(self, tools: Dict[str, ToolSpec]):
        self._executor = SentinelToolExecutor(tools)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Process tool calls from the last AIMessage in state.

        Compatible with LangGraph's node interface: takes state dict,
        returns dict with 'messages' key containing ToolMessages.

        Args:
            state: LangGraph state dict. Must have 'messages' key with
                   the last message being an AIMessage with tool_calls.

        Returns:
            Dict with 'messages' key containing list of ToolMessage results.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_message = messages[-1]

        # Extract tool_calls from the last message
        tool_calls = self._extract_tool_calls(last_message)
        if not tool_calls:
            return {"messages": []}

        # Execute each tool call through the sentinel executor
        results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            result = self._executor.execute_tool(tool_name, tool_args, tool_call_id=tool_call_id)
            results.append(self._to_tool_message(result))

        return {"messages": results}

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Async variant of __call__."""
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_message = messages[-1]
        tool_calls = self._extract_tool_calls(last_message)
        if not tool_calls:
            return {"messages": []}

        results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            result = await self._executor.async_execute_tool(
                tool_name, tool_args, tool_call_id=tool_call_id
            )
            results.append(self._to_tool_message(result))

        return {"messages": results}

    @staticmethod
    def _extract_tool_calls(message: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from an AIMessage or dict."""
        # LangChain AIMessage
        if hasattr(message, "tool_calls") and message.tool_calls:
            return [
                {
                    "name": tc.get("name", tc.get("function", {}).get("name", "")),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in message.tool_calls
            ]

        # Dict-based message (e.g., from state)
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])
            return [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in tool_calls
            ]

        return []

    @staticmethod
    def _to_tool_message(result: ToolResult) -> Any:
        """Convert ToolResult to a ToolMessage (or dict if langchain not available)."""
        if result.blocked:
            content = json.dumps({
                "error": result.error,
                "blocked": True,
                "remediation": result.remediation,
            })
        else:
            content = json.dumps(result.output) if not isinstance(result.output, str) else result.output

        if _LANGCHAIN_AVAILABLE:
            return ToolMessage(
                content=content,
                tool_call_id=result.tool_call_id,
                name=result.tool_name,
            )
        else:
            return {
                "role": "tool",
                "content": content,
                "tool_call_id": result.tool_call_id,
                "name": result.tool_name,
            }

    @property
    def executor(self) -> SentinelToolExecutor:
        """Access the underlying executor."""
        return self._executor
