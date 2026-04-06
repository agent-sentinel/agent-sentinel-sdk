"""Tests for auto_register_tools and discover_tools."""
import sys
import types
from unittest.mock import patch, MagicMock, ANY

import pytest

from agent_sentinel.integrations.registry import discover_tools, auto_register_tools
from agent_sentinel.integrations.tools import sentinel_tool
from agent_sentinel.guard import guarded_action


class TestDiscoverTools:
    def test_discover_sentinel_tools(self):
        # Create a test module with sentinel tools
        mod = types.ModuleType("test_mod")

        @sentinel_tool(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @sentinel_tool(name="issue_refund", is_commit=True, requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        mod.lookup_order = lookup_order
        mod.issue_refund = issue_refund

        tools = discover_tools(modules=[mod])
        assert len(tools) == 2

        names = {t["name"] for t in tools}
        assert names == {"lookup_order", "issue_refund"}

        lookup = next(t for t in tools if t["name"] == "lookup_order")
        assert lookup["produces_evidence"] is True

        refund = next(t for t in tools if t["name"] == "issue_refund")
        assert refund["is_commit"] is True
        assert refund["requires"] == ["lookup_order"]

    def test_discover_guarded_actions(self):
        mod = types.ModuleType("test_mod2")

        @guarded_action(name="my_action", produces_evidence=True)
        def my_action():
            pass

        mod.my_action = my_action

        tools = discover_tools(modules=[mod])
        assert len(tools) == 1
        assert tools[0]["name"] == "my_action"
        assert tools[0]["produces_evidence"] is True

    def test_discover_empty_module(self):
        mod = types.ModuleType("empty_mod")
        tools = discover_tools(modules=[mod])
        assert tools == []

    def test_no_duplicates(self):
        mod = types.ModuleType("dup_mod")

        @sentinel_tool(name="same_tool")
        def tool_a():
            pass

        @sentinel_tool(name="same_tool")
        def tool_b():
            pass

        mod.tool_a = tool_a
        mod.tool_b = tool_b

        tools = discover_tools(modules=[mod])
        assert len(tools) == 1


class TestAutoRegisterTools:
    def test_register_posts_to_platform(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "registered": 2, "updated": 0, "total": 2}
        mock_response.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_response

        mod = types.ModuleType("reg_mod")

        @sentinel_tool(name="lookup", produces_evidence=True)
        def lookup():
            pass

        mod.lookup = lookup

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = auto_register_tools(
                platform_url="https://api.example.com",
                api_token="test-token",
                agent_id="agent-1",
                modules=[mod],
            )

        assert result["status"] == "success"
        mock_httpx.post.assert_called_once()
        call_kwargs = mock_httpx.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["agent_id"] == "agent-1"
        assert len(payload["actions"]) == 1
        assert payload["actions"][0]["name"] == "lookup"

    def test_register_no_tools(self):
        mod = types.ModuleType("empty")
        result = auto_register_tools(
            platform_url="https://api.example.com",
            api_token="test-token",
            agent_id="agent-1",
            modules=[mod],
        )
        assert result["total"] == 0
