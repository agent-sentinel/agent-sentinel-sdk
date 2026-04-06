"""Tests for LangGraph integration."""
import json
import pytest

from agent_sentinel.integrations.langgraph import SentinelToolNode
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.policy import PolicyEngine


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


def _make_state_with_tool_calls(tool_calls):
    """Create a LangGraph-like state with an AIMessage containing tool_calls."""
    message = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    return {"messages": [message]}


class TestSentinelToolNode:
    def test_basic_execution(self):
        def greet(name: str):
            return {"greeting": f"Hello {name}"}

        node = SentinelToolNode({"greet": (greet, {})})
        state = _make_state_with_tool_calls([
            {"name": "greet", "args": {"name": "World"}, "id": "tc_1"},
        ])

        result = node(state)
        messages = result["messages"]
        assert len(messages) == 1

        # Check content (could be ToolMessage or dict depending on langchain availability)
        msg = messages[0]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        assert json.loads(content) == {"greeting": "Hello World"}

    def test_multiple_tool_calls(self):
        def add(a: int, b: int):
            return {"sum": a + b}

        def multiply(a: int, b: int):
            return {"product": a * b}

        node = SentinelToolNode({
            "add": (add, {}),
            "multiply": (multiply, {}),
        })
        state = _make_state_with_tool_calls([
            {"name": "add", "args": {"a": 2, "b": 3}, "id": "tc_1"},
            {"name": "multiply", "args": {"a": 4, "b": 5}, "id": "tc_2"},
        ])

        result = node(state)
        messages = result["messages"]
        assert len(messages) == 2

    def test_evidence_chain(self):
        def lookup(order_id: str):
            return {"order_id": order_id, "status": "found"}

        def refund(order_id: str, amount: float):
            return {"refunded": True}

        node = SentinelToolNode({
            "lookup": (lookup, {"produces_evidence": True}),
            "refund": (refund, {"is_commit": True, "requires": ["lookup"]}),
        })

        # Refund without evidence should be blocked
        state = _make_state_with_tool_calls([
            {"name": "refund", "args": {"order_id": "123", "amount": 50}, "id": "tc_1"},
        ])
        result = node(state)
        msg = result["messages"][0]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        parsed = json.loads(content)
        assert parsed["blocked"] is True
        assert parsed["remediation"] is not None

        # Lookup first
        state = _make_state_with_tool_calls([
            {"name": "lookup", "args": {"order_id": "123"}, "id": "tc_2"},
        ])
        node(state)

        # Now refund should succeed
        state = _make_state_with_tool_calls([
            {"name": "refund", "args": {"order_id": "123", "amount": 50}, "id": "tc_3"},
        ])
        result = node(state)
        msg = result["messages"][0]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        parsed = json.loads(content)
        assert parsed["refunded"] is True

    def test_policy_block(self):
        PolicyEngine.configure(denied_actions=["dangerous_tool"])

        def dangerous_tool():
            return "bad"

        node = SentinelToolNode({"dangerous_tool": (dangerous_tool, {})})
        state = _make_state_with_tool_calls([
            {"name": "dangerous_tool", "args": {}, "id": "tc_1"},
        ])

        result = node(state)
        msg = result["messages"][0]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        parsed = json.loads(content)
        assert parsed["blocked"] is True

    def test_unknown_tool(self):
        node = SentinelToolNode({"known": (lambda: "ok", {})})
        state = _make_state_with_tool_calls([
            {"name": "unknown", "args": {}, "id": "tc_1"},
        ])

        result = node(state)
        msg = result["messages"][0]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        parsed = json.loads(content)
        assert parsed["blocked"] is True
        assert "Unknown tool" in parsed["error"]

    def test_empty_state(self):
        node = SentinelToolNode({})
        result = node({"messages": []})
        assert result == {"messages": []}

    def test_no_tool_calls_in_message(self):
        node = SentinelToolNode({"a": (lambda: None, {})})
        result = node({"messages": [{"role": "user", "content": "hello"}]})
        assert result == {"messages": []}

    def test_executor_accessible(self):
        node = SentinelToolNode({"a": (lambda: None, {})})
        assert node.executor is not None
        assert "a" in node.executor.get_tool_names()
