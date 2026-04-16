"""Tests for SentinelToolExecutor, OpenAISentinelTools, AnthropicSentinelTools."""
import json
import pytest
from types import SimpleNamespace

from agent_sentinel.integrations.tool_executor import SentinelToolExecutor, ToolResult
from agent_sentinel.integrations.openai import OpenAISentinelTools
from agent_sentinel.integrations.anthropic_tools import AnthropicSentinelTools
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.policy import PolicyEngine


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


def _make_openai_tool_call(tc_id: str, name: str, arguments: dict):
    """Create a mock OpenAI tool_call object."""
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_anthropic_block(block_id: str, name: str, input_data: dict):
    """Create a mock Anthropic tool_use block."""
    return SimpleNamespace(id=block_id, type="tool_use", name=name, input=input_data)


# =========================================================================
# Base SentinelToolExecutor Tests
# =========================================================================

class TestSentinelToolExecutor:
    def test_register_and_execute(self):
        def greet(name: str):
            return f"Hello {name}"

        executor = SentinelToolExecutor({"greet": (greet, {})})
        result = executor.execute_tool("greet", {"name": "World"}, tool_call_id="tc_1")
        assert result.output == "Hello World"
        assert result.blocked is False

    def test_unknown_tool(self):
        executor = SentinelToolExecutor()
        result = executor.execute_tool("nonexistent", {}, tool_call_id="tc_1")
        assert result.blocked is True
        assert "Unknown tool" in result.error

    def test_policy_block_returns_result(self):
        PolicyEngine.configure(denied_actions=["dangerous_tool"])

        def dangerous_tool():
            return "bad"

        executor = SentinelToolExecutor({"dangerous_tool": (dangerous_tool, {})})
        result = executor.execute_tool("dangerous_tool", {}, tool_call_id="tc_1")
        assert result.blocked is True
        assert result.error is not None

    def test_evidence_enforcement(self):
        def lookup():
            return {"id": "123"}

        def commit(id: str):
            return {"done": True}

        executor = SentinelToolExecutor({
            "lookup": (lookup, {"produces_evidence": True}),
            "commit": (commit, {"requires": ["lookup"]}),
        })

        # Should block — no evidence yet
        result = executor.execute_tool("commit", {"id": "123"}, tool_call_id="tc_1")
        assert result.blocked is True
        assert result.remediation is not None

        # Produce evidence
        executor.execute_tool("lookup", {}, tool_call_id="tc_2")

        # Should succeed now
        result = executor.execute_tool("commit", {"id": "123"}, tool_call_id="tc_3")
        assert result.blocked is False
        assert result.output == {"done": True}

    def test_get_tool_names(self):
        executor = SentinelToolExecutor({
            "a": (lambda: None, {}),
            "b": (lambda: None, {}),
        })
        assert sorted(executor.get_tool_names()) == ["a", "b"]

    def test_get_tool_configs(self):
        executor = SentinelToolExecutor({
            "lookup": (lambda: None, {"produces_evidence": True}),
        })
        configs = executor.get_tool_configs()
        assert configs["lookup"]["produces_evidence"] is True

    def test_tool_result_to_dict(self):
        result = ToolResult(tool_call_id="tc_1", tool_name="test", output={"ok": True})
        d = result.to_dict()
        assert d["tool_call_id"] == "tc_1"
        assert d["output"] == {"ok": True}


# =========================================================================
# OpenAI Executor Tests
# =========================================================================

class TestOpenAISentinelTools:
    def test_execute_tool_call(self):
        def lookup(order_id: str):
            return {"order_id": order_id, "status": "found"}

        tools = OpenAISentinelTools({"lookup": (lookup, {"produces_evidence": True})})
        tc = _make_openai_tool_call("tc_1", "lookup", {"order_id": "ORD-123"})

        result = tools.execute(tc)
        assert result.blocked is False
        assert result.output["status"] == "found"
        assert result.tool_call_id == "tc_1"

    def test_to_tool_message_success(self):
        result = ToolResult(tool_call_id="tc_1", tool_name="test", output={"ok": True})
        msg = OpenAISentinelTools.to_tool_message(result)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert json.loads(msg["content"]) == {"ok": True}

    def test_to_tool_message_blocked(self):
        result = ToolResult(
            tool_call_id="tc_1", tool_name="test",
            blocked=True, error="Policy violation",
            remediation={"reason_code": "MISSING_EVIDENCE"},
        )
        msg = OpenAISentinelTools.to_tool_message(result)
        content = json.loads(msg["content"])
        assert content["blocked"] is True
        assert content["remediation"]["reason_code"] == "MISSING_EVIDENCE"

    def test_evidence_chain(self):
        def lookup(order_id: str):
            return {"order_id": order_id}

        def refund(order_id: str, amount: float):
            return {"refunded": True}

        tools = OpenAISentinelTools({
            "lookup": (lookup, {"produces_evidence": True}),
            "refund": (refund, {"is_commit": True, "requires": ["lookup"]}),
        })

        # Block without evidence
        tc = _make_openai_tool_call("tc_1", "refund", {"order_id": "123", "amount": 50})
        result = tools.execute(tc)
        assert result.blocked is True

        # Produce evidence
        tc = _make_openai_tool_call("tc_2", "lookup", {"order_id": "123"})
        tools.execute(tc)

        # Succeed with evidence
        tc = _make_openai_tool_call("tc_3", "refund", {"order_id": "123", "amount": 50})
        result = tools.execute(tc)
        assert result.blocked is False


# =========================================================================
# Anthropic Executor Tests
# =========================================================================

class TestAnthropicSentinelTools:
    def test_execute_block(self):
        def lookup(order_id: str):
            return {"order_id": order_id}

        tools = AnthropicSentinelTools({"lookup": (lookup, {})})
        block = _make_anthropic_block("tu_1", "lookup", {"order_id": "ORD-123"})

        result = tools.execute(block)
        assert result.blocked is False
        assert result.output["order_id"] == "ORD-123"
        assert result.tool_call_id == "tu_1"

    def test_to_tool_result_block_success(self):
        result = ToolResult(tool_call_id="tu_1", tool_name="test", output={"ok": True})
        block = AnthropicSentinelTools.to_tool_result_block(result)
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tu_1"
        assert "is_error" not in block

    def test_to_tool_result_block_blocked(self):
        result = ToolResult(
            tool_call_id="tu_1", tool_name="test",
            blocked=True, error="Missing evidence",
            remediation={"reason_code": "MISSING_EVIDENCE"},
        )
        block = AnthropicSentinelTools.to_tool_result_block(result)
        assert block["is_error"] is True
        content = json.loads(block["content"])
        assert content["blocked"] is True

    def test_evidence_chain(self):
        def lookup(order_id: str):
            return {"order_id": order_id}

        def refund(order_id: str):
            return {"refunded": True}

        tools = AnthropicSentinelTools({
            "lookup": (lookup, {"produces_evidence": True}),
            "refund": (refund, {"requires": ["lookup"]}),
        })

        # Block
        block = _make_anthropic_block("tu_1", "refund", {"order_id": "123"})
        result = tools.execute(block)
        assert result.blocked is True

        # Evidence
        block = _make_anthropic_block("tu_2", "lookup", {"order_id": "123"})
        tools.execute(block)

        # Succeed
        block = _make_anthropic_block("tu_3", "refund", {"order_id": "123"})
        result = tools.execute(block)
        assert result.blocked is False
