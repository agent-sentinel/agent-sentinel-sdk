"""Tests for @sentinel_tool decorator."""
import pytest

from agent_sentinel.integrations.tools import sentinel_tool
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError


@pytest.fixture(autouse=True)
def reset():
    EvidenceTracker.reset_session()
    yield
    EvidenceTracker.reset_session()


class TestSentinelTool:
    def test_basic_execution(self):
        @sentinel_tool(name="lookup_order")
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        result = lookup_order(order_id="123")
        assert result["order_id"] == "123"

    def test_produces_evidence(self):
        @sentinel_tool(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        lookup_order(order_id="123")
        assert EvidenceTracker.has_evidence("lookup_order")

    def test_requires_blocks(self):
        @sentinel_tool(name="issue_refund", requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        with pytest.raises(EvidenceViolationError):
            issue_refund(amount=50.0)

    def test_evidence_chain(self):
        @sentinel_tool(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @sentinel_tool(name="issue_refund", is_commit=True, requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        lookup_order(order_id="123")
        result = issue_refund(amount=50.0)
        assert result["refunded"] == 50.0

    def test_config_attribute(self):
        @sentinel_tool(name="my_tool", produces_evidence=True, is_commit=False)
        def my_tool():
            pass

        assert hasattr(my_tool, "_sentinel_tool_config")
        assert my_tool._sentinel_tool_config["produces_evidence"] is True
        assert my_tool._sentinel_tool_config["name"] == "my_tool"

    def test_original_attribute(self):
        def original_fn():
            return "original"

        wrapped = sentinel_tool(name="test")(original_fn)
        assert hasattr(wrapped, "_sentinel_tool_original")
        assert wrapped._sentinel_tool_original is original_fn

    def test_guarded_attribute(self):
        @sentinel_tool(name="test_tool")
        def test_tool():
            pass

        assert hasattr(test_tool, "_sentinel_guarded")
        assert test_tool._sentinel_guarded is True
