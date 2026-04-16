"""Tests for CrewAI integration evidence support."""
import sys
import types
import pytest

# Mock crewai module before importing agent_sentinel.integrations.crewai
# This avoids the pydantic/chromadb crash on Python 3.14
_crewai_mock = types.ModuleType("crewai")
_crewai_mock.Crew = type("Crew", (), {})  # type: ignore
_crewai_mock.Agent = type("Agent", (), {})  # type: ignore
_crewai_mock.Task = type("Task", (), {})  # type: ignore
sys.modules.setdefault("crewai", _crewai_mock)

from agent_sentinel.integrations.crewai import wrap_crew_action
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.errors import EvidenceViolationError


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


class TestWrapCrewActionEvidence:
    def test_produces_evidence(self):
        @wrap_crew_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        lookup_order(order_id="ORD-123")
        assert EvidenceTracker.has_evidence("lookup_order")

    def test_requires_blocks(self):
        @wrap_crew_action(name="issue_refund", requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        with pytest.raises(EvidenceViolationError):
            issue_refund(amount=50.0)

    def test_evidence_chain(self):
        @wrap_crew_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @wrap_crew_action(name="issue_refund", is_commit=True, requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        lookup_order(order_id="ORD-123")
        result = issue_refund(amount=50.0)
        assert result["refunded"] == 50.0

    def test_grounding_rules(self):
        @wrap_crew_action(
            name="lookup_order",
            produces_evidence=True,
        )
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @wrap_crew_action(
            name="issue_refund",
            is_commit=True,
            requires=["lookup_order"],
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
            },
        )
        def issue_refund(order_id: str, amount: float):
            return {"refunded": True}

        lookup_order(order_id="ORD-123")
        result = issue_refund(order_id="ORD-123", amount=50.0)
        assert result["refunded"] is True

        # Wrong order_id should block
        EvidenceTracker.reset_session()
        lookup_order(order_id="ORD-123")
        with pytest.raises(EvidenceViolationError):
            issue_refund(order_id="ORD-WRONG", amount=50.0)
