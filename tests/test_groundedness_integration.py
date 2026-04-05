"""Integration tests for groundedness checks with guarded_action."""
import pytest

from agent_sentinel.guard import guarded_action
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


class TestDecoratorGroundedness:
    def test_grounded_action_succeeds(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id, "amount": 99.99}

        @guarded_action(
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

    def test_ungrounded_action_blocked(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @guarded_action(
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
        with pytest.raises(EvidenceViolationError) as exc_info:
            issue_refund(order_id="ORD-WRONG", amount=50.0)

        assert exc_info.value.remediation.reason_code == "UNGROUNDED_ARGUMENT"

    def test_no_grounding_rules_no_check(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @guarded_action(name="issue_refund", requires=["lookup_order"])
        def issue_refund(order_id: str):
            return {"ok": True}

        lookup_order(order_id="ORD-123")
        result = issue_refund(order_id="ORD-ANYTHING")
        assert result["ok"] is True


class TestPolicyGroundedness:
    def test_policy_level_grounding(self):
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
            grounding_rules={
                "issue_refund": {
                    "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                }
            },
        )
        EvidenceTracker.record_evidence("lookup_order", result={"order_id": "ORD-123"})
        PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "ORD-123"})

    def test_policy_level_grounding_blocked(self):
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
            grounding_rules={
                "issue_refund": {
                    "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                }
            },
        )
        EvidenceTracker.record_evidence("lookup_order", result={"order_id": "ORD-123"})
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "ORD-WRONG"})
        assert exc_info.value.remediation.reason_code == "UNGROUNDED_ARGUMENT"
