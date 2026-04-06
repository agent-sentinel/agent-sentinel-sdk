"""Tests for groundedness checks — verifying commit action args match evidence."""
import pytest
import time as time_mod

from agent_sentinel.evidence import EvidenceTracker


@pytest.fixture(autouse=True)
def reset():
    EvidenceTracker.reset_session()
    yield
    EvidenceTracker.reset_session()


class TestCheckGroundedness:
    def test_grounded_single_field(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "amount": 99.99},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "refund_amount": 50.0},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is True
        assert ungrounded == []

    def test_ungrounded_value_mismatch(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "amount": 99.99},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-999"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is False
        assert len(ungrounded) == 1
        assert ungrounded[0]["field"] == "order_id"

    def test_ungrounded_no_evidence(self):
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is False
        assert len(ungrounded) == 1

    def test_grounded_from_kwargs(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"status": "found"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id", "source_from": "kwargs"}},
        )
        assert grounded is True

    def test_multiple_grounding_rules(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "customer_id": "CUST-456"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "customer_id": "CUST-456"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "customer_id": {"source_action": "lookup_order", "source_field": "customer_id"},
            },
        )
        assert grounded is True
        assert ungrounded == []

    def test_partial_grounding(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123", "customer_id": "CUST-456"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "customer_id": "CUST-WRONG"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "customer_id": {"source_action": "lookup_order", "source_field": "customer_id"},
            },
        )
        assert grounded is False
        assert len(ungrounded) == 1
        assert ungrounded[0]["field"] == "customer_id"

    def test_grounded_across_actions(self):
        EvidenceTracker.record_evidence("lookup_order", result={"order_id": "ORD-123"})
        EvidenceTracker.record_evidence("verify_identity", result={"user_id": "USER-789"})
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "user_id": "USER-789"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "user_id": {"source_action": "verify_identity", "source_field": "user_id"},
            },
        )
        assert grounded is True

    def test_empty_rules_always_grounded(self):
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"anything": "value"},
            grounding_rules={},
        )
        assert grounded is True
        assert ungrounded == []

    def test_grounded_with_max_age(self):
        EvidenceTracker.record_evidence("lookup_order", result={"order_id": "ORD-123"})
        time_mod.sleep(0.05)
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
            max_age_seconds=0,
        )
        assert grounded is False

    def test_nested_result_field(self):
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order": {"id": "ORD-123", "status": "active"}},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order.id"}},
        )
        assert grounded is True
