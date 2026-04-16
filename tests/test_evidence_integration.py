"""Integration tests for evidence system with guarded_action and PolicyEngine."""
import pytest

from agent_sentinel.guard import guarded_action
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError
from agent_sentinel.intervention import InterventionTracker, InterventionType


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all state before and after each test."""
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    InterventionTracker.clear()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    InterventionTracker.clear()


# =========================================================================
# Decorator-level evidence tests
# =========================================================================

class TestGuardedActionEvidence:
    def test_produces_evidence_records_on_success(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"status": "found", "order_id": order_id}

        result = lookup_order("123")
        assert result["status"] == "found"
        assert EvidenceTracker.has_evidence("lookup_order")

    def test_produces_evidence_not_recorded_on_error(self):
        @guarded_action(name="failing_lookup", produces_evidence=True)
        def failing_lookup():
            raise ValueError("not found")

        with pytest.raises(ValueError):
            failing_lookup()

        assert not EvidenceTracker.has_evidence("failing_lookup")

    def test_requires_blocks_when_missing(self):
        @guarded_action(name="issue_refund", is_commit=True, requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        with pytest.raises(EvidenceViolationError) as exc_info:
            issue_refund(50.0)

        error = exc_info.value
        assert error.recoverable is True
        assert "lookup_order" in error.remediation.missing_requirements

    def test_requires_succeeds_when_evidence_present(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"found": True}

        @guarded_action(name="issue_refund", is_commit=True, requires=["lookup_order"])
        def issue_refund(amount: float):
            return {"refunded": amount}

        lookup_order("123")
        result = issue_refund(50.0)
        assert result["refunded"] == 50.0

    def test_full_chain(self):
        """Test a complete evidence chain: produce -> consume."""
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id, "amount": 100}

        @guarded_action(name="verify_identity", produces_evidence=True)
        def verify_identity(user_id: str):
            return {"verified": True}

        @guarded_action(
            name="issue_refund",
            is_commit=True,
            requires=["lookup_order", "verify_identity"],
        )
        def issue_refund(order_id: str, amount: float):
            return {"refunded": True}

        # Should fail - no evidence yet
        with pytest.raises(EvidenceViolationError) as exc_info:
            issue_refund("123", 50.0)
        assert set(exc_info.value.remediation.missing_requirements) == {
            "lookup_order", "verify_identity"
        }

        # Produce first piece of evidence
        lookup_order("123")

        # Should still fail - missing verify_identity
        with pytest.raises(EvidenceViolationError) as exc_info:
            issue_refund("123", 50.0)
        assert exc_info.value.remediation.missing_requirements == ["verify_identity"]

        # Produce second piece
        verify_identity("user_456")

        # Should succeed now
        result = issue_refund("123", 50.0)
        assert result["refunded"] is True


# =========================================================================
# Policy-level evidence tests
# =========================================================================

class TestPolicyEngineEvidence:
    def test_policy_evidence_requirements_block(self):
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
        )

        with pytest.raises(EvidenceViolationError):
            PolicyEngine.check_action("issue_refund", 0.0)

    def test_policy_evidence_requirements_pass(self):
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
        )

        EvidenceTracker.record_evidence("lookup_order")
        # Should not raise
        PolicyEngine.check_action("issue_refund", 0.0)

    def test_policy_evidence_max_age(self):
        import time

        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
            evidence_max_age_seconds={"issue_refund": 0},  # Immediately stale
        )

        EvidenceTracker.record_evidence("lookup_order")
        time.sleep(0.05)

        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0)
        assert exc_info.value.remediation.stale_evidence == ["lookup_order"]

    def test_unconfigured_actions_pass(self):
        """Actions not in evidence_requirements should pass through."""
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
        )

        # This action has no evidence requirements
        PolicyEngine.check_action("send_email", 0.0)


# =========================================================================
# Intervention recording tests
# =========================================================================

class TestEvidenceInterventions:
    def test_missing_evidence_records_intervention(self):
        @guarded_action(name="issue_refund", requires=["lookup_order"])
        def issue_refund():
            return True

        with pytest.raises(EvidenceViolationError):
            issue_refund()

        interventions = InterventionTracker.get_interventions(limit=1)
        assert len(interventions) == 1
        intervention = interventions[0]
        assert intervention.intervention_type == InterventionType.MISSING_EVIDENCE
        assert intervention.remediation_payload is not None
        assert intervention.remediation_payload["reason_code"] == "MISSING_EVIDENCE"
