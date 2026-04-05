"""Tests for argument constraint enforcement in PolicyEngine."""
import pytest

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


class TestPolicyArgumentConstraints:
    def test_valid_args_pass(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number", "maximum": 1000}},
                    "required": ["amount"],
                }
            }
        )
        PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": 500})

    def test_invalid_type_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": "not_a_number"})
        assert exc_info.value.remediation.reason_code == "ARGUMENT_CONSTRAINT_VIOLATED"
        assert len(exc_info.value.remediation.argument_violations) == 1

    def test_exceeds_maximum_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number", "maximum": 1000}},
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": 5000})
        assert "maximum" in exc_info.value.remediation.argument_violations[0].lower() or "1000" in exc_info.value.remediation.argument_violations[0]

    def test_missing_required_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "required": ["order_id", "amount"],
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "123"})
        assert "amount" in exc_info.value.remediation.argument_violations[0]

    def test_no_kwargs_skips_validation(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        PolicyEngine.check_action("issue_refund", 0.0)

    def test_unconstrained_action_passes(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        PolicyEngine.check_action("send_email", 0.0, kwargs={"to": "user@example.com"})

    def test_multiple_violations(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {
                        "amount": {"type": "number", "maximum": 1000},
                        "reason": {"enum": ["defective", "wrong_item"]},
                    },
                    "required": ["order_id"],
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action(
                "issue_refund", 0.0,
                kwargs={"amount": 5000, "reason": "just_because"},
            )
        assert len(exc_info.value.remediation.argument_violations) == 3
