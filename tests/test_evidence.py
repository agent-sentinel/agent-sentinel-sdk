"""Tests for the EvidenceTracker module and evidence error classes."""
import time
import threading

import pytest

from agent_sentinel.evidence import EvidenceTracker, EvidenceRecord
from agent_sentinel.errors import (
    EvidenceViolationError,
    RemediationPayload,
    PolicyViolationError,
)


@pytest.fixture(autouse=True)
def reset_evidence():
    """Reset evidence tracker before and after each test."""
    EvidenceTracker.reset_session()
    yield
    EvidenceTracker.reset_session()


# =========================================================================
# EvidenceRecord Tests
# =========================================================================

class TestEvidenceRecord:
    def test_create_record(self):
        record = EvidenceRecord(action_name="lookup_order")
        assert record.action_name == "lookup_order"
        assert record.timestamp > 0
        assert record.args_hash is None
        assert record.result_summary is None

    def test_to_dict(self):
        record = EvidenceRecord(
            action_name="lookup_order",
            result_summary={"status": "found"},
        )
        d = record.to_dict()
        assert d["action_name"] == "lookup_order"
        assert d["result_summary"] == {"status": "found"}
        assert "timestamp" in d


# =========================================================================
# EvidenceTracker Tests
# =========================================================================

class TestEvidenceTracker:
    def test_record_and_has_evidence(self):
        assert not EvidenceTracker.has_evidence("lookup_order")
        EvidenceTracker.record_evidence("lookup_order")
        assert EvidenceTracker.has_evidence("lookup_order")

    def test_has_evidence_wrong_action(self):
        EvidenceTracker.record_evidence("lookup_order")
        assert not EvidenceTracker.has_evidence("verify_identity")

    def test_get_evidence(self):
        EvidenceTracker.record_evidence("lookup_order", kwargs={"id": "123"})
        EvidenceTracker.record_evidence("lookup_order", kwargs={"id": "456"})
        records = EvidenceTracker.get_evidence("lookup_order")
        assert len(records) == 2
        # Most recent first
        assert records[0].timestamp >= records[1].timestamp

    def test_check_requirements_all_met(self):
        EvidenceTracker.record_evidence("lookup_order")
        EvidenceTracker.record_evidence("verify_identity")
        all_met, missing, stale = EvidenceTracker.check_requirements(
            ["lookup_order", "verify_identity"]
        )
        assert all_met
        assert missing == []
        assert stale == []

    def test_check_requirements_missing(self):
        EvidenceTracker.record_evidence("lookup_order")
        all_met, missing, stale = EvidenceTracker.check_requirements(
            ["lookup_order", "verify_identity"]
        )
        assert not all_met
        assert missing == ["verify_identity"]
        assert stale == []

    def test_check_requirements_stale(self):
        EvidenceTracker.record_evidence("lookup_order")
        # Simulate stale evidence by checking with max_age_seconds=0
        time.sleep(0.05)
        all_met, missing, stale = EvidenceTracker.check_requirements(
            ["lookup_order"], max_age_seconds=0
        )
        assert not all_met
        assert missing == []
        assert stale == ["lookup_order"]

    def test_max_age_filtering(self):
        EvidenceTracker.record_evidence("lookup_order")
        assert EvidenceTracker.has_evidence("lookup_order", max_age_seconds=10)
        time.sleep(0.05)
        assert not EvidenceTracker.has_evidence("lookup_order", max_age_seconds=0)

    def test_reset_run(self):
        EvidenceTracker.record_evidence("lookup_order")
        assert EvidenceTracker.get_run_evidence_count() == 1
        assert EvidenceTracker.get_session_evidence_count() == 1
        EvidenceTracker.reset_run()
        assert EvidenceTracker.get_run_evidence_count() == 0
        # Session evidence preserved
        assert EvidenceTracker.get_session_evidence_count() == 1

    def test_reset_session(self):
        EvidenceTracker.record_evidence("lookup_order")
        EvidenceTracker.reset_session()
        assert EvidenceTracker.get_run_evidence_count() == 0
        assert EvidenceTracker.get_session_evidence_count() == 0

    def test_thread_safety(self):
        """Ensure concurrent recording doesn't corrupt state."""
        errors = []

        def record_many(prefix: str, count: int):
            try:
                for i in range(count):
                    EvidenceTracker.record_evidence(f"{prefix}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=("a", 100)),
            threading.Thread(target=record_many, args=("b", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert EvidenceTracker.get_run_evidence_count() == 200


# =========================================================================
# RemediationPayload Tests
# =========================================================================

class TestRemediationPayload:
    def test_create_payload(self):
        payload = RemediationPayload(
            reason_code="MISSING_EVIDENCE",
            missing_requirements=["lookup_order"],
            required_prior_actions=["lookup_order", "verify_identity"],
            retry_guidance="Execute lookup_order first",
        )
        assert payload.reason_code == "MISSING_EVIDENCE"
        assert payload.missing_requirements == ["lookup_order"]

    def test_to_dict(self):
        payload = RemediationPayload(reason_code="STALE_EVIDENCE")
        d = payload.to_dict()
        assert d["reason_code"] == "STALE_EVIDENCE"
        assert d["missing_requirements"] == []

    def test_defaults(self):
        payload = RemediationPayload(reason_code="TEST")
        assert payload.safe_alternatives == []
        assert payload.stale_evidence == []
        assert payload.argument_violations == []


# =========================================================================
# EvidenceViolationError Tests
# =========================================================================

class TestEvidenceViolationError:
    def test_is_policy_violation_subclass(self):
        error = EvidenceViolationError(
            message="test",
            missing_requirements=["lookup_order"],
        )
        assert isinstance(error, PolicyViolationError)

    def test_recoverable(self):
        error = EvidenceViolationError(message="test")
        assert error.recoverable is True

    def test_error_code(self):
        error = EvidenceViolationError(message="test")
        assert error.error_code == "EVIDENCE_VIOLATION"

    def test_missing_evidence_reason_code(self):
        error = EvidenceViolationError(
            message="test",
            missing_requirements=["lookup_order"],
        )
        assert error.remediation.reason_code == "MISSING_EVIDENCE"

    def test_stale_evidence_reason_code(self):
        error = EvidenceViolationError(
            message="test",
            stale_evidence=["lookup_order"],
        )
        assert error.remediation.reason_code == "STALE_EVIDENCE"

    def test_argument_constraint_reason_code(self):
        error = EvidenceViolationError(
            message="test",
            argument_violations=["amount must be <= 1000"],
        )
        assert error.remediation.reason_code == "ARGUMENT_CONSTRAINT_VIOLATED"

    def test_auto_retry_guidance(self):
        error = EvidenceViolationError(
            message="test",
            missing_requirements=["lookup_order"],
        )
        assert "lookup_order" in error.remediation.retry_guidance

    def test_remediation_in_details(self):
        error = EvidenceViolationError(
            message="test",
            missing_requirements=["lookup_order"],
            required_prior_actions=["lookup_order"],
        )
        assert "remediation" in error.details
        assert error.details["remediation"]["reason_code"] == "MISSING_EVIDENCE"

    def test_to_dict(self):
        error = EvidenceViolationError(message="test error")
        d = error.to_dict()
        assert d["error_code"] == "EVIDENCE_VIOLATION"
        assert d["recoverable"] is True
        assert "remediation" in d["details"]
