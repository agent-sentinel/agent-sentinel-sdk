"""Compliance request/response timestamps are timezone-aware UTC.

They used ``datetime.utcnow`` (naive, deprecated since Python 3.12); the
serialized form now carries an explicit ``+00:00`` offset.
"""

from datetime import UTC, datetime, timedelta

from agent_sentinel.compliance import ApprovalRequest, ApprovalResponse, ApprovalStatus


def test_approval_request_created_at_is_aware_utc():
    request = ApprovalRequest(
        action_name="a", action_description="d", inputs={}, estimated_cost=0.0
    )
    assert request.created_at.tzinfo is UTC
    assert datetime.now(UTC) - request.created_at < timedelta(seconds=5)
    assert request.to_dict()["created_at"].endswith("+00:00")


def test_approval_response_approved_at_is_aware_utc():
    response = ApprovalResponse(request_id="r", status=ApprovalStatus.APPROVED)
    assert response.approved_at.tzinfo is UTC
    assert response.to_dict()["approved_at"].endswith("+00:00")
