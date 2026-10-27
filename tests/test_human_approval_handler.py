"""Tests for HumanApprovalHandler delegation to ApprovalClient."""
from unittest.mock import patch, AsyncMock
import pytest

from agent_sentinel.compliance import HumanApprovalHandler
from agent_sentinel.approval import ApprovalClient, ApprovalResponse, ApprovalStatus


def _approved_response():
    return ApprovalResponse(
        approval_id="test-id",
        status=ApprovalStatus.APPROVED,
        decided_at=None,
        decided_by_email=None,
        decision_notes=None,
        info_request_message=None,
    )


def _denied_response():
    return ApprovalResponse(
        approval_id="test-id",
        status=ApprovalStatus.REJECTED,
        decided_at=None,
        decided_by_email=None,
        decision_notes="Too risky",
        info_request_message=None,
    )


class TestHumanApprovalHandlerDelegation:
    def setup_method(self):
        ApprovalClient.reset()

    def teardown_method(self):
        ApprovalClient.reset()

    def test_delegates_to_approval_client_when_configured(self):
        """When ApprovalClient is configured, sync approval goes to platform."""
        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
        )
        handler = HumanApprovalHandler(callback=lambda req: True)

        with patch.object(
            ApprovalClient, "request_approval_sync", return_value=_approved_response()
        ) as mock_sync:
            result = handler.request_approval_sync(
                action_name="delete_file",
                estimated_cost=0.05,
                context={"path": "/tmp/test"},
            )

        mock_sync.assert_called_once()
        assert result is True

    def test_denied_response_returns_false(self):
        """When the platform denies, sync returns False."""
        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
        )
        handler = HumanApprovalHandler(callback=lambda req: True)

        with patch.object(
            ApprovalClient, "request_approval_sync", return_value=_denied_response()
        ):
            result = handler.request_approval_sync(
                action_name="drop_table",
                estimated_cost=0.0,
            )

        assert result is False

    def test_falls_back_to_local_callback_when_not_configured(self):
        """When ApprovalClient is not configured, local callback is used."""
        callback_called = []
        def local_callback(req):
            callback_called.append(req)
            return True

        handler = HumanApprovalHandler(callback=local_callback)
        result = handler.request_approval_sync(
            action_name="read_file",
            estimated_cost=0.01,
        )

        assert len(callback_called) == 1
        assert result is True

    @pytest.mark.asyncio
    async def test_async_delegates_to_approval_client_when_configured(self):
        """Async path also delegates when client is configured."""
        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
        )
        handler = HumanApprovalHandler(callback=lambda req: True)

        with patch.object(
            ApprovalClient, "request_approval_async", new_callable=AsyncMock, return_value=_approved_response()
        ) as mock_async:
            result = await handler.request_approval_async(
                action_name="delete_file",
                estimated_cost=0.05,
            )

        mock_async.assert_called_once()
        assert result is True
