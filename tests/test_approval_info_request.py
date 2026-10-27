"""Tests for ApprovalClient INFO_REQUESTED handling."""
from unittest.mock import MagicMock, patch

import pytest

from agent_sentinel.approval import ApprovalClient, ApprovalResponse, ApprovalStatus


def _resp(status, info_message=None):
    return ApprovalResponse(
        approval_id="test-id",
        status=ApprovalStatus(status),
        decided_at=None,
        decided_by_email=None,
        decision_notes=None,
        info_request_message=info_message,
    )


class TestInfoRequestHandler:
    def setup_method(self):
        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
            poll_interval=0.01,
        )

    def teardown_method(self):
        ApprovalClient.reset()

    def test_handler_called_and_response_posted(self):
        """Handler is called with question; response is posted to platform."""
        calls = []

        def handler(msg):
            calls.append(msg)
            return "The answer is 42"

        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
            poll_interval=0.01,
            info_request_handler=handler,
        )

        with (
            patch.object(ApprovalClient, "submit_request", return_value="test-id"),
            patch.object(
                ApprovalClient,
                "poll_status",
                side_effect=[_resp("info_requested", "What is the answer?"), _resp("approved")],
            ),
            patch.object(
                ApprovalClient, "respond_to_info_request", return_value=_resp("pending")
            ) as mock_respond,
        ):
            resp = ApprovalClient.request_approval_sync(action_name="act", estimated_cost=0.0)

        assert resp.status == ApprovalStatus.APPROVED
        assert calls == ["What is the answer?"]
        mock_respond.assert_called_once_with("test-id", "The answer is 42")

    def test_no_handler_continues_polling(self):
        """Without a handler, INFO_REQUESTED just continues polling (backward-compat)."""
        with (
            patch.object(ApprovalClient, "submit_request", return_value="test-id"),
            patch.object(
                ApprovalClient,
                "poll_status",
                side_effect=[_resp("info_requested", "question?"), _resp("approved")],
            ),
            patch.object(ApprovalClient, "respond_to_info_request") as mock_respond,
        ):
            resp = ApprovalClient.request_approval_sync(action_name="act", estimated_cost=0.0)

        assert resp.status == ApprovalStatus.APPROVED
        mock_respond.assert_not_called()

    def test_handler_returning_none_skips_respond(self):
        """If handler returns None, respond_to_info_request is NOT called."""
        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
            poll_interval=0.01,
            info_request_handler=lambda msg: None,
        )

        with (
            patch.object(ApprovalClient, "submit_request", return_value="test-id"),
            patch.object(
                ApprovalClient,
                "poll_status",
                side_effect=[_resp("info_requested", "q?"), _resp("approved")],
            ),
            patch.object(ApprovalClient, "respond_to_info_request") as mock_respond,
        ):
            resp = ApprovalClient.request_approval_sync(action_name="act", estimated_cost=0.0)

        assert resp.status == ApprovalStatus.APPROVED
        mock_respond.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_polling_calls_async_handler(self):
        """Async polling also calls handler (sync or async) correctly."""
        calls = []

        async def async_handler(msg):
            calls.append(msg)
            return "async answer"

        ApprovalClient.configure(
            platform_url="http://localhost:19302",
            api_token="test-token",
            poll_interval=0.01,
            info_request_handler=async_handler,
        )

        with (
            patch.object(ApprovalClient, "submit_request", return_value="test-id"),
            patch.object(
                ApprovalClient,
                "poll_status",
                side_effect=[_resp("info_requested", "async q?"), _resp("approved")],
            ),
            patch.object(
                ApprovalClient, "respond_to_info_request", return_value=_resp("pending")
            ) as mock_respond,
        ):
            resp = await ApprovalClient.request_approval_async(action_name="act", estimated_cost=0.0)

        assert resp.status == ApprovalStatus.APPROVED
        assert calls == ["async q?"]
        mock_respond.assert_called_once_with("test-id", "async answer")
