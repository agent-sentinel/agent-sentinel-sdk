"""
Tests for t12: SDK identity headers (agent, run, harness, principal) and one
auth scheme.

Covers:
- ExecutionContext carries harness/principal alongside agent_id/run_id.
- configure() accepts harness/principal (explicit args and env vars) and
  stores them in the execution context.
- build_identity_headers() is the single helper used to build headers for
  every platform call: Authorization: Bearer, User-Agent, and X-Sentinel-*
  identity headers, omitted when unset.
- sync.py, policy.py, kill_switch.py, mcp.py, and approval.py all use the
  helper and no longer send the ApiKey scheme.
- Request bodies keep sending agent_id/run_id for compatibility.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import agent_sentinel
from agent_sentinel import __version__
from agent_sentinel.approval import ApprovalClient, ApprovalRequest, RiskLevel, ApprovalPriority
from agent_sentinel.context import ExecutionContext, build_identity_headers, _current_context
from agent_sentinel.kill_switch import KillSwitchClient
from agent_sentinel.mcp import MCPClient
from agent_sentinel.policy import PolicyEngine, RemotePolicyConfig, PolicyCache
from agent_sentinel.sync import BackgroundSync, SyncConfig


@pytest.fixture(autouse=True)
def _clean_clients():
    """conftest.py resets PolicyEngine/CostTracker; these singletons need the same treatment here."""
    KillSwitchClient.reset()
    ApprovalClient.reset()
    yield
    KillSwitchClient.reset()
    ApprovalClient.reset()


class TestExecutionContextIdentity:
    """harness/principal live in the execution context alongside agent_id/run_id."""

    def test_defaults_to_none(self):
        with ExecutionContext(agent_id="a1", run_id="r1"):
            assert ExecutionContext.get_harness() is None
            assert ExecutionContext.get_principal() is None

    def test_stores_harness_and_principal(self):
        with ExecutionContext(
            agent_id="a1", run_id="r1", harness="langchain", principal="user@example.com"
        ):
            assert ExecutionContext.get_harness() == "langchain"
            assert ExecutionContext.get_principal() == "user@example.com"
            ctx = ExecutionContext.current()
            assert ctx.agent_id == "a1"
            assert ctx.run_id == "r1"
            assert ctx.harness == "langchain"
            assert ctx.principal == "user@example.com"

    def test_nested_context_overrides(self):
        with ExecutionContext(harness="outer", principal="outer-user"):
            with ExecutionContext(harness="inner", principal="inner-user"):
                assert ExecutionContext.get_harness() == "inner"
                assert ExecutionContext.get_principal() == "inner-user"
            assert ExecutionContext.get_harness() == "outer"
            assert ExecutionContext.get_principal() == "outer-user"

    def test_no_context_returns_none(self):
        assert ExecutionContext.get_harness() is None
        assert ExecutionContext.get_principal() is None


class TestConfigureHarnessPrincipal:
    """configure() accepts harness/principal explicitly and via env vars."""

    def _configure(self, **kwargs):
        with patch("agent_sentinel.sync.enable_remote_sync"), patch(
            "httpx.get", side_effect=Exception("network error")
        ):
            agent_sentinel.configure(
                api_key="test-key",
                platform_url="https://api.test.com",
                **kwargs,
            )

    def teardown_method(self, _method):
        # configure() leaves an ExecutionContext open via __enter__() with no
        # matching __exit__(); undo it directly, as test_fail_closed.py does
        # for the same top-level configure() calls.
        _current_context.set(None)

    def test_configure_accepts_harness_and_principal(self):
        self._configure(harness="crewai", principal="svc-account")

        assert ExecutionContext.get_harness() == "crewai"
        assert ExecutionContext.get_principal() == "svc-account"

    def test_configure_reads_env_vars_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_HARNESS", "autogen")
        monkeypatch.setenv("AGENT_SENTINEL_PRINCIPAL", "user-42")

        self._configure()

        assert ExecutionContext.get_harness() == "autogen"
        assert ExecutionContext.get_principal() == "user-42"

    def test_configure_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_HARNESS", "autogen")
        monkeypatch.setenv("AGENT_SENTINEL_PRINCIPAL", "env-user")

        self._configure(harness="explicit-harness", principal="explicit-user")

        assert ExecutionContext.get_harness() == "explicit-harness"
        assert ExecutionContext.get_principal() == "explicit-user"

    def test_configure_without_harness_or_principal_leaves_them_unset(self):
        self._configure()

        assert ExecutionContext.get_harness() is None
        assert ExecutionContext.get_principal() is None


class TestBuildIdentityHeaders:
    """The single header-building helper shared by every platform call."""

    def test_bearer_scheme_and_user_agent(self):
        headers = build_identity_headers("secret-key")

        assert headers["Authorization"] == "Bearer secret-key"
        assert headers["User-Agent"] == f"agentsentinel-sdk/{__version__}"

    def test_no_apikey_scheme(self):
        headers = build_identity_headers("secret-key")

        assert headers["Authorization"].startswith("Bearer ")
        assert "ApiKey" not in headers["Authorization"]

    def test_omits_identity_headers_when_unset(self):
        headers = build_identity_headers("secret-key")

        assert "X-Sentinel-Agent-Id" not in headers
        assert "X-Sentinel-Run-Id" not in headers
        assert "X-Sentinel-Harness" not in headers
        assert "X-Sentinel-Principal" not in headers

    def test_pulls_identity_from_execution_context(self):
        with ExecutionContext(
            agent_id="a1", run_id="r1", harness="langchain", principal="user@example.com"
        ):
            headers = build_identity_headers("secret-key")

        assert headers["X-Sentinel-Agent-Id"] == "a1"
        assert headers["X-Sentinel-Run-Id"] == "r1"
        assert headers["X-Sentinel-Harness"] == "langchain"
        assert headers["X-Sentinel-Principal"] == "user@example.com"

    def test_explicit_agent_and_run_id_override_context(self):
        with ExecutionContext(agent_id="ctx-agent", run_id="ctx-run"):
            headers = build_identity_headers(
                "secret-key", agent_id="explicit-agent", run_id="explicit-run"
            )

        assert headers["X-Sentinel-Agent-Id"] == "explicit-agent"
        assert headers["X-Sentinel-Run-Id"] == "explicit-run"

    def test_harness_and_principal_always_come_from_context(self):
        """No call site threads harness/principal through explicitly -- only context supplies them."""
        with ExecutionContext(harness="crewai", principal="user-1"):
            headers = build_identity_headers(
                "secret-key", agent_id="explicit-agent", run_id="explicit-run"
            )

        assert headers["X-Sentinel-Harness"] == "crewai"
        assert headers["X-Sentinel-Principal"] == "user-1"


class TestSyncHeaders:
    """sync.py uses build_identity_headers() for both ingest and interventions."""

    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_ingest_uses_bearer_and_identity_headers(self, mock_httpx_client):
        mock_response = Mock(status_code=202, text="Accepted")
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            agent_id="agent-1",
            run_id="run-1",
        )
        sync = BackgroundSync(config)

        with ExecutionContext(harness="langchain", principal="user@example.com"):
            result = sync._upload_batch([{"id": "e1"}])

        assert result is True
        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["User-Agent"] == f"agentsentinel-sdk/{__version__}"
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Run-Id"] == "run-1"
        assert headers["X-Sentinel-Harness"] == "langchain"
        assert headers["X-Sentinel-Principal"] == "user@example.com"
        assert headers["Content-Type"] == "application/json"

        # Request body still sends agent_id/run_id for compatibility.
        payload = kwargs["json"]
        assert payload["agent_id"] == "agent-1"
        assert payload["run_id"] == "run-1"

    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_ingest_run_id_override_beats_config_run_id(self, mock_httpx_client):
        mock_response = Mock(status_code=202, text="Accepted")
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            agent_id="agent-1",
            run_id="config-run",
        )
        sync = BackgroundSync(config)

        sync._upload_batch([{"id": "e1"}], run_id="batch-run")

        _, kwargs = mock_client.post.call_args
        assert kwargs["headers"]["X-Sentinel-Run-Id"] == "batch-run"
        assert kwargs["json"]["run_id"] == "batch-run"

    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.post")
    def test_intervention_upload_uses_bearer_and_identity_headers(self, mock_post):
        mock_post.return_value = Mock(status_code=201)

        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            agent_id="agent-1",
            run_id="run-1",
        )
        sync = BackgroundSync(config)

        with ExecutionContext(harness="crewai", principal="svc-account"):
            result = sync._upload_intervention_batch([{"id": "i1"}])

        assert result is True
        _, kwargs = mock_post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["User-Agent"] == f"agentsentinel-sdk/{__version__}"
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Run-Id"] == "run-1"
        assert headers["X-Sentinel-Harness"] == "crewai"
        assert headers["X-Sentinel-Principal"] == "svc-account"
        assert headers["Content-Type"] == "application/json"


class TestPolicyHeaders:
    """policy.py's remote sync uses build_identity_headers()."""

    @patch("httpx.get")
    def test_policy_sync_uses_bearer_and_identity_headers(self, mock_get):
        mock_get.return_value = Mock(status_code=200, json=lambda: {"policies": []})

        PolicyEngine._remote_config = RemotePolicyConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            agent_id="agent-1",
            run_id="run-1",
        )

        PolicyEngine._fetch_policies_from_platform()

        _, kwargs = mock_get.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["User-Agent"] == f"agentsentinel-sdk/{__version__}"
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Run-Id"] == "run-1"


class TestKillSwitchHeaders:
    """kill_switch.py no longer sends the ApiKey scheme."""

    @patch("httpx.Client")
    def test_check_uses_bearer_and_identity_headers(self, mock_httpx_client):
        mock_response = Mock(status_code=200, json=lambda: {"is_killed": False})
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        KillSwitchClient.configure(platform_url="https://api.example.com", api_token="test-token")

        with ExecutionContext(agent_id="agent-1", harness="langchain"):
            KillSwitchClient.is_killed("agent", "agent-1")

        _, kwargs = mock_client.get.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Harness"] == "langchain"


class TestMCPHeaders:
    """mcp.py's client no longer sends the ApiKey scheme."""

    @pytest.mark.asyncio
    async def test_client_headers_use_bearer_and_identity_headers(self):
        with ExecutionContext(agent_id="agent-1", run_id="run-1", harness="langchain"):
            client = MCPClient(platform_url="https://api.example.com", api_token="test-token")
            http_client = await client._get_client()

        try:
            assert http_client.headers["authorization"] == "Bearer test-token"
            assert "ApiKey" not in http_client.headers["authorization"]
            assert http_client.headers["user-agent"] == f"agentsentinel-sdk/{__version__}"
            assert http_client.headers["x-sentinel-agent-id"] == "agent-1"
            assert http_client.headers["x-sentinel-run-id"] == "run-1"
            assert http_client.headers["x-sentinel-harness"] == "langchain"
            assert http_client.headers["content-type"] == "application/json"
        finally:
            await client.close()


class TestApprovalHeaders:
    """approval.py's client no longer sends the ApiKey scheme on any endpoint."""

    @patch("httpx.Client")
    def test_submit_request_uses_bearer_and_request_identity(self, mock_httpx_client):
        mock_response = Mock(status_code=201, json=lambda: {"id": "approval-1"})
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        ApprovalClient.configure(platform_url="https://api.example.com", api_token="test-token")

        request = ApprovalRequest(
            action_name="transfer_funds",
            agent_id="agent-1",
            run_id="run-1",
            risk_level=RiskLevel.HIGH,
            priority=ApprovalPriority.HIGH,
        )
        with ExecutionContext(harness="crewai", principal="svc-account"):
            ApprovalClient.submit_request(request)

        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Run-Id"] == "run-1"
        assert headers["X-Sentinel-Harness"] == "crewai"
        assert headers["X-Sentinel-Principal"] == "svc-account"
        assert headers["Content-Type"] == "application/json"

    @patch("httpx.Client")
    def test_check_approval_required_uses_bearer_and_call_identity(self, mock_httpx_client):
        mock_response = Mock(status_code=200, json=lambda: {"requires_approval": False})
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        ApprovalClient.configure(platform_url="https://api.example.com", api_token="test-token")

        ApprovalClient.check_approval_required(
            action_name="transfer_funds", agent_id="agent-1", run_id="run-1"
        )

        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
        assert headers["X-Sentinel-Agent-Id"] == "agent-1"
        assert headers["X-Sentinel-Run-Id"] == "run-1"

    @patch("httpx.Client")
    def test_poll_status_uses_bearer_scheme(self, mock_httpx_client):
        mock_response = Mock(
            status_code=200,
            json=lambda: {"id": "approval-1", "status": "pending"},
        )
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        ApprovalClient.configure(platform_url="https://api.example.com", api_token="test-token")

        ApprovalClient.poll_status("approval-1")

        _, kwargs = mock_client.get.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]

    @patch("httpx.Client")
    def test_cancel_request_uses_bearer_scheme(self, mock_httpx_client):
        mock_response = Mock(
            status_code=200,
            json=lambda: {"id": "approval-1", "status": "cancelled"},
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        ApprovalClient.configure(platform_url="https://api.example.com", api_token="test-token")

        ApprovalClient.cancel_request("approval-1")

        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]

    @patch("httpx.Client")
    def test_respond_to_info_request_uses_bearer_scheme(self, mock_httpx_client):
        mock_response = Mock(
            status_code=200,
            json=lambda: {"id": "approval-1", "status": "pending"},
        )
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx_client.return_value = mock_client

        ApprovalClient.configure(platform_url="https://api.example.com", api_token="test-token")

        ApprovalClient.respond_to_info_request("approval-1", "answer")

        _, kwargs = mock_client.post.call_args
        headers = kwargs["headers"]

        assert headers["Authorization"] == "Bearer test-token"
        assert "ApiKey" not in headers["Authorization"]
