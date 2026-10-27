"""
Platform URL defaults (0.5.0).

Every public entry point that talks to the platform resolves its base URL
through ``agent_sentinel._defaults.resolve_platform_url``: explicit argument >
``AGENT_SENTINEL_PLATFORM_URL`` > ``AGENT_SENTINEL_URL`` > the hosted
platform. Callers no longer have to pass ``platform_url`` at all, and
existing positional calls keep working.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_sentinel import _defaults
from agent_sentinel._defaults import (
    DEFAULT_PLATFORM_URL,
    resolve_api_token,
    resolve_platform_url,
)
from agent_sentinel.errors import ConfigurationError

ENV_URL_VARS = ("AGENT_SENTINEL_PLATFORM_URL", "AGENT_SENTINEL_URL")
STAGING = "https://staging.example.test"
SELF_HOSTED = "https://sentinel.internal.example"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_URL_VARS + ("AGENT_SENTINEL_API_KEY",):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TestResolvePlatformUrl:
    def test_default_is_hosted_platform(self):
        assert resolve_platform_url(None) == DEFAULT_PLATFORM_URL
        assert DEFAULT_PLATFORM_URL.startswith("https://")
        assert not DEFAULT_PLATFORM_URL.endswith("/")

    def test_agent_sentinel_url_overrides_default(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        assert resolve_platform_url(None) == SELF_HOSTED

    def test_platform_url_env_beats_agent_sentinel_url(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", STAGING)
        assert resolve_platform_url(None) == STAGING

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", STAGING)
        assert resolve_platform_url("http://localhost:8000") == "http://localhost:8000"

    def test_strips_whitespace_and_trailing_slashes(self, monkeypatch):
        assert resolve_platform_url("  http://localhost:8000//  ") == (
            "http://localhost:8000"
        )
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", f" {STAGING}/ ")
        assert resolve_platform_url(None) == STAGING

    def test_blank_values_fall_through(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", "   ")
        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        assert resolve_platform_url("  ") == SELF_HOSTED
        monkeypatch.setenv("AGENT_SENTINEL_URL", "")
        assert resolve_platform_url("") == DEFAULT_PLATFORM_URL


class TestResolveApiToken:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        assert resolve_api_token("explicit") == "explicit"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", " env-key ")
        assert resolve_api_token(None) == "env-key"

    def test_missing_raises_configuration_error(self):
        with pytest.raises(ConfigurationError) as exc:
            resolve_api_token(None)
        assert "AGENT_SENTINEL_API_KEY" in str(exc.value)


def test_host_spelled_only_in_defaults_module():
    """The hosted platform host lives in exactly one place in the package."""
    import pathlib

    import agent_sentinel

    pkg = pathlib.Path(agent_sentinel.__file__).parent
    host = DEFAULT_PLATFORM_URL.split("://", 1)[1]
    hits = sorted(
        str(p.relative_to(pkg))
        for p in pkg.rglob("*")
        if p.suffix in {".py", ".md"} and host in p.read_text(encoding="utf-8")
    )
    assert hits == ["_defaults.py"]


def test_mcp_proxy_uses_shared_default():
    pytest.importorskip("mcp")
    from agent_sentinel import mcp_proxy

    assert mcp_proxy.DEFAULT_URL == _defaults.DEFAULT_PLATFORM_URL
    config = mcp_proxy.load_config({"AGENT_SENTINEL_API_KEY": "k"})
    assert config.url == f"{DEFAULT_PLATFORM_URL}/mcp"


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


class TestSyncDefaults:
    def test_sync_config_without_url_uses_default(self):
        from agent_sentinel.sync import SyncConfig

        cfg = SyncConfig(api_token="tok")
        assert cfg.platform_url == DEFAULT_PLATFORM_URL
        assert cfg.api_token == "tok"

    def test_sync_config_env_override_and_api_key_env(self, monkeypatch):
        from agent_sentinel.sync import SyncConfig

        monkeypatch.setenv("AGENT_SENTINEL_URL", STAGING)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        cfg = SyncConfig()
        assert cfg.platform_url == STAGING
        assert cfg.api_token == "env-key"

    def test_sync_config_positional_backcompat(self):
        from agent_sentinel.sync import SyncConfig

        cfg = SyncConfig("http://localhost:8000/", "tok", "run-1", "agent-1", 5.0)
        assert cfg.platform_url == "http://localhost:8000"
        assert cfg.api_token == "tok"
        assert cfg.run_id == "run-1"
        assert cfg.agent_id == "agent-1"
        assert cfg.flush_interval == 5.0

    def test_sync_config_missing_token_raises(self):
        from agent_sentinel.sync import SyncConfig

        with pytest.raises(ConfigurationError):
            SyncConfig()

    def test_enable_remote_sync_without_url(self, monkeypatch):
        from agent_sentinel import sync

        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        try:
            s = sync.enable_remote_sync(sync_mode="manual")
            assert s.config.platform_url == DEFAULT_PLATFORM_URL
            assert s.config.api_token == "env-key"
        finally:
            sync._global_sync = None

    def test_enable_remote_sync_positional_backcompat(self):
        from agent_sentinel import sync

        try:
            s = sync.enable_remote_sync(
                "http://localhost:8000", "tok", "run-1", "agent-1", 3.0, "manual"
            )
            assert s.config.platform_url == "http://localhost:8000"
            assert s.config.api_token == "tok"
            assert s.config.run_id == "run-1"
            assert s.config.agent_id == "agent-1"
            assert s.config.flush_interval == 3.0
        finally:
            sync._global_sync = None

    def test_manual_flush_without_url(self, monkeypatch):
        from agent_sentinel import sync

        seen = {}

        class FakeManualSync:
            def __init__(self, config):
                seen["config"] = config

            def flush(self):
                seen["flushed"] = True

        monkeypatch.setattr(sync, "ManualSync", FakeManualSync)
        sync.manual_flush(api_token="tok")
        assert seen["config"].platform_url == DEFAULT_PLATFORM_URL
        assert seen["flushed"] is True

        sync.manual_flush("http://localhost:8000", "tok", "run-9")
        assert seen["config"].platform_url == "http://localhost:8000"
        assert seen["config"].run_id == "run-9"


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class TestPolicyDefaults:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        from agent_sentinel import policy
        from agent_sentinel.policy import PolicyEngine

        monkeypatch.setattr(policy, "HTTPX_AVAILABLE", True)
        monkeypatch.setattr(PolicyEngine, "_sync_policies_once", classmethod(lambda c: None))
        monkeypatch.setattr(PolicyEngine, "_start_background_sync", classmethod(lambda c: None))

    def test_enable_remote_sync_without_url(self):
        from agent_sentinel.policy import PolicyEngine

        PolicyEngine.enable_remote_sync(api_token="tok")
        assert PolicyEngine._remote_config.platform_url == DEFAULT_PLATFORM_URL

    def test_enable_remote_sync_env_override(self, monkeypatch):
        from agent_sentinel.policy import PolicyEngine

        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", STAGING + "/")
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        PolicyEngine.enable_remote_sync()
        assert PolicyEngine._remote_config.platform_url == STAGING
        assert PolicyEngine._remote_config.api_token == "env-key"

    def test_enable_remote_sync_positional_backcompat(self):
        from agent_sentinel.policy import PolicyEngine

        PolicyEngine.enable_remote_sync("http://localhost:8000/", "tok", "agent-1", "run-1")
        cfg = PolicyEngine._remote_config
        assert cfg.platform_url == "http://localhost:8000"
        assert cfg.api_token == "tok"
        assert cfg.agent_id == "agent-1"
        assert cfg.run_id == "run-1"

    def test_remote_policy_config_positional_backcompat(self):
        from agent_sentinel.policy import RemotePolicyConfig

        cfg = RemotePolicyConfig("http://localhost:8000", "tok", "agent-1")
        assert (cfg.platform_url, cfg.api_token, cfg.agent_id) == (
            "http://localhost:8000",
            "tok",
            "agent-1",
        )
        assert RemotePolicyConfig(api_token="t").platform_url == DEFAULT_PLATFORM_URL


# ---------------------------------------------------------------------------
# Kill switch / approvals
# ---------------------------------------------------------------------------


class TestKillSwitchDefaults:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from agent_sentinel.kill_switch import KillSwitchClient

        KillSwitchClient.reset()
        yield
        KillSwitchClient.reset()

    def test_configure_without_url(self):
        from agent_sentinel.kill_switch import KillSwitchClient

        KillSwitchClient.configure(api_token="tok")
        assert KillSwitchClient._config.platform_url == DEFAULT_PLATFORM_URL

    def test_configure_env_override(self, monkeypatch):
        from agent_sentinel.kill_switch import KillSwitchClient

        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        KillSwitchClient.configure()
        assert KillSwitchClient._config.platform_url == SELF_HOSTED
        assert KillSwitchClient._config.api_token == "env-key"

    def test_configure_positional_backcompat(self):
        from agent_sentinel.kill_switch import KillSwitchClient, KillSwitchConfig

        KillSwitchClient.configure("http://localhost:8000/", "tok", 1.0, True)
        cfg = KillSwitchClient._config
        assert (cfg.platform_url, cfg.api_token, cfg.cache_ttl) == (
            "http://localhost:8000",
            "tok",
            1.0,
        )
        raw = KillSwitchConfig("http://x.test", "tok", 2.0)
        assert (raw.platform_url, raw.api_token, raw.cache_ttl) == (
            "http://x.test",
            "tok",
            2.0,
        )

    def test_configure_missing_token_raises(self):
        from agent_sentinel.kill_switch import KillSwitchClient

        with pytest.raises(ConfigurationError):
            KillSwitchClient.configure()


class TestApprovalDefaults:
    @pytest.fixture(autouse=True)
    def _reset(self):
        from agent_sentinel.approval import ApprovalClient

        ApprovalClient.reset()
        yield
        ApprovalClient.reset()

    def test_configure_without_url(self):
        from agent_sentinel.approval import ApprovalClient

        ApprovalClient.configure(api_token="tok")
        assert ApprovalClient._config.platform_url == DEFAULT_PLATFORM_URL

    def test_configure_env_override(self, monkeypatch):
        from agent_sentinel.approval import ApprovalClient

        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", STAGING)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        ApprovalClient.configure()
        assert ApprovalClient._config.platform_url == STAGING
        assert ApprovalClient._config.api_token == "env-key"

    def test_configure_positional_backcompat(self):
        from agent_sentinel.approval import ApprovalClient, ApprovalConfig

        ApprovalClient.configure("http://localhost:8000/", "tok", 1.5, 60)
        cfg = ApprovalClient._config
        assert (cfg.platform_url, cfg.api_token, cfg.poll_interval, cfg.default_timeout) == (
            "http://localhost:8000",
            "tok",
            1.5,
            60,
        )
        raw = ApprovalConfig("http://x.test", "tok", 2.0)
        assert (raw.platform_url, raw.api_token, raw.poll_interval) == (
            "http://x.test",
            "tok",
            2.0,
        )


# ---------------------------------------------------------------------------
# Cost backend
# ---------------------------------------------------------------------------


class TestRemoteCostBackendDefaults:
    def test_init_without_url(self):
        from agent_sentinel.cost_backend import RemoteCostBackend

        backend = RemoteCostBackend(api_key="k")
        assert backend._platform_url == DEFAULT_PLATFORM_URL
        assert backend._api_key == "k"

    def test_init_env(self, monkeypatch):
        from agent_sentinel.cost_backend import RemoteCostBackend

        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        backend = RemoteCostBackend()
        assert backend._platform_url == SELF_HOSTED
        assert backend._api_key == "env-key"

    def test_init_positional_backcompat(self):
        from agent_sentinel.cost_backend import RemoteCostBackend

        backend = RemoteCostBackend("http://localhost:8000/", "k", 2.0)
        assert backend._platform_url == "http://localhost:8000"
        assert backend._api_key == "k"
        assert backend._timeout == 2.0

    def test_backend_from_env_remote_needs_only_api_key(self, monkeypatch):
        from agent_sentinel.cost_backend import RemoteCostBackend, backend_from_env

        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "k")
        backend = backend_from_env()
        assert isinstance(backend, RemoteCostBackend)
        assert backend._platform_url == DEFAULT_PLATFORM_URL

    def test_backend_from_env_remote_honours_url_override(self, monkeypatch):
        from agent_sentinel.cost_backend import backend_from_env

        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "k")
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", STAGING)
        assert backend_from_env()._platform_url == STAGING

    def test_backend_from_env_remote_still_requires_api_key(self, monkeypatch):
        from agent_sentinel.cost_backend import backend_from_env

        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        with pytest.raises(ValueError, match="AGENT_SENTINEL_API_KEY"):
            backend_from_env()


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


class TestMCPClientDefaults:
    @pytest.fixture(autouse=True)
    def _httpx(self, monkeypatch):
        from agent_sentinel import mcp

        monkeypatch.setattr(mcp, "HTTPX_AVAILABLE", True)

    def test_client_without_url(self):
        from agent_sentinel.mcp import MCPClient

        client = MCPClient(api_token="tok")
        assert client.platform_url == DEFAULT_PLATFORM_URL
        assert client.api_base == f"{DEFAULT_PLATFORM_URL}/api/v1/mcp"

    def test_client_env(self, monkeypatch):
        from agent_sentinel.mcp import MCPClient

        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "env-key")
        client = MCPClient()
        assert client.platform_url == SELF_HOSTED
        assert client.api_token == "env-key"

    def test_client_positional_backcompat(self):
        from agent_sentinel.mcp import MCPClient

        client = MCPClient("http://localhost:8000/", "tok", 7.0)
        assert client.platform_url == "http://localhost:8000"
        assert client.api_token == "tok"
        assert client.timeout == 7.0

    def test_set_default_client_without_url(self):
        from agent_sentinel import mcp

        try:
            mcp.set_default_client(api_token="tok")
            assert mcp.get_default_client().platform_url == DEFAULT_PLATFORM_URL
            mcp.set_default_client("http://localhost:8000", "tok", 4.0)
            assert mcp.get_default_client().platform_url == "http://localhost:8000"
            assert mcp.get_default_client().timeout == 4.0
        finally:
            mcp._default_client = None


# ---------------------------------------------------------------------------
# Tool auto-registration
# ---------------------------------------------------------------------------


class TestAutoRegisterDefaults:
    def _patched(self):
        response = MagicMock()
        response.json.return_value = {"registered": 1, "updated": 0}
        return (
            patch(
                "agent_sentinel.integrations.registry.discover_tools",
                return_value=[{"name": "t"}],
            ),
            patch("httpx.post", return_value=response),
        )

    def test_without_url(self):
        from agent_sentinel.integrations.registry import auto_register_tools

        p_discover, p_post = self._patched()
        with p_discover, p_post as post:
            auto_register_tools(api_token="tok", agent_id="agent-1")
        assert post.call_args.args[0] == f"{DEFAULT_PLATFORM_URL}/api/v1/actions/register"
        assert post.call_args.kwargs["headers"] == {"Authorization": "ApiKey tok"}

    def test_positional_backcompat(self):
        from agent_sentinel.integrations.registry import auto_register_tools

        p_discover, p_post = self._patched()
        with p_discover, p_post as post:
            auto_register_tools("http://localhost:8000/", "tok", "agent-1")
        assert post.call_args.args[0] == "http://localhost:8000/api/v1/actions/register"
        assert post.call_args.kwargs["json"]["agent_id"] == "agent-1"

    def test_agent_id_still_required(self):
        from agent_sentinel.integrations.registry import auto_register_tools

        with pytest.raises(ConfigurationError, match="agent_id"):
            auto_register_tools(api_token="tok")


# ---------------------------------------------------------------------------
# Top-level configure()
# ---------------------------------------------------------------------------


class TestConfigureDefaults:
    @pytest.fixture
    def captured(self, monkeypatch):
        from agent_sentinel import sync
        from agent_sentinel.context import ExecutionContext
        from agent_sentinel.policy import PolicyEngine

        seen = {}
        monkeypatch.setattr(
            sync, "enable_remote_sync", lambda **kw: seen.setdefault("sync", kw)
        )
        monkeypatch.setattr(
            PolicyEngine,
            "enable_remote_sync",
            classmethod(lambda cls, **kw: seen.setdefault("policy", kw)),
        )
        monkeypatch.setattr(ExecutionContext, "__enter__", lambda self: self)
        return seen

    def test_configure_without_url(self, captured):
        import agent_sentinel

        agent_sentinel.configure(api_key="tok")
        assert captured["sync"]["platform_url"] == DEFAULT_PLATFORM_URL
        assert captured["policy"]["platform_url"] == DEFAULT_PLATFORM_URL

    def test_configure_env_override(self, captured, monkeypatch):
        import agent_sentinel

        monkeypatch.setenv("AGENT_SENTINEL_URL", SELF_HOSTED)
        agent_sentinel.configure("tok")
        assert captured["sync"]["platform_url"] == SELF_HOSTED

    def test_configure_positional_backcompat(self, captured):
        import agent_sentinel

        agent_sentinel.configure("tok", "http://localhost:8000/", "agent-1")
        assert captured["sync"]["platform_url"] == "http://localhost:8000"
        assert captured["sync"]["api_token"] == "tok"
        assert captured["sync"]["agent_id"] == "agent-1"
