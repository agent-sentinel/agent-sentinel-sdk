"""
Tests for ``agentsentinel-runtime run`` (agent_sentinel.runtime.cli).

The runtime serves the ``agentsentinel-mcp`` proxy with a config file. These
tests drive a real MCP client session through ``cli.serve`` (and so through
``mcp_proxy.serve``) over in-memory streams, with only the HTTP transport
replaced by an ``httpx.MockTransport`` fake of the platform's ``/mcp``, the
same pattern as test_mcp_proxy.py. They skip without the ``mcp`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
from functools import partial
from pathlib import Path

import pytest
import yaml

if importlib.util.find_spec("mcp") is not None:
    import anyio
    import httpx
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from agent_sentinel.runtime import cli
    from agent_sentinel.runtime import config as runtime_config


@pytest.fixture
def _require_mcp():
    pytest.importorskip("mcp")


pytestmark = pytest.mark.usefixtures("_require_mcp")

SENTINEL = "as_sentinel_KEYLEAK_7d41b84a6c"
ENV_SENTINEL = "as_envkey_ENVLEAK_0c9b2e51aa"
HARNESS = "cursor-bg"

SEND_TOOL = {
    "name": "send",
    "description": "send description",
    "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
}


class FakeUpstream:
    """A minimal stand-in for the platform's stateless JSON ``/mcp``."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict]] = []

    def methods(self) -> list[str]:
        return [body.get("method") for _, _, body in self.requests]

    def mock_transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            self.requests.append((str(request.url), dict(request.headers), body))
            if "id" not in body:
                return httpx.Response(202)
            method = body["method"]
            if method == "initialize":
                result = {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "agent-sentinel", "version": "test"},
                }
            elif method == "tools/list":
                result = {"tools": [SEND_TOOL]}
            elif method == "tools/call":
                result = {
                    "content": [{"type": "text", "text": "delivered"}],
                    "isError": False,
                }
            else:
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {"code": -32601, "message": "Method not found"},
                    },
                )
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}
            )

        return httpx.MockTransport(handler)


def _write_config(tmp_path, **overrides) -> str:
    workspace = tmp_path / "work"
    workspace.mkdir(exist_ok=True)
    key_dir = tmp_path / "secrets"
    key_dir.mkdir(exist_ok=True)
    key_file = key_dir / "key"
    key_file.write_text(f"  {SENTINEL}\n")
    os.chmod(key_file, 0o600)
    doc = {
        "url": "https://api.example.test",
        "workspace": str(workspace),
        "api_key_file": str(key_file),
        "harness": HARNESS,
    }
    doc.update(overrides)
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(doc))
    os.chmod(path, 0o600)
    return str(path)


async def _round_trip(cfg, transport):
    """initialize -> tools/list -> tools/call through the runtime's serve."""
    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                partial(cli.serve, cfg, *server_streams, http_transport=transport)
            )
            async with ClientSession(*client_streams) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                called = await session.call_tool("send", {"to": "a@example.com"})
            tg.cancel_scope.cancel()
    return init, tools, called


@pytest.fixture
def no_listening(monkeypatch):
    """Make any socket bind or listen fail loudly, and record the attempt."""
    attempts: list[str] = []

    def refuse(name):
        def method(self, *args, **kwargs):
            attempts.append(name)
            raise OSError(f"socket.{name} is forbidden in the runtime")

        return method

    monkeypatch.setattr(socket.socket, "bind", refuse("bind"))
    monkeypatch.setattr(socket.socket, "listen", refuse("listen"))
    return attempts


async def test_round_trip_is_outbound_only(tmp_path, no_listening):
    cfg = runtime_config.load_runtime_config(_write_config(tmp_path))
    fake = FakeUpstream()

    init, tools, called = await _round_trip(cfg, fake.mock_transport())

    assert init.serverInfo.name == "agentsentinel-mcp"
    assert [t.name for t in tools.tools] == ["send"]
    # call_tool reports local failures as isError, so check the real answer.
    assert called.isError is False
    assert called.content[0].text == "delivered"
    assert fake.methods() == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    ]
    assert no_listening == []

    # The patch is live: a bind attempt in this process would have failed.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="forbidden"):
            probe.bind(("127.0.0.1", 0))
    finally:
        probe.close()


async def test_upstream_gets_the_file_key_and_harness_header(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SENTINEL_API_KEY", ENV_SENTINEL)
    cfg = runtime_config.load_runtime_config(_write_config(tmp_path))
    fake = FakeUpstream()

    _, _, called = await _round_trip(cfg, fake.mock_transport())

    assert called.isError is False
    assert fake.requests
    for url, headers, _ in fake.requests:
        assert url == "https://api.example.test/mcp"
        assert headers["authorization"] == f"ApiKey {SENTINEL}"
        assert headers["x-sentinel-harness"] == HARNESS
    # Nothing the model sees carries the key.
    assert SENTINEL not in called.model_dump_json()


def test_run_uses_the_key_file_and_warns_about_the_env_key(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("AGENT_SENTINEL_API_KEY", ENV_SENTINEL)
    captured = []

    async def fake_run_stdio(cfg):
        captured.append(cfg)

    monkeypatch.setattr(cli, "_run_stdio", fake_run_stdio)

    assert cli.main(["run", "--config", _write_config(tmp_path)]) == 0

    assert len(captured) == 1
    assert captured[0].api_key == SENTINEL
    assert captured[0].proxy_config().api_key == SENTINEL
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert "AGENT_SENTINEL_API_KEY" in err
    assert "api_key_file" in err
    for secret in (SENTINEL, ENV_SENTINEL, "KEYLEAK", "ENVLEAK"):
        assert secret not in err


def test_run_without_env_key_prints_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_SENTINEL_API_KEY", raising=False)

    async def fake_run_stdio(cfg):
        return None

    monkeypatch.setattr(cli, "_run_stdio", fake_run_stdio)

    assert cli.main(["run", "--config", _write_config(tmp_path)]) == 0
    assert capsys.readouterr() == ("", "")


def test_run_with_a_bad_config_exits_2_without_the_key(tmp_path, monkeypatch, capsys):
    path = _write_config(tmp_path)
    os.chmod(tmp_path / "secrets" / "key", 0o644)
    started = []

    async def fake_run_stdio(cfg):
        started.append(cfg)

    monkeypatch.setattr(cli, "_run_stdio", fake_run_stdio)

    assert cli.main(["run", "--config", path]) == 2

    assert started == []
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith("agentsentinel-runtime: api_key_file must not be readable")
    assert SENTINEL not in err and "KEYLEAK" not in err


def test_run_failure_is_one_line_without_the_key(tmp_path, monkeypatch, capsys):
    async def failing_run_stdio(cfg):
        raise RuntimeError(f"boom {cfg.api_key}")

    monkeypatch.setattr(cli, "_run_stdio", failing_run_stdio)

    assert cli.main(["run", "--config", _write_config(tmp_path)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "agentsentinel-runtime: stopped (RuntimeError)\n"


def test_console_script_is_declared():
    from importlib.metadata import entry_points

    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("agentsentinel-runtime") == "agent_sentinel.runtime.cli:main"
    assert scripts.get("agentsentinel-mcp") == "agent_sentinel.mcp_proxy:main"


# --- outbound-only, statically ---------------------------------------------------
# The round trip above uses in-memory streams, so it cannot open a socket
# either way. This check fails if listening code is ever added.

_LISTENING = re.compile(
    r"\.bind\(|\.listen\(|create_server|start_server|start_unix_server|"
    r"socketserver|http\.server|serve_forever|uvicorn|SO_REUSEADDR"
)


def _listening_hits(text: str) -> list[str]:
    return _LISTENING.findall(text)


def test_runtime_and_proxy_source_have_no_listening_code():
    import agent_sentinel.mcp_proxy as proxy_module
    import agent_sentinel.runtime as runtime_pkg

    files = [Path(proxy_module.__file__)]
    files += sorted(Path(runtime_pkg.__file__).parent.glob("*.py"))
    assert len(files) >= 4
    for path in files:
        assert _listening_hits(path.read_text()) == [], path.name


def test_listening_check_catches_a_listener():
    assert _listening_hits("sock.bind(('', 0)); sock.listen(1)")
    assert _listening_hits("await asyncio.start_server(handler)")


# --- agentsentinel-mcp messages are unchanged by the shared validators ----------

_GOOD = {"AGENT_SENTINEL_API_KEY": "as_test_key"}


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "AGENT_SENTINEL_API_KEY is not set; use an agent-bound API key"),
        (
            {"AGENT_SENTINEL_API_KEY": "bad key"},
            "AGENT_SENTINEL_API_KEY contains characters that cannot be sent in an HTTP header",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_URL": "http://[::1"},
            "AGENT_SENTINEL_URL is not a valid URL",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_URL": "nope"},
            "AGENT_SENTINEL_URL must be an absolute http(s) URL",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_URL": "https://u:p@h.example"},
            "AGENT_SENTINEL_URL must not contain credentials",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_URL": "https://h.example/?q=1"},
            "AGENT_SENTINEL_URL must not contain a query or fragment",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_URL": "http://h.example"},
            "AGENT_SENTINEL_URL must use https (http is allowed only for localhost and 127.0.0.1)",
        ),
        (
            {**_GOOD, "AGENT_SENTINEL_HARNESS": "bad\x01"},
            "AGENT_SENTINEL_HARNESS contains characters that cannot be sent in an HTTP header",
        ),
        # Order: key before URL, URL before harness.
        (
            {"AGENT_SENTINEL_URL": "nope"},
            "AGENT_SENTINEL_API_KEY is not set; use an agent-bound API key",
        ),
        (
            {
                **_GOOD,
                "AGENT_SENTINEL_URL": "nope",
                "AGENT_SENTINEL_HARNESS": "bad\x01",
            },
            "AGENT_SENTINEL_URL must be an absolute http(s) URL",
        ),
    ],
)
def test_mcp_proxy_messages_are_unchanged(environ, message):
    from agent_sentinel import mcp_proxy

    with pytest.raises(mcp_proxy.ConfigError, match=f"^{re.escape(message)}$"):
        mcp_proxy.load_config(environ)
