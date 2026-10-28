"""
Tests for the ``agentsentinel-mcp`` stdio proxy (agent_sentinel.mcp_proxy).

The proxy needs the optional ``mcp`` extra. Every proxy test calls
``pytest.importorskip("mcp")`` via the ``_require_mcp`` fixture, so a core-only
install skips them; the import-isolation test at the bottom runs everywhere,
because importing ``agent_sentinel`` must never need ``mcp``.

The upstream is an in-process fake of the platform's stateless ``/mcp``:
behind ``httpx.MockTransport`` for the in-process tests, and served by uvicorn
on 127.0.0.1 for the end-to-end stdio test that checks no byte leaks the key.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import partial

import pytest

if importlib.util.find_spec("mcp") is not None:
    import anyio
    import httpx
    from mcp import ClientSession, McpError, types
    from mcp.shared.memory import create_client_server_memory_streams

    from agent_sentinel import mcp_proxy

API_KEY = "as_test_agent_bound_9f3c2e7d41b84a6c"
HARNESS = "claude-code"

SEND_TOOL = {
    "name": "send",
    "description": "send description",
    "inputSchema": {
        "type": "object",
        "properties": {"to": {"type": "string"}},
        "required": ["to"],
    },
}
CAPPED_TOOL = {
    "name": "capped",
    "description": "over its action budget",
    "inputSchema": {"type": "object"},
}


class FakeUpstream:
    """A minimal stand-in for the platform's stateless JSON ``/mcp``."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[tuple[dict[str, str], dict]] = []

    def methods(self) -> list[str]:
        return [body.get("method") for _, body in self.requests]

    def respond(self, headers: dict[str, str], raw: bytes) -> tuple[int, bytes]:
        body = json.loads(raw) if raw else {}
        self.requests.append((headers, body))
        if self.status != 200:
            return self.status, json.dumps(
                {"detail": "denied", "reason_code": "agent_bound_key_required"}
            ).encode()
        if "id" not in body:
            return 202, b""
        method = body["method"]
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "agent-sentinel", "version": "test"},
            }
        elif method == "tools/list":
            result = {"tools": [SEND_TOOL, CAPPED_TOOL]}
        elif method == "tools/call" and body["params"]["name"] == "send":
            result = {
                "content": [{"type": "text", "text": "delivered ✓"}],
                "isError": False,
                "_meta": {"echo": body["params"].get("_meta")},
            }
        elif method == "tools/call":
            result = {
                "content": [{"type": "text", "text": "denied: action budget"}],
                "structuredContent": {
                    "decision": "deny",
                    "reason_code": "action_budget_exceeded",
                },
                "isError": True,
            }
        else:
            return 200, json.dumps(
                {"jsonrpc": "2.0", "id": body["id"],
                 "error": {"code": -32601, "message": "Method not found"}}
            ).encode()
        return 200, json.dumps(
            {"jsonrpc": "2.0", "id": body["id"], "result": result}
        ).encode()

    def mock_transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            status, content = self.respond(dict(request.headers), request.content)
            return httpx.Response(
                status, headers={"content-type": "application/json"}, content=content
            )

        return httpx.MockTransport(handler)

    async def asgi(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        raw = b""
        while True:
            message = await receive()
            raw += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        status, content = self.respond(headers, raw)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": content})


@pytest.fixture
def _require_mcp():
    pytest.importorskip("mcp")


def _config(harness: str | None = HARNESS):
    return mcp_proxy.ProxyConfig(
        url="https://api.example.test/mcp", api_key=API_KEY, harness=harness
    )


@asynccontextmanager
async def _harness(transport, config=None):
    """An MCP client session (the 'harness') connected to the proxy in-process."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                partial(
                    mcp_proxy.serve,
                    config or _config(),
                    *server_streams,
                    http_transport=transport,
                )
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


# --- proxying ---------------------------------------------------------------


@pytest.mark.usefixtures("_require_mcp")
async def test_initialize_is_local_and_upstream_is_lazy_and_reused():
    fake = FakeUpstream()
    async with _harness(fake.mock_transport()) as session:
        assert fake.requests == []  # initialize answered locally

        await session.list_tools()
        await session.list_tools()
        await session.call_tool("send", {"to": "a@example.com"})

    assert fake.methods().count("initialize") == 1
    assert fake.methods().count("tools/list") == 2
    assert fake.methods().count("tools/call") == 1


@pytest.mark.usefixtures("_require_mcp")
async def test_tools_list_passes_through():
    fake = FakeUpstream()
    async with _harness(fake.mock_transport()) as session:
        result = await session.list_tools()

    assert [t.model_dump(exclude_none=True) for t in result.tools] == [
        SEND_TOOL,
        CAPPED_TOOL,
    ]


@pytest.mark.usefixtures("_require_mcp")
async def test_tools_call_passes_through_results_and_meta():
    fake = FakeUpstream()
    approval = {"approval_request_id": "12345678-1234-4234-8234-123456789abc"}
    async with _harness(fake.mock_transport()) as session:
        ok = await session.call_tool("send", {"to": "a@example.com"}, meta=approval)
        refused = await session.call_tool("capped", {})

    assert ok.isError is False
    assert ok.content[0].text == "delivered ✓"
    assert ok.meta == {"echo": approval}

    sent = [b for _, b in fake.requests if b.get("method") == "tools/call"]
    assert sent[0]["params"]["arguments"] == {"to": "a@example.com"}
    assert sent[0]["params"]["_meta"] == approval

    assert refused.isError is True
    assert refused.structuredContent == {
        "decision": "deny",
        "reason_code": "action_budget_exceeded",
    }
    assert refused.content[0].text == "denied: action budget"


@pytest.mark.usefixtures("_require_mcp")
async def test_auth_and_harness_headers_sent_upstream():
    fake = FakeUpstream()
    async with _harness(fake.mock_transport()) as session:
        await session.list_tools()
        await session.call_tool("send", {"to": "a@example.com"})

    assert fake.requests
    for headers, _ in fake.requests:
        assert headers["authorization"] == f"ApiKey {API_KEY}"
        assert headers["x-sentinel-harness"] == HARNESS


@pytest.mark.usefixtures("_require_mcp")
async def test_harness_header_omitted_when_unset():
    fake = FakeUpstream()
    async with _harness(fake.mock_transport(), _config(harness=None)) as session:
        await session.list_tools()

    for headers, _ in fake.requests:
        assert "x-sentinel-harness" not in headers
        assert headers["authorization"] == f"ApiKey {API_KEY}"


@pytest.mark.usefixtures("_require_mcp")
@pytest.mark.parametrize("status", [401, 403])
async def test_upstream_auth_rejection_maps_to_mcp_error_and_tool_error(status):
    fake = FakeUpstream(status=status)
    async with _harness(fake.mock_transport()) as session:
        with pytest.raises(McpError) as listed:
            await session.list_tools()
        called = await session.call_tool("send", {"to": "a@example.com"})

    message = listed.value.error.message
    assert f"HTTP {status}" in message
    assert "agent_bound_key_required" in message
    assert API_KEY not in message

    assert called.isError is True
    text = called.content[0].text
    assert f"HTTP {status}" in text
    assert API_KEY not in text
    assert "authorization" not in text.lower()


@pytest.mark.usefixtures("_require_mcp")
async def test_transport_failure_maps_to_mcp_error_and_tool_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    async with _harness(httpx.MockTransport(refuse)) as session:
        with pytest.raises(McpError) as listed:
            await session.list_tools()
        called = await session.call_tool("send", {"to": "a@example.com"})

    assert "unreachable" in listed.value.error.message
    assert called.isError is True
    assert "unreachable" in called.content[0].text


@pytest.mark.usefixtures("_require_mcp")
async def test_upstream_session_recovers_after_a_failed_initialize():
    fake = FakeUpstream(status=503)
    async with _harness(fake.mock_transport()) as session:
        with pytest.raises(McpError):
            await session.list_tools()
        fake.status = 200
        result = await session.list_tools()

    assert [t.name for t in result.tools] == ["send", "capped"]


# --- configuration ----------------------------------------------------------


@pytest.mark.usefixtures("_require_mcp")
def test_config_defaults_and_local_http():
    config = mcp_proxy.load_config({"AGENT_SENTINEL_API_KEY": API_KEY})
    assert config.url == "https://api.agentsentinel.dev/mcp"
    assert config.harness is None
    assert API_KEY not in repr(config)

    for host in ("localhost", "127.0.0.1"):
        config = mcp_proxy.load_config({
            "AGENT_SENTINEL_API_KEY": API_KEY,
            "AGENT_SENTINEL_URL": f"http://{host}:8000/",
            "AGENT_SENTINEL_HARNESS": "cursor",
        })
        assert config.url == f"http://{host}:8000/mcp"
        assert config.harness == "cursor"


@pytest.mark.usefixtures("_require_mcp")
def test_missing_key_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("AGENT_SENTINEL_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_SENTINEL_URL", raising=False)

    assert mcp_proxy.main() != 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "AGENT_SENTINEL_API_KEY" in captured.err


@pytest.mark.usefixtures("_require_mcp")
@pytest.mark.parametrize(
    "url",
    [
        "http://api.agentsentinel.dev",
        "http://10.0.0.5:8000",
        "http://localhost.evil.example",
        "ftp://localhost",
        f"https://{API_KEY}@api.agentsentinel.dev",
        API_KEY,
    ],
)
def test_disallowed_url_exits_nonzero_without_the_key(monkeypatch, capsys, url):
    monkeypatch.setenv("AGENT_SENTINEL_API_KEY", API_KEY)
    monkeypatch.setenv("AGENT_SENTINEL_URL", url)

    assert mcp_proxy.main() != 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "AGENT_SENTINEL_URL" in captured.err
    assert API_KEY not in captured.err


@pytest.mark.usefixtures("_require_mcp")
def test_missing_key_exits_nonzero_as_a_process():
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_SENTINEL_")}
    # Optional LangChain emits this third-party compatibility warning on 3.14.
    # Keep exact assertions on the proxy's own diagnostic below.
    env["PYTHONWARNINGS"] = "ignore:Core Pydantic V1 functionality:UserWarning:langchain_core.utils.pydantic"
    proc = subprocess.run(
        [sys.executable, "-m", "agent_sentinel.mcp_proxy"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert proc.stdout == b""
    assert proc.stderr.count(b"\n") == 1


# --- end to end: no secret in any stdout/stderr byte or JSON-RPC message -----


@pytest.fixture
def http_upstream(_require_mcp):
    """FakeUpstream served over real HTTP on 127.0.0.1."""
    uvicorn = pytest.importorskip("uvicorn")
    fake = FakeUpstream()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    # interface="asgi3": autodetection misreads a bound method as ASGI2.
    server = uvicorn.Server(
        uvicorn.Config(
            fake.asgi,
            interface="asgi3",
            log_level="critical",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "fake upstream did not start"
        time.sleep(0.02)
    try:
        yield fake, f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def _run_stdio_proxy(url: str, messages: list[dict], expect_ids: set[int]):
    """Drive ``python -m agent_sentinel.mcp_proxy`` over real stdio."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_SENTINEL_")}
    env.update({
        "AGENT_SENTINEL_API_KEY": API_KEY,
        "AGENT_SENTINEL_URL": url,
        "AGENT_SENTINEL_HARNESS": HARNESS,
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_sentinel.mcp_proxy"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    watchdog = threading.Timer(60, proc.kill)
    watchdog.start()
    try:
        for message in messages:
            proc.stdin.write(json.dumps(message).encode() + b"\n")
        proc.stdin.flush()
        stdout = b""
        replies: dict[int, dict] = {}
        while not expect_ids <= replies.keys():
            line = proc.stdout.readline()
            assert line, "proxy closed stdout before answering"
            stdout += line
            reply = json.loads(line)
            if "id" in reply:
                replies[reply["id"]] = reply
        proc.stdin.close()
        stdout += proc.stdout.read()
        stderr = proc.stderr.read()
        returncode = proc.wait(timeout=30)
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
    return returncode, stdout, stderr, replies


def _session_messages() -> list[dict]:
    return [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-harness", "version": "0"},
        }},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "send", "arguments": {"to": "a@example.com"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "capped", "arguments": {}}},
    ]


@pytest.mark.parametrize("status", [200, 403])
def test_stdio_proxy_never_leaks_the_key(http_upstream, status):
    fake, url = http_upstream
    fake.status = status

    returncode, stdout, stderr, replies = _run_stdio_proxy(
        url, _session_messages(), {1, 2, 3, 4}
    )

    assert returncode == 0, stderr
    # The key really was used upstream, so its absence below is not vacuous.
    assert fake.requests, replies
    assert all(h["authorization"] == f"ApiKey {API_KEY}" for h, _ in fake.requests)
    assert all(h["x-sentinel-harness"] == HARNESS for h, _ in fake.requests)

    key = API_KEY.encode()
    assert key not in stdout
    assert key not in stderr
    for reply in replies.values():
        assert API_KEY not in json.dumps(reply)
    for line in stdout.splitlines():
        assert API_KEY not in json.dumps(json.loads(line))

    assert replies[1]["result"]["serverInfo"]["name"] == "agentsentinel-mcp"
    if status == 200:
        assert [t["name"] for t in replies[2]["result"]["tools"]] == ["send", "capped"]
        assert replies[3]["result"]["isError"] is False
        assert replies[4]["result"]["isError"] is True
        assert replies[4]["result"]["structuredContent"]["reason_code"] == (
            "action_budget_exceeded"
        )
    else:
        assert "HTTP 403" in replies[2]["error"]["message"]
        assert replies[3]["result"]["isError"] is True
        assert "HTTP 403" in replies[3]["result"]["content"][0]["text"]


# --- import isolation (runs without mcp) ------------------------------------


def test_import_agent_sentinel_does_not_need_mcp():
    """``import agent_sentinel`` works with ``mcp`` absent and never loads the proxy."""
    code = (
        "import sys\n"
        "sys.modules['mcp'] = None\n"  # makes any `import mcp` raise ImportError
        "import agent_sentinel\n"
        "import agent_sentinel.mcp\n"
        "assert 'agent_sentinel.mcp_proxy' not in sys.modules\n"
        "try:\n"
        "    import agent_sentinel.mcp_proxy\n"
        "except ImportError as exc:\n"
        "    assert 'agentsentinel-sdk[mcp]' in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('proxy imported without mcp')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
