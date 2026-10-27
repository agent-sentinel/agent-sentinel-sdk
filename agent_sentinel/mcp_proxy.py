"""
``agentsentinel-mcp``: a stdio MCP server that proxies to AgentSentinel's ``/mcp``.

MCP-native harnesses (Claude Code, Cursor, ...) register this command like any
other stdio MCP server. It answers ``initialize`` locally, then forwards
``tools/list`` and ``tools/call`` to the platform's streamable-HTTP ``/mcp``
endpoint using an agent-bound API key read from the environment. The key only
ever travels in the upstream ``Authorization`` header: it is never written to
stdout, stderr, a log line or any MCP message, so the model never sees it.

Configuration (environment):

- ``AGENT_SENTINEL_API_KEY`` (required): an agent-bound API key.
- ``AGENT_SENTINEL_URL`` (default: the hosted platform): the
  platform base URL; the proxy targets ``<url>/mcp``. ``https`` is required
  unless the host is ``localhost`` or ``127.0.0.1``.
- ``AGENT_SENTINEL_HARNESS`` (optional): sent upstream as ``X-Sentinel-Harness``.

Needs the optional extra: ``pip install "agentsentinel-sdk[mcp]"`` (Python 3.10+).
This module is never imported by ``agent_sentinel`` itself.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

try:
    import anyio
    import httpx
    from anyio.abc import TaskGroup, TaskStatus
    from mcp import ClientSession, McpError, types
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "agentsentinel-mcp needs the optional MCP extra: "
        'pip install "agentsentinel-sdk[mcp]" (Python 3.10+)'
    ) from exc

from . import __version__
from ._defaults import DEFAULT_PLATFORM_URL

logger = logging.getLogger("agent_sentinel.mcp_proxy")

DEFAULT_URL = DEFAULT_PLATFORM_URL
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1"})

# JSON-RPC code for failures reaching the platform (HTTP 4xx/5xx, transport
# errors). Server-defined range; the platform's own /mcp does not emit it.
UPSTREAM_ERROR_CODE = -32001
# Errors that mean "could not get an answer from the platform" rather than
# "the platform answered with a JSON-RPC error". tools/call reports these as
# an isError tool result; a genuine upstream JSON-RPC error passes through.
_LOCAL_FAILURE_CODES = frozenset(
    {UPSTREAM_ERROR_CODE, types.CONNECTION_CLOSED, httpx.codes.REQUEST_TIMEOUT}
)

_KEY_CHARS = re.compile(r"^[\x21-\x7e]+$")  # printable ASCII, no spaces
_HEADER_CHARS = re.compile(r"^[\x20-\x7e]+$")  # printable ASCII
_REASON_CODE = re.compile(r"^[a-z0-9_]{1,64}$")

# Sends one tools/call upstream with the proxy's error mapping applied.
CallForward = Callable[[types.CallToolRequestParams], Awaitable[types.CallToolResult]]
# Optional post-processing of a tools/call result (agentsentinel-runtime's
# approval wait). Receives the original params, the upstream result and the
# forwarder; returns the result for the client.
ApprovalWaiter = Callable[
    [types.CallToolRequestParams, types.CallToolResult, CallForward],
    Awaitable[types.CallToolResult],
]


class ConfigError(Exception):
    """Invalid proxy configuration. Messages never contain the API key."""


@dataclass(frozen=True)
class ProxyConfig:
    """Resolved proxy settings. ``api_key`` is excluded from ``repr``."""

    url: str
    api_key: str = field(repr=False)
    harness: Optional[str] = None

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"ApiKey {self.api_key}",
            "User-Agent": f"agentsentinel-mcp/{__version__}",
        }
        if self.harness:
            headers["X-Sentinel-Harness"] = self.harness
        return headers


def load_config(environ: Mapping[str, str]) -> ProxyConfig:
    """Read and validate the proxy configuration from ``environ``.

    Raises ConfigError with a one-line message. No message echoes a value
    from the environment, so a key pasted into the wrong variable cannot leak.
    """
    api_key = check_api_key(
        environ.get("AGENT_SENTINEL_API_KEY", ""), "AGENT_SENTINEL_API_KEY"
    )
    base = check_base_url(environ.get("AGENT_SENTINEL_URL", ""), "AGENT_SENTINEL_URL")
    harness = check_harness(
        environ.get("AGENT_SENTINEL_HARNESS", ""), "AGENT_SENTINEL_HARNESS"
    )
    return ProxyConfig(url=f"{base}/mcp", api_key=api_key, harness=harness)


# Shared validators (also used by agent_sentinel.runtime). ``name`` is the
# setting's name as the user wrote it; messages never include the value.


def check_api_key(value: str, name: str) -> str:
    """Return the stripped key, or raise ConfigError if empty or unsendable."""
    api_key = value.strip()
    if not api_key:
        raise ConfigError(f"{name} is not set; use an agent-bound API key")
    if not _KEY_CHARS.match(api_key):
        raise ConfigError(
            f"{name} contains characters that cannot be sent in an HTTP header"
        )
    return api_key


def check_base_url(value: str, name: str) -> str:
    """Return the platform base URL (no trailing slash, no ``/mcp``).

    An empty value means ``DEFAULT_URL``. ``https`` is required unless the
    host is ``localhost`` or ``127.0.0.1``.
    """
    base = (value.strip() or DEFAULT_URL).rstrip("/")
    try:
        parts = urlsplit(base)
        hostname = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError:
        raise ConfigError(f"{name} is not a valid URL") from None
    if not hostname:
        raise ConfigError(f"{name} must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{name} must not contain credentials")
    if parts.query or parts.fragment:
        raise ConfigError(f"{name} must not contain a query or fragment")
    scheme = parts.scheme.lower()
    if not (scheme == "https" or (scheme == "http" and hostname in _LOCAL_HOSTS)):
        raise ConfigError(
            f"{name} must use https "
            "(http is allowed only for localhost and 127.0.0.1)"
        )
    return base


def check_harness(value: Optional[str], name: str) -> Optional[str]:
    """Return the stripped harness name, or None when empty."""
    harness = (value or "").strip() or None
    if harness is not None and not _HEADER_CHARS.match(harness):
        raise ConfigError(
            f"{name} contains characters that cannot be sent in an HTTP header"
        )
    return harness


class _UpstreamErrorTransport(httpx.AsyncBaseTransport):
    """Turn HTTP errors and transport failures into JSON-RPC error responses.

    The official streamable-HTTP client raises on a non-2xx status inside its
    own task group, which tears the whole client down instead of failing the
    one request. Answering with a JSON-RPC error for the request's id keeps
    the session usable and surfaces the failure as an McpError on that call.
    The message carries only the status and the platform's reason code, never
    the key or response headers.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            response = await self._inner.handle_async_request(request)
        except httpx.TransportError as exc:
            return _error_response(
                request,
                f"AgentSentinel is unreachable ({type(exc).__name__})",
            )
        if response.status_code < 400:
            return response
        try:
            await response.aread()
            reason_code = _reason_code(response.content)
        except httpx.HTTPError:
            reason_code = None
        finally:
            await response.aclose()
        status = response.status_code
        detail = f"HTTP {status}, {reason_code}" if reason_code else f"HTTP {status}"
        if status in (401, 403):
            message = f"AgentSentinel rejected the API key ({detail})"
        else:
            message = f"AgentSentinel request failed ({detail})"
        return _error_response(request, message)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _reason_code(body: bytes) -> Optional[str]:
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    code = data.get("reason_code")
    if code is None and isinstance(data.get("detail"), dict):
        code = data["detail"].get("reason_code")
    if isinstance(code, str) and _REASON_CODE.match(code):
        return code
    return None


def _error_response(request: httpx.Request, message: str) -> httpx.Response:
    request_id = None
    try:
        payload = json.loads(request.content)
        if isinstance(payload, dict):
            request_id = payload.get("id")
    except (ValueError, httpx.RequestNotRead):
        pass
    if request_id is None:
        # A notification (or a non-JSON-RPC request) expects no response.
        return httpx.Response(202, request=request)
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": UPSTREAM_ERROR_CODE, "message": message},
    }
    return httpx.Response(
        200,
        headers={"Content-Type": "application/json"},
        content=json.dumps(body).encode(),
        request=request,
    )


def _find_mcp_error(exc: BaseException) -> Optional[McpError]:
    if isinstance(exc, McpError):
        return exc
    # anyio task groups wrap failures in (Base)ExceptionGroup; duck-typed so
    # Python 3.10 (the exceptiongroup backport) is covered too.
    inner_exceptions = getattr(exc, "exceptions", None)
    if isinstance(inner_exceptions, tuple):
        for inner in inner_exceptions:
            found = _find_mcp_error(inner)
            if found is not None:
                return found
    return None


def _upstream_error(exc: BaseException) -> types.ErrorData:
    found = _find_mcp_error(exc)
    if found is not None:
        return found.error
    return types.ErrorData(
        code=UPSTREAM_ERROR_CODE,
        message=f"AgentSentinel request failed ({type(exc).__name__})",
    )


class _Upstream:
    """A lazily opened, reused MCP client session to the platform.

    The official client's context managers must be entered and exited in one
    task, so a dedicated owner task holds them open for the proxy's lifetime.
    Request handlers share the session; if it ends, the next call reopens it.
    """

    def __init__(
        self,
        config: ProxyConfig,
        task_group: TaskGroup,
        http_transport: Optional[httpx.AsyncBaseTransport],
    ) -> None:
        self._config = config
        self._task_group = task_group
        self._http_transport = http_transport
        self._lock = anyio.Lock()
        self._stop = anyio.Event()
        self._session: Optional[ClientSession] = None

    async def session(self) -> ClientSession:
        async with self._lock:
            if self._session is None:
                self._session = await self._task_group.start(self._own)
            return self._session

    def close(self) -> None:
        self._stop.set()

    async def _own(self, *, task_status: TaskStatus[ClientSession]) -> None:
        inner = self._http_transport or httpx.AsyncHTTPTransport()
        http_client = httpx.AsyncClient(
            headers=self._config.headers(),
            timeout=httpx.Timeout(30.0, read=300.0),
            transport=_UpstreamErrorTransport(inner),
            follow_redirects=False,
        )
        started = False
        session: Optional[ClientSession] = None
        try:
            async with http_client, streamable_http_client(
                self._config.url, http_client=http_client
            ) as (read_stream, write_stream, _), ClientSession(
                read_stream, write_stream
            ) as session:
                await session.initialize()
                started = True
                task_status.started(session)
                await self._stop.wait()
        except Exception:
            if not started:
                raise
            logger.debug("upstream MCP session ended; it reopens on the next call")
        finally:
            if session is not None and self._session is session:
                self._session = None


def build_server(
    upstream: _Upstream, approval_waiter: Optional[ApprovalWaiter] = None
) -> Server[Any, Any]:
    """The stdio-facing server: tools/list and tools/call go upstream verbatim.

    ``approval_waiter``, when set, sees every tools/call result before the
    client does; with None the result is returned unchanged.
    """
    server: Server[Any, Any] = Server(
        "agentsentinel-mcp",
        version=__version__,
        instructions="Governed tools from AgentSentinel.",
    )

    # Requests are rebuilt from their params: the parsed incoming request keeps
    # its JSON-RPC envelope (jsonrpc, id) as extra fields, which send_request
    # supplies itself. The params, including _meta, are forwarded unchanged.
    async def list_tools(req: types.ListToolsRequest) -> types.ServerResult:
        try:
            session = await upstream.session()
            result = await session.send_request(
                types.ClientRequest(types.ListToolsRequest(params=req.params)),
                types.ListToolsResult,
            )
        except Exception as exc:  # noqa: BLE001 - every failure becomes an MCP error
            raise McpError(_upstream_error(exc)) from None
        return types.ServerResult(result)

    async def forward(params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            session = await upstream.session()
            return await session.send_request(
                types.ClientRequest(types.CallToolRequest(params=params)),
                types.CallToolResult,
            )
        except Exception as exc:  # noqa: BLE001 - every failure becomes a tool result
            error = _upstream_error(exc)
            if _find_mcp_error(exc) is not None and error.code not in _LOCAL_FAILURE_CODES:
                raise McpError(error) from None
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=error.message)],
                isError=True,
            )

    async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
        result = await forward(req.params)
        if approval_waiter is not None:
            result = await approval_waiter(req.params, result, forward)
        return types.ServerResult(result)

    server.request_handlers[types.ListToolsRequest] = list_tools
    server.request_handlers[types.CallToolRequest] = call_tool
    return server


async def serve(
    config: ProxyConfig,
    read_stream: Any,
    write_stream: Any,
    *,
    http_transport: Optional[httpx.AsyncBaseTransport] = None,
    approval_waiter: Optional[ApprovalWaiter] = None,
) -> None:
    """Run the proxy over the given MCP streams until the client disconnects.

    ``http_transport`` replaces the network transport (tests pass an
    ``httpx.MockTransport``); headers and error mapping still apply.
    ``approval_waiter`` is passed to ``build_server``.
    """
    async with anyio.create_task_group() as task_group:
        upstream = _Upstream(config, task_group, http_transport)
        server = build_server(upstream, approval_waiter)
        try:
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
        finally:
            upstream.close()


async def _run_stdio(config: ProxyConfig) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await serve(config, read_stream, write_stream)


def main() -> int:
    """Console entry point for ``agentsentinel-mcp``."""
    try:
        config = load_config(os.environ)
    except ConfigError as exc:
        print(f"agentsentinel-mcp: {exc}", file=sys.stderr)
        return 2
    try:
        anyio.run(_run_stdio, config)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback or the key
        print(f"agentsentinel-mcp: stopped ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
