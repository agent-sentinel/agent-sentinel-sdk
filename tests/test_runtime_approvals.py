"""
Tests for the runtime's approval wait (agent_sentinel.runtime.approvals).

A real MCP client session drives ``cli.serve`` (and so ``mcp_proxy.serve``)
over in-memory streams. Only the HTTP transport is replaced: one
``httpx.MockTransport`` fakes both the platform's ``/mcp`` and
``/api/v1/approvals/{id}/poll``. Sleep and clock are faked so nothing waits
in real time. They skip without the ``mcp`` extra.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from functools import partial
from typing import Any

import pytest

if importlib.util.find_spec("mcp") is not None:
    import anyio
    import anyio.lowlevel
    import httpx
    from mcp import ClientSession, McpError, types
    from mcp.shared.memory import create_client_server_memory_streams

    from agent_sentinel import mcp_proxy
    from agent_sentinel.runtime import approvals, cli
    from agent_sentinel.runtime.config import ApprovalsConfig, RuntimeConfig


@pytest.fixture
def _require_mcp():
    pytest.importorskip("mcp")


pytestmark = pytest.mark.usefixtures("_require_mcp")

API_KEY = "as_SENTINEL_key_must_not_leak"
LEAK_EMAIL = "leak-sentinel@example.test"
BASE_URL = "https://api.example.test"
APPROVAL_ID = "3f2b8c1e-7a4d-4e9b-9c0a-5d6e7f8a9b0c"
POLL_URL = f"{BASE_URL}/api/v1/approvals/{APPROVAL_ID}/poll"
ARGS = {"to": "a@example.com", "count": 3, "nested": {"k": ["v"]}}

SEND_TOOL = {
    "name": "send",
    "description": "send description",
    "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
}


def _require_approval(approval_id: Any = APPROVAL_ID, decision: str = "require_approval"):
    """A governed tools/call result in the shape of ``_decision_result``."""
    structured = {"decision": decision, "reason_code": "needs_human_review"}
    if approval_id is not None:
        structured["approval_request_id"] = approval_id
    return {
        "content": [{"type": "text", "text": "Approval required (needs_human_review)."}],
        "isError": True,
        "structuredContent": structured,
    }


DELIVERED = {
    "content": [{"type": "text", "text": "delivered after approval"}],
    "isError": False,
}


def _poll_body(status: str, **extra) -> dict:
    body = {
        "id": APPROVAL_ID,
        "status": status,
        "decided_at": None if status == "pending" else "2026-10-27T10:00:00Z",
        "decided_by_email": None if status == "pending" else LEAK_EMAIL,
        "decision_notes": None,
        "info_request_message": None,
        "approved_with_modifications": None,
    }
    body.update(extra)
    return body


class FakePlatform:
    """``/mcp`` plus the approval poll endpoint, behind one MockTransport."""

    def __init__(
        self,
        polls: list[Any],
        *,
        first: dict | None = None,
        retry: dict | None = None,
    ) -> None:
        self.polls = list(polls)
        self.first = first if first is not None else _require_approval()
        self.retry = retry if retry is not None else DELIVERED
        self.tool_calls: list[dict] = []
        self.poll_requests: list[httpx.Request] = []
        self.other_urls: list[str] = []

    def mock_transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp" and request.url.host == "api.example.test":
            return self._mcp(request)
        if str(request.url) == POLL_URL and request.method == "GET":
            self.poll_requests.append(request)
            item = self.polls.pop(0)
            if isinstance(item, Exception):
                raise item
            status, body = item
            if isinstance(body, (bytes, str)):
                return httpx.Response(status, content=body)
            headers = {"Location": "https://evil.example.test/steal"} if status == 302 else {}
            return httpx.Response(status, json=body, headers=headers)
        self.other_urls.append(f"{request.method} {request.url}")
        return httpx.Response(404, json={"detail": "Not Found"})

    def _mcp(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
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
            params = body["params"]
            self.tool_calls.append(params)
            meta = params.get("_meta") or {}
            result = self.retry if "approval_request_id" in meta else self.first
        else:
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                },
            )
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


class FakeTime:
    """``sleep(d)`` advances the clock by ``d`` without waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await anyio.lowlevel.checkpoint()


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(approvals, "_sleep", fake.sleep)
    monkeypatch.setattr(approvals, "_clock", fake.clock)
    return fake


def _config(*, wait: bool = True, timeout: int = 900, interval: int = 5) -> RuntimeConfig:
    return RuntimeConfig(
        url=BASE_URL,
        workspace="/tmp/work",
        api_key_file="/tmp/secrets/key",
        harness="cursor-bg",
        approvals=ApprovalsConfig(
            wait=wait, timeout_seconds=timeout, poll_interval_seconds=interval
        ),
        api_key=API_KEY,
    )


async def _call(serve, meta: dict | None = None) -> types.CallToolResult:
    """initialize -> tools/call through ``serve(read_stream, write_stream)``."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(serve, *server_streams))
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                result = await session.call_tool("send", ARGS, meta=meta)
            tg.cancel_scope.cancel()
    return result


def _runtime(platform: FakePlatform, **cfg) -> Any:
    return partial(cli.serve, _config(**cfg), http_transport=platform.mock_transport())


def _text(result: types.CallToolResult) -> str:
    assert len(result.content) == 1
    return result.content[0].text


def _leaks(text: str) -> list[str]:
    needles = {API_KEY, LEAK_EMAIL, "decided_by_email"}
    needles |= {API_KEY[i : i + 8] for i in range(len(API_KEY) - 7)}
    return sorted(n for n in needles if n in text)


def _assert_result_clean(result: types.CallToolResult) -> None:
    assert _leaks(result.model_dump_json(by_alias=True)) == []


@pytest.fixture(autouse=True)
def _no_leaks_in_logs_or_output(caplog, capsys):
    """Every test: no log record and no stdout/stderr carries a sentinel."""
    caplog.set_level(logging.DEBUG)
    yield
    for record in caplog.records:
        assert _leaks(record.getMessage()) == [], record.name
    out, err = capsys.readouterr()
    assert _leaks(out + err) == []


# --- approved: retried once ------------------------------------------------------


async def test_pending_pending_approved_retries_once_with_merged_meta(fake_time):
    platform = FakePlatform(
        [(200, _poll_body("pending")), (200, _poll_body("pending")), (200, _poll_body("approved"))]
    )

    result = await _call(_runtime(platform), meta={"orig": "x"})

    # The retry's upstream result, verbatim.
    assert result.model_dump() == types.CallToolResult.model_validate(DELIVERED).model_dump()
    assert len(platform.tool_calls) == 2
    first, retry = platform.tool_calls
    assert first["name"] == retry["name"] == "send"
    assert first["arguments"] == retry["arguments"] == ARGS
    assert first["_meta"] == {"orig": "x"}
    assert retry["_meta"] == {"orig": "x", "approval_request_id": APPROVAL_ID}
    assert len(platform.poll_requests) == 3
    for request in platform.poll_requests:
        assert request.headers["authorization"] == f"ApiKey {API_KEY}"
        assert request.headers["x-sentinel-harness"] == "cursor-bg"
    assert fake_time.sleeps == [5, 5]  # polls at once, then every interval
    assert platform.other_urls == []
    _assert_result_clean(result)


async def test_approved_retry_without_original_meta(fake_time):
    platform = FakePlatform([(200, _poll_body("approved"))])

    result = await _call(_runtime(platform))

    assert result.isError is False
    assert platform.tool_calls[1]["_meta"] == {"approval_request_id": APPROVAL_ID}
    assert platform.tool_calls[1]["arguments"] == ARGS


async def test_second_require_approval_on_retry_is_returned_as_is(fake_time):
    second = _require_approval()
    platform = FakePlatform([(200, _poll_body("approved"))], retry=second)

    result = await _call(_runtime(platform))

    assert result.model_dump() == types.CallToolResult.model_validate(second).model_dump()
    assert len(platform.tool_calls) == 2
    assert len(platform.poll_requests) == 1  # no polling after the retry


# --- terminal statuses -----------------------------------------------------------


@pytest.mark.parametrize("status", ["rejected", "expired", "cancelled"])
async def test_terminal_status_is_an_error_with_notes(fake_time, status):
    platform = FakePlatform(
        [(200, _poll_body("pending")), (200, _poll_body(status, decision_notes="Too risky"))]
    )

    result = await _call(_runtime(platform))

    assert result.isError is True
    assert _text(result) == f"Approval {status}: Too risky (approval {APPROVAL_ID})"
    assert result.structuredContent is None
    assert len(platform.tool_calls) == 1
    _assert_result_clean(result)


@pytest.mark.parametrize("status", ["rejected", "expired", "cancelled"])
async def test_terminal_status_without_notes(fake_time, status):
    platform = FakePlatform([(200, _poll_body(status))])

    result = await _call(_runtime(platform))

    assert result.isError is True
    assert _text(result) == f"Approval {status} (approval {APPROVAL_ID})"
    assert len(platform.tool_calls) == 1


async def test_approver_text_never_echoes_the_key(fake_time):
    platform = FakePlatform(
        [(200, _poll_body("rejected", decision_notes=f"key was {API_KEY}"))]
    )

    result = await _call(_runtime(platform))

    assert _text(result) == f"Approval rejected: key was [redacted] (approval {APPROVAL_ID})"


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        (f"see {API_KEY[5:13]} end", "see [redacted] end"),
        (f"see {API_KEY[2:20]}.", "see [redacted]."),
        (f"{API_KEY[0:9]} and {API_KEY[-8:]}", "[redacted] and [redacted]"),
        (f"keep {API_KEY[5:12]} here", f"keep {API_KEY[5:12]} here"),
    ],
)
async def test_approver_text_never_echoes_part_of_the_key(fake_time, notes, expected):
    platform = FakePlatform([(200, _poll_body("rejected", decision_notes=notes))])

    result = await _call(_runtime(platform))

    assert _text(result) == f"Approval rejected: {expected} (approval {APPROVAL_ID})"
    _assert_result_clean(result)


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ("ping user@host now", "ping user@host now"),
        ("tag x@y", "tag x@y"),
        ("mail a.b@c.io today", "mail [redacted] today"),
        ("write to ops@team.example.com.", "write to [redacted]."),
        ("ask jürgen@bücher.de", "ask [redacted]"),
        (f"to a-{API_KEY[3:11]}@ops.example.com", "to [redacted]"),
    ],
)
async def test_approver_text_scrubs_only_real_email_addresses(fake_time, notes, expected):
    platform = FakePlatform([(200, _poll_body("rejected", decision_notes=notes))])

    result = await _call(_runtime(platform))

    assert _text(result) == f"Approval rejected: {expected} (approval {APPROVAL_ID})"
    _assert_result_clean(result)


@pytest.mark.parametrize(
    "text",
    ["a@" + "a" * 10_000, "a" * 5_000 + "@" + "a" * 5_000, "a@" + "a." * 5_000 + "1"],
)
def test_email_pattern_stays_fast_on_long_near_misses(text):
    started = time.perf_counter()
    assert approvals._EMAIL.sub("[redacted]", text) == text
    assert time.perf_counter() - started < 2.0


async def test_approver_text_never_exposes_the_approver_email(fake_time):
    platform = FakePlatform(
        [
            (
                200,
                _poll_body(
                    "rejected",
                    decision_notes=f"Contact {LEAK_EMAIL.upper()} or ops@example.com",
                ),
            )
        ]
    )

    result = await _call(_runtime(platform))

    assert _text(result) == (
        f"Approval rejected: Contact [redacted] or [redacted] (approval {APPROVAL_ID})"
    )
    _assert_result_clean(result)


async def test_info_request_message_never_exposes_the_approver_email(fake_time):
    platform = FakePlatform(
        [
            (
                200,
                _poll_body(
                    "info_requested",
                    decided_by_email=LEAK_EMAIL,
                    info_request_message=f"Ask {LEAK_EMAIL} which account",
                ),
            )
        ]
    )

    result = await _call(_runtime(platform))

    assert _text(result) == (
        f"The approver asked for more information about approval {APPROVAL_ID}; "
        "the call did not run: Ask [redacted] which account"
    )
    _assert_result_clean(result)


async def test_info_requested_is_an_error_with_the_message(fake_time):
    platform = FakePlatform(
        [(200, _poll_body("info_requested", info_request_message="Which account?"))]
    )

    result = await _call(_runtime(platform))

    assert result.isError is True
    assert _text(result) == (
        f"The approver asked for more information about approval {APPROVAL_ID}; "
        "the call did not run: Which account?"
    )
    assert len(platform.tool_calls) == 1
    _assert_result_clean(result)


# --- timeout ---------------------------------------------------------------------


async def test_timeout_names_the_id_and_does_not_cancel(fake_time):
    platform = FakePlatform([(200, _poll_body("pending"))] * 10)

    result = await _call(_runtime(platform, timeout=10, interval=5))

    assert result.isError is True
    assert _text(result) == (
        f"Approval {APPROVAL_ID} is still pending after 10s. It was not cancelled; "
        f'retry the same call with _meta.approval_request_id = "{APPROVAL_ID}" '
        "once it is approved."
    )
    assert len(platform.poll_requests) == 2
    assert fake_time.sleeps == [5, 5]
    assert len(platform.tool_calls) == 1
    assert platform.other_urls == []  # in particular, no /cancel
    _assert_result_clean(result)


async def test_timeout_shorter_than_the_interval_still_polls_once(fake_time):
    platform = FakePlatform([(200, _poll_body("pending"))] * 3)

    result = await _call(_runtime(platform, timeout=3, interval=5))

    assert result.isError is True
    assert fake_time.sleeps == [3]
    assert len(platform.poll_requests) == 1


async def test_approval_arriving_after_the_deadline_is_a_timeout(fake_time):
    platform = FakePlatform([])
    handle = platform._handle

    def slow_poll(request: httpx.Request) -> httpx.Response:
        if str(request.url) == POLL_URL:
            fake_time.now += 20  # the response comes back 20s later
            platform.polls.append((200, _poll_body("approved")))
        return handle(request)

    serve = partial(
        cli.serve, _config(timeout=3, interval=1), http_transport=httpx.MockTransport(slow_poll)
    )

    result = await _call(serve)

    assert result.isError is True
    assert _text(result) == (
        f"Approval {APPROVAL_ID} is still pending after 3s. It was not cancelled; "
        f'retry the same call with _meta.approval_request_id = "{APPROVAL_ID}" '
        "once it is approved."
    )
    assert len(platform.poll_requests) == 1
    assert len(platform.tool_calls) == 1  # no retry
    assert platform.other_urls == []  # not cancelled
    assert fake_time.sleeps == []  # nothing polled or slept past the deadline


async def test_a_hung_poll_is_bounded_by_the_deadline(monkeypatch):
    # Real clock: the request never answers, the 1s deadline still holds.
    monkeypatch.setattr(approvals, "_clock", anyio.current_time)
    hung = anyio.Event()

    class HangingTransport(httpx.AsyncBaseTransport):
        def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
            self.inner = inner

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if str(request.url) == POLL_URL:
                await hung.wait()
            return await self.inner.handle_async_request(request)

    platform = FakePlatform([])
    serve = partial(
        cli.serve,
        _config(timeout=1, interval=1),
        http_transport=HangingTransport(platform.mock_transport()),
    )

    with anyio.fail_after(10):
        result = await _call(serve)

    assert result.isError is True
    assert _text(result) == (
        f"Approval {APPROVAL_ID} is still pending after 1s. It was not cancelled; "
        f'retry the same call with _meta.approval_request_id = "{APPROVAL_ID}" '
        "once it is approved."
    )
    assert len(platform.tool_calls) == 1


# --- poll failures ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            (401, {"detail": {"reason_code": "agent_bound_key_required"}}),
            "AgentSentinel rejected the API key (HTTP 401, agent_bound_key_required)",
        ),
        ((403, {"detail": "Forbidden"}), "AgentSentinel rejected the API key (HTTP 403)"),
        (
            (404, {"detail": "Approval request not found"}),
            "AgentSentinel request failed (HTTP 404)",
        ),
        ((500, b"oops"), "AgentSentinel request failed (HTTP 500)"),
        ((302, {}), "AgentSentinel request failed (HTTP 302)"),
        ("connect_error", "AgentSentinel is unreachable (ConnectError)"),
        (
            (200, b"not json"),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, b"[1, 2]"),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, _poll_body("weird")),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, {**_poll_body("pending"), "status": []}),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, _poll_body("rejected", decided_by_email=["x"])),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, _poll_body("approved", id="00000000-0000-0000-0000-000000000000")),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
        (
            (200, _poll_body("rejected", decision_notes={"x": 1})),
            "AgentSentinel request failed (invalid approval poll response)",
        ),
    ],
)
async def test_poll_failure_is_an_upstream_style_error(fake_time, response, message):
    if response == "connect_error":
        response = httpx.ConnectError(f"refused {API_KEY}")
    platform = FakePlatform([(200, _poll_body("pending")), response])

    result = await _call(_runtime(platform))

    assert result.isError is True
    assert _text(result) == message
    assert len(platform.poll_requests) == 2  # stops at the failure
    assert len(platform.tool_calls) == 1  # no retry
    assert platform.other_urls == []  # a redirect is never followed
    _assert_result_clean(result)


# --- pass-through ----------------------------------------------------------------


async def test_wait_disabled_passes_require_approval_through(fake_time):
    platform = FakePlatform([])

    result = await _call(_runtime(platform, wait=False))

    expected = types.CallToolResult.model_validate(_require_approval())
    assert result.model_dump() == expected.model_dump()
    assert platform.poll_requests == []
    assert len(platform.tool_calls) == 1


async def test_mcp_proxy_without_a_waiter_passes_require_approval_through(fake_time):
    platform = FakePlatform([])
    proxy_config = _config().proxy_config()
    serve = partial(mcp_proxy.serve, proxy_config, http_transport=platform.mock_transport())

    result = await _call(serve)

    expected = types.CallToolResult.model_validate(_require_approval())
    assert result.model_dump() == expected.model_dump()
    assert platform.poll_requests == []


@pytest.mark.parametrize(
    "first",
    [
        _require_approval(approval_id=None),
        _require_approval(approval_id="not-a-uuid"),
        _require_approval(approval_id="../../admin"),
        _require_approval(approval_id=42),
        _require_approval(decision="deny"),
        DELIVERED,
    ],
)
async def test_other_results_pass_through_unchanged(fake_time, first):
    platform = FakePlatform([], first=first)

    result = await _call(_runtime(platform))

    assert result.model_dump() == types.CallToolResult.model_validate(first).model_dump()
    assert platform.poll_requests == []
    assert len(platform.tool_calls) == 1


# --- cancellation ----------------------------------------------------------------


async def test_cancelling_the_held_request_stops_polling(monkeypatch):
    platform = FakePlatform([(200, _poll_body("pending"))] * 10)
    first_poll_done = anyio.Event()
    never = anyio.Event()
    sleeps_cancelled: list[bool] = []
    now = [0.0]

    async def blocking_sleep(delay: float) -> None:
        if platform.poll_requests:
            first_poll_done.set()
            try:
                await never.wait()
            except anyio.get_cancelled_exc_class():
                sleeps_cancelled.append(True)
                raise
        now[0] += delay
        await anyio.lowlevel.checkpoint()

    monkeypatch.setattr(approvals, "_sleep", blocking_sleep)
    monkeypatch.setattr(approvals, "_clock", lambda: now[0])

    errors: list[McpError] = []
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(_runtime(platform), *server_streams))
            async with ClientSession(*client_streams) as session:
                await session.initialize()

                async def held_call() -> None:
                    try:
                        await session.call_tool("send", ARGS)
                    except McpError as exc:
                        errors.append(exc)

                async with anyio.create_task_group() as calls:
                    calls.start_soon(held_call)
                    with anyio.fail_after(5):
                        await first_poll_done.wait()
                    request_id = session._request_id - 1
                    await session.send_notification(
                        types.ClientNotification(
                            types.CancelledNotification(
                                params=types.CancelledNotificationParams(requestId=request_id)
                            )
                        )
                    )
                for _ in range(20):
                    await anyio.lowlevel.checkpoint()
            tg.cancel_scope.cancel()

    assert [e.error.message for e in errors] == ["Request cancelled"]
    assert sleeps_cancelled == [True]
    assert len(platform.poll_requests) == 1
    assert len(platform.tool_calls) == 1  # no retry
    assert platform.other_urls == []  # the approval is not cancelled upstream


def test_waiter_repr_has_no_key():
    waiter = approvals.ApprovalWaiter(
        _config().proxy_config(), BASE_URL, ApprovalsConfig()
    )
    assert API_KEY not in repr(waiter)
