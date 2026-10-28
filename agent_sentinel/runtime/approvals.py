"""
Human-in-the-loop wait for ``agentsentinel-runtime run``.

When a governed ``tools/call`` comes back with ``require_approval``, the
platform has already notified the org's approvers (Slack, email, webhook).
The runtime holds the call, polls ``GET {url}/api/v1/approvals/{id}/poll``
outbound with the same agent-bound key, and once the approval is granted
retries the call once with ``_meta.approval_request_id``. Every other outcome
becomes an ``isError`` tool result.

Rules: outbound HTTPS only; the key and the approver's email never reach a
result, a log line or any output; a timeout never cancels the approval, so
the model can retry the same call later.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import anyio
import httpx
from mcp import types

from ..mcp_proxy import CallForward, ProxyConfig, _reason_code

if TYPE_CHECKING:
    from .config import ApprovalsConfig

logger = logging.getLogger("agent_sentinel.runtime.approvals")

# Patchable defaults, looked up at call time (tests replace them).
_sleep: Callable[[float], Awaitable[Any]] = anyio.sleep
_clock: Callable[[], float] = anyio.current_time

_TERMINAL_ERROR_STATUSES = frozenset({"rejected", "expired", "cancelled"})
_KNOWN_STATUSES = frozenset({"pending", "approved", "info_requested"}) | _TERMINAL_ERROR_STATUSES
_INVALID_RESPONSE = "AgentSentinel request failed (invalid approval poll response)"
# Any email address in approver text (the approver's own included) is scrubbed:
# a domain with a dot and a TLD of 2+ letters, so a bare ``user@host`` is kept.
# The lookbehind starts a match only where a run starts, keeping it linear.
_EMAIL = re.compile(
    r"(?<![^\s@<>()\[\],;:\"'])[^\s@<>()\[\],;:\"']+@"
    r"(?:[^\s@<>()\[\],;:\"'.]+\.)+[^\W\d_]{2,}"
)
# Any run of this many characters of the key is scrubbed, not just the whole key.
_KEY_WINDOW = 8
_REDACTED = "[redacted]"


def _scrub_key(value: str, key: str) -> str:
    """``value`` with every run of 8+ characters of ``key`` replaced once."""
    if not key:
        return value
    if len(key) < _KEY_WINDOW:
        return value.replace(key, _REDACTED)
    hit = [False] * len(value)
    for i in range(len(key) - _KEY_WINDOW + 1):
        window = key[i : i + _KEY_WINDOW]
        start = value.find(window)
        while start != -1:
            hit[start : start + _KEY_WINDOW] = [True] * _KEY_WINDOW
            start = value.find(window, start + 1)
    out: list[str] = []
    for index, char in enumerate(value):
        if not hit[index]:
            out.append(char)
        elif index == 0 or not hit[index - 1]:
            out.append(_REDACTED)
    return "".join(out)


class _PollFailed(Exception):
    """The poll could not produce a status. ``message`` is safe to show."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _error_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)], isError=True
    )


def _approval_id(result: types.CallToolResult) -> str | None:
    """The canonical approval id of a ``require_approval`` result, else None."""
    structured = result.structuredContent
    if not isinstance(structured, dict):
        return None
    if structured.get("decision") != "require_approval":
        return None
    raw = structured.get("approval_request_id")
    if not isinstance(raw, str):
        return None
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return None


class ApprovalWaiter:
    """An ``mcp_proxy.ApprovalWaiter``: hold, poll, retry once.

    Holds no per-call state, so concurrent held calls are independent.
    """

    def __init__(
        self,
        proxy_config: ProxyConfig,
        base_url: str,
        approvals: ApprovalsConfig,
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[Any]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._proxy_config = proxy_config
        self._base_url = base_url.rstrip("/")
        self._approvals = approvals
        self._http_transport = http_transport
        self._sleep = sleep
        self._clock = clock

    def __repr__(self) -> str:
        return f"ApprovalWaiter(base_url={self._base_url!r})"

    async def __call__(
        self,
        params: types.CallToolRequestParams,
        result: types.CallToolResult,
        forward: CallForward,
    ) -> types.CallToolResult:
        approval_id = _approval_id(result)
        if approval_id is None:
            return result
        try:
            body = await self._wait(approval_id)
        except _PollFailed as exc:
            return _error_result(exc.message)
        if body is None:
            return _error_result(
                f"Approval {approval_id} is still pending after "
                f"{self._approvals.timeout_seconds}s. It was not cancelled; retry "
                f'the same call with _meta.approval_request_id = "{approval_id}" '
                "once it is approved."
            )

        status = body["status"]
        approver_email = body.get("decided_by_email")
        if status == "approved":
            meta = params.meta.model_dump(by_alias=True, exclude_none=True) if params.meta else {}
            meta["approval_request_id"] = approval_id
            retry = params.model_copy(
                update={"meta": types.RequestParams.Meta.model_validate(meta)}
            )
            # forward, never the waiter again: at most one retry per held call.
            return await forward(retry)
        if status == "info_requested":
            message = self._redact(body.get("info_request_message"), approver_email)
            text = (
                f"The approver asked for more information about approval "
                f"{approval_id}; the call did not run"
            )
            return _error_result(f"{text}: {message}" if message else f"{text}.")
        # rejected / expired / cancelled
        notes = self._redact(body.get("decision_notes"), approver_email)
        text = f"Approval {status}"
        if notes:
            text += f": {notes}"
        return _error_result(f"{text} (approval {approval_id})")

    def _redact(self, value: str | None, approver_email: str | None) -> str | None:
        """Approver free text, with any part of the key and any email address replaced."""
        if not value:
            return None
        # Emails first: scrubbing a key fragment inside an address would leave
        # the rest of it unmatched. The key scrub runs last, so it always holds.
        if approver_email:
            value = re.sub(re.escape(approver_email), _REDACTED, value, flags=re.IGNORECASE)
        value = _EMAIL.sub(_REDACTED, value)
        return _scrub_key(value, self._proxy_config.api_key)

    async def _wait(self, approval_id: str) -> dict[str, Any] | None:
        """Poll until a non-pending status (its body) or the timeout (None).

        Polls once straight away, then every interval. Nothing is accepted
        after the deadline: each request is bounded by the time left, and a
        response that arrives late counts as a timeout.
        """
        sleep = self._sleep or _sleep
        clock = self._clock or _clock
        interval = self._approvals.poll_interval_seconds
        deadline = clock() + self._approvals.timeout_seconds
        headers = {**self._proxy_config.headers(), "Accept": "application/json"}
        url = f"{self._base_url}/api/v1/approvals/{approval_id}/poll"
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(30.0),
            transport=self._http_transport or httpx.AsyncHTTPTransport(),
            follow_redirects=False,
        ) as client:
            while True:
                remaining = deadline - clock()
                if remaining <= 0:
                    return None
                body = None
                with anyio.move_on_after(remaining):
                    body = await self._poll(client, url, approval_id)
                if body is None or clock() >= deadline:
                    return None
                logger.debug("approval %s: %s", approval_id, body["status"])
                if body["status"] != "pending":
                    return body
                remaining = deadline - clock()
                if remaining <= 0:
                    return None
                await sleep(min(interval, remaining))

    async def _poll(
        self, client: httpx.AsyncClient, url: str, approval_id: str
    ) -> dict[str, Any]:
        # Only httpx and decoding errors are caught: cancellation (the client
        # cancelled the held request) propagates and stops the polling.
        try:
            response = await client.get(url)
        except httpx.TransportError as exc:
            raise _PollFailed(f"AgentSentinel is unreachable ({type(exc).__name__})") from None
        except httpx.HTTPError as exc:
            raise _PollFailed(f"AgentSentinel request failed ({type(exc).__name__})") from None

        status = response.status_code
        if status != 200:
            reason_code = _reason_code(response.content)
            detail = f"HTTP {status}, {reason_code}" if reason_code else f"HTTP {status}"
            if status in (401, 403):
                raise _PollFailed(f"AgentSentinel rejected the API key ({detail})")
            raise _PollFailed(f"AgentSentinel request failed ({detail})")

        try:
            body = response.json()
        except ValueError:
            raise _PollFailed(_INVALID_RESPONSE) from None
        if not isinstance(body, dict):
            raise _PollFailed(_INVALID_RESPONSE)
        try:
            same_id = isinstance(body.get("id"), str) and str(uuid.UUID(body["id"])) == approval_id
        except ValueError:
            same_id = False
        status_value = body.get("status")
        if not same_id or not isinstance(status_value, str) or status_value not in _KNOWN_STATUSES:
            raise _PollFailed(_INVALID_RESPONSE)
        for name in ("decision_notes", "info_request_message", "decided_by_email"):
            if body.get(name) is not None and not isinstance(body[name], str):
                raise _PollFailed(_INVALID_RESPONSE)
        # Allow-list: only these fields are used. decided_by_email is kept only
        # to scrub it from the approver's text; it is never shown.
        return {
            "status": status_value,
            "decision_notes": body.get("decision_notes"),
            "info_request_message": body.get("info_request_message"),
            "decided_by_email": body.get("decided_by_email"),
        }
