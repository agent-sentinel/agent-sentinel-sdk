"""
Execution Context: Runtime attribution for agent actions.

Provides a context manager that sets agent_id, task_id, and mission_id
for all guarded_action calls within its scope. Uses contextvars for
thread-safe, async-safe propagation.

Use cases:
- Workflow activities where context is known at runtime, not import time
- Serverless handlers with per-invocation attribution
- Request-scoped web handlers
- Any runtime where decorator params are insufficient
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class _ContextData:
    """Immutable snapshot of execution context values."""
    agent_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    mission_id: str | None = None
    harness: str | None = None
    principal: str | None = None


_current_context: contextvars.ContextVar[_ContextData | None] = contextvars.ContextVar(
    "agent_sentinel_context", default=None
)


class ExecutionContext:
    """
    Context manager that sets agent_id, task_id, and mission_id for
    all guarded_action calls within its scope.

    Values are automatically cleaned up on exit, even if an exception
    is raised. Nested contexts are supported -- inner values override
    outer values for the duration of the inner block.

    Usage::

        with ExecutionContext(agent_id="writer-1", task_id="task-42", mission_id="m-7"):
            # all guarded_action calls here inherit these values
            await my_tool()

        # values are cleaned up here

    Can also be used in async contexts::

        async with ExecutionContext(agent_id="reader-2"):
            await some_guarded_action()
    """

    def __init__(
        self,
        agent_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        mission_id: str | None = None,
        harness: str | None = None,
        principal: str | None = None,
    ):
        self._data = _ContextData(
            agent_id=agent_id,
            run_id=run_id,
            task_id=task_id,
            mission_id=mission_id,
            harness=harness,
            principal=principal,
        )
        self._token: contextvars.Token | None = None

    # -- Sync context manager --

    def __enter__(self) -> ExecutionContext:
        self._token = _current_context.set(self._data)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None

    # -- Async context manager --

    async def __aenter__(self) -> ExecutionContext:
        return self.__enter__()

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.__exit__(_exc_type, _exc_val, _exc_tb)

    # -- Static accessors --

    @staticmethod
    def current() -> _ContextData | None:
        """Return the current execution context, or None if not set."""
        return _current_context.get()

    @staticmethod
    def get_agent_id() -> str | None:
        """Return the current agent_id, or None."""
        ctx = _current_context.get()
        return ctx.agent_id if ctx else None

    @staticmethod
    def get_run_id() -> str | None:
        """Return the current run_id, or None."""
        ctx = _current_context.get()
        return ctx.run_id if ctx else None

    @staticmethod
    def get_task_id() -> str | None:
        """Return the current task_id, or None."""
        ctx = _current_context.get()
        return ctx.task_id if ctx else None

    @staticmethod
    def get_mission_id() -> str | None:
        """Return the current mission_id, or None."""
        ctx = _current_context.get()
        return ctx.mission_id if ctx else None

    @staticmethod
    def get_harness() -> str | None:
        """Return the current harness, or None."""
        ctx = _current_context.get()
        return ctx.harness if ctx else None

    @staticmethod
    def get_principal() -> str | None:
        """Return the current principal, or None."""
        ctx = _current_context.get()
        return ctx.principal if ctx else None


def build_identity_headers(
    api_token: str,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    """
    Build the standard auth + identity headers sent with every platform request.

    Every SDK HTTP call authenticates the same way (``Authorization: Bearer
    <api_token>``) and identifies its caller via ``X-Sentinel-*`` headers so
    the platform can bind identity to the key instead of trusting body
    strings. ``agent_id``/``run_id`` arguments let a caller override the
    current ``ExecutionContext`` (e.g. a sync batch grouped under a specific
    run); ``harness``/``principal`` always come from the context, since no
    call site threads them through explicitly. Headers with no resolved
    value are omitted rather than sent empty.

    Note: imports ``__version__`` lazily -- ``agent_sentinel/__init__.py``
    imports this module before it finishes defining ``__version__``, so a
    top-level import here would be circular.
    """
    from . import __version__

    ctx = ExecutionContext.current()
    resolved_agent_id = agent_id or (ctx.agent_id if ctx else None)
    resolved_run_id = run_id or (ctx.run_id if ctx else None)
    harness = ctx.harness if ctx else None
    principal = ctx.principal if ctx else None

    headers = {
        "Authorization": f"Bearer {api_token}",
        "User-Agent": f"agentsentinel-sdk/{__version__}",
    }
    if resolved_agent_id:
        headers["X-Sentinel-Agent-Id"] = resolved_agent_id
    if resolved_run_id:
        headers["X-Sentinel-Run-Id"] = resolved_run_id
    if harness:
        headers["X-Sentinel-Harness"] = harness
    if principal:
        headers["X-Sentinel-Principal"] = principal
    return headers
