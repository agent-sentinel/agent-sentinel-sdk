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
from typing import Optional


@dataclass(frozen=True)
class _ContextData:
    """Immutable snapshot of execution context values."""
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    task_id: Optional[str] = None
    mission_id: Optional[str] = None


_current_context: contextvars.ContextVar[Optional[_ContextData]] = contextvars.ContextVar(
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
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        mission_id: Optional[str] = None,
    ):
        self._data = _ContextData(
            agent_id=agent_id,
            run_id=run_id,
            task_id=task_id,
            mission_id=mission_id,
        )
        self._token: Optional[contextvars.Token] = None

    # -- Sync context manager --

    def __enter__(self) -> "ExecutionContext":
        self._token = _current_context.set(self._data)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None

    # -- Async context manager --

    async def __aenter__(self) -> "ExecutionContext":
        return self.__enter__()

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.__exit__(_exc_type, _exc_val, _exc_tb)

    # -- Static accessors --

    @staticmethod
    def current() -> Optional[_ContextData]:
        """Return the current execution context, or None if not set."""
        return _current_context.get()

    @staticmethod
    def get_agent_id() -> Optional[str]:
        """Return the current agent_id, or None."""
        ctx = _current_context.get()
        return ctx.agent_id if ctx else None

    @staticmethod
    def get_run_id() -> Optional[str]:
        """Return the current run_id, or None."""
        ctx = _current_context.get()
        return ctx.run_id if ctx else None

    @staticmethod
    def get_task_id() -> Optional[str]:
        """Return the current task_id, or None."""
        ctx = _current_context.get()
        return ctx.task_id if ctx else None

    @staticmethod
    def get_mission_id() -> Optional[str]:
        """Return the current mission_id, or None."""
        ctx = _current_context.get()
        return ctx.mission_id if ctx else None
