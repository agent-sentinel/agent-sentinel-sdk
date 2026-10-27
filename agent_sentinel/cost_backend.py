"""
Cost Backend Module: pluggable storage for budget counters.

The original CostTracker kept all state in a process-local class attribute.
That is correct for a single-process agent using the decorator, and wrong for
anything multi-process — a proxy, a worker pool, a CLI invoked per tool call —
where every process starts with a zero balance and budget enforcement silently
stops working.

This module introduces a backend seam so the storage can change without the
CostTracker API changing:

    InMemoryCostBackend  default; identical behaviour to the original
    FileCostBackend      single-host, multi-process, no dependencies
    RemoteCostBackend     t21: reserves against the platform's org/group
                           budgets instead of local counters

InMemoryCostBackend and FileCostBackend import nothing outside the standard
library. RemoteCostBackend needs httpx (optional, like the rest of the SDK's
network calls) and reaches into the rest of the package for enforcement-mode
and identity helpers -- always via a lazy import inside its methods, so this
module stays a leaf the rest of the SDK can import without a cycle.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable

from ._defaults import resolve_api_token, resolve_platform_url

try:  # POSIX
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # Windows
    _HAS_FCNTL = False

try:  # Windows
    import msvcrt

    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger("agent_sentinel.cost_backend")


@dataclass
class CostSnapshot:
    """Point-in-time view of all counters."""

    session_total: float = 0.0
    run_total: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_costs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_total": self.session_total,
            "run_total": self.run_total,
            "action_counts": dict(self.action_counts),
            "action_costs": dict(self.action_costs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CostSnapshot":
        return cls(
            session_total=float(data.get("session_total", 0.0)),
            run_total=float(data.get("run_total", 0.0)),
            action_counts={k: int(v) for k, v in (data.get("action_counts") or {}).items()},
            action_costs={k: float(v) for k, v in (data.get("action_costs") or {}).items()},
        )


@runtime_checkable
class CostBackend(Protocol):
    """
    Storage protocol for cost counters.

    Implementations must make add_cost atomic with respect to concurrent
    callers, because it is the operation budget enforcement depends on.
    """

    def add_cost(self, action: str, cost_usd: float) -> None: ...

    def snapshot(self) -> CostSnapshot: ...

    def reset_run(self) -> None: ...

    def reset_all(self) -> None: ...

    def try_reserve(
        self,
        action: str,
        cost_usd: float,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budget: Optional[float] = None,
    ) -> Optional[BudgetRejection]: ...


@dataclass
class BudgetRejection:
    """Which budget a reservation failed against, and by how much."""

    budget_type: str  # "session" | "run" | "action"
    spent: float      # what the total would have become
    limit: float


def _evaluate_budgets(
    snapshot: CostSnapshot,
    action: str,
    cost_usd: float,
    session_budget: Optional[float],
    run_budget: Optional[float],
    action_budget: Optional[float],
) -> Optional[BudgetRejection]:
    """
    Check a proposed spend against every configured budget.

    Shared by both backends so the comparison semantics cannot drift between
    the in-memory and file implementations.
    """
    cost_usd = _validate_cost(cost_usd)

    if session_budget is not None:
        projected = snapshot.session_total + cost_usd
        if projected > session_budget:
            return BudgetRejection("session", projected, session_budget)

    if run_budget is not None:
        projected = snapshot.run_total + cost_usd
        if projected > run_budget:
            return BudgetRejection("run", projected, run_budget)

    if action_budget is not None:
        projected = snapshot.action_costs.get(action, 0.0) + cost_usd
        if projected > action_budget:
            return BudgetRejection("action", projected, action_budget)

    return None


def _validate_cost(cost_usd: float) -> float:
    """
    Reject a spend that would corrupt every future budget comparison.

    NaN is the dangerous one: every ``>`` against NaN is False, so one NaN
    spend silently and permanently disables every budget on both entry
    points. Negative and infinite values are rejected for the same reason.
    """
    try:
        value = float(cost_usd)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cost_usd must be numeric, got {cost_usd!r}") from exc
    if not math.isfinite(value):
        raise ValueError(
            f"cost_usd must be finite, got {value!r}. A non-finite spend "
            f"would disable every budget comparison permanently."
        )
    if value < 0:
        raise ValueError(f"cost_usd must not be negative, got {value!r}")
    return value


def _apply_cost(snapshot: CostSnapshot, action: str, cost_usd: float) -> None:
    """Increment every counter for one spend. Shared by both backends."""
    cost_usd = _validate_cost(cost_usd)
    snapshot.session_total += cost_usd
    snapshot.run_total += cost_usd
    snapshot.action_counts[action] = snapshot.action_counts.get(action, 0) + 1
    snapshot.action_costs[action] = snapshot.action_costs.get(action, 0.0) + cost_usd


class InMemoryCostBackend:
    """
    Process-local counters guarded by a lock.

    This is the default and reproduces the original CostTracker behaviour
    exactly. Correct for a single-process agent; counters are not visible to
    other processes and are lost on exit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = CostSnapshot()

    def add_cost(self, action: str, cost_usd: float) -> None:
        with self._lock:
            _apply_cost(self._snapshot, action, cost_usd)

    def try_reserve(
        self,
        action: str,
        cost_usd: float,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budget: Optional[float] = None,
    ) -> Optional[BudgetRejection]:
        """
        Check every budget and record the spend, atomically.

        Returns None when the spend was permitted and recorded, or a
        BudgetRejection naming the budget that refused it (in which case
        nothing was recorded).
        """
        with self._lock:
            rejection = _evaluate_budgets(
                self._snapshot, action, cost_usd,
                session_budget, run_budget, action_budget,
            )
            if rejection is not None:
                return rejection
            _apply_cost(self._snapshot, action, cost_usd)
            return None

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot.from_dict(self._snapshot.to_dict())

    def reset_run(self) -> None:
        with self._lock:
            self._snapshot.run_total = 0.0

    def reset_all(self) -> None:
        with self._lock:
            self._snapshot = CostSnapshot()


class FileCostBackend:
    """
    Counters shared across processes on one host via a lock file.

    Every mutation takes an exclusive OS-level lock, re-reads state from disk,
    applies the change, and writes atomically. That read-modify-write under
    lock makes each individual counter update safe: no write is lost when
    concurrent workers increment the same counter.

    IMPORTANT — what this does NOT give you. Budget *enforcement* is currently
    check-then-act across two separate lock acquisitions: PolicyEngine reads
    the total under one lock, the action runs, and the spend is recorded under
    a later lock. Concurrent workers can therefore all observe the same
    pre-spend total and all pass the same budget check, overshooting the cap.
    Atomic enforcement needs a check-and-reserve primitive on this protocol
    (compare and increment inside one _locked_update); until that exists,
    treat multi-process budgets as approximate, not hard caps.

    Suitable for a single-host proxy or worker pool. For enforcement across
    multiple hosts, implement CostBackend against Redis or Postgres — the
    contract is the four methods above.

    Args:
        path: state file location. Parent directories are created.
        timeout: seconds to wait for the lock before raising TimeoutError.
            None waits indefinitely.
    """

    def __init__(
        self, path: "os.PathLike[str] | str", timeout: Optional[float] = 5.0
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._timeout = timeout
        self._thread_lock = threading.Lock()

    # -- locking ---------------------------------------------------------

    def _acquire(self, handle) -> None:
        """
        Take the exclusive lock, honouring self._timeout.

        A wedged holder must not hang every other worker forever, so the
        blocking flock is replaced by a bounded non-blocking retry loop.
        """
        if self._timeout is None:
            # Explicit opt-in to waiting forever.
            if _HAS_FCNTL:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif _HAS_MSVCRT:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return

        deadline = time.monotonic() + self._timeout
        delay = 0.001
        while True:
            try:
                if _HAS_FCNTL:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif _HAS_MSVCRT:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Neither locking API available; the thread lock is the
                    # only guard. Stated rather than silently pretended away.
                    pass
                return
            except OSError:
                if time.monotonic() >= deadline:
                    # The SDK's own TimeoutError, not the builtin. The builtin
                    # is not an AgentSentinelError, so it escapes every handler
                    # in guard.py and tool_executor.py and hard-crashes the
                    # caller's agent instead of degrading to a blocked result.
                    # agent_sentinel.TimeoutError shadows the builtin name, so
                    # a user writing `except agent_sentinel.TimeoutError` would
                    # otherwise catch nothing.
                    from .errors import TimeoutError as SentinelTimeoutError

                    raise SentinelTimeoutError(
                        f"Timed out after {self._timeout}s waiting for the cost "
                        f"state lock at {self._lock_path}. Another process may "
                        f"be wedged while holding it.",
                        timeout_seconds=self._timeout,
                        operation="cost_lock",
                    )
                time.sleep(delay)
                delay = min(delay * 2, 0.05)

    def _release(self, handle) -> None:
        if _HAS_FCNTL:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif _HAS_MSVCRT:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    # -- io --------------------------------------------------------------

    def _read(self) -> CostSnapshot:
        """
        Load counters from disk.

        "File absent" and "file present but unreadable" are different
        situations and only the first legitimately means zero. Serving a zeroed
        snapshot for a corrupt file silently resets every budget — and because
        the next write persists those zeros, permanently. One byte written by
        anything with access to the file would disable enforcement with no
        trace.

        A corrupt file is therefore quarantined (renamed, not deleted — it is
        evidence of what the counters said) and logged at ERROR. The next call
        starts from a clean zero file, so the agent recovers on the following
        action rather than wedging forever.
        """
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                return CostSnapshot.from_dict(json.load(fh))
        except FileNotFoundError:
            # No file yet: genuinely zero.
            return CostSnapshot()
        except (
            json.JSONDecodeError,
            ValueError,
            # Valid JSON of the wrong shape ([1,2,3], null, "str") reaches
            # from_dict and blows up on .get / float().
            AttributeError,
            TypeError,
        ) as exc:
            self._quarantine_corrupt_state(exc)
            return CostSnapshot()

    def _quarantine_corrupt_state(self, exc: BaseException) -> None:
        """Move an unreadable state file aside so it cannot be overwritten."""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        target = self._path.with_suffix(self._path.suffix + f".corrupt.{stamp}")
        try:
            os.replace(self._path, target)
            logger.error(
                "Cost state at %s was unreadable (%s: %s) and has been "
                "quarantined to %s. Budget counters restart from zero; any "
                "spend recorded before the corruption is NOT enforced. "
                "Inspect the quarantined file to see what the counters held.",
                self._path, type(exc).__name__, exc, target,
            )
        except OSError as move_error:
            logger.error(
                "Cost state at %s is unreadable (%s: %s) and could not be "
                "quarantined (%s). Budget counters restart from zero and "
                "enforcement is degraded until the file is repaired.",
                self._path, type(exc).__name__, exc, move_error,
            )

    def _write(self, snapshot: CostSnapshot) -> None:
        # Write to a temp file in the same directory, then rename. Rename is
        # atomic on POSIX and on Windows via os.replace, so a reader never sees
        # a half-written file.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=self._path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(snapshot.to_dict(), fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _locked_update(self, mutate) -> None:
        with self._thread_lock:
            with self._lock_path.open("a+") as lock_handle:
                self._acquire(lock_handle)
                try:
                    snapshot = self._read()
                    mutate(snapshot)
                    self._write(snapshot)
                finally:
                    self._release(lock_handle)

    # -- CostBackend -----------------------------------------------------

    def add_cost(self, action: str, cost_usd: float) -> None:
        self._locked_update(lambda snapshot: _apply_cost(snapshot, action, cost_usd))

    def try_reserve(
        self,
        action: str,
        cost_usd: float,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budget: Optional[float] = None,
    ) -> Optional[BudgetRejection]:
        """
        Check every budget and record the spend inside a single lock hold.

        This is what makes a multi-process budget a real cap. Reading the
        total under one lock and incrementing under a later one lets N workers
        all observe the same pre-spend total and all pass the same check,
        overshooting the limit by up to (workers x cost). Comparing and
        incrementing without releasing the lock in between removes that window.

        Returns None when the spend was permitted and recorded, or a
        BudgetRejection naming the refusing budget (nothing recorded).
        """
        outcome: list[Optional[BudgetRejection]] = [None]

        def mutate(snapshot: CostSnapshot) -> None:
            rejection = _evaluate_budgets(
                snapshot, action, cost_usd,
                session_budget, run_budget, action_budget,
            )
            outcome[0] = rejection
            if rejection is None:
                _apply_cost(snapshot, action, cost_usd)

        self._locked_update(mutate)
        return outcome[0]

    def snapshot(self) -> CostSnapshot:
        with self._thread_lock:
            with self._lock_path.open("a+") as lock_handle:
                self._acquire(lock_handle)
                try:
                    return self._read()
                finally:
                    self._release(lock_handle)

    def reset_run(self) -> None:
        self._locked_update(lambda snapshot: setattr(snapshot, "run_total", 0.0))

    def reset_all(self) -> None:
        def mutate(snapshot: CostSnapshot) -> None:
            snapshot.session_total = 0.0
            snapshot.run_total = 0.0
            snapshot.action_counts.clear()
            snapshot.action_costs.clear()

        self._locked_update(mutate)


class RemoteCostBackend:
    """
    Reserves budget against the platform instead of tracking counters locally
    (t21): every ``try_reserve`` is a ``POST /api/v1/budgets/reserve`` call,
    checked atomically server-side against the caller's organization monthly
    budget and any ``GroupBudget`` matching the active group
    (``CostTracker.get_group()``). The reservation is held for 15 minutes and
    reconciled when the real ``cost_usd`` reaches ingest.

    Enforcement moves server-side, so ``session_budget``/``run_budget``/
    ``action_budget`` are accepted for ``CostBackend`` protocol compatibility
    but not evaluated locally -- the platform is authoritative. ``add_cost``/
    ``snapshot``/``reset_run``/``reset_all`` operate on a local, in-memory
    mirror purely so ``CostTracker.get_session_total()`` and friends have
    something to report; that mirror is not what enforcement reads.

    Network failure follows the caller's enforcement mode (t11), the same
    fail-open/fail-closed contract as ``KillSwitchClient``: "fail_open"
    (the default) permits the reservation -- the spend is recorded on the
    local mirror and the action proceeds, since nothing was held
    server-side to settle later; "fail_closed" raises so the caller blocks.
    The mode is read from ``PolicyEngine.get_enforcement_mode()`` at call
    time, since the ``CostBackend`` protocol has no way to thread a per-call
    ``fail_closed`` override down to the backend -- a per-call override
    passed to ``PolicyEngine.check_and_reserve(..., fail_closed=...)`` is
    therefore not seen by a network-failure decision made here.

    Opt-in via ``AGENT_SENTINEL_COST_BACKEND=remote`` (api_key read from
    ``AGENT_SENTINEL_API_KEY`` by ``backend_from_env()``), or construct
    directly::

        CostTracker.set_backend(RemoteCostBackend(api_key="sk_live_..."))

    ``platform_url`` defaults to the hosted platform;
    ``AGENT_SENTINEL_PLATFORM_URL`` / ``AGENT_SENTINEL_URL`` override it
    (self-hosted or staging). ``api_key`` defaults to
    ``AGENT_SENTINEL_API_KEY``.
    """

    def __init__(
        self,
        platform_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self._platform_url = resolve_platform_url(platform_url)
        self._api_key = resolve_api_token(api_key)
        self._timeout = timeout
        self._local = InMemoryCostBackend()

    def add_cost(self, action: str, cost_usd: float) -> None:
        self._local.add_cost(action, cost_usd)

    def snapshot(self) -> CostSnapshot:
        return self._local.snapshot()

    def reset_run(self) -> None:
        self._local.reset_run()

    def reset_all(self) -> None:
        self._local.reset_all()

    def try_reserve(
        self,
        action: str,
        cost_usd: float,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budget: Optional[float] = None,
    ) -> Optional[BudgetRejection]:
        """
        Reserve ``cost_usd`` against the platform's org/group budgets.

        Returns None when the platform accepted the reservation (recorded on
        the local mirror too), or a BudgetRejection naming the org or group
        budget that refused it. A reservation that cannot be completed at all
        (network error, timeout, unexpected status) is resolved by the
        current enforcement mode -- see the class docstring.
        """
        cost_usd = _validate_cost(cost_usd)

        if not HTTPX_AVAILABLE:
            return self._handle_unreachable(
                action, cost_usd, RuntimeError("httpx is not installed")
            )

        from .context import ExecutionContext, build_identity_headers
        from .cost import CostTracker

        url = f"{self._platform_url}/api/v1/budgets/reserve"
        headers = build_identity_headers(self._api_key)
        headers["Content-Type"] = "application/json"
        payload: Dict[str, object] = {
            "action_name": action,
            "estimated_cost_usd": cost_usd,
        }
        group_id = CostTracker.get_group()
        if group_id:
            payload["group_id"] = group_id
        run_id = ExecutionContext.get_run_id()
        if run_id:
            payload["run_id"] = run_id

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
            return self._handle_unreachable(action, cost_usd, exc)

        if response.status_code == 201:
            self._local.add_cost(action, cost_usd)
            return None

        if response.status_code == 402:
            detail: Dict[str, object] = {}
            try:
                detail = response.json().get("detail") or {}
            except Exception:  # noqa: BLE001 - malformed body, fall back below
                detail = {}
            return BudgetRejection(
                budget_type=str(detail.get("budget_type", "organization_monthly")),
                spent=float(detail.get("spent", cost_usd)),
                limit=float(detail.get("limit", 0.0)),
            )

        return self._handle_unreachable(
            action,
            cost_usd,
            RuntimeError(f"unexpected status {response.status_code} from {url}"),
        )

    def _handle_unreachable(
        self, action: str, cost_usd: float, exc: Exception
    ) -> Optional[BudgetRejection]:
        """
        Resolve a reservation call that could not be completed at all.

        Mirrors KillSwitchClient.is_killed: fail_open logs and permits (the
        spend is recorded on the local mirror so totals stay consistent even
        though nothing was held server-side); fail_closed raises NetworkError
        so PolicyEngine.check_and_reserve blocks the action.
        """
        from .errors import NetworkError
        from .policy import PolicyEngine

        if PolicyEngine.get_enforcement_mode() == "fail_closed":
            raise NetworkError(
                f"Budget reservation for '{action}' could not be completed "
                f"against the platform ({exc}), and enforcement mode is "
                f"'fail_closed'.",
                endpoint="/api/v1/budgets/reserve",
            )

        logger.warning(
            "Budget reservation for '%s' failed (%s); failing open and "
            "permitting the action (enforcement mode 'fail_open').",
            action, exc,
        )
        self._local.add_cost(action, cost_usd)
        return None


def backend_from_env() -> Optional[CostBackend]:
    """
    Build a backend from environment variables, or None to keep the default.

    AGENT_SENTINEL_COST_BACKEND    "memory" (default), "file", or "remote"
    AGENT_SENTINEL_COST_PATH       state file path when backend is "file"
    AGENT_SENTINEL_API_KEY         platform API key when backend is "remote"
    AGENT_SENTINEL_PLATFORM_URL /  optional override of the hosted platform
    AGENT_SENTINEL_URL             URL (self-hosted or staging)

    Returns:
        A CostBackend, or None if no backend is configured via the environment.

    Raises:
        ValueError: if the backend name is unrecognised, "file" is selected
            without a path, or "remote" is selected without an API key. Failing loudly here is deliberate: a typo that
            silently fell back to per-process memory would disable budget
            enforcement across a worker pool without any signal.
    """
    name = os.environ.get("AGENT_SENTINEL_COST_BACKEND", "").strip().lower()
    if not name or name == "memory":
        return None

    if name == "file":
        path = os.environ.get("AGENT_SENTINEL_COST_PATH", "").strip()
        if not path:
            raise ValueError(
                "AGENT_SENTINEL_COST_BACKEND=file requires "
                "AGENT_SENTINEL_COST_PATH to be set."
            )
        return FileCostBackend(path)

    if name == "remote":
        api_key = os.environ.get("AGENT_SENTINEL_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "AGENT_SENTINEL_COST_BACKEND=remote requires "
                "AGENT_SENTINEL_API_KEY to be set."
            )
        return RemoteCostBackend(api_key=api_key)

    raise ValueError(
        f"Unknown AGENT_SENTINEL_COST_BACKEND: {name!r}. "
        "Expected 'memory', 'file', or 'remote'."
    )
