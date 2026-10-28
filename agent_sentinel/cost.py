"""
Cost Tracking Module: counters for budget enforcement.

Phase 2 Implementation:
- Thread-safe cost accumulation
- Run-level and session-level tracking
- Support for nested contexts (runs within sessions)
- Simple reset mechanism for testing

Storage is pluggable (see cost_backend.py). The default is process-local
memory, which is correct for a single-process agent using @guarded_action.
Any multi-process deployment — a proxy, a worker pool, a per-call CLI — needs
shared state, or each process starts with a zero balance and budget
enforcement silently stops working:

    from agent_sentinel.cost import CostTracker
    from agent_sentinel.cost_backend import FileCostBackend

    CostTracker.set_backend(FileCostBackend("/var/lib/sentinel/costs.json"))

or via the environment, which is the usual path for a container:

    AGENT_SENTINEL_COST_BACKEND=file
    AGENT_SENTINEL_COST_PATH=/var/lib/sentinel/costs.json
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .cost_backend import (
    CostBackend,
    CostSnapshot,
    InMemoryCostBackend,
    backend_from_env,
)

logger = logging.getLogger("agent_sentinel.cost")


@dataclass
class CostCounter:
    """
    Accumulates costs at different scopes.

    Scopes:
    - session: Entire lifetime of the application
    - run: A single agent execution (can be reset between runs)
    - action: Individual action costs (tracked in ledger)

    Retained for backwards compatibility. Live state is held by the configured
    CostBackend; this remains importable for callers that type against it.
    """
    session_total: float = 0.0
    run_total: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    action_costs: dict[str, float] = field(default_factory=dict)


class CostTracker:
    """
    Thread-safe accessor for cost counters across the SDK.

    Usage:
        # Record a cost
        CostTracker.add_cost("search_web", 0.02)

        # Check current run total
        total = CostTracker.get_run_total()

        # Reset for new run
        CostTracker.reset_run()

    Storage is delegated to a CostBackend. The default backend is
    process-local; call set_backend() or set the AGENT_SENTINEL_COST_BACKEND
    environment variable to share counters across processes.
    """

    _backend: CostBackend | None = None
    _env_backend_checked: bool = False
    # Group/mission attribution (upstream). Independent of the storage
    # backend: it labels where costs came from, it does not hold counters.
    _group_id: str | None = None

    # -- backend management ----------------------------------------------

    @classmethod
    def _get_backend(cls) -> CostBackend:
        """
        Resolve the active backend, consulting the environment once.

        An invalid environment configuration is logged and falls back to memory
        rather than raising, so a bad env var cannot crash a user's agent at
        first spend. The log is at ERROR because budget enforcement is
        degraded when this happens.
        """
        if cls._backend is None and not cls._env_backend_checked:
            cls._env_backend_checked = True
            try:
                env_backend = backend_from_env()
                if env_backend is not None:
                    cls._backend = env_backend
                    logger.info(
                        "CostTracker using backend from environment: %s",
                        type(env_backend).__name__,
                    )
            except ValueError as exc:
                logger.error(
                    "Invalid cost backend configuration, falling back to "
                    "process-local memory. Budget enforcement will NOT be "
                    "shared across processes. %s",
                    exc,
                )

        if cls._backend is None:
            cls._backend = InMemoryCostBackend()

        return cls._backend

    @classmethod
    def set_backend(cls, backend: CostBackend | None) -> None:
        """
        Install a cost backend.

        Args:
            backend: Any object implementing the CostBackend protocol.
                Passing None restores the default process-local backend and
                re-enables environment detection.
        """
        cls._backend = backend
        if backend is None:
            cls._env_backend_checked = False

    @classmethod
    def get_backend(cls) -> CostBackend:
        """Return the active backend, resolving the default if unset."""
        return cls._get_backend()

    # -- counters ---------------------------------------------------------

    @classmethod
    def add_cost(cls, action: str, cost_usd: float) -> None:
        """
        Add a cost to both session and run totals.

        Thread-safe. With a shared backend, also safe across processes.

        Args:
            action: Name of the action being costed
            cost_usd: Cost in USD to add
        """
        cls._get_backend().add_cost(action, cost_usd)

    @classmethod
    def try_reserve(
        cls,
        action: str,
        cost_usd: float,
        session_budget: float | None = None,
        run_budget: float | None = None,
        action_budget: float | None = None,
    ):
        """
        Atomically check a spend against every budget and record it.

        Prefer this over ``get_*_total()`` followed by ``add_cost()`` whenever
        the result gates execution: the two-call form is check-then-act, and
        concurrent workers can all pass the check before any of them records,
        overshooting the cap.

        Backends predating this method fall back to the non-atomic path so a
        third-party CostBackend keeps working.

        Returns:
            None if permitted and recorded, else a BudgetRejection naming the
            budget that refused it.
        """
        backend = cls._get_backend()

        reserve = getattr(backend, "try_reserve", None)
        if callable(reserve):
            return reserve(
                action, cost_usd, session_budget, run_budget, action_budget
            )

        # Fallback for a custom backend that predates the protocol addition.
        # Non-atomic by construction; logged so the weaker guarantee is visible.
        from .cost_backend import _evaluate_budgets

        logger.debug(
            "CostBackend %s has no try_reserve; falling back to non-atomic "
            "check-then-act. Concurrent budget enforcement is approximate.",
            type(backend).__name__,
        )
        rejection = _evaluate_budgets(
            backend.snapshot(), action, cost_usd,
            session_budget, run_budget, action_budget,
        )
        if rejection is None:
            backend.add_cost(action, cost_usd)
        return rejection

    @classmethod
    def get_session_total(cls) -> float:
        """
        Get total cost for the entire session (application lifetime).

        Returns:
            Total cost in USD since application started
        """
        return cls._get_backend().snapshot().session_total

    @classmethod
    def get_run_total(cls) -> float:
        """
        Get total cost for the current run.

        A "run" is typically one agent execution. Call reset_run() between runs.

        Returns:
            Total cost in USD for current run
        """
        return cls._get_backend().snapshot().run_total

    @classmethod
    def get_action_stats(cls, action: str | None = None) -> dict:
        """
        Get statistics for actions.

        Args:
            action: If specified, returns stats for that action only.
                   If None, returns all action stats.

        Returns:
            Dict with 'count' and 'total_cost' for the action(s)
        """
        snapshot = cls._get_backend().snapshot()
        if action:
            return {
                "count": snapshot.action_counts.get(action, 0),
                "total_cost": snapshot.action_costs.get(action, 0.0),
            }
        return {
            "counts": dict(snapshot.action_counts),
            "costs": dict(snapshot.action_costs),
        }

    @classmethod
    def reset_run(cls) -> None:
        """
        Reset the run-level counter.

        Call this at the start of each new agent run.
        Does NOT reset session total or action statistics.
        """
        cls._get_backend().reset_run()

    @classmethod
    def reset_all(cls) -> None:
        """
        Reset all counters.

        Useful for testing. In production, you typically only reset_run()
        between executions.
        """
        cls._get_backend().reset_all()

    @classmethod
    def set_group(cls, group_id: str) -> None:
        """Set the active group/mission ID for cost attribution.

        Propagates to all subsequent ingest payloads as mission_id.

        Temporal activity pattern::

            CostTracker.set_group(workflow.info().workflow_id)
        """
        cls._group_id = group_id

    @classmethod
    def get_group(cls) -> str | None:
        """Return the current group/mission ID, or None if not set."""
        return cls._group_id

    @classmethod
    def get_snapshot(cls) -> dict:
        """
        Get a complete snapshot of all cost data.

        Returns:
            Dict with session_total, run_total, group_id, and action statistics
        """
        snapshot = cls._get_backend().snapshot().to_dict()
        snapshot["group_id"] = cls._group_id
        return snapshot


__all__ = ["CostCounter", "CostTracker", "CostSnapshot"]
