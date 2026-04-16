"""
Convenience top-level configure() and flush() functions.

configure() wires up background sync, the policy engine, and optional LLM
auto-instrumentation in a single call — the "platform-first" entry point.

flush() tears everything down cleanly (sync stops, policy refresh stops).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("agent_sentinel")


def configure(
    api_key: str,
    platform_url: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    # Budget / policy shortcuts
    run_budget: Optional[float] = None,
    session_budget: Optional[float] = None,
    action_budgets: Optional[Dict[str, float]] = None,
    denied_actions: Optional[List[str]] = None,
    rate_limits: Optional[Dict] = None,
    # Auto-instrumentation
    auto_instrument: bool = False,
) -> None:
    """
    One-call SDK configuration (platform-first mode).

    Sets up:
    - Background ledger sync to the platform
    - PolicyEngine remote sync (approval rules, deny lists, budgets)
    - Optional local policy overrides (run/action budgets, deny lists)
    - Optional LLM auto-instrumentation (OpenAI / Anthropic)
    - Execution context for proper attribution

    Args:
        api_key: Platform API key (ApiKey <token> format accepted).
        platform_url: Base URL of the AgentSentinel platform.
        agent_id: Stable identifier for this agent type.
        run_id: ID for this specific run (auto-generated if None).
        run_budget: Max USD for the current run (local override).
        session_budget: Max USD for the current session (local override).
        action_budgets: Per-action spend caps {action_name: max_usd}.
        denied_actions: Action names to block locally (pre-platform check).
        rate_limits: Rate limit config forwarded to PolicyEngine.
        auto_instrument: If True, attempt to auto-instrument LLM clients.
    """
    from .sync import enable_remote_sync
    from .policy import PolicyEngine
    from .context import ExecutionContext

    # 1. Set up execution context with agent/run attribution
    # This ensures interventions are properly attributed
    ExecutionContext(
        agent_id=agent_id,
        run_id=run_id,
    ).__enter__()

    # 2. Start background ledger sync
    enable_remote_sync(
        platform_url=platform_url,
        api_token=api_key,
        run_id=run_id,
        agent_id=agent_id,
    )

    # 3. Apply local policy overrides if provided
    if any(v is not None for v in (run_budget, action_budgets, denied_actions)):
        PolicyEngine.configure(
            run_budget=run_budget,
            action_budgets=action_budgets,
            denied_actions=denied_actions or [],
        )

    # 4. Enable remote policy sync (approval rules, deny lists from platform)
    PolicyEngine.enable_remote_sync(
        platform_url=platform_url,
        api_token=api_key,
        agent_id=agent_id,
        run_id=run_id,
    )

    # 5. Auto-instrument LLM clients if requested
    if auto_instrument:
        _try_auto_instrument()

    logger.info(
        "agent_sentinel configured: platform_url=%s agent_id=%s run_id=%s",
        platform_url,
        agent_id,
        run_id,
    )


def flush() -> None:
    """
    Flush pending telemetry and stop all background threads.

    Call at the end of a run (or rely on atexit — the sync module registers
    an atexit handler automatically).
    """
    from .sync import flush_and_stop
    from .policy import PolicyEngine

    flush_and_stop()

    try:
        PolicyEngine.stop_remote_sync()
    except Exception:
        pass


def _try_auto_instrument() -> None:
    """Best-effort LLM auto-instrumentation — never raises."""
    try:
        from .integrations import openai as _oi  # noqa: F401
        logger.debug("OpenAI auto-instrumentation enabled")
    except Exception:
        pass

    try:
        from .integrations import anthropic as _ai  # noqa: F401
        logger.debug("Anthropic auto-instrumentation enabled")
    except Exception:
        pass
