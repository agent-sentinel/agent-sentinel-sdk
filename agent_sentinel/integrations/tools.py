"""
Generic tool decorator for AgentSentinel.

Provides @sentinel_tool as a convenience alias for @guarded_action
with tool-specific framing and discovery attributes.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..guard import guarded_action


def sentinel_tool(
    name: Optional[str] = None,
    cost_usd: float = 0.0,
    tags: Optional[list[str]] = None,
    produces_evidence: bool = False,
    is_commit: bool = False,
    requires: Optional[list[str]] = None,
    argument_constraints: Optional[dict] = None,
    evidence_max_age_seconds: Optional[int] = None,
    grounding_rules: Optional[dict] = None,
):
    """
    Decorator to register a function as a sentinel-guarded tool.

    This is a convenience alias for @guarded_action with tool-specific
    framing. Decorated functions are discoverable by auto_register_tools().

    Usage:
        @sentinel_tool(produces_evidence=True)
        def lookup_order(order_id: str) -> dict:
            return {"order_id": order_id, "status": "found"}

        @sentinel_tool(is_commit=True, requires=["lookup_order"])
        def issue_refund(order_id: str, amount: float) -> dict:
            return {"refunded": True}
    """
    config = {
        "name": name,
        "cost_usd": cost_usd,
        "tags": tags,
        "produces_evidence": produces_evidence,
        "is_commit": is_commit,
        "requires": requires,
        "argument_constraints": argument_constraints,
        "evidence_max_age_seconds": evidence_max_age_seconds,
        "grounding_rules": grounding_rules,
    }

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Delegate to guarded_action for all enforcement logic
        wrapper = guarded_action(**{k: v for k, v in config.items() if v is not None or k == "name"})(func)
        # Tag for discovery
        wrapper._sentinel_tool_config = config  # type: ignore[attr-defined]
        wrapper._sentinel_tool_original = func  # type: ignore[attr-defined]
        return wrapper

    return decorator
