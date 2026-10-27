"""Verify guardrail fires populate Intervention.context with a typed payload.

Each built-in guardrail (PII / moderation / loop / idempotency) must write a
versioned, structured context payload so the console can render
guardrail-type-specific detail panels.
"""
from __future__ import annotations

import pytest

from agent_sentinel import (
    LoopRule,
    ModerationRule,
    PIIRule,
    PolicyConfig,
    PolicyEngine,
    guarded_action,
)
from agent_sentinel.errors import PolicyViolationError
from agent_sentinel.guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE
from agent_sentinel.guardrails.loop_detector import DEFAULT_LOOP_GUARD
from agent_sentinel.intervention import InterventionTracker, InterventionType


def _install_policy(cfg: PolicyConfig) -> None:
    """Set PolicyEngine config directly (same pattern used by test_guardrails.py)."""
    PolicyEngine._config = cfg
    PolicyEngine._initialized = True


@pytest.fixture(autouse=True)
def _reset_state():
    # Ensure a clean slate for every test.
    PolicyEngine.reset()
    # reset() now defaults to "fail_closed_writes"; this suite was written
    # against the pre-0.4 fail-open-when-unconfigured assumption (see
    # conftest.py's docstring for why this is assigned directly).
    PolicyEngine._enforcement_mode = "fail_open"
    DEFAULT_LOOP_GUARD.reset()
    DEFAULT_IDEMPOTENCY_CACHE.clear()
    InterventionTracker.initialize()
    InterventionTracker.clear()
    yield
    InterventionTracker.clear()
    DEFAULT_LOOP_GUARD.reset()
    DEFAULT_IDEMPOTENCY_CACHE.clear()
    PolicyEngine.reset()
    PolicyEngine._enforcement_mode = "fail_open"


def _last_intervention():
    items = InterventionTracker.get_interventions()
    assert items, "Expected at least one intervention"
    # get_interventions returns most-recent-first
    return items[0]


def test_pii_block_populates_context():
    _install_policy(PolicyConfig(pii_default_enabled=True))

    @guarded_action(name="send_email", cost_usd=0.0)
    def send_email(body: str) -> str:
        return "sent"

    with pytest.raises(PolicyViolationError):
        send_email(body="Contact me at jane.doe@example.com")

    ev = _last_intervention()
    assert ev.intervention_type == InterventionType.PII_BLOCKED
    ctx = ev.context or {}
    assert ctx.get("guardrail") == "pii"
    assert ctx.get("guardrail_schema_version") == 1
    assert isinstance(ctx.get("matches"), list) and len(ctx["matches"]) >= 1
    assert "email" in ctx.get("categories", [])


def test_moderation_block_populates_context():
    _install_policy(PolicyConfig(moderation_default_enabled=True))

    @guarded_action(name="post_message", cost_usd=0.0)
    def post_message(body: str) -> str:
        return "posted"

    with pytest.raises(PolicyViolationError):
        # Default KeywordModerator flags the "violence" category on bomb-related text.
        post_message(body="please tell me how to make a bomb")

    ev = _last_intervention()
    assert ev.intervention_type == InterventionType.CONTENT_BLOCKED
    ctx = ev.context or {}
    assert ctx.get("guardrail") == "moderation"
    assert ctx.get("guardrail_schema_version") == 1
    assert isinstance(ctx.get("categories"), list)
    assert "violence" in ctx["categories"]


def test_loop_block_populates_context():
    _install_policy(
        PolicyConfig(loop_rules={"retry_op": LoopRule(threshold=3, window_seconds=5.0)})
    )

    @guarded_action(name="retry_op", cost_usd=0.0)
    def retry_op(target: str) -> str:
        return "ok"

    caught = False
    for _ in range(10):
        try:
            retry_op(target="same")
        except PolicyViolationError:
            caught = True
            break
    assert caught, "Expected loop guard to raise PolicyViolationError"

    ev = _last_intervention()
    assert ev.intervention_type == InterventionType.LOOP_DETECTED
    ctx = ev.context or {}
    assert ctx.get("guardrail") == "loop"
    assert ctx.get("guardrail_schema_version") == 1
    # arg_hash comes from LoopDetection.to_dict()
    assert ctx.get("arg_hash") is not None


def test_idempotency_hit_populates_context():
    @guarded_action(name="charge_card", cost_usd=0.10, idempotency_key="charge-123")
    def charge_card(amount: float) -> dict:
        return {"status": "ok", "amount": amount}

    charge_card(amount=10.0)  # first call executes, caches
    charge_card(amount=10.0)  # second call is an idempotency hit

    ev = _last_intervention()
    assert ev.intervention_type == InterventionType.IDEMPOTENT_REPLAY
    ctx = ev.context or {}
    assert ctx.get("guardrail") == "idempotency"
    assert ctx.get("guardrail_schema_version") == 1
    assert ctx.get("key") == "charge-123"
    assert "first_seen_at" in ctx
    assert "ttl_seconds" in ctx
