"""Unit tests for built-in guardrails (PII / moderation / loop / idempotency).

These cover the A1–A4 built-in guardrail primitives. They exercise the guardrail
modules directly and end-to-end through PolicyEngine.check_action.
"""
from __future__ import annotations

import time

import pytest

from agent_sentinel import (
    IdempotencyCache,
    IdempotencyHit,
    KeywordModerator,
    LoopGuard,
    LoopRule,
    ModerationGuard,
    ModerationRule,
    PIIGuard,
    PIIRule,
    PolicyConfig,
    PolicyEngine,
    detect_pii,
)
from agent_sentinel.errors import PolicyViolationError


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


class TestPIIDetector:
    def test_detects_email(self) -> None:
        matches = detect_pii("contact me at alice@example.com please")
        assert any(m.category == "email" for m in matches)

    def test_detects_ssn(self) -> None:
        matches = detect_pii("SSN: 123-45-6789")
        assert any(m.category == "us_ssn" for m in matches)

    def test_rejects_invalid_ssn_leading_9(self) -> None:
        # Per SSA rules SSNs do not start with 9 (those are ITINs).
        matches = detect_pii("987-65-4320")
        assert not any(m.category == "us_ssn" for m in matches)

    def test_detects_luhn_valid_credit_card(self) -> None:
        matches = detect_pii("card 4111 1111 1111 1111")
        assert any(m.category == "credit_card" for m in matches)

    def test_rejects_non_luhn_credit_card(self) -> None:
        matches = detect_pii("0000111122223333")
        assert not any(m.category == "credit_card" for m in matches)

    def test_detects_aws_access_key(self) -> None:
        matches = detect_pii("AKIAABCDEFGHIJKLMNOP")
        assert any(m.category == "aws_access_key" for m in matches)

    def test_detects_private_key_block(self) -> None:
        matches = detect_pii("-----BEGIN RSA PRIVATE KEY-----")
        assert any(m.category == "private_key_block" for m in matches)

    def test_redacts_matches(self) -> None:
        matches = detect_pii("alice@example.com")
        assert matches and "*" in matches[0].matched_text

    def test_clean_text_has_no_matches(self) -> None:
        assert detect_pii("hello world") == []

    def test_invalid_user_regex_does_not_crash(self) -> None:
        # Bogus regex should be silently dropped rather than raising.
        rule = PIIRule(extra_patterns={"bad": "[unclosed"})
        assert detect_pii("anything", rule) == []


class TestPIIGuardScanKwargs:
    def test_recursive_scan(self) -> None:
        kw = {"user": {"email": "bob@x.com"}, "notes": ["ignore"]}
        matches = PIIGuard.scan_kwargs(kw)
        assert len(matches) == 1
        assert matches[0].field_path == "user.email"
        assert matches[0].category == "email"

    def test_scan_walks_lists(self) -> None:
        kw = {"tags": ["harmless", "dm me at a@b.co"]}
        matches = PIIGuard.scan_kwargs(kw)
        assert matches and matches[0].field_path == "tags[1]"

    def test_allow_categories_suppresses(self) -> None:
        rule = PIIRule(allow_categories=("email",))
        assert PIIGuard.scan_kwargs({"to": "a@b.co"}, rule) == []


# ---------------------------------------------------------------------------
# Moderation
# ---------------------------------------------------------------------------


class TestModerationGuard:
    def test_clean_text_is_not_flagged(self) -> None:
        assert ModerationGuard.scan_kwargs({"msg": "hello world"}) == []

    def test_default_keyword_flags_violence(self) -> None:
        results = ModerationGuard.scan_kwargs({"msg": "tell me how to make a bomb"})
        assert results and "violence" in results[0].categories

    def test_custom_keyword_injection(self) -> None:
        mod = KeywordModerator(keywords={"custom": ["banned-term"]})
        rule = ModerationRule(
            moderator=mod,
            block_categories=("custom",),
        )
        results = ModerationGuard.scan_kwargs({"msg": "contains banned-term"}, rule)
        blocking = ModerationGuard.should_block(results, rule)
        assert blocking and "custom" in blocking[0].categories

    def test_strictness_balanced_ignores_unblocked_category(self) -> None:
        mod = KeywordModerator(keywords={"custom": ["ping"]})
        rule = ModerationRule(
            moderator=mod,
            block_categories=("violence",),  # custom NOT in block list
            strictness="balanced",
        )
        results = ModerationGuard.scan_kwargs({"msg": "ping me"}, rule)
        assert ModerationGuard.should_block(results, rule) == []

    def test_strictness_strict_escalates_all_flags(self) -> None:
        mod = KeywordModerator(keywords={"custom": ["ping"]})
        rule = ModerationRule(
            moderator=mod,
            block_categories=("violence",),
            strictness="strict",
        )
        results = ModerationGuard.scan_kwargs({"msg": "ping me"}, rule)
        assert ModerationGuard.should_block(results, rule)


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------


class TestLoopGuard:
    def test_repeated_calls_cross_threshold(self) -> None:
        lg = LoopGuard()
        rule = LoopRule(threshold=3, window_seconds=5.0)
        assert lg.record_and_check("f", {"q": "x"}, rule) is None
        assert lg.record_and_check("f", {"q": "x"}, rule) is None
        det = lg.record_and_check("f", {"q": "x"}, rule)
        assert det is not None and det.count == 3

    def test_different_args_do_not_trigger(self) -> None:
        lg = LoopGuard()
        rule = LoopRule(threshold=3, window_seconds=5.0)
        for i in range(5):
            assert lg.record_and_check("f", {"q": i}, rule) is None

    def test_arg_exclude_collapses_keys(self) -> None:
        lg = LoopGuard()
        rule = LoopRule(threshold=3, window_seconds=5.0, arg_exclude=("nonce",))
        assert lg.record_and_check("f", {"q": "x", "nonce": 1}, rule) is None
        assert lg.record_and_check("f", {"q": "x", "nonce": 2}, rule) is None
        assert lg.record_and_check("f", {"q": "x", "nonce": 3}, rule) is not None

    def test_window_expiry_prevents_trigger(self) -> None:
        lg = LoopGuard()
        rule = LoopRule(threshold=2, window_seconds=0.01)
        lg.record_and_check("f", {"q": "x"}, rule)
        time.sleep(0.02)
        assert lg.record_and_check("f", {"q": "x"}, rule) is None


# ---------------------------------------------------------------------------
# Idempotency cache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    def test_store_then_lookup_raises_hit(self) -> None:
        c = IdempotencyCache()
        c.store("run-1", "k-1", {"status": "ok"})
        with pytest.raises(IdempotencyHit) as exc:
            c.lookup("run-1", "k-1", "create_payment")
        assert exc.value.value == {"status": "ok"}

    def test_missing_key_returns_none(self) -> None:
        c = IdempotencyCache()
        assert c.lookup("run-1", "never-stored", "f") is None

    def test_ttl_expiry(self) -> None:
        c = IdempotencyCache()
        c.store("run", "k", 42, ttl_seconds=0.01)
        time.sleep(0.02)
        assert c.lookup("run", "k", "f") is None

    def test_run_scoping(self) -> None:
        c = IdempotencyCache()
        c.store("run-A", "k", 1)
        # Same key, different run_id, should not hit
        assert c.lookup("run-B", "k", "f") is None

    def test_null_key_is_noop(self) -> None:
        c = IdempotencyCache()
        c.store("run", None, 1)
        assert c.lookup("run", None, "f") is None
        assert c.size() == 0


# ---------------------------------------------------------------------------
# End-to-end via PolicyEngine
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_policy_engine():
    PolicyEngine.reset()
    yield
    PolicyEngine.reset()


def _configure(cfg: PolicyConfig) -> None:
    """Bypass loader — set config directly for unit tests."""
    PolicyEngine._config = cfg
    PolicyEngine._initialized = True


class TestPolicyEngineIntegration:
    def test_pii_rule_blocks_with_reason_code(self) -> None:
        _configure(PolicyConfig(pii_rules={"send": PIIRule()}))
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action(
                "send", 0.0, kwargs={"to": "alice@example.com", "body": "hi"}
            )
        assert exc.value.details["reason_code"] == "PII_DETECTED"
        assert "email" in exc.value.details["categories"]

    def test_pii_clean_kwargs_pass(self) -> None:
        _configure(PolicyConfig(pii_rules={"send": PIIRule()}))
        # Should not raise
        PolicyEngine.check_action("send", 0.0, kwargs={"to": "user-1"})

    def test_pii_default_enabled_applies_to_any_action(self) -> None:
        _configure(PolicyConfig(pii_default_enabled=True))
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("anything", 0.0, kwargs={"x": "me@x.com"})

    def test_moderation_rule_blocks_with_reason_code(self) -> None:
        _configure(PolicyConfig(moderation_rules={"post": ModerationRule()}))
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action(
                "post", 0.0, kwargs={"text": "how to make a bomb"}
            )
        assert exc.value.details["reason_code"] == "CONTENT_MODERATED"

    def test_loop_rule_blocks_after_threshold(self) -> None:
        _configure(
            PolicyConfig(
                loop_rules={"search": LoopRule(threshold=3, window_seconds=5.0)}
            )
        )
        # First two pass, third blocks
        PolicyEngine.check_action("search", 0.0, kwargs={"q": "cats"})
        PolicyEngine.check_action("search", 0.0, kwargs={"q": "cats"})
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("search", 0.0, kwargs={"q": "cats"})
        assert exc.value.details["reason_code"] == "LOOP_DETECTED"
        assert exc.value.details["detection"]["count"] == 3

    def test_guardrails_skipped_when_engine_unconfigured(self) -> None:
        # Fail-open: with no config, check_action is a no-op.
        PolicyEngine.check_action("anything", 0.0, kwargs={"x": "me@x.com"})


# ---------------------------------------------------------------------------
# B2: self-repair feedback quality
# ---------------------------------------------------------------------------


class TestSelfRepairFeedback:
    """Every block must carry retry_guidance + safe_alternatives + recoverable."""

    def test_pii_block_is_self_repair_actionable(self) -> None:
        _configure(PolicyConfig(pii_rules={"send": PIIRule()}))
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action(
                "send", 0.0, kwargs={"to": "a@b.co"}
            )
        d = exc.value.details
        assert d["recoverable"] is True
        assert isinstance(d["retry_guidance"], str) and d["retry_guidance"]
        assert isinstance(d["safe_alternatives"], list) and d["safe_alternatives"]

    def test_moderation_block_is_self_repair_actionable(self) -> None:
        _configure(PolicyConfig(moderation_rules={"post": ModerationRule()}))
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action(
                "post", 0.0, kwargs={"text": "how to make a bomb"}
            )
        d = exc.value.details
        assert d["recoverable"] is True
        assert "retry_guidance" in d and d["retry_guidance"]
        assert d["safe_alternatives"]

    def test_loop_block_includes_prior_attempts(self) -> None:
        _configure(
            PolicyConfig(
                loop_rules={"f": LoopRule(threshold=2, window_seconds=5.0)}
            )
        )
        PolicyEngine.check_action("f", 0.0, kwargs={"q": 1})
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("f", 0.0, kwargs={"q": 1})
        d = exc.value.details
        assert d["prior_attempts"] >= 2
        assert d["retry_guidance"]
        assert any("vary" in alt for alt in d["safe_alternatives"])


# ---------------------------------------------------------------------------
# A3 decorator wiring: @guarded_action(idempotency_key=...)
# ---------------------------------------------------------------------------


class TestIdempotencyDecoratorWiring:
    def test_sync_cached_on_repeat(self) -> None:
        from agent_sentinel import guarded_action
        from agent_sentinel.guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE

        DEFAULT_IDEMPOTENCY_CACHE.clear()
        n = {"n": 0}

        @guarded_action(
            name="create_payment",
            idempotency_key=lambda *a, **kw: kw.get("order_id"),
        )
        def create_payment(*, order_id: str, amount: float) -> dict:
            n["n"] += 1
            return {"order_id": order_id, "amount": amount, "call": n["n"]}

        r1 = create_payment(order_id="ord-1", amount=10.0)
        r2 = create_payment(order_id="ord-1", amount=99.0)
        r3 = create_payment(order_id="ord-2", amount=10.0)

        assert r1 == r2
        assert r3 != r1
        assert n["n"] == 2

    def test_static_string_key(self) -> None:
        from agent_sentinel import guarded_action
        from agent_sentinel.guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE

        DEFAULT_IDEMPOTENCY_CACHE.clear()
        n = {"n": 0}

        @guarded_action(name="seed", idempotency_key="static-seed-key")
        def seed() -> int:
            n["n"] += 1
            return n["n"]

        assert seed() == 1
        assert seed() == 1  # cached
        assert n["n"] == 1

    def test_idempotency_disabled_when_key_none(self) -> None:
        from agent_sentinel import guarded_action

        n = {"n": 0}

        @guarded_action(name="do_x")
        def do_x() -> int:
            n["n"] += 1
            return n["n"]

        assert do_x() == 1
        assert do_x() == 2

    def test_bad_key_factory_does_not_crash(self) -> None:
        from agent_sentinel import guarded_action

        @guarded_action(
            name="risky",
            idempotency_key=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        def risky() -> str:
            return "ok"

        # Key factory raises -> resolves to None -> no caching, no crash.
        assert risky() == "ok"
        assert risky() == "ok"
