"""
Extended coverage tests for policy module.

Tests cover:
- PolicyConfig creation and validation
- RateLimiter edge cases and window management
- PolicyCache file operations and TTL
- PolicyEngine configuration and state management
- Budget enforcement (session, run, action-level)
- Rate limiting integration
- Action allow/deny lists
"""
from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from agent_sentinel.policy import (
    PolicyEngine,
    PolicyConfig,
    RateLimiter,
    PolicyCache,
)
from agent_sentinel.cost import CostTracker
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError


class TestPolicyConfig:
    """Test PolicyConfig data structure."""

    def test_policy_config_creation(self) -> None:
        """Test creating a basic policy config."""
        config = PolicyConfig(
            run_budget=10.0,
            session_budget=100.0,
            denied_actions=["dangerous"],
        )
        
        assert config.run_budget == 10.0
        assert config.session_budget == 100.0
        assert config.denied_actions == ["dangerous"]
        assert config.action_budgets == {}
        assert config.allowed_actions is None

    def test_policy_config_with_action_budgets(self) -> None:
        """Test policy config with per-action budgets."""
        config = PolicyConfig(
            run_budget=10.0,
            action_budgets={"search": 5.0, "analyze": 3.0},
        )
        
        assert config.action_budgets["search"] == 5.0
        assert config.action_budgets["analyze"] == 3.0

    def test_policy_config_with_allowed_actions(self) -> None:
        """Test policy config with action whitelist."""
        config = PolicyConfig(
            allowed_actions=["search", "summarize", "translate"],
        )
        
        assert set(config.allowed_actions) == {"search", "summarize", "translate"}

    def test_policy_config_with_rate_limits(self) -> None:
        """Test policy config with rate limiting."""
        config = PolicyConfig(
            rate_limits={
                "search": {"max_count": 10, "window_seconds": 60},
                "analyze": {"max_count": 5, "window_seconds": 30},
            }
        )
        
        assert config.rate_limits["search"]["max_count"] == 10
        assert config.rate_limits["analyze"]["window_seconds"] == 30


class TestRateLimiterAdvanced:
    """Advanced rate limiter tests."""

    def test_rate_limiter_concurrent_actions(self) -> None:
        """Test rate limiter with concurrent different actions."""
        limiter = RateLimiter()
        
        # Each action has independent limits
        for i in range(2):  # Only 2 calls since limit is 2
            limiter.check_rate_limit("action_a", max_count=5, window_seconds=60)
            limiter.check_rate_limit("action_b", max_count=2, window_seconds=60)
        
        # action_a has space
        limiter.check_rate_limit("action_a", max_count=5, window_seconds=60)
        
        # action_b should fail (already at limit)
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("action_b", max_count=2, window_seconds=60)

    def test_rate_limiter_window_precision(self) -> None:
        """Test rate limiter respects exact window boundaries."""
        limiter = RateLimiter()
        
        # Use very small window
        limiter.check_rate_limit("test", max_count=1, window_seconds=0.1)
        
        # Should fail immediately
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("test", max_count=1, window_seconds=0.1)
        
        # Wait for window to expire
        time.sleep(0.15)
        
        # Should succeed now
        limiter.check_rate_limit("test", max_count=1, window_seconds=0.1)

    def test_rate_limiter_counter_accuracy(self) -> None:
        """Test that counter accurately tracks calls."""
        limiter = RateLimiter()
        
        # Make exactly max_count calls
        max_count = 7
        for i in range(max_count):
            limiter.check_rate_limit("test", max_count=max_count, window_seconds=60)
        
        # One more should fail
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("test", max_count=max_count, window_seconds=60)

    def test_rate_limiter_case_sensitive_action_names(self) -> None:
        """Test that action names are case-sensitive."""
        limiter = RateLimiter()
        
        limiter.check_rate_limit("ACTION", max_count=1, window_seconds=60)
        
        # Different case is different action
        limiter.check_rate_limit("action", max_count=1, window_seconds=60)
        limiter.check_rate_limit("Action", max_count=1, window_seconds=60)


class TestPolicyCacheAdvanced:
    """Advanced policy cache tests."""

    def test_cache_with_large_policies(self, tmp_path) -> None:
        """Test cache handles large policy lists."""
        cache = PolicyCache(tmp_path)
        
        # Create large policy list
        policies = [
            {
                "id": f"policy-{i}",
                "enabled": True,
                "run_budget": float(i),
                "name": f"policy_{i}",
            }
            for i in range(100)
        ]
        
        cache.save(policies, ttl=3600)
        loaded = cache.load()
        
        assert len(loaded) == 100
        assert loaded[50]["id"] == "policy-50"

    def test_cache_file_permissions(self, tmp_path) -> None:
        """Test cache file is created with proper permissions."""
        cache = PolicyCache(tmp_path)
        
        policies = [{"id": "test", "enabled": True}]
        cache.save(policies, ttl=3600)
        
        assert cache.cache_file.exists()
        # File should be readable and writable
        assert cache.cache_file.stat().st_mode & 0o644

    def test_cache_corrupted_json_partial(self, tmp_path) -> None:
        """Test cache handles partially corrupted JSON."""
        cache = PolicyCache(tmp_path)
        
        # Write partially valid JSON
        cache.cache_file.write_text('{"policies": [{"id": "1"')
        
        # Should return None gracefully
        result = cache.load()
        assert result is None

    def test_cache_with_special_characters(self, tmp_path) -> None:
        """Test cache handles policies with special characters."""
        cache = PolicyCache(tmp_path)
        
        policies = [
            {
                "id": "policy-1",
                "name": "Policy with émojis 🎉",
                "description": "Special chars: <>\"'{}[]",
            }
        ]
        
        cache.save(policies, ttl=3600)
        loaded = cache.load()
        
        assert loaded[0]["name"] == "Policy with émojis 🎉"


class TestPolicyEngineConfiguration:
    """Test PolicyEngine configuration and state management."""

    def test_configure_run_budget_only(self) -> None:
        """Test configuring just run budget."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(run_budget=5.0)
        config = PolicyEngine.get_config()
        
        assert config.run_budget == 5.0
        assert config.session_budget is None

    def test_configure_multiple_budgets(self) -> None:
        """Test configuring session and run budgets."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            session_budget=100.0,
            run_budget=10.0,
        )
        config = PolicyEngine.get_config()
        
        assert config.session_budget == 100.0
        assert config.run_budget == 10.0

    def test_configure_action_specific_limits(self) -> None:
        """Test configuring per-action budgets and rate limits."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            action_budgets={"expensive": 2.0, "cheap": 0.5},
            rate_limits={
                "frequent": {"max_count": 100, "window_seconds": 60},
                "rare": {"max_count": 5, "window_seconds": 60},
            }
        )
        config = PolicyEngine.get_config()
        
        assert config.action_budgets["expensive"] == 2.0
        assert config.rate_limits["frequent"]["max_count"] == 100

    def test_configure_action_lists(self) -> None:
        """Test configuring allowed and denied actions."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            denied_actions=["dangerous", "forbidden"],
            allowed_actions=["search", "analyze"],
        )
        config = PolicyEngine.get_config()
        
        assert "dangerous" in config.denied_actions
        assert "search" in config.allowed_actions

    def test_is_configured_state(self) -> None:
        """Test is_configured() reflects state."""
        PolicyEngine.reset()
        
        assert not PolicyEngine.is_configured()
        
        PolicyEngine.configure(run_budget=1.0)
        assert PolicyEngine.is_configured()
        
        PolicyEngine.reset()
        assert not PolicyEngine.is_configured()


class TestBudgetEnforcement:
    """Test budget enforcement at different levels."""

    def test_run_budget_enforcement(self) -> None:
        """Test run-level budget is enforced."""
        PolicyEngine.reset()
        CostTracker.reset_all()
        
        PolicyEngine.configure(run_budget=1.0)
        
        # First action within budget
        PolicyEngine.check_action("action1", cost=0.6)
        CostTracker.add_cost("action1", 0.6)
        
        # Second action exceeds budget
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("action2", cost=0.5)
        
        PolicyEngine.reset()
        CostTracker.reset_all()

    def test_session_budget_enforcement(self) -> None:
        """Test session-level budget is enforced."""
        PolicyEngine.reset()
        CostTracker.reset_all()
        
        PolicyEngine.configure(session_budget=2.0)
        
        # Multiple actions within budget
        PolicyEngine.check_action("action1", cost=0.5)
        CostTracker.add_cost("action1", 0.5)
        
        PolicyEngine.check_action("action2", cost=0.5)
        CostTracker.add_cost("action2", 0.5)
        
        PolicyEngine.check_action("action3", cost=0.5)
        CostTracker.add_cost("action3", 0.5)
        
        # Next action exceeds session budget
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("action4", cost=1.0)
        
        PolicyEngine.reset()
        CostTracker.reset_all()

    def test_action_specific_budget(self) -> None:
        """Test action-specific budgets are enforced."""
        PolicyEngine.reset()
        CostTracker.reset_all()
        
        PolicyEngine.configure(
            action_budgets={"expensive": 1.0}
        )
        
        # Within action budget
        PolicyEngine.check_action("expensive", cost=0.6)
        CostTracker.add_cost("expensive", 0.6)
        
        # Exceeds action budget
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("expensive", cost=0.5)
        
        PolicyEngine.reset()
        CostTracker.reset_all()

    def test_strict_mode_disabled(self) -> None:
        """Test behavior when strict_mode is disabled - it still enforces budgets."""
        PolicyEngine.reset()
        
        # Note: strict_mode doesn't prevent budget checks, it affects error handling
        # Budget enforcement is always active regardless of strict_mode
        PolicyEngine.configure(
            run_budget=0.1,
            strict_mode=False,
        )
        
        # This will still raise because budget enforcement is independent of strict_mode
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("action", cost=1.0)
        
        PolicyEngine.reset()


class TestActionListEnforcement:
    """Test allow and deny lists."""

    def test_denied_action_raises_error(self) -> None:
        """Test that denied actions raise PolicyViolationError."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            denied_actions=["forbidden", "blocked"]
        )
        
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("forbidden", cost=0.01)
        
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("blocked", cost=0.01)
        
        PolicyEngine.reset()

    def test_allowed_actions_whitelist(self) -> None:
        """Test that allowed_actions works as whitelist."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            allowed_actions=["search", "analyze"]
        )
        
        # Allowed actions work
        PolicyEngine.check_action("search", cost=0.01)
        PolicyEngine.check_action("analyze", cost=0.01)
        
        # Non-allowed action fails
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("delete", cost=0.01)
        
        PolicyEngine.reset()

    def test_allowed_and_denied_combined(self) -> None:
        """Test combining allowed and denied lists."""
        PolicyEngine.reset()
        
        PolicyEngine.configure(
            allowed_actions=["read", "write", "admin"],
            denied_actions=["admin"],  # Even though in allowed list
        )
        
        PolicyEngine.check_action("read", cost=0.01)
        PolicyEngine.check_action("write", cost=0.01)
        
        # Denied takes precedence
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("admin", cost=0.01)
        
        PolicyEngine.reset()


class TestPolicyIntegration:
    """Integration tests combining multiple policy features."""

    def test_complete_policy_scenario(self) -> None:
        """Test realistic policy scenario."""
        PolicyEngine.reset()
        CostTracker.reset_all()
        
        PolicyEngine.configure(
            session_budget=10.0,
            run_budget=2.0,
            action_budgets={"premium": 1.0},
            denied_actions=["delete_all"],
            rate_limits={
                "frequent_action": {"max_count": 5, "window_seconds": 60}
            }
        )
        
        # Normal action succeeds
        PolicyEngine.check_action("normal_action", cost=0.5)
        CostTracker.add_cost("normal_action", 0.5)
        
        # Premium action within budget
        PolicyEngine.check_action("premium", cost=0.8)
        CostTracker.add_cost("premium", 0.8)
        
        # Another premium action exceeds action budget
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("premium", cost=0.3)
        
        # Denied action fails
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("delete_all", cost=0.0)
        
        # Rate limited action succeeds up to limit
        for _ in range(5):
            PolicyEngine.check_action("frequent_action", cost=0.01)
        
        # Exceeds rate limit
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("frequent_action", cost=0.01)
        
        PolicyEngine.reset()
        CostTracker.reset_all()

    def test_policy_reset_clears_state(self) -> None:
        """Test that reset clears all policy state."""
        PolicyEngine.configure(run_budget=1.0)
        assert PolicyEngine.is_configured()
        
        PolicyEngine.reset()
        assert not PolicyEngine.is_configured()
        
        # Should be able to reconfigure
        PolicyEngine.configure(run_budget=2.0)
        assert PolicyEngine.is_configured()
        assert PolicyEngine.get_config().run_budget == 2.0

