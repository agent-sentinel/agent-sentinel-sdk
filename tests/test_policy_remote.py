"""
Tests for policy engine with remote sync, caching, and rate limiting.

Tests:
- Remote policy sync
- Policy caching
- Cache expiration
- Rate limiting
- Policy merging (global -> agent -> run)
- Background refresh
- Fail-open on network errors
"""
from __future__ import annotations

import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from agent_sentinel.policy import (
    PolicyEngine,
    PolicyConfig,
    RemotePolicyConfig,
    PolicyCache,
    RateLimiter,
)
from agent_sentinel.errors import PolicyViolationError, BudgetExceededError
from agent_sentinel.cost import CostTracker


@pytest.fixture(autouse=True)
def reset_policy_engine():
    """Reset policy engine before and after each test."""
    PolicyEngine.reset()
    CostTracker.reset_all()
    yield
    PolicyEngine.reset()
    CostTracker.reset_all()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "policy_cache"
    cache_dir.mkdir()
    return cache_dir


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_allows_within_limit(self):
        """Test that actions within rate limit are allowed."""
        limiter = RateLimiter()
        
        # Should allow 3 calls within limit of 5
        limiter.check_rate_limit("test_action", max_count=5, window_seconds=60)
        limiter.check_rate_limit("test_action", max_count=5, window_seconds=60)
        limiter.check_rate_limit("test_action", max_count=5, window_seconds=60)
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test that actions exceeding rate limit are blocked."""
        limiter = RateLimiter()
        
        # Max 3 calls per 60 seconds
        limiter.check_rate_limit("test_action", max_count=3, window_seconds=60)
        limiter.check_rate_limit("test_action", max_count=3, window_seconds=60)
        limiter.check_rate_limit("test_action", max_count=3, window_seconds=60)
        
        # 4th call should fail
        with pytest.raises(PolicyViolationError) as exc:
            limiter.check_rate_limit("test_action", max_count=3, window_seconds=60)
        
        assert "Rate limit exceeded" in str(exc.value)
        assert "test_action" in str(exc.value)
    
    def test_rate_limiter_window_expiry(self):
        """Test that rate limiter window expires correctly."""
        limiter = RateLimiter()
        
        # Max 2 calls per 1 second
        limiter.check_rate_limit("test_action", max_count=2, window_seconds=1)
        limiter.check_rate_limit("test_action", max_count=2, window_seconds=1)
        
        # Should fail immediately
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("test_action", max_count=2, window_seconds=1)
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should succeed now
        limiter.check_rate_limit("test_action", max_count=2, window_seconds=1)
    
    def test_rate_limiter_different_actions(self):
        """Test that rate limits are per-action."""
        limiter = RateLimiter()
        
        # Each action has its own limit
        limiter.check_rate_limit("action1", max_count=1, window_seconds=60)
        limiter.check_rate_limit("action2", max_count=1, window_seconds=60)
        
        # First action is maxed
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("action1", max_count=1, window_seconds=60)
        
        # Second action is maxed
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("action2", max_count=1, window_seconds=60)
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter()
        
        limiter.check_rate_limit("test_action", max_count=1, window_seconds=60)
        
        # Should fail
        with pytest.raises(PolicyViolationError):
            limiter.check_rate_limit("test_action", max_count=1, window_seconds=60)
        
        # Reset and try again
        limiter.reset()
        limiter.check_rate_limit("test_action", max_count=1, window_seconds=60)


class TestPolicyCache:
    """Test policy caching functionality."""
    
    def test_cache_save_and_load(self, temp_cache_dir):
        """Test saving and loading policies from cache."""
        cache = PolicyCache(temp_cache_dir)
        
        policies = [
            {
                "id": "policy-1",
                "enabled": True,
                "run_budget": 1.0,
            }
        ]
        
        # Save to cache
        cache.save(policies, ttl=3600)
        
        # Load from cache
        loaded = cache.load()
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0]["id"] == "policy-1"
    
    def test_cache_expiration(self, temp_cache_dir):
        """Test that cache expires after TTL."""
        cache = PolicyCache(temp_cache_dir)
        
        policies = [{"id": "policy-1"}]
        
        # Save with very short TTL
        cache.save(policies, ttl=0.1)
        
        # Should load immediately
        loaded = cache.load()
        assert loaded is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should return None (expired)
        loaded = cache.load()
        assert loaded is None
    
    def test_cache_missing_file(self, temp_cache_dir):
        """Test loading when cache file doesn't exist."""
        cache = PolicyCache(temp_cache_dir)
        
        loaded = cache.load()
        assert loaded is None
    
    def test_cache_clear(self, temp_cache_dir):
        """Test clearing the cache."""
        cache = PolicyCache(temp_cache_dir)
        
        policies = [{"id": "policy-1"}]
        cache.save(policies, ttl=3600)
        
        # Verify it exists
        assert cache.cache_file.exists()
        
        # Clear
        cache.clear()
        
        # Should be gone
        assert not cache.cache_file.exists()
    
    def test_cache_invalid_json(self, temp_cache_dir):
        """Test handling of corrupted cache file."""
        cache = PolicyCache(temp_cache_dir)
        
        # Write invalid JSON
        cache.cache_file.write_text("not valid json{]")
        
        # Should return None and not crash
        loaded = cache.load()
        assert loaded is None


class TestPolicyEngineRateLimiting:
    """Test rate limiting integration in PolicyEngine."""
    
    def test_rate_limit_enforcement(self):
        """Test that PolicyEngine enforces rate limits."""
        PolicyEngine.configure(
            rate_limits={
                "test_action": {
                    "max_count": 2,
                    "window_seconds": 60
                }
            }
        )
        
        # First two should succeed
        PolicyEngine.check_action("test_action", cost=0.01)
        PolicyEngine.check_action("test_action", cost=0.01)
        
        # Third should fail
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("test_action", cost=0.01)
        
        assert "Rate limit exceeded" in str(exc.value)
    
    def test_rate_limit_with_budgets(self):
        """Test rate limiting combined with budgets."""
        PolicyEngine.configure(
            run_budget=1.0,
            rate_limits={
                "test_action": {
                    "max_count": 10,
                    "window_seconds": 60
                }
            }
        )
        
        # Should hit rate limit before budget (10 * 0.05 = 0.5 < 1.0)
        for i in range(10):
            PolicyEngine.check_action("test_action", cost=0.05)
        
        # 11th should fail on rate limit
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("test_action", cost=0.05)


class TestRemotePolicySync:
    """Test remote policy synchronization."""
    
    @patch('httpx.get')
    def test_fetch_policies_success(self, mock_get, temp_cache_dir):
        """Test successful policy fetch from platform."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "policy-1",
                "enabled": True,
                "run_budget": 1.0,
                "session_budget": 10.0,
                "action_budgets": {},
                "denied_actions": [],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            }
        ]
        mock_get.return_value = mock_response
        
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999  # Don't auto-refresh during test
            )
        
        # Verify fetch was called
        mock_get.assert_called()
        
        # Verify policy was applied
        assert PolicyEngine.is_configured()
        config = PolicyEngine.get_config()
        assert config.run_budget == 1.0
        assert config.session_budget == 10.0
    
    @patch('httpx.get')
    def test_fetch_policies_with_agent_id(self, mock_get, temp_cache_dir):
        """Test fetching policies with agent_id parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        # Use temp cache dir to ensure cache miss
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                agent_id="my-agent-123",
                refresh_interval=9999
            )
        
        # Verify agent_id was passed
        assert mock_get.called
        call_args = mock_get.call_args
        assert call_args[1]["params"]["agent_id"] == "my-agent-123"
    
    @patch('httpx.get')
    def test_fetch_policies_network_error(self, mock_get, temp_cache_dir):
        """Test handling of network errors (fail-open)."""
        mock_get.side_effect = Exception("Network error")
        
        # Use temp cache dir to ensure no cache
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            # Should not crash
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999
            )
        
        # Should remain fail-open (not configured) since fetch failed and no cache
        assert not PolicyEngine.is_configured()
    
    @patch('httpx.get')
    def test_fetch_policies_401_error(self, mock_get, temp_cache_dir):
        """Test handling of authentication errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        # Use temp cache dir to ensure no cache
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="invalid-token",
                refresh_interval=9999
            )
        
        # Should not crash and remain fail-open
        assert not PolicyEngine.is_configured()


class TestPolicyCaching:
    """Test policy caching integration."""
    
    @patch('httpx.get')
    def test_cache_hit(self, mock_get, temp_cache_dir):
        """Test that cache is used when available."""
        # Pre-populate cache
        cache = PolicyCache(temp_cache_dir)
        policies = [
            {
                "id": "cached-policy",
                "enabled": True,
                "run_budget": 5.0,
                "session_budget": None,
                "action_budgets": {},
                "denied_actions": [],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            }
        ]
        cache.save(policies, ttl=3600)
        
        # Enable sync - should use cache, not call network
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999
            )
        
        # Should NOT have called the API (cache hit)
        mock_get.assert_not_called()
        
        # Should have applied cached policy
        config = PolicyEngine.get_config()
        assert config.run_budget == 5.0
    
    @patch('httpx.get')
    def test_cache_miss_fetches_remote(self, mock_get, temp_cache_dir):
        """Test that cache miss triggers remote fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "remote-policy",
                "enabled": True,
                "run_budget": 3.0,
                "session_budget": None,
                "action_budgets": {},
                "denied_actions": [],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            }
        ]
        mock_get.return_value = mock_response
        
        cache = PolicyCache(temp_cache_dir)
        
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999
            )
        
        # Should have called API (cache miss)
        mock_get.assert_called()
        
        # Should have applied remote policy
        config = PolicyEngine.get_config()
        assert config.run_budget == 3.0


class TestPolicyMerging:
    """Test policy merging logic (global -> agent -> run)."""
    
    @patch('httpx.get')
    def test_merge_multiple_policies(self, mock_get, temp_cache_dir):
        """Test merging global, agent, and run policies."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            # Global policy
            {
                "id": "global-policy",
                "enabled": True,
                "run_budget": 10.0,
                "session_budget": 100.0,
                "action_budgets": {"action1": 5.0},
                "denied_actions": ["bad_action"],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            },
            # Agent policy (more restrictive)
            {
                "id": "agent-policy",
                "enabled": True,
                "run_budget": 5.0,  # More restrictive
                "session_budget": None,
                "action_budgets": {"action1": 3.0, "action2": 1.0},
                "denied_actions": ["another_bad"],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "agent",
                "target_id": "my-agent",
                "updated_at": "2025-01-01T00:00:00"
            },
        ]
        mock_get.return_value = mock_response
        
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                agent_id="my-agent",
                refresh_interval=9999
            )
        
        config = PolicyEngine.get_config()
        
        # Should use most restrictive budget
        assert config.run_budget == 5.0
        
        # Should keep global session budget
        assert config.session_budget == 100.0
        
        # Should merge action budgets (most restrictive)
        assert config.action_budgets["action1"] == 3.0
        assert config.action_budgets["action2"] == 1.0
        
        # Should union denied actions
        assert "bad_action" in config.denied_actions
        assert "another_bad" in config.denied_actions
    
    @patch('httpx.get')
    def test_merge_allowed_actions_intersection(self, mock_get, temp_cache_dir):
        """Test that allowed actions are intersected."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "policy-1",
                "enabled": True,
                "run_budget": None,
                "session_budget": None,
                "action_budgets": {},
                "denied_actions": [],
                "allowed_actions": ["action1", "action2", "action3"],
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            },
            {
                "id": "policy-2",
                "enabled": True,
                "run_budget": None,
                "session_budget": None,
                "action_budgets": {},
                "denied_actions": [],
                "allowed_actions": ["action2", "action3", "action4"],
                "rate_limits": {},
                "scope": "agent",
                "target_id": "my-agent",
                "updated_at": "2025-01-01T00:00:00"
            },
        ]
        mock_get.return_value = mock_response
        
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                agent_id="my-agent",
                refresh_interval=9999
            )
        
        config = PolicyEngine.get_config()
        
        # Should be intersection (only action2 and action3)
        assert config.allowed_actions is not None
        assert set(config.allowed_actions) == {"action2", "action3"}


class TestBackgroundRefresh:
    """Test background policy refresh."""
    
    @patch('httpx.get')
    def test_background_refresh_starts(self, mock_get):
        """Test that background refresh thread starts."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        PolicyEngine.enable_remote_sync(
            platform_url="https://api.test.com",
            api_token="test-token",
            refresh_interval=0.5  # Very short for testing
        )
        
        # Verify thread is running
        assert PolicyEngine._sync_thread is not None
        assert PolicyEngine._sync_thread.is_alive()
        
        # Stop it
        PolicyEngine.stop_remote_sync()
        
        # Give it time to stop
        time.sleep(0.1)
    
    @patch('httpx.get')
    def test_background_refresh_updates_policies(self, mock_get, temp_cache_dir):
        """Test that background refresh calls the fetch function periodically."""
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "id": "policy-1",
                    "enabled": True,
                    "run_budget": 1.0,
                    "session_budget": None,
                    "action_budgets": {},
                    "denied_actions": [],
                    "allowed_actions": None,
                    "rate_limits": {},
                    "scope": "global",
                    "target_id": None,
                    "updated_at": "2025-01-01T00:00:00"
                }
            ]
            return mock_response
        
        mock_get.side_effect = side_effect
        
        cache = PolicyCache(temp_cache_dir)
        cache_patch = patch('agent_sentinel.policy.PolicyCache', return_value=cache)
        cache_patch.start()
        
        try:
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=0.2,  # Refresh every 0.2 seconds
                cache_ttl=0.05  # Very short cache TTL
            )
            
            # Should be called once initially
            initial_calls = call_count[0]
            assert initial_calls >= 1
            
            # Wait for at least one refresh cycle
            time.sleep(0.4)
            
            # Should have been called at least once more
            assert call_count[0] > initial_calls
        finally:
            # Clean up
            PolicyEngine.stop_remote_sync()
            cache_patch.stop()


class TestPolicyEngineIntegration:
    """Integration tests for complete policy enforcement."""
    
    def test_complete_policy_enforcement(self):
        """Test complete policy with budgets, denials, and rate limits."""
        PolicyEngine.configure(
            run_budget=1.0,
            session_budget=10.0,
            action_budgets={"expensive": 0.5},
            denied_actions=["forbidden"],
            rate_limits={
                "limited": {
                    "max_count": 2,
                    "window_seconds": 60
                }
            }
        )
        
        # Test denied action
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("forbidden", cost=0.01)
        
        # Test rate limiting
        PolicyEngine.check_action("limited", cost=0.01)
        PolicyEngine.check_action("limited", cost=0.01)
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("limited", cost=0.01)
        
        # Test action budget
        PolicyEngine.check_action("expensive", cost=0.3)
        CostTracker.add_cost("expensive", 0.3)
        
        # Within limit
        PolicyEngine.check_action("expensive", cost=0.15)
        CostTracker.add_cost("expensive", 0.15)
        
        # Over action budget
        with pytest.raises(BudgetExceededError):
            PolicyEngine.check_action("expensive", cost=0.1)
    
    @patch('httpx.get')
    def test_remote_policies_with_fail_open(self, mock_get, temp_cache_dir):
        """Test that network failures don't break agent."""
        # First call succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "policy-1",
                "enabled": True,
                "run_budget": 1.0,
                "session_budget": None,
                "action_budgets": {},
                "denied_actions": ["bad_action"],
                "allowed_actions": None,
                "rate_limits": {},
                "scope": "global",
                "target_id": None,
                "updated_at": "2025-01-01T00:00:00"
            }
        ]
        mock_get.return_value = mock_response
        
        cache = PolicyCache(temp_cache_dir)
        with patch('agent_sentinel.policy.PolicyCache', return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999
            )
        
        # Policy should be enforced
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("bad_action", cost=0.01)
        
        # Now simulate network failure on refresh
        mock_get.side_effect = Exception("Network error")
        
        # Agent should still work (policies remain active)
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("bad_action", cost=0.01)

