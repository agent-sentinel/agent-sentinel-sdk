"""
Policy Engine: Budget enforcement and action control.

Phase 2 Implementation:
- Load policies from callguard.yaml or code definitions
- Check budgets before action execution
- Support for session and run-level limits
- Action-specific budgets and deny lists

Phase 3 Implementation:
- Remote policy synchronization from platform
- Local policy caching with expiration
- Periodic background refresh
- Rate limiting support
- Time-based policy enforcement
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from .cost import CostTracker
from .errors import BudgetExceededError, PolicyViolationError

logger = logging.getLogger("agent_sentinel")

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning(
        "PyYAML not installed. Policy loading from callguard.yaml disabled. "
        "Install with: pip install pyyaml"
    )

# Try to import httpx for remote sync
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.debug("httpx not installed. Remote policy sync disabled.")


@dataclass
class PolicyConfig:
    """
    Configuration for policy enforcement.
    
    Attributes:
        session_budget: Max cost in USD for entire session (None = unlimited)
        run_budget: Max cost in USD for current run (None = unlimited)
        action_budgets: Dict mapping action name to max cost per action
        denied_actions: List of action names that are blocked
        allowed_actions: If set, only these actions are permitted (allowlist mode)
        rate_limits: Dict mapping action name to rate limit config
        strict_mode: If True, any policy violation stops execution
        require_approval: If True, actions may require human approval
        approval_actions: List of action names that require approval
        approval_threshold_usd: Actions costing more than this require approval
        approval_timeout_seconds: How long to wait for approval
    """
    session_budget: Optional[float] = None
    run_budget: Optional[float] = None
    action_budgets: Dict[str, float] = field(default_factory=dict)
    denied_actions: List[str] = field(default_factory=list)
    allowed_actions: Optional[List[str]] = None
    rate_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    strict_mode: bool = True
    # Approval Inbox settings
    require_approval: bool = False
    approval_actions: List[str] = field(default_factory=list)
    approval_threshold_usd: Optional[float] = None
    approval_timeout_seconds: int = 3600


@dataclass
class RemotePolicyConfig:
    """Configuration for remote policy synchronization."""
    platform_url: str
    api_token: str
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    refresh_interval: float = 300.0  # 5 minutes default
    cache_ttl: float = 600.0  # 10 minutes cache TTL
    enabled: bool = True


class RateLimiter:
    """
    Rate limiter for action execution.
    
    Tracks action counts within time windows to enforce rate limits.
    """
    
    def __init__(self):
        self._action_history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def check_rate_limit(
        self,
        action: str,
        max_count: int,
        window_seconds: int
    ) -> None:
        """
        Check if action is within rate limit.
        
        Args:
            action: Name of the action
            max_count: Maximum number of calls allowed in window
            window_seconds: Time window in seconds
        
        Raises:
            PolicyViolationError: If rate limit exceeded
        """
        with self._lock:
            now = time.time()
            
            # Get or create history for this action
            if action not in self._action_history:
                self._action_history[action] = []
            
            history = self._action_history[action]
            
            # Remove timestamps outside the window
            cutoff = now - window_seconds
            history[:] = [ts for ts in history if ts > cutoff]
            
            # Check if we've exceeded the limit
            if len(history) >= max_count:
                oldest = history[0]
                wait_time = int(oldest + window_seconds - now)
                raise PolicyViolationError(
                    f"Rate limit exceeded for action '{action}': "
                    f"{max_count} calls per {window_seconds}s. "
                    f"Retry in {wait_time}s"
                )
            
            # Record this action
            history.append(now)
    
    def reset(self) -> None:
        """Reset all rate limit tracking."""
        with self._lock:
            self._action_history.clear()


class PolicyCache:
    """
    Local cache for remote policies with expiration.
    
    Stores policies to disk and manages cache invalidation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use .agent-sentinel directory
            sentinel_home = os.getenv("AGENT_SENTINEL_HOME", ".agent-sentinel")
            self.cache_dir = Path(sentinel_home)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "policy_cache.json"
    
    def save(self, policies: List[Dict[str, Any]], ttl: float) -> None:
        """
        Save policies to cache with expiration time.
        
        Args:
            policies: List of policy dictionaries from platform
            ttl: Time to live in seconds
        """
        try:
            now = datetime.now(timezone.utc)
            cache_data = {
                "policies": policies,
                "cached_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=ttl)).isoformat()
            }
            
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved {len(policies)} policies to cache")
        
        except Exception as e:
            logger.error(f"Failed to save policy cache: {e}")
    
    def load(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load policies from cache if not expired.
        
        Returns:
            List of policy dictionaries or None if cache invalid/expired
        """
        try:
            if not self.cache_file.exists():
                logger.debug("No policy cache found")
                return None
            
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check expiration
            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            # Ensure both datetimes are timezone-aware
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expires_at:
                logger.debug("Policy cache expired")
                return None
            
            policies = cache_data["policies"]
            logger.debug(f"Loaded {len(policies)} policies from cache")
            return policies
        
        except Exception as e:
            logger.error(f"Failed to load policy cache: {e}")
            return None
    
    def clear(self) -> None:
        """Clear the policy cache."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.debug("Policy cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear policy cache: {e}")


class PolicyEngine:
    """
    Singleton policy engine for budget enforcement.
    
    Phase 3 Features:
    - Remote policy synchronization from platform
    - Local policy caching
    - Rate limiting
    - Background policy refresh
    
    Usage:
        # Load from YAML file (local)
        PolicyEngine.load_from_yaml("callguard.yaml")
        
        # Enable remote sync
        PolicyEngine.enable_remote_sync(
            platform_url="https://api.agentsentinel.dev",
            api_token="your-jwt-token",
            refresh_interval=300  # 5 minutes
        )
        
        # Or configure in code
        PolicyEngine.configure(
            run_budget=1.0,
            denied_actions=["dangerous_operation"]
        )
        
        # Check before action execution
        PolicyEngine.check_action("search_web", cost=0.02)
    """
    
    _config: Optional[PolicyConfig] = None
    _initialized: bool = False
    _rate_limiter: RateLimiter = RateLimiter()
    
    # Remote sync state
    _remote_config: Optional[RemotePolicyConfig] = None
    _policy_cache: Optional[PolicyCache] = None
    _sync_thread: Optional[threading.Thread] = None
    _stop_sync: threading.Event = threading.Event()
    _sync_lock: threading.Lock = threading.Lock()
    
    @classmethod
    def configure(
        cls,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budgets: Optional[Dict[str, float]] = None,
        denied_actions: Optional[List[str]] = None,
        allowed_actions: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, Dict[str, int]]] = None,
        strict_mode: bool = True
    ) -> None:
        """
        Configure policy engine programmatically.
        
        Args:
            session_budget: Max USD for entire session
            run_budget: Max USD for current run
            action_budgets: Dict of action -> max cost
            denied_actions: List of blocked actions
            allowed_actions: If set, allowlist mode (only these permitted)
            rate_limits: Dict of action -> {"max_count": N, "window_seconds": M}
            strict_mode: If True, violations stop execution
        """
        cls._config = PolicyConfig(
            session_budget=session_budget,
            run_budget=run_budget,
            action_budgets=action_budgets or {},
            denied_actions=denied_actions or [],
            allowed_actions=allowed_actions,
            rate_limits=rate_limits or {},
            strict_mode=strict_mode
        )
        cls._initialized = True
        logger.info("Policy engine configured programmatically")
    
    @classmethod
    def load_from_yaml(cls, path: Optional[str] = None) -> None:
        """
        Load policy configuration from YAML file.
        
        Args:
            path: Path to callguard.yaml file. If None, searches for:
                  1. ./callguard.yaml
                  2. ./.agent-sentinel/callguard.yaml
                  3. $AGENT_SENTINEL_HOME/callguard.yaml
        
        YAML Format:
            budgets:
              session: 10.0
              run: 1.0
              actions:
                expensive_action: 0.5
            
            denied_actions:
              - dangerous_operation
              - delete_database
            
            allowed_actions:  # If present, only these are allowed
              - safe_operation
              - read_only
            
            rate_limits:
              api_call:
                max_count: 10
                window_seconds: 60
            
            strict_mode: true
        """
        if not YAML_AVAILABLE:
            logger.error("Cannot load YAML: PyYAML not installed")
            return
        
        # Find the config file
        if path:
            config_path = Path(path)
        else:
            # Search common locations
            candidates = [
                Path("callguard.yaml"),
                Path(".agent-sentinel/callguard.yaml"),
            ]
            
            sentinel_home = os.getenv("AGENT_SENTINEL_HOME")
            if sentinel_home:
                candidates.append(Path(sentinel_home) / "callguard.yaml")
            
            config_path = None
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
        
        if not config_path or not config_path.exists():
            logger.info("No callguard.yaml found, policy engine not configured")
            return
        
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning(f"Empty policy file: {config_path}")
                return
            
            # Parse budgets
            budgets = data.get("budgets", {})
            session_budget = budgets.get("session")
            run_budget = budgets.get("run")
            action_budgets = budgets.get("actions", {})
            
            # Parse action controls
            denied_actions = data.get("denied_actions", [])
            allowed_actions = data.get("allowed_actions")
            rate_limits = data.get("rate_limits", {})
            strict_mode = data.get("strict_mode", True)
            
            cls.configure(
                session_budget=session_budget,
                run_budget=run_budget,
                action_budgets=action_budgets,
                denied_actions=denied_actions,
                allowed_actions=allowed_actions,
                rate_limits=rate_limits,
                strict_mode=strict_mode
            )
            
            logger.info(f"Policy engine loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load policy from {config_path}: {e}")
    
    @classmethod
    def enable_remote_sync(
        cls,
        platform_url: str,
        api_token: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        refresh_interval: float = 300.0,
        cache_ttl: float = 600.0
    ) -> None:
        """
        Enable remote policy synchronization from platform.
        
        Policies are downloaded on startup and refreshed periodically.
        Uses local cache to minimize network calls and handle offline scenarios.
        
        Args:
            platform_url: Platform API base URL
            api_token: JWT token for authentication
            agent_id: Optional agent identifier for agent-specific policies
            run_id: Optional run identifier for run-specific policies
            refresh_interval: Seconds between policy refreshes (default: 300 = 5 min)
            cache_ttl: Cache time-to-live in seconds (default: 600 = 10 min)
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed. Remote policy sync disabled.")
            return
        
        cls._remote_config = RemotePolicyConfig(
            platform_url=platform_url.rstrip("/"),
            api_token=api_token,
            agent_id=agent_id,
            run_id=run_id,
            refresh_interval=refresh_interval,
            cache_ttl=cache_ttl,
            enabled=True
        )
        
        cls._policy_cache = PolicyCache()
        
        # Initial sync - try cache first, then remote
        cls._sync_policies_once()
        
        # Start background refresh thread
        cls._start_background_sync()
        
        logger.info("Remote policy sync enabled")
    
    @classmethod
    def _sync_policies_once(cls) -> None:
        """
        Sync policies once: try cache first, then remote.
        """
        if not cls._remote_config or not cls._policy_cache:
            return
        
        with cls._sync_lock:
            # Try loading from cache first
            cached_policies = cls._policy_cache.load()
            if cached_policies:
                cls._apply_remote_policies(cached_policies)
                logger.debug("Applied policies from cache")
                return
            
            # Cache miss or expired - fetch from platform
            policies = cls._fetch_policies_from_platform()
            if policies:
                cls._policy_cache.save(policies, cls._remote_config.cache_ttl)
                cls._apply_remote_policies(policies)
                logger.info(f"Downloaded and applied {len(policies)} policies from platform")
            else:
                logger.warning("Failed to fetch policies from platform")
    
    @classmethod
    def _fetch_policies_from_platform(cls) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch policies from platform API.
        
        Returns:
            List of policy dictionaries or None on failure
        """
        if not cls._remote_config or not HTTPX_AVAILABLE:
            return None
        
        try:
            url = f"{cls._remote_config.platform_url}/api/v1/policies/sync"
            
            params = {}
            if cls._remote_config.agent_id:
                params["agent_id"] = cls._remote_config.agent_id
            if cls._remote_config.run_id:
                params["run_id"] = cls._remote_config.run_id
            
            headers = {
                "Authorization": f"Bearer {cls._remote_config.api_token}"
            }
            
            response = httpx.get(url, params=params, headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch policies: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching policies: {e}")
            return None
    
    @classmethod
    def _apply_remote_policies(cls, policies: List[Dict[str, Any]]) -> None:
        """
        Apply remote policies by merging them with local config.
        
        Remote policies override local settings. Multiple policies are merged
        in order (global -> agent -> run).
        """
        if not policies:
            return
        
        # Start with empty or existing config
        if cls._config is None:
            cls._config = PolicyConfig()
            cls._initialized = True
        
        # Merge all policies in order
        for policy in policies:
            if not policy.get("enabled", True):
                continue
            
            # Merge budgets (most restrictive wins)
            if policy.get("run_budget") is not None:
                if cls._config.run_budget is None:
                    cls._config.run_budget = policy["run_budget"]
                else:
                    cls._config.run_budget = min(cls._config.run_budget, policy["run_budget"])
            
            if policy.get("session_budget") is not None:
                if cls._config.session_budget is None:
                    cls._config.session_budget = policy["session_budget"]
                else:
                    cls._config.session_budget = min(cls._config.session_budget, policy["session_budget"])
            
            # Merge action budgets
            if policy.get("action_budgets"):
                for action, budget in policy["action_budgets"].items():
                    if action not in cls._config.action_budgets:
                        cls._config.action_budgets[action] = budget
                    else:
                        # Most restrictive
                        cls._config.action_budgets[action] = min(
                            cls._config.action_budgets[action],
                            budget
                        )
            
            # Merge denied actions (union)
            if policy.get("denied_actions"):
                cls._config.denied_actions = list(set(
                    cls._config.denied_actions + policy["denied_actions"]
                ))
            
            # Allowed actions (intersection if both exist, otherwise override)
            if policy.get("allowed_actions"):
                if cls._config.allowed_actions is None:
                    cls._config.allowed_actions = policy["allowed_actions"]
                else:
                    # Intersection - only allow what both permit
                    cls._config.allowed_actions = list(set(
                        cls._config.allowed_actions
                    ) & set(policy["allowed_actions"]))
            
            # Merge rate limits
            if policy.get("rate_limits"):
                for action, limits in policy["rate_limits"].items():
                    if action not in cls._config.rate_limits:
                        cls._config.rate_limits[action] = limits
                    else:
                        # Most restrictive
                        existing = cls._config.rate_limits[action]
                        cls._config.rate_limits[action] = {
                            "max_count": min(
                                existing.get("max_count", 999999),
                                limits.get("max_count", 999999)
                            ),
                            "window_seconds": existing.get("window_seconds", limits.get("window_seconds", 60))
                        }
            
            # Merge approval settings (any policy can enable approval)
            if policy.get("require_approval"):
                cls._config.require_approval = True
            
            # Merge approval actions (union)
            if policy.get("approval_actions"):
                cls._config.approval_actions = list(set(
                    cls._config.approval_actions + policy["approval_actions"]
                ))
            
            # Approval threshold (most restrictive - lowest threshold)
            if policy.get("approval_threshold_usd") is not None:
                if cls._config.approval_threshold_usd is None:
                    cls._config.approval_threshold_usd = policy["approval_threshold_usd"]
                else:
                    cls._config.approval_threshold_usd = min(
                        cls._config.approval_threshold_usd,
                        policy["approval_threshold_usd"]
                    )
            
            # Approval timeout (use the policy's timeout)
            if policy.get("approval_timeout_seconds"):
                cls._config.approval_timeout_seconds = policy["approval_timeout_seconds"]
    
    @classmethod
    def _start_background_sync(cls) -> None:
        """Start background thread for periodic policy refresh."""
        if cls._sync_thread and cls._sync_thread.is_alive():
            return
        
        cls._stop_sync.clear()
        cls._sync_thread = threading.Thread(
            target=cls._background_sync_loop,
            daemon=True,
            name="PolicySync"
        )
        cls._sync_thread.start()
        logger.debug("Started background policy sync thread")
    
    @classmethod
    def _background_sync_loop(cls) -> None:
        """Background loop that periodically refreshes policies."""
        if not cls._remote_config:
            return
        
        interval = cls._remote_config.refresh_interval
        
        while not cls._stop_sync.wait(timeout=interval):
            try:
                cls._sync_policies_once()
            except Exception as e:
                logger.error(f"Error in policy sync loop: {e}")
    
    @classmethod
    def stop_remote_sync(cls) -> None:
        """Stop background policy synchronization."""
        if cls._sync_thread:
            cls._stop_sync.set()
            cls._sync_thread.join(timeout=5.0)
            logger.info("Stopped remote policy sync")
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if policy engine has been configured."""
        return cls._initialized and cls._config is not None
    
    @classmethod
    def check_action(cls, action: str, cost: float) -> None:
        """
        Check if an action is permitted under current policy.
        
        This is called BEFORE the action executes. If any policy is
        violated, raises an exception to prevent execution.
        
        Args:
            action: Name of the action to check
            cost: Proposed cost of the action in USD
        
        Raises:
            PolicyViolationError: If action is denied or rate limit exceeded
            BudgetExceededError: If budget would be exceeded
        """
        # If not configured, allow everything (fail-open)
        if not cls.is_configured():
            return
        
        config = cls._config
        
        # 1. Check denied/allowed lists
        if action in config.denied_actions:
            raise PolicyViolationError(
                f"Action '{action}' is on the denied list"
            )
        
        if config.allowed_actions and action not in config.allowed_actions:
            raise PolicyViolationError(
                f"Action '{action}' is not on the allowed list. "
                f"Permitted: {config.allowed_actions}"
            )
        
        # 2. Check rate limits
        if action in config.rate_limits:
            limits = config.rate_limits[action]
            max_count = limits.get("max_count", 999999)
            window_seconds = limits.get("window_seconds", 60)
            cls._rate_limiter.check_rate_limit(action, max_count, window_seconds)
        
        # 3. Check session budget
        if config.session_budget is not None:
            current_session = CostTracker.get_session_total()
            if current_session + cost > config.session_budget:
                raise BudgetExceededError(
                    f"Session budget exceeded: {current_session:.4f} + {cost:.4f} "
                    f"> {config.session_budget:.4f} USD",
                    spent=current_session + cost,
                    limit=config.session_budget,
                    budget_type="session"
                )
        
        # 4. Check run budget
        if config.run_budget is not None:
            current_run = CostTracker.get_run_total()
            if current_run + cost > config.run_budget:
                raise BudgetExceededError(
                    f"Run budget exceeded: {current_run:.4f} + {cost:.4f} "
                    f"> {config.run_budget:.4f} USD",
                    spent=current_run + cost,
                    limit=config.run_budget,
                    budget_type="run"
                )
        
        # 5. Check action-specific budget
        if action in config.action_budgets:
            action_limit = config.action_budgets[action]
            action_stats = CostTracker.get_action_stats(action)
            action_total = action_stats["total_cost"]
            
            if action_total + cost > action_limit:
                raise BudgetExceededError(
                    f"Action '{action}' budget exceeded: {action_total:.4f} + {cost:.4f} "
                    f"> {action_limit:.4f} USD",
                    spent=action_total + cost,
                    limit=action_limit,
                    budget_type="action"
                )
    
    @classmethod
    def get_config(cls) -> Optional[PolicyConfig]:
        """
        Get current policy configuration.
        
        Returns:
            PolicyConfig object or None if not configured
        """
        return cls._config
    
    @classmethod
    def requires_approval(cls, action: str, cost: float) -> bool:
        """
        Check if an action requires human approval.
        
        Args:
            action: Name of the action
            cost: Cost of the action in USD
        
        Returns:
            True if human approval is required
        """
        if not cls.is_configured():
            return False
        
        config = cls._config
        
        # Check if approval is enabled
        if not config.require_approval:
            return False
        
        # Check if this specific action requires approval
        if config.approval_actions and action in config.approval_actions:
            return True
        
        # Check if cost exceeds threshold
        if config.approval_threshold_usd is not None:
            if cost >= config.approval_threshold_usd:
                return True
        
        # If no specific actions listed and no threshold, 
        # require_approval=True means all actions need approval
        if not config.approval_actions and config.approval_threshold_usd is None:
            return True
        
        return False
    
    @classmethod
    def get_approval_timeout(cls) -> int:
        """Get the configured approval timeout in seconds."""
        if cls._config:
            return cls._config.approval_timeout_seconds
        return 3600  # Default 1 hour
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset policy engine.
        
        Useful for testing. In production, you typically configure once
        at startup.
        """
        cls.stop_remote_sync()
        cls._config = None
        cls._initialized = False
        cls._remote_config = None
        cls._policy_cache = None
        cls._rate_limiter.reset()
