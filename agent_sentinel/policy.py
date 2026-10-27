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
from .context import build_identity_headers
from ._defaults import resolve_api_token, resolve_platform_url
from .errors import (
    BudgetExceededError,
    PolicyViolationError,
    EvidenceViolationError,
    PolicyNotConfiguredError,
    NoPolicyForWriteError,
)
from .intervention import InterventionTracker, InterventionType, InterventionOutcome

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
    approval_tags: List[str] = field(default_factory=list)
    approval_risk_levels: List[str] = field(default_factory=list)
    approval_threshold_usd: Optional[float] = None
    approval_timeout_seconds: int = 3600
    # Evidence graph settings (action correctness enforcement)
    evidence_requirements: Dict[str, List[str]] = field(default_factory=dict)
    evidence_max_age_seconds: Dict[str, int] = field(default_factory=dict)
    commit_actions: List[str] = field(default_factory=list)
    evidence_actions: List[str] = field(default_factory=list)
    argument_constraints: Dict[str, dict] = field(default_factory=dict)
    grounding_rules: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)
    # Built-in guardrails (A1-A4): opt-in, per-action.
    # Map action name -> rule instance. See agent_sentinel.guardrails for rule shapes.
    pii_rules: Dict[str, Any] = field(default_factory=dict)
    moderation_rules: Dict[str, Any] = field(default_factory=dict)
    loop_rules: Dict[str, Any] = field(default_factory=dict)
    # Global toggle: scan every action with the default rule when no per-action rule exists.
    pii_default_enabled: bool = False
    moderation_default_enabled: bool = False
    loop_default_enabled: bool = False
    # Budget warning thresholds (percentage of budget consumed at which to emit a BUDGET_WARNING)
    budget_warning_thresholds: List[int] = field(default_factory=lambda: [80])


@dataclass
class RemotePolicyConfig:
    """Configuration for remote policy synchronization.

    ``platform_url`` defaults to the hosted platform (overridable with
    AGENT_SENTINEL_PLATFORM_URL / AGENT_SENTINEL_URL) and ``api_token`` to
    AGENT_SENTINEL_API_KEY.
    """
    platform_url: Optional[str] = None
    api_token: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    refresh_interval: float = 300.0  # 5 minutes default
    cache_ttl: float = 600.0  # 10 minutes cache TTL
    enabled: bool = True

    def __post_init__(self) -> None:
        self.platform_url = resolve_platform_url(self.platform_url)
        self.api_token = resolve_api_token(self.api_token)


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
                wait_time = max(0, int(oldest + window_seconds - now))
                raise PolicyViolationError(
                    f"Rate limit exceeded for action '{action}': "
                    f"{max_count} calls per {window_seconds}s. "
                    f"Retry in {wait_time}s",
                    policy_rule="rate_limits",
                    action_name=action,
                    reason_code="RATE_LIMITED",
                    retry_after_seconds=float(wait_time),
                    # Unlike a deny list, this clears on its own.
                    recoverable=True,
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
    
    def load_stale(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load policies from cache even if expired (stale fallback).

        Used when the platform is unreachable and we need any policy data.

        Returns:
            List of policy dictionaries or None if no cache exists
        """
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            policies = cache_data["policies"]
            logger.warning(f"Using stale policy cache ({len(policies)} policies) - platform unreachable")
            return policies

        except Exception as e:
            logger.error(f"Failed to load stale policy cache: {e}")
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
            api_token="your-api-key",
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

    # Enforcement mode. "fail_open" permits everything when unconfigured, so
    # a config or telemetry problem never breaks a caller's agent — the
    # pre-0.4 default. "fail_closed" blocks every action instead, which is
    # what a deployed control needs — a proxy that cannot find its policy and
    # silently passes every action is an undetected control failure.
    # "fail_closed_writes" — the default since 0.4.0 — splits the difference
    # using the action's declared impact (guarded_action(impact=...), default
    # "medium"): impact "medium"/"high" actions block with
    # NoPolicyForWriteError the same as fail_closed, while "read"/"low"
    # actions proceed with a warning the same as fail_open. This applies
    # whenever the policy layer cannot vouch for an action — unconfigured, a
    # remote sync that has never produced a usable cache, or policy
    # evaluation itself raising (see _handle_unconfigured and
    # _raise_evaluation_error).
    #
    # This is a breaking change from the 0.3.x "fail_open" default, kept
    # available for one more minor version via
    # AGENT_SENTINEL_ENFORCEMENT=fail_open.
    _enforcement_mode: str = "fail_closed_writes"
    _enforcement_mode_explicit: bool = False
    _env_enforcement_checked: bool = False
    _unconfigured_warning_emitted: bool = False
    # Set via set_enforcement_mode(mode, pin=True) / configure(fail_closed=...).
    # A pinned mode survives reset() (see reset()'s clear_pin argument): tests
    # and hosts that reset engine state between runs must keep the
    # operator-configured enforcement mode until configure() changes it or
    # conftest clears the pin, not silently revert to the default.
    _enforcement_mode_pinned: bool = False

    VALID_ENFORCEMENT_MODES = ("fail_open", "fail_closed_writes", "fail_closed")
    # "read"/"low" proceed under fail_closed_writes; anything else -- declared
    # "medium"/"high" or an unrecognised value -- is treated as a write.
    _LOW_IMPACT_LEVELS = frozenset({"read", "low"})

    @classmethod
    def set_enforcement_mode(cls, mode: str, pin: bool = False) -> None:
        """
        Set behaviour for actions checked while the policy layer cannot
        vouch for them (unconfigured, or policy evaluation itself raises).

        Args:
            mode: "fail_open" to permit unpoliced actions regardless of
                impact; "fail_closed_writes" (the default since 0.4.0) to
                block impact "medium"/"high" actions with
                NoPolicyForWriteError and permit "read"/"low" ones with a
                warning; or "fail_closed" to block every action with
                PolicyNotConfiguredError regardless of declared impact.
            pin: If True, this mode survives reset() (see reset()'s
                clear_pin argument) instead of reverting to the default.

        Raises:
            ValueError: if mode is not a recognised enforcement mode.
        """
        normalized = (mode or "").strip().lower()
        if normalized not in cls.VALID_ENFORCEMENT_MODES:
            raise ValueError(
                f"Unknown enforcement mode: {mode!r}. "
                f"Expected one of {cls.VALID_ENFORCEMENT_MODES}."
            )
        cls._enforcement_mode = normalized
        cls._enforcement_mode_explicit = True
        cls._unconfigured_warning_emitted = False
        if pin:
            cls._enforcement_mode_pinned = True
        logger.info("Policy enforcement mode set to '%s'", normalized)

    @classmethod
    def resolve_enforcement_mode(cls, fail_closed: Optional[bool] = None) -> str:
        """
        Resolve the effective enforcement mode for a single check.

        Precedence: an explicit ``fail_closed`` argument (e.g. from
        ``guarded_action(fail_closed=...)``) beats the process-wide mode
        (``set_enforcement_mode()`` / ``AGENT_SENTINEL_ENFORCEMENT`` / the
        "fail_closed_writes" default). ``fail_closed`` is a two-way switch --
        True resolves to "fail_closed", False to "fail_open"; there is no
        per-call way to force "fail_closed_writes", since it is already the
        process-wide default. Unlike ``set_enforcement_mode()``, this does
        not mutate global state -- callers that want a lasting change should
        pin it instead.
        """
        if fail_closed is not None:
            return "fail_closed" if fail_closed else "fail_open"
        return cls.get_enforcement_mode()

    @classmethod
    def get_enforcement_mode(cls) -> str:
        """
        Return the active enforcement mode, consulting the environment once.

        AGENT_SENTINEL_ENFORCEMENT may be set to "fail_open",
        "fail_closed_writes", or "fail_closed". An explicit
        set_enforcement_mode() call always wins over the environment.

        An unrecognised value raises ValueError, every call, not just the
        first: an operator who typed it was turning enforcement on, and
        falling back to fail_open would land permissive.
        """
        if not cls._enforcement_mode_explicit and not cls._env_enforcement_checked:
            raw = os.environ.get("AGENT_SENTINEL_ENFORCEMENT", "").strip().lower()
            if raw:
                if raw in cls.VALID_ENFORCEMENT_MODES:
                    cls._enforcement_mode = raw
                    cls._env_enforcement_checked = True
                    logger.info(
                        "Policy enforcement mode set from environment: '%s'", raw
                    )
                else:
                    # Deliberately do NOT mark the env as checked. Setting the
                    # flag before raising made the typo throw exactly once and
                    # then silently run fail_open for the rest of the process —
                    # the precise outcome this raise exists to prevent.
                    # Raise rather than fall back. An operator who typed this
                    # was trying to turn enforcement ON; silently landing on
                    # fail_open is the wrong direction to fail for a security
                    # control, and it matches backend_from_env's contract for
                    # the same class of typo.
                    raise ValueError(
                        f"Invalid AGENT_SENTINEL_ENFORCEMENT={raw!r}. "
                        f"Expected one of {cls.VALID_ENFORCEMENT_MODES}."
                    )
            else:
                cls._env_enforcement_checked = True
        return cls._enforcement_mode

    @classmethod
    def _is_write_impact(cls, impact: Optional[str]) -> bool:
        """
        True for anything "fail_closed_writes" should block: declared
        "medium"/"high" impact, a missing impact (guarded_action's own
        default is "medium"), or an unrecognised value -- an unrecognised
        impact string is exactly the kind of degraded state
        "fail_closed_writes" exists to be conservative about.
        """
        return (impact or "medium") not in cls._LOW_IMPACT_LEVELS

    @classmethod
    def _handle_unconfigured(
        cls,
        action: str,
        fail_closed: Optional[bool] = None,
        impact: str = "medium",
    ) -> None:
        """
        Decide what happens when an action is checked with no policy loaded.

        Returns normally to permit the action; raises to block it. Under
        "fail_closed_writes" (the default), the outcome also depends on the
        action's declared impact.

        Args:
            action: Name of the action being checked.
            fail_closed: Per-call override. See resolve_enforcement_mode().
            impact: The action's declared impact ("read", "low", "medium",
                or "high"). Only consulted under "fail_closed_writes".

        Raises:
            PolicyNotConfiguredError: when the resolved enforcement mode is
                "fail_closed".
            NoPolicyForWriteError: when the resolved enforcement mode is
                "fail_closed_writes" and impact is "medium"/"high".
        """
        mode = cls.resolve_enforcement_mode(fail_closed)
        if mode == "fail_closed":
            raise PolicyNotConfiguredError(action_name=action)
        if mode == "fail_closed_writes" and cls._is_write_impact(impact):
            raise NoPolicyForWriteError(action_name=action, impact=impact)

        # Warn once per configuration, not per call: an agent making thousands
        # of tool calls would otherwise flood the log and the warning would be
        # ignored, which defeats the point of raising it.
        if not cls._unconfigured_warning_emitted:
            cls._unconfigured_warning_emitted = True
            if mode == "fail_closed_writes":
                logger.warning(
                    "Policy engine is not configured; permitting action "
                    "'%s' (impact=%r) because enforcement mode is "
                    "'fail_closed_writes', which only blocks impact "
                    "'medium'/'high' actions. Load a policy, or set "
                    "enforcement mode to 'fail_closed' to block every "
                    "action regardless of declared impact.",
                    action, impact,
                )
            else:
                logger.warning(
                    "Policy engine is not configured; permitting all actions "
                    "(enforcement mode 'fail_open'). No budget, deny list, rate "
                    "limit, or evidence rule is being enforced. Load a policy, or "
                    "set enforcement mode to 'fail_closed' to block instead."
                )


    @classmethod
    def configure(
        cls,
        session_budget: Optional[float] = None,
        run_budget: Optional[float] = None,
        action_budgets: Optional[Dict[str, float]] = None,
        denied_actions: Optional[List[str]] = None,
        allowed_actions: Optional[List[str]] = None,
        rate_limits: Optional[Dict[str, Dict[str, int]]] = None,
        strict_mode: bool = True,
        evidence_requirements: Optional[Dict[str, List[str]]] = None,
        evidence_max_age_seconds: Optional[Dict[str, int]] = None,
        commit_actions: Optional[List[str]] = None,
        evidence_actions: Optional[List[str]] = None,
        argument_constraints: Optional[Dict[str, dict]] = None,
        grounding_rules: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        fail_closed: Optional[bool] = None,
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
            evidence_requirements: Dict of action -> list of required prior actions
            evidence_max_age_seconds: Dict of action -> max evidence age in seconds
            commit_actions: List of state-changing commit actions
            evidence_actions: List of evidence-producing actions
            argument_constraints: Dict of action -> JSON Schema for argument validation
            fail_closed: Pin the enforcement mode for this process (see
                set_enforcement_mode(pin=True)). True pins "fail_closed",
                False pins "fail_open", None (default) leaves it unchanged.
                Passing this alone, with no budget/deny-list/allow-list
                argument, does NOT install a policy: doing so would mark the
                engine "configured" with an empty, permissive PolicyConfig,
                which defeats the point of fail_closed.
        """
        if fail_closed is not None:
            cls.set_enforcement_mode(
                "fail_closed" if fail_closed else "fail_open", pin=True
            )

        has_policy_args = any(
            v is not None
            for v in (
                session_budget,
                run_budget,
                action_budgets,
                denied_actions,
                allowed_actions,
                rate_limits,
                evidence_requirements,
                evidence_max_age_seconds,
                commit_actions,
                evidence_actions,
                argument_constraints,
                grounding_rules,
            )
        )
        if not has_policy_args:
            logger.debug(
                "PolicyEngine.configure() called with no policy arguments; "
                "no PolicyConfig installed (intentional no-op, e.g. a "
                "fail_closed-only call)"
            )
            return

        cls._config = PolicyConfig(
            session_budget=session_budget,
            run_budget=run_budget,
            action_budgets=action_budgets or {},
            denied_actions=denied_actions or [],
            allowed_actions=allowed_actions,
            rate_limits=rate_limits or {},
            strict_mode=strict_mode,
            evidence_requirements=evidence_requirements or {},
            evidence_max_age_seconds=evidence_max_age_seconds or {},
            commit_actions=commit_actions or [],
            evidence_actions=evidence_actions or [],
            argument_constraints=argument_constraints or {},
            grounding_rules=grounding_rules or {},
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

            # Parse evidence graph settings
            evidence = data.get("evidence", {})
            evidence_requirements = evidence.get("requirements", {})
            evidence_max_age_seconds = evidence.get("max_age_seconds", {})
            commit_actions = evidence.get("commit_actions", [])
            evidence_actions = evidence.get("evidence_actions", [])
            argument_constraints = evidence.get("argument_constraints", {})
            grounding_rules = evidence.get("grounding_rules", {})

            cls.configure(
                session_budget=session_budget,
                run_budget=run_budget,
                action_budgets=action_budgets,
                denied_actions=denied_actions,
                allowed_actions=allowed_actions,
                rate_limits=rate_limits,
                strict_mode=strict_mode,
                evidence_requirements=evidence_requirements,
                evidence_max_age_seconds=evidence_max_age_seconds,
                commit_actions=commit_actions,
                evidence_actions=evidence_actions,
                argument_constraints=argument_constraints,
                grounding_rules=grounding_rules,
            )
            
            logger.info(f"Policy engine loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load policy from {config_path}: {e}")
    
    @classmethod
    def enable_remote_sync(
        cls,
        platform_url: Optional[str] = None,
        api_token: Optional[str] = None,
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
            platform_url: Platform API base URL. Defaults to the hosted
                platform; AGENT_SENTINEL_PLATFORM_URL / AGENT_SENTINEL_URL
                override it.
            api_token: JWT token or API key. Defaults to AGENT_SENTINEL_API_KEY.
            agent_id: Optional agent identifier for agent-specific policies
            run_id: Optional run identifier for run-specific policies
            refresh_interval: Seconds between policy refreshes (default: 300 = 5 min)
            cache_ttl: Cache time-to-live in seconds (default: 600 = 10 min)
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed. Remote policy sync disabled.")
            return
        
        cls._remote_config = RemotePolicyConfig(
            platform_url=platform_url,
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
        Sync policies once: try cache first, then remote, then stale cache.
        """
        if not cls._remote_config or not cls._policy_cache:
            return

        with cls._sync_lock:
            # Try loading from fresh cache first
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
                # Platform unreachable — fall back to stale cache if available
                stale = cls._policy_cache.load_stale()
                if stale:
                    cls._apply_remote_policies(stale)
                else:
                    logger.warning("Failed to fetch policies from platform and no cache available")
    
    @classmethod
    def _fetch_policies_from_platform(cls) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch policies from platform API.

        Returns:
            List of policy dictionaries or None on failure.
            Handles both the legacy list response and the current envelope
            format {"policies": [...], "conflicts": [...]}.
            Logs any conflicts returned by the platform as warnings.
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

            headers = build_identity_headers(
                cls._remote_config.api_token,
                agent_id=cls._remote_config.agent_id,
                run_id=cls._remote_config.run_id,
            )

            response = httpx.get(url, params=params, headers=headers, timeout=10.0)

            if response.status_code != 200:
                logger.error(f"Failed to fetch policies: {response.status_code}")
                return None

            data = response.json()

            # Handle envelope format {"policies": [...], "conflicts": [...]}
            if isinstance(data, dict):
                policies = data.get("policies", [])
                conflicts = data.get("conflicts", [])
                if conflicts:
                    logger.warning(
                        f"Policy conflicts detected ({len(conflicts)}) — "
                        f"active policies have settings that conflict and were silently merged. "
                        f"Review and resolve in the console:"
                    )
                    for c in conflicts:
                        logger.warning(
                            f"  CONFLICT [{c.get('field')}] {c.get('description')} "
                            f"(resolved to: {c.get('resolved_value')})"
                        )
                return policies

            # Legacy: plain list
            if isinstance(data, list):
                return data

            logger.error(f"Unexpected policy sync response format: {type(data)}")
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

            # Merge approval tags (union)
            if policy.get("approval_tags"):
                cls._config.approval_tags = list(set(
                    cls._config.approval_tags + policy["approval_tags"]
                ))

            # Merge approval risk levels (union)
            if policy.get("approval_risk_levels"):
                cls._config.approval_risk_levels = list(set(
                    cls._config.approval_risk_levels + policy["approval_risk_levels"]
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

            # Merge evidence requirements (union of all requirements per action)
            if policy.get("evidence_requirements"):
                for action, required in policy["evidence_requirements"].items():
                    if action not in cls._config.evidence_requirements:
                        cls._config.evidence_requirements[action] = list(required)
                    else:
                        cls._config.evidence_requirements[action] = list(set(
                            cls._config.evidence_requirements[action] + required
                        ))

            # Merge evidence max age (most restrictive = minimum)
            if policy.get("evidence_max_age_seconds"):
                for action, max_age in policy["evidence_max_age_seconds"].items():
                    if action not in cls._config.evidence_max_age_seconds:
                        cls._config.evidence_max_age_seconds[action] = max_age
                    else:
                        cls._config.evidence_max_age_seconds[action] = min(
                            cls._config.evidence_max_age_seconds[action], max_age
                        )

            # Merge commit/evidence action lists (union)
            if policy.get("commit_actions"):
                cls._config.commit_actions = list(set(
                    cls._config.commit_actions + policy["commit_actions"]
                ))
            if policy.get("evidence_actions"):
                cls._config.evidence_actions = list(set(
                    cls._config.evidence_actions + policy["evidence_actions"]
                ))

            # Merge argument constraints (per-action, last policy wins)
            if policy.get("argument_constraints"):
                for action, constraints in policy["argument_constraints"].items():
                    cls._config.argument_constraints[action] = constraints

            # Merge grounding rules (per-action, last policy wins)
            if policy.get("grounding_rules"):
                for action, rules in policy["grounding_rules"].items():
                    cls._config.grounding_rules[action] = rules
    
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
                cls._refresh_from_platform()
            except Exception as e:
                logger.error(f"Error in policy sync loop: {e}")

    @classmethod
    def _refresh_from_platform(cls) -> None:
        """
        Force-fetch policies from platform and update cache.
        Used by background sync to always get fresh data, bypassing cache.
        Also resets _config so deleted policies don't persist.
        """
        if not cls._remote_config or not cls._policy_cache:
            return

        with cls._sync_lock:
            policies = cls._fetch_policies_from_platform()
            if policies is not None:
                cls._policy_cache.save(policies, cls._remote_config.cache_ttl)
                cls._config = None  # Reset so stale merged state doesn't persist
                cls._initialized = False
                cls._apply_remote_policies(policies)
                logger.debug(f"Background refresh: applied {len(policies)} policies from platform")
            else:
                logger.warning("Background refresh: platform unreachable, keeping current config")
    
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
    def check_action(
        cls,
        action: str,
        cost: float,
        kwargs: Optional[Dict[str, Any]] = None,
        fail_closed: Optional[bool] = None,
        impact: str = "medium",
    ) -> None:
        """
        Check if an action is permitted under current policy.

        This is called BEFORE the action executes. If any policy is
        violated, raises an exception to prevent execution.

        Args:
            action: Name of the action to check
            cost: Proposed cost of the action in USD
            kwargs: Optional action arguments (for evidence and argument constraint checks)
            fail_closed: Per-call enforcement override. See
                resolve_enforcement_mode().
            impact: Declared impact ("read", "low", "medium", or "high").
                Only consulted when the policy layer cannot vouch for the
                action (unconfigured or an evaluation error), under the
                default "fail_closed_writes" mode.

        Raises:
            PolicyViolationError: If action is denied, rate limit exceeded,
                the policy layer could not vouch for a write-impact action
                under fail_closed_writes (reason_code NO_POLICY_FOR_WRITE),
                or policy evaluation itself raised unexpectedly under
                fail_closed (reason_code EVALUATION_ERROR)
            BudgetExceededError: If budget would be exceeded
            EvidenceViolationError: If required evidence is missing or stale
        """
        # No policy loaded: behaviour depends on enforcement mode and, under
        # fail_closed_writes (the default), on the action's declared impact.
        if not cls.is_configured():
            cls._handle_unconfigured(action, fail_closed=fail_closed, impact=impact)
            return

        config = cls._config

        try:
            # 1. Check denied/allowed lists
            if action in config.denied_actions:
                raise PolicyViolationError(
                    f"Action '{action}' is on the denied list",
                    policy_rule="denied_actions",
                    action_name=action,
                    reason_code="DENIED_ACTION",
                    recoverable=False,
                )

            if config.allowed_actions and action not in config.allowed_actions:
                raise PolicyViolationError(
                    f"Action '{action}' is not on the allowed list. "
                    f"Permitted: {config.allowed_actions}",
                    policy_rule="allowed_actions",
                    action_name=action,
                    reason_code="NOT_IN_ALLOWLIST",
                    safe_alternatives=list(config.allowed_actions),
                    recoverable=False,
                )

            # 2. Check rate limits
            if action in config.rate_limits:
                limits = config.rate_limits[action]
                max_count = limits.get("max_count", 999999)
                window_seconds = limits.get("window_seconds", 60)
                cls._rate_limiter.check_rate_limit(action, max_count, window_seconds)

            # 3-5. Budgets. Reads and compares without reserving, so concurrent
            # callers can each pass before any of them records. check_and_reserve()
            # closes that window; this path stays for callers that only want to ask.
            cls._check_budgets_readonly(action, cost, config)

            # 6-8. Evidence, argument constraints, groundedness.
            cls._check_evidence_and_args(action, config, kwargs)
        except (PolicyViolationError, BudgetExceededError, EvidenceViolationError):
            raise
        except Exception as exc:  # policy evaluation itself is broken
            cls._raise_evaluation_error(action, fail_closed, exc, impact=impact)

    @classmethod
    def check_and_reserve(
        cls,
        action: str,
        cost: float,
        kwargs: Optional[Dict[str, Any]] = None,
        fail_closed: Optional[bool] = None,
        impact: str = "medium",
    ) -> None:
        """
        Check an action and atomically reserve its cost against every budget.

        Same guarantees as ``check_action`` for deny lists, allowlists, rate
        limits, guardrails and evidence, but the budget comparison and the
        spend record happen inside a single lock hold. That is what makes a
        budget a real cap when workers run concurrently: reading the total and
        recording the spend as two steps lets every worker pass the same check
        before any of them records.

        The caller must NOT separately record this cost — reserving records it.

        Args:
            action: Name of the action to check
            cost: Proposed cost of the action in USD
            kwargs: Optional action arguments
            fail_closed: Per-call enforcement override. See
                resolve_enforcement_mode().
            impact: Declared impact ("read", "low", "medium", or "high").
                See check_action.

        Raises:
            PolicyViolationError, BudgetExceededError, EvidenceViolationError,
            PolicyNotConfiguredError: as check_action.
        """
        if not cls.is_configured():
            cls._handle_unconfigured(action, fail_closed=fail_closed, impact=impact)
            # Unpoliced but still counted, so totals stay accurate.
            CostTracker.add_cost(action, cost)
            return

        config = cls._config

        # Everything that can reject runs before the reservation, or a blocked
        # action consumes budget it never spent.
        try:
            cls._check_non_budget_rules(action, config)
            cls._check_evidence_and_args(action, config, kwargs)
        except (PolicyViolationError, BudgetExceededError, EvidenceViolationError):
            raise
        except Exception as exc:  # policy evaluation itself is broken
            cls._raise_evaluation_error(action, fail_closed, exc, impact=impact)

        try:
            rejection = CostTracker.try_reserve(
                action,
                cost,
                session_budget=config.session_budget,
                run_budget=config.run_budget,
                action_budget=config.action_budgets.get(action),
            )
        except (PolicyViolationError, BudgetExceededError, EvidenceViolationError):
            raise
        except Exception as exc:  # policy evaluation itself is broken
            cls._raise_evaluation_error(action, fail_closed, exc, impact=impact)
            return  # unreachable: _raise_evaluation_error always raises

        if rejection is not None:
            raise BudgetExceededError(
                f"{rejection.budget_type.capitalize()} budget exceeded: "
                f"{rejection.spent:.4f} > {rejection.limit:.4f} USD",
                spent=rejection.spent,
                limit=rejection.limit,
                budget_type=rejection.budget_type,
            )

    @classmethod
    def _raise_evaluation_error(
        cls,
        action: str,
        fail_closed: Optional[bool],
        exc: Exception,
        impact: str = "medium",
    ) -> None:
        """
        Handle an unexpected exception raised while evaluating policy.

        Under "fail_closed", always wraps it in a PolicyViolationError
        (reason_code EVALUATION_ERROR) so a bug in policy evaluation blocks
        the action rather than crashing past the control entirely. Under the
        default "fail_closed_writes", the same wrap happens but only for
        impact "medium"/"high" (reason_code NO_POLICY_FOR_WRITE instead) --
        a broken evaluator is exactly the "can't vouch for this" state that
        mode exists to be conservative about for writes, while reads keep
        working. Under "fail_open", re-raises unchanged — today's behaviour,
        preserved.

        Always raises; never returns.
        """
        mode = cls.resolve_enforcement_mode(fail_closed)
        if mode == "fail_open":
            raise exc
        if mode == "fail_closed_writes":
            if not cls._is_write_impact(impact):
                raise exc
            raise NoPolicyForWriteError(
                action_name=action,
                impact=impact,
                message=(
                    f"Policy evaluation for '{action}' raised an unexpected "
                    f"error ({exc}) and enforcement mode is "
                    f"'fail_closed_writes', so the impact={impact!r} action "
                    "was blocked."
                ),
            ) from exc
        raise PolicyViolationError(
            f"Policy evaluation for '{action}' raised an unexpected error: {exc}",
            policy_rule="evaluation_error",
            action_name=action,
            reason_code="EVALUATION_ERROR",
            recoverable=False,
        ) from exc

    @classmethod
    def _check_non_budget_rules(cls, action: str, config: Any) -> None:
        """Deny list, allowlist and rate limit. Shared by both entry points."""
        if action in config.denied_actions:
            raise PolicyViolationError(
                f"Action '{action}' is on the denied list",
                policy_rule="denied_actions",
                action_name=action,
                reason_code="DENIED_ACTION",
                recoverable=False,
            )

        if config.allowed_actions and action not in config.allowed_actions:
            raise PolicyViolationError(
                f"Action '{action}' is not on the allowed list. "
                f"Permitted: {config.allowed_actions}",
                policy_rule="allowed_actions",
                action_name=action,
                reason_code="NOT_IN_ALLOWLIST",
                safe_alternatives=list(config.allowed_actions),
                recoverable=False,
            )

        if action in config.rate_limits:
            limits = config.rate_limits[action]
            cls._rate_limiter.check_rate_limit(
                action,
                limits.get("max_count", 999999),
                limits.get("window_seconds", 60),
            )

    @classmethod
    def _check_budgets_readonly(cls, action: str, cost: float, config: Any) -> None:
        """Budget comparison without reserving. See check_and_reserve."""
        # 3. Check session budget
        if config.session_budget is not None:
            current_session = CostTracker.get_session_total()
            session_projected = current_session + cost
            if config.session_budget > 0:
                session_pct = (session_projected / config.session_budget) * 100
                warning_thresholds = getattr(config, "budget_warning_thresholds", [80])
                for threshold in sorted(warning_thresholds, reverse=True):
                    if session_pct >= threshold:
                        InterventionTracker.record(
                            intervention_type=InterventionType.BUDGET_WARNING,
                            outcome=InterventionOutcome.WARNED,
                            action_name=action,
                            reason=(
                                f"Session budget {threshold}% threshold reached "
                                f"({session_pct:.1f}% used, ${session_projected:.4f} of ${config.session_budget:.4f})"
                            ),
                        )
                        break  # Only emit the highest crossed threshold
            if session_projected > config.session_budget:
                raise BudgetExceededError(
                    f"Session budget exceeded: {current_session:.4f} + {cost:.4f} "
                    f"> {config.session_budget:.4f} USD",
                    spent=session_projected,
                    limit=config.session_budget,
                    budget_type="session"
                )

        # 4. Check run budget
        if config.run_budget is not None:
            current_run = CostTracker.get_run_total()
            run_projected = current_run + cost
            if config.run_budget > 0:
                run_pct = (run_projected / config.run_budget) * 100
                warning_thresholds = getattr(config, "budget_warning_thresholds", [80])
                for threshold in sorted(warning_thresholds, reverse=True):
                    if run_pct >= threshold:
                        InterventionTracker.record(
                            intervention_type=InterventionType.BUDGET_WARNING,
                            outcome=InterventionOutcome.WARNED,
                            action_name=action,
                            reason=(
                                f"Run budget {threshold}% threshold reached "
                                f"({run_pct:.1f}% used, ${run_projected:.4f} of ${config.run_budget:.4f})"
                            ),
                        )
                        break  # Only emit the highest crossed threshold
            if run_projected > config.run_budget:
                raise BudgetExceededError(
                    f"Run budget exceeded: {current_run:.4f} + {cost:.4f} "
                    f"> {config.run_budget:.4f} USD",
                    spent=run_projected,
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
    def _check_evidence_and_args(
        cls, action: str, config: Any, kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Evidence requirements, argument constraints and groundedness.

        Split out so check_and_reserve can run these BEFORE reserving budget:
        a reservation must be the last thing that can fail.
        """
        # 6. Check evidence requirements
        if action in config.evidence_requirements:
            from .evidence import EvidenceTracker
            required = config.evidence_requirements[action]
            max_age = config.evidence_max_age_seconds.get(action)
            all_met, missing, stale = EvidenceTracker.check_requirements(required, max_age)
            if not all_met:
                raise EvidenceViolationError(
                    message=f"Action '{action}' requires evidence from: {missing + stale}",
                    action_name=action,
                    missing_requirements=missing,
                    required_prior_actions=required,
                    stale_evidence=stale,
                )

        # 7. Check argument constraints
        if action in config.argument_constraints and kwargs:
            from .constraints import validate_constraints
            violations = validate_constraints(kwargs, config.argument_constraints[action])
            if violations:
                raise EvidenceViolationError(
                    message=f"Action '{action}' argument constraints violated: {violations}",
                    action_name=action,
                    argument_violations=violations,
                )

        # 7b. PII guardrail (built-in)
        pii_rule = config.pii_rules.get(action)
        if pii_rule is None and config.pii_default_enabled:
            from .guardrails.pii import PIIRule as _PIIRule
            pii_rule = _PIIRule()
        if pii_rule is not None and kwargs:
            from .guardrails.pii import PIIGuard
            matches = PIIGuard.scan_kwargs(kwargs, pii_rule)
            if matches:
                cats = sorted({m.category for m in matches})
                fields = sorted({m.field_path for m in matches})
                retry_hint = (
                    f"Remove PII (categories={cats}) from fields={fields} "
                    "before retrying. Replace with tokenized references, "
                    "anonymized IDs, or placeholder values."
                )
                err = PolicyViolationError(
                    f"Action '{action}' blocked by PII guardrail: "
                    f"categories={cats} fields={fields}",
                    policy_rule="pii_guard",
                    action_name=action,
                    details={
                        "reason_code": "PII_DETECTED",
                        "categories": cats,
                        "fields": fields,
                        "matches": [m.to_dict() for m in matches],
                        "retry_guidance": retry_hint,
                        "safe_alternatives": [
                            "use an anonymized or tokenized identifier",
                            "route through an approved redaction helper",
                            "escalate for human review via approval inbox",
                        ],
                        "recoverable": True,
                    },
                )
                raise err

        # 7c. Content moderation guardrail (built-in)
        mod_rule = config.moderation_rules.get(action)
        if mod_rule is None and config.moderation_default_enabled:
            from .guardrails.moderation import ModerationRule as _ModerationRule
            mod_rule = _ModerationRule()
        if mod_rule is not None and kwargs:
            from .guardrails.moderation import ModerationGuard
            results = ModerationGuard.scan_kwargs(kwargs, mod_rule)
            blocking = ModerationGuard.should_block(results, mod_rule)
            if blocking:
                cats = sorted({c for r in blocking for c in r.categories})
                fields = sorted({r.field_path for r in blocking})
                retry_hint = (
                    f"Rephrase or remove disallowed content (categories={cats}) "
                    f"from fields={fields}. Do not paraphrase to bypass moderation."
                )
                raise PolicyViolationError(
                    f"Action '{action}' blocked by content moderation: "
                    f"categories={cats} fields={fields}",
                    policy_rule="moderation_guard",
                    action_name=action,
                    details={
                        "reason_code": "CONTENT_MODERATED",
                        "categories": cats,
                        "fields": fields,
                        "results": [r.to_dict() for r in blocking],
                        "retry_guidance": retry_hint,
                        "safe_alternatives": [
                            "respond with a refusal and escalate to a human",
                            "clarify intent to the user and request a different request",
                        ],
                        "recoverable": True,
                    },
                )

        # 7d. Loop protection guardrail (built-in)
        loop_rule = config.loop_rules.get(action)
        if loop_rule is None and config.loop_default_enabled:
            from .guardrails.loop_detector import LoopRule as _LoopRule
            loop_rule = _LoopRule()
        if loop_rule is not None:
            from .guardrails.loop_detector import DEFAULT_LOOP_GUARD
            det = DEFAULT_LOOP_GUARD.record_and_check(action, kwargs, loop_rule)
            if det is not None:
                raise PolicyViolationError(
                    f"Action '{action}' blocked by loop protection: "
                    f"{det.count} repeats in {det.window_seconds}s",
                    policy_rule="loop_guard",
                    action_name=action,
                    details={
                        "reason_code": "LOOP_DETECTED",
                        "detection": det.to_dict(),
                        "retry_guidance": det.break_out_hint,
                        "prior_attempts": det.count,
                        "safe_alternatives": [
                            "vary the arguments before retrying",
                            "escalate to a human via the approval inbox",
                            "stop retrying and return the current best-effort result",
                        ],
                        "recoverable": True,
                    },
                )

        # 8. Check groundedness
        if action in config.grounding_rules and kwargs:
            from .evidence import EvidenceTracker
            grounded, ungrounded_details = EvidenceTracker.check_groundedness(
                action_kwargs=kwargs,
                grounding_rules=config.grounding_rules[action],
                max_age_seconds=config.evidence_max_age_seconds.get(action),
            )
            if not grounded:
                field_names = [d["field"] for d in ungrounded_details]
                error = EvidenceViolationError(
                    message=f"Action '{action}' arguments not grounded in evidence: {field_names}",
                    action_name=action,
                    argument_violations=[
                        f"'{d['field']}' value '{d['actual_value']}' not found in {d['expected_source']}"
                        for d in ungrounded_details
                    ],
                    retry_guidance=f"Ensure these arguments match prior evidence: {field_names}",
                )
                error.remediation.reason_code = "UNGROUNDED_ARGUMENT"
                raise error

    @classmethod
    def apply_config(cls, config: Optional[PolicyConfig]) -> None:
        """
        Install a previously captured PolicyConfig.

        The counterpart to get_config(). configure() only accepts a subset of
        PolicyConfig's fields, so it cannot faithfully restore a snapshot —
        approval settings and evidence rules would be silently dropped. Use
        this to save and restore policy around code that calls reset():

            saved = PolicyEngine.get_config()
            ...                       # something that may reset the engine
            PolicyEngine.apply_config(saved)

        Passing None leaves the engine unconfigured, matching reset().
        """
        cls._config = config
        cls._initialized = config is not None

    @classmethod
    def get_config(cls) -> Optional[PolicyConfig]:
        """
        Get current policy configuration.
        
        Returns:
            PolicyConfig object or None if not configured
        """
        return cls._config
    
    @classmethod
    def requires_approval(
        cls,
        action: str,
        cost: float,
        tags: Optional[List[str]] = None,
        risk_level: Optional[str] = None,
    ) -> bool:
        """
        Check if an action requires human approval.

        Args:
            action: Name of the action
            cost: Cost of the action in USD
            tags: Tags associated with the action
            risk_level: Risk level of the action (e.g. "LOW", "MEDIUM", "HIGH", "CRITICAL")

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

        # Check if any of the action's tags match approval_tags
        if config.approval_tags and tags:
            if any(t in config.approval_tags for t in tags):
                return True

        # Check if the action's risk level matches approval_risk_levels
        if config.approval_risk_levels and risk_level:
            if risk_level.upper() in [r.upper() for r in config.approval_risk_levels]:
                return True

        # Check if cost exceeds threshold
        if config.approval_threshold_usd is not None:
            if cost >= config.approval_threshold_usd:
                return True

        # If no specific criteria configured,
        # require_approval=True means all actions need approval
        if (not config.approval_actions
            and not config.approval_tags
            and not config.approval_risk_levels
            and config.approval_threshold_usd is None):
            return True

        return False
    
    @classmethod
    def get_approval_timeout(cls) -> int:
        """Get the configured approval timeout in seconds."""
        if cls._config:
            return cls._config.approval_timeout_seconds
        return 3600  # Default 1 hour
    
    @classmethod
    def reset(cls, *, clear_pin: bool = False) -> None:
        """
        Reset policy engine.
        
        Useful for testing. In production, you typically configure once
        at startup.

        Args:
            clear_pin: If True, also clear an enforcement mode pinned via
                set_enforcement_mode(pin=True) / configure(fail_closed=...),
                reverting to the SDK's own default ("fail_closed_writes",
                see the _enforcement_mode class attribute above). Defaults
                to False: the pin must survive reset() so tests and hosts
                that reset engine state between runs keep the
                operator-configured enforcement mode until configure()
                changes it or conftest clears the pin. Test isolation
                should pass True so a pinned mode cannot leak between
                tests.
        """
        cls.stop_remote_sync()
        cls._config = None
        cls._initialized = False
        cls._remote_config = None
        cls._policy_cache = None
        cls._rate_limiter.reset()
        # Enforcement state is global too: leaving it set meant opting
        # into fail_closed poisoned everything later in the process. Revert
        # to the SDK's own default rather than a hard-coded "fail_open" --
        # a bare reset() is not an explicit opt-in to fail-open, and this is
        # reachable by any host that resets engine state between agent runs.
        if clear_pin or not cls._enforcement_mode_pinned:
            cls._enforcement_mode = "fail_closed_writes"
            cls._enforcement_mode_explicit = False
            cls._env_enforcement_checked = False
            cls._enforcement_mode_pinned = False
        cls._unconfigured_warning_emitted = False
