"""
Kill Switch: Emergency shutdown for agents, runs, and missions.

Provides an SDK client that checks the platform's kill switch state
before executing guarded actions. Follows the SDK's fail-open philosophy:
if the platform is unreachable, actions proceed normally. Only blocks
when a kill is positively confirmed.
"""
from __future__ import annotations

import time
import logging
import threading
from typing import Optional, Dict
from dataclasses import dataclass

from .errors import AgentKilledError

logger = logging.getLogger("agent_sentinel")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class KillSwitchConfig:
    """Configuration for the kill switch client."""
    platform_url: str
    api_token: str
    cache_ttl: float = 5.0  # seconds to cache kill switch state
    enabled: bool = True


class KillSwitchClient:
    """
    Client for checking kill switch state against the platform.

    Fail-open: if the platform is unreachable, assumes not killed.
    Caches results briefly to avoid excessive network calls.

    Usage::

        KillSwitchClient.configure(
            platform_url="https://api.agentsentinel.dev",
            api_token="your-api-key",
        )

        # Check manually
        if KillSwitchClient.is_killed("agent", "writer-1"):
            print("Agent is killed!")

        # Or raise if killed (used by guard.py automatically)
        KillSwitchClient.check_or_raise(agent_id="writer-1")
    """

    _config: Optional[KillSwitchConfig] = None
    _cache: Dict[str, tuple[bool, float]] = {}  # key -> (is_killed, expires_at)
    _lock = threading.Lock()

    @classmethod
    def configure(
        cls,
        platform_url: str,
        api_token: str,
        cache_ttl: float = 5.0,
        enabled: bool = True,
    ) -> None:
        """Configure the kill switch client."""
        with cls._lock:
            cls._config = KillSwitchConfig(
                platform_url=platform_url.rstrip("/"),
                api_token=api_token,
                cache_ttl=cache_ttl,
                enabled=enabled,
            )
            cls._cache.clear()
        logger.info(f"Kill switch client configured for {platform_url}")

    @classmethod
    def is_configured(cls) -> bool:
        """Check if the kill switch client is configured and enabled."""
        return cls._config is not None and cls._config.enabled

    @classmethod
    def is_killed(cls, target_type: str, target_id: str) -> bool:
        """
        Check if a target is killed.

        Args:
            target_type: "agent", "run", or "mission"
            target_id: The identifier to check

        Returns:
            True if the target is killed, False otherwise.
            Returns False (fail-open) if platform is unreachable.
        """
        if not cls.is_configured() or not HTTPX_AVAILABLE:
            return False

        cache_key = f"{target_type}:{target_id}"

        # Check cache
        with cls._lock:
            if cache_key in cls._cache:
                killed, expires_at = cls._cache[cache_key]
                if time.monotonic() < expires_at:
                    return killed

        # Cache miss or expired -- query platform
        try:
            url = f"{cls._config.platform_url}/api/v1/kill-switch/check"
            headers = {"Authorization": f"ApiKey {cls._config.api_token}"}
            params = {"target_type": target_type, "target_id": target_id}

            with httpx.Client(timeout=5.0) as client:
                response = client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                killed = bool(data.get("is_killed", False))

                # Update cache
                with cls._lock:
                    cls._cache[cache_key] = (
                        killed,
                        time.monotonic() + cls._config.cache_ttl,
                    )

                return killed
            else:
                logger.warning(
                    f"Kill switch check returned {response.status_code}, "
                    f"failing open for {target_type}:{target_id}"
                )
                return False

        except Exception as e:
            # Fail-open: platform unreachable -> assume not killed
            logger.warning(
                f"Kill switch check failed ({e}), "
                f"failing open for {target_type}:{target_id}"
            )
            return False

    @classmethod
    def check_or_raise(
        cls,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> None:
        """
        Check kill switch for all provided targets. Raises AgentKilledError
        if any target is killed.

        Args:
            agent_id: Agent identifier to check
            run_id: Run identifier to check
            mission_id: Mission identifier to check

        Raises:
            AgentKilledError: If any target is killed
        """
        if not cls.is_configured():
            return

        for target_type, target_id in [
            ("agent", agent_id),
            ("run", run_id),
            ("mission", mission_id),
        ]:
            if target_id and cls.is_killed(target_type, target_id):
                raise AgentKilledError(
                    f"{target_type} '{target_id}' has been killed",
                    target_type=target_type,
                    target_id=target_id,
                )

    @classmethod
    def reset(cls) -> None:
        """Reset configuration and cache (for testing)."""
        with cls._lock:
            cls._config = None
            cls._cache.clear()
