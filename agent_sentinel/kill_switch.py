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
from dataclasses import dataclass

from .context import build_identity_headers
from .errors import AgentKilledError, KillSwitchUnavailableError
from ._defaults import resolve_api_token, resolve_platform_url

logger = logging.getLogger("agent_sentinel")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class KillSwitchConfig:
    """Configuration for the kill switch client.

    ``platform_url`` defaults to the hosted platform (overridable with
    AGENT_SENTINEL_PLATFORM_URL / AGENT_SENTINEL_URL) and ``api_token`` to
    AGENT_SENTINEL_API_KEY.
    """
    platform_url: str | None = None
    api_token: str | None = None
    cache_ttl: float = 5.0  # seconds to cache kill switch state
    enabled: bool = True

    def __post_init__(self) -> None:
        self.platform_url = resolve_platform_url(self.platform_url)
        self.api_token = resolve_api_token(self.api_token)


class KillSwitchClient:
    """
    Client for checking kill switch state against the platform.

    Fail-open: if the platform is unreachable, assumes not killed.
    Caches results briefly to avoid excessive network calls.

    Usage::

        KillSwitchClient.configure(api_token="your-api-key")

        # Check manually
        if KillSwitchClient.is_killed("agent", "writer-1"):
            print("Agent is killed!")

        # Or raise if killed (used by guard.py automatically)
        KillSwitchClient.check_or_raise(agent_id="writer-1")
    """

    _config: KillSwitchConfig | None = None
    _cache: dict[str, tuple[bool, float]] = {}  # key -> (is_killed, expires_at)
    _lock = threading.Lock()

    @classmethod
    def configure(
        cls,
        platform_url: str | None = None,
        api_token: str | None = None,
        cache_ttl: float = 5.0,
        enabled: bool = True,
    ) -> None:
        """Configure the kill switch client.

        ``platform_url`` defaults to the hosted platform (or the
        AGENT_SENTINEL_PLATFORM_URL / AGENT_SENTINEL_URL override);
        ``api_token`` defaults to AGENT_SENTINEL_API_KEY.
        """
        with cls._lock:
            cls._config = KillSwitchConfig(
                platform_url=platform_url,
                api_token=api_token,
                cache_ttl=cache_ttl,
                enabled=enabled,
            )
            cls._cache.clear()
        logger.info(f"Kill switch client configured for {cls._config.platform_url}")

    @classmethod
    def is_configured(cls) -> bool:
        """Check if the kill switch client is configured and enabled."""
        return cls._config is not None and cls._config.enabled

    @classmethod
    def is_killed(
        cls, target_type: str, target_id: str, fail_closed: bool = False
    ) -> bool:
        """
        Check if a target is killed.

        Args:
            target_type: "agent", "run", or "mission"
            target_id: The identifier to check
            fail_closed: If True, raise KillSwitchUnavailableError instead of
                assuming not-killed when the check itself cannot be completed
                (httpx missing, platform unreachable, non-200 response).

        Returns:
            True if the target is killed, False otherwise.
            Returns False (fail-open) if platform is unreachable and
            fail_closed is False.

        Raises:
            KillSwitchUnavailableError: if fail_closed is True and the check
                could not be completed.
        """
        if not cls.is_configured():
            return False

        if not HTTPX_AVAILABLE:
            if fail_closed:
                raise KillSwitchUnavailableError(
                    "httpx is not installed; kill switch cannot be checked "
                    f"for {target_type}:{target_id}.",
                    target_type=target_type,
                    target_id=target_id,
                )
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
            headers = build_identity_headers(cls._config.api_token)
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
                if fail_closed:
                    raise KillSwitchUnavailableError(
                        f"Kill switch check returned {response.status_code} "
                        f"for {target_type}:{target_id}.",
                        target_type=target_type,
                        target_id=target_id,
                    )
                logger.warning(
                    f"Kill switch check returned {response.status_code}, "
                    f"failing open for {target_type}:{target_id}"
                )
                return False

        except KillSwitchUnavailableError:
            raise
        except Exception as e:
            if fail_closed:
                raise KillSwitchUnavailableError(
                    f"Kill switch check failed ({e}) for "
                    f"{target_type}:{target_id}.",
                    target_type=target_type,
                    target_id=target_id,
                ) from e
            # Fail-open: platform unreachable -> assume not killed
            logger.warning(
                f"Kill switch check failed ({e}), "
                f"failing open for {target_type}:{target_id}"
            )
            return False

    @classmethod
    def check_or_raise(
        cls,
        agent_id: str | None = None,
        run_id: str | None = None,
        mission_id: str | None = None,
        fail_closed: bool = False,
    ) -> None:
        """
        Check kill switch for all provided targets. Raises AgentKilledError
        if any target is killed.

        Args:
            agent_id: Agent identifier to check
            run_id: Run identifier to check
            mission_id: Mission identifier to check
            fail_closed: If True, a kill switch check that cannot complete
                raises KillSwitchUnavailableError instead of assuming
                not-killed. See is_killed().

        Raises:
            AgentKilledError: If any target is killed
            KillSwitchUnavailableError: If fail_closed is True and a check
                could not be completed
        """
        if not cls.is_configured():
            return

        for target_type, target_id in [
            ("agent", agent_id),
            ("run", run_id),
            ("mission", mission_id),
        ]:
            if target_id and cls.is_killed(target_type, target_id, fail_closed=fail_closed):
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
