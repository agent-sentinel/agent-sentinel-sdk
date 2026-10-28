"""
Shared connection defaults for every SDK entry point that talks to the platform.

``DEFAULT_PLATFORM_URL`` is the only place in the package that spells the
hosted platform host. Everything else resolves its base URL through
``resolve_platform_url`` so callers never have to pass one:

    explicit argument > AGENT_SENTINEL_PLATFORM_URL > AGENT_SENTINEL_URL > default

Set either environment variable to point the SDK at a self-hosted or staging
platform.
"""
from __future__ import annotations

import os

from .errors import ConfigurationError

DEFAULT_PLATFORM_URL = "https://api.agentsentinel.dev"

PLATFORM_URL_ENV_VARS = ("AGENT_SENTINEL_PLATFORM_URL", "AGENT_SENTINEL_URL")
API_KEY_ENV_VAR = "AGENT_SENTINEL_API_KEY"


def _clean_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def resolve_platform_url(explicit: str | None = None) -> str:
    """Return the platform base URL (no trailing slash).

    Precedence: ``explicit`` > ``AGENT_SENTINEL_PLATFORM_URL`` >
    ``AGENT_SENTINEL_URL`` > the hosted platform. Blank values are skipped.
    """
    url = _clean_url(explicit)
    if url:
        return url
    for var in PLATFORM_URL_ENV_VARS:
        url = _clean_url(os.environ.get(var))
        if url:
            return url
    return DEFAULT_PLATFORM_URL


def resolve_api_token(explicit: str | None = None) -> str:
    """Return ``explicit`` if given, else ``AGENT_SENTINEL_API_KEY``.

    An explicitly passed value (even an empty string) is returned unchanged,
    preserving pre-0.5 behaviour for callers that always pass a token.

    Raises:
        ConfigurationError: if no token was passed and the env var is unset.
    """
    if explicit is not None:
        return explicit
    token = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not token:
        raise ConfigurationError(
            "No platform API key: pass api_token/api_key or set "
            f"{API_KEY_ENV_VAR}.",
            config_key=API_KEY_ENV_VAR,
        )
    return token
