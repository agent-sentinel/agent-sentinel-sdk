"""
Shared test isolation for the SDK suite.

Two pieces of global state can leak between tests and, worse, in from the
developer's or CI runner's shell:

- ``AGENT_SENTINEL_ENFORCEMENT`` — exporting ``fail_closed`` turns 412 passing
  tests into 55 failures, because an unconfigured PolicyEngine starts blocking.
- ``AGENT_SENTINEL_COST_BACKEND`` / ``AGENT_SENTINEL_COST_PATH`` — pointing at
  a file backend makes the whole run share one on-disk counter.

A pinned enforcement mode (``set_enforcement_mode(pin=True)`` /
``configure(fail_closed=...)``) deliberately survives a plain
``PolicyEngine.reset()`` so it doesn't revert mid-process. Test isolation
therefore needs ``clear_pin=True`` here, or a pinned mode would leak into
every test that runs after it.

Since 0.4.0 the SDK's own out-of-the-box default is ``"fail_closed_writes"``
(see the ``PolicyEngine._enforcement_mode`` class attribute): a process that
never calls ``reset()`` or ``set_enforcement_mode()`` now blocks impact
"medium"/"high" actions when unconfigured. ``PolicyEngine.reset()`` itself
now reverts an unpinned mode to that same default (see its docstring) -- a
bare ``reset()`` is not an explicit opt-in to fail-open. Most of this suite
predates 0.4.0 and was written against the old fail-open-when-unconfigured
assumption and doesn't care about enforcement mode at all, so this fixture
pins ``"fail_open"`` explicitly for every test, right after ``reset()``.
It assigns ``PolicyEngine._enforcement_mode`` directly rather than calling
``set_enforcement_mode()``: the public setter marks the mode "explicit",
which would stop a test-supplied ``AGENT_SENTINEL_ENFORCEMENT`` from being
picked up by ``get_enforcement_mode()``. ``test_fail_closed.py`` is where
"fail_closed_writes" itself gets exercised, documenting the *new* default --
its tests call ``set_enforcement_mode("fail_closed_writes")`` directly.

Neutralising them here means the suite result depends on the code under test,
not on the ambient environment.
"""
from __future__ import annotations

import pytest

from agent_sentinel.cost import CostTracker
from agent_sentinel.policy import PolicyEngine

_SENTINEL_ENV_VARS = (
    "AGENT_SENTINEL_ENFORCEMENT",
    "AGENT_SENTINEL_COST_BACKEND",
    "AGENT_SENTINEL_COST_PATH",
    # 0.5.0: platform URL / API key now fall back to these, so an exported
    # value would silently redirect clients built without explicit arguments.
    "AGENT_SENTINEL_PLATFORM_URL",
    "AGENT_SENTINEL_URL",
    "AGENT_SENTINEL_API_KEY",
)


@pytest.fixture(autouse=True)
def isolate_sentinel_globals(monkeypatch):
    """Clear ambient env and global engine/counter state around every test."""
    for var in _SENTINEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    PolicyEngine.reset(clear_pin=True)
    # reset() now reverts to the SDK's "fail_closed_writes" default, not
    # "fail_open" -- bypass the public setter so this doesn't also mark the
    # mode "explicit" (see the module docstring above).
    PolicyEngine._enforcement_mode = "fail_open"
    CostTracker.set_backend(None)

    yield

    PolicyEngine.reset(clear_pin=True)
    PolicyEngine._enforcement_mode = "fail_open"
    CostTracker.set_backend(None)
