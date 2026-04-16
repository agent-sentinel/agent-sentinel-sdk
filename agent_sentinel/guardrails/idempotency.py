"""
Idempotency guardrail.

Provides a keyed cache so that an action called twice with the same
idempotency key (scoped to a run) returns the prior result instead of
executing again. This is a first-class "idempotency" primitive — distinct from
our existing `is_commit`-style commit-repeat detection, which only flags
duplicate commits; here we *dedup and replay*.

Design:
- `IdempotencyCache` is a simple in-process thread-safe dict keyed by
  (run_id, key). A TTL sweep removes stale entries.
- Integration point: the `@guarded_action` decorator consults the cache
  before executing; on hit, raises `IdempotencyHit` which the decorator
  catches to short-circuit and return the cached value.
- Hits are also recorded as an intervention (new InterventionType.IDEMPOTENT_REPLAY).

Scope: in-process only. Distributed idempotency (shared across workers)
would require a backing store on the platform; that is a separate ticket.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class _Entry:
    value: Any
    stored_at: float
    expires_at: float


class IdempotencyHit(Exception):
    """
    Raised by the cache when an idempotency key was previously seen.
    Carries the prior result so the decorator can return it transparently.
    """

    def __init__(self, action: str, key: str, value: Any, stored_at: float) -> None:
        super().__init__(f"Idempotency hit for action={action!r} key={key!r}")
        self.action = action
        self.key = key
        self.value = value
        self.stored_at = stored_at


class IdempotencyCache:
    """
    In-process idempotency cache.

    Usage (from guard.py):
        hit = cache.lookup(run_id, key)                 # raises IdempotencyHit if present
        result = fn(*args, **kwargs)
        cache.store(run_id, key, result, ttl_seconds)
    """

    def __init__(self, default_ttl_seconds: float = 3600.0) -> None:
        self._entries: Dict[Tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def lookup(self, run_id: Optional[str], key: Optional[str], action: str) -> None:
        """
        If (run_id, key) has a live entry, raise IdempotencyHit.
        Otherwise return None.
        """
        if not key:
            return None
        scope = run_id or "__no_run__"
        now = time.time()
        with self._lock:
            entry = self._entries.get((scope, key))
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop((scope, key), None)
                return None
        raise IdempotencyHit(action=action, key=key, value=entry.value, stored_at=entry.stored_at)

    def store(
        self,
        run_id: Optional[str],
        key: Optional[str],
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if not key:
            return
        scope = run_id or "__no_run__"
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = time.time()
        with self._lock:
            self._entries[(scope, key)] = _Entry(
                value=value,
                stored_at=now,
                expires_at=now + ttl,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def purge_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for k, v in list(self._entries.items()):
                if v.expires_at <= now:
                    self._entries.pop(k, None)
                    removed += 1
        return removed


# Module-level default used by guard.py when a key is supplied
DEFAULT_IDEMPOTENCY_CACHE = IdempotencyCache()


def make_key(fn: Optional[Callable[..., str]], args: tuple, kwargs: dict) -> Optional[str]:
    """
    Resolve an idempotency key from a callable or return it directly.

    - If fn is None -> None
    - If fn is a string -> returned as-is (static key)
    - If fn is callable -> invoked with (*args, **kwargs); returned value (or None)
    """
    if fn is None:
        return None
    if isinstance(fn, str):
        return fn or None
    try:
        k = fn(*args, **kwargs)
    except Exception:
        return None
    return str(k) if k else None
