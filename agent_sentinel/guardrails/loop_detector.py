"""
Loop-protection guardrail.

Detects tight loops where an agent invokes the same action with
semantically-identical arguments N times within a sliding time window.
This provides "loop protection" as a first-class primitive
(distinct from rate limits, which are count-based regardless of args).

Design:
- `LoopGuard` keeps per-action, per-arg-hash deques of timestamps.
- `record_and_check(...)` is called from PolicyEngine.check_action
  *before* execution; if the sliding window shows >= threshold repeats,
  it returns a LoopDetection with the last-N arg diffs for self-repair.
- Singleton-lite: a module-level default instance is used by the engine;
  tests can instantiate their own.

Not concerned with cross-process loops (each process has its own state).
Distributed loop detection is out of scope for the SDK primitive.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


def _arg_hash(kwargs: Optional[Dict[str, Any]]) -> str:
    """Deterministic hash of kwargs for loop-equality comparison."""
    if not kwargs:
        return "∅"
    try:
        blob = json.dumps(kwargs, sort_keys=True, default=str)
    except Exception:
        blob = repr(sorted(kwargs.items(), key=lambda kv: kv[0]))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class LoopRule:
    """
    Per-action loop-detection rule.

    Attributes:
        threshold: Number of identical-argument repeats to trigger detection.
        window_seconds: Sliding window size.
        arg_exclude: Argument keys to exclude from the equality hash
            (e.g. per-call timestamps, idempotency nonces).
    """
    threshold: int = 5
    window_seconds: float = 10.0
    arg_exclude: Iterable[str] = field(default_factory=tuple)


@dataclass
class LoopDetection:
    """Result returned when a loop is detected."""
    action: str
    arg_hash: str
    count: int
    window_seconds: float
    first_seen_at: float
    last_seen_at: float
    recent_args: List[Dict[str, Any]] = field(default_factory=list)
    break_out_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "arg_hash": self.arg_hash,
            "count": self.count,
            "window_seconds": self.window_seconds,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "recent_args": self.recent_args,
            "break_out_hint": self.break_out_hint,
        }


class LoopGuard:
    """
    Per-action sliding-window loop detector.

    Thread-safe. Keeps at most `max_history` timestamps + arg-snapshots per
    (action, arg_hash) key to bound memory.
    """

    def __init__(self, max_history: int = 32) -> None:
        self._max_history = max_history
        self._state: Dict[Tuple[str, str], Deque[Tuple[float, Dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._state.clear()

    def record_and_check(
        self,
        action: str,
        kwargs: Optional[Dict[str, Any]],
        rule: LoopRule,
    ) -> Optional[LoopDetection]:
        """
        Record an attempted call and return a LoopDetection if the rule's
        threshold is exceeded in the sliding window. Otherwise returns None.
        """
        effective_kwargs = self._strip(kwargs, rule.arg_exclude)
        h = _arg_hash(effective_kwargs)
        now = time.monotonic()
        cutoff = now - rule.window_seconds

        with self._lock:
            key = (action, h)
            dq = self._state.get(key)
            if dq is None:
                dq = deque(maxlen=self._max_history)
                self._state[key] = dq
            # Evict old entries
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            dq.append((now, dict(effective_kwargs or {})))
            if len(dq) < rule.threshold:
                return None
            snapshot_ts = [ts for ts, _ in dq]
            snapshot_args = [args for _, args in list(dq)[-rule.threshold:]]

        return LoopDetection(
            action=action,
            arg_hash=h,
            count=len(snapshot_ts),
            window_seconds=rule.window_seconds,
            first_seen_at=snapshot_ts[0],
            last_seen_at=snapshot_ts[-1],
            recent_args=snapshot_args,
            break_out_hint=(
                f"Action '{action}' was called {len(snapshot_ts)} times with "
                f"identical arguments in {rule.window_seconds}s. Break the loop: "
                "vary the arguments, escalate to a human, or stop retrying."
            ),
        )

    @staticmethod
    def _strip(
        kwargs: Optional[Dict[str, Any]],
        exclude: Iterable[str],
    ) -> Dict[str, Any]:
        if not kwargs:
            return {}
        excl = set(exclude)
        return {k: v for k, v in kwargs.items() if k not in excl}


# Module-level default used by PolicyEngine
DEFAULT_LOOP_GUARD = LoopGuard()
