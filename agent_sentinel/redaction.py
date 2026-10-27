"""
Redaction Module: Strip secrets out of tool-call inputs before they are
persisted to the local ledger or uploaded to the platform.

Guard.py records `{"args": args, "kwargs": kwargs}` verbatim for every
guarded action. If a caller passes an API key, password, or token as a
tool argument, that value would otherwise be written to disk (ledger.py)
and later uploaded (sync.py) as-is. This module intercepts those values
first.

Two independent mechanisms:
- `redact_inputs()`: replaces values whose *key* looks sensitive, or whose
  *value* matches a common secret shape, with a `[redacted:<kind>]`
  placeholder. Oversized values are truncated rather than redacted.
- `ledger_input_capture_enabled()`: a hard off switch
  (`AGENT_SENTINEL_LEDGER_INPUTS=none`) that disables input capture
  entirely, for callers who don't want tool arguments persisted at all.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

LEDGER_INPUTS_ENV = "AGENT_SENTINEL_LEDGER_INPUTS"

DEFAULT_MAX_BYTES = 8192

# Key names that should never have their value persisted, regardless of shape.
_SENSITIVE_KEY = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|authorization|credential|private[_-]?key)"
)

# Common secret value shapes, checked when the key itself looks innocuous
# (e.g. a "notes" field that happens to contain a pasted token).
_VALUE_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("stripe_key", re.compile(r"sk_live_[A-Za-z0-9]+")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{10,}")),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[abp]-[A-Za-z0-9-]+")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")),
]


def ledger_input_capture_enabled() -> bool:
    """Whether tool-call inputs should be captured at all.

    `AGENT_SENTINEL_LEDGER_INPUTS=none` (case-insensitive) disables input
    capture entirely, independent of secret redaction below.
    """
    return os.getenv(LEDGER_INPUTS_ENV, "").strip().lower() != "none"


def _match_value_kind(text: str) -> Optional[str]:
    for kind, pattern in _VALUE_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def _truncate(text: str, max_bytes: int) -> Tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    overflow = len(encoded) - max_bytes
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}...[truncated {overflow} bytes]", True


def _redact_value(value: Any, max_bytes: int, key_redacted: bool) -> Tuple[Any, bool]:
    if key_redacted:
        return (None if value is None else "[redacted:key]"), value is not None

    if isinstance(value, str):
        kind = _match_value_kind(value)
        if kind:
            return f"[redacted:{kind}]", True
        return _truncate(value, max_bytes)

    if isinstance(value, dict):
        return _redact_dict(value, max_bytes)

    if isinstance(value, (list, tuple)):
        changed = False
        out = []
        for item in value:
            new_item, item_changed = _redact_value(item, max_bytes, key_redacted=False)
            out.append(new_item)
            changed = changed or item_changed
        return (tuple(out) if isinstance(value, tuple) else out), changed

    return value, False


def _redact_dict(mapping: Dict[str, Any], max_bytes: int) -> Tuple[Dict[str, Any], bool]:
    out: Dict[str, Any] = {}
    changed = False
    for key, value in mapping.items():
        key_redacted = isinstance(key, str) and bool(_SENSITIVE_KEY.search(key))
        new_value, was_changed = _redact_value(value, max_bytes, key_redacted)
        out[key] = new_value
        changed = changed or was_changed
    return out, changed


def redact_inputs(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Tuple[Tuple[Any, ...], Dict[str, Any], bool]:
    """
    Redact secrets out of tool-call args/kwargs before they're persisted.

    Replaces values whose key matches a sensitive-key pattern (api_key,
    secret, token, password, passwd, authorization, credential,
    private_key) and any string value matching a common secret shape with
    `[redacted:<kind>]`. Oversized string values are truncated instead of
    redacted, with `...[truncated N bytes]` appended. Recurses into nested
    dicts/lists/tuples. Never mutates the inputs.

    Returns (redacted_args, redacted_kwargs, redacted) where `redacted` is
    True if anything was replaced or truncated.
    """
    changed = False
    new_args: List[Any] = []
    for value in args or ():
        new_value, was_changed = _redact_value(value, max_bytes, key_redacted=False)
        new_args.append(new_value)
        changed = changed or was_changed

    new_kwargs, kwargs_changed = _redact_dict(kwargs or {}, max_bytes)
    changed = changed or kwargs_changed

    return tuple(new_args), new_kwargs, changed
