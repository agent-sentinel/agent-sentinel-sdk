"""
PII (Personally Identifiable Information) detection guardrail.

Stdlib-only regex detector for common PII classes. Designed to be called
from PolicyEngine.check_action *before* an action executes, so that kwargs
containing PII can be blocked with structured, actionable feedback.

Detected categories (default):
- email, us_ssn, credit_card, phone_us, api_key_like, aws_access_key,
  private_key_block

The detector is intentionally conservative (favours precision over recall)
to avoid false positives on internal IDs. For higher-recall needs, callers
can swap in their own regex set via PIIRule.

Design:
- `detect_pii(text, rule)` is a pure function returning list[PIIMatch].
- `PIIGuard.scan_kwargs(kwargs, rule)` walks a kwargs dict recursively and
  returns all matches.
- Credit cards are additionally Luhn-validated to reduce false positives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

# ---------------------------------------------------------------------------
# Regex library
# ---------------------------------------------------------------------------

# Compiled once at import time; patterns are conservative.
_EMAIL = re.compile(
    r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
)
_US_SSN = re.compile(
    r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"
)
# 13-19 digit candidate; Luhn-validated downstream.
_CREDIT_CARD_CANDIDATE = re.compile(
    r"\b(?:\d[ -]?){12,18}\d\b"
)
_PHONE_US = re.compile(
    r"\b(?:\+?1[ \-.]?)?\(?[2-9]\d{2}\)?[ \-.]?[2-9]\d{2}[ \-.]?\d{4}\b"
)
# Generic high-entropy API-key-like token: 32+ chars of base64/hex.
_API_KEY_LIKE = re.compile(
    r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9_\-]{24,}\b"
    r"|\b[A-Za-z0-9]{40,}\b"
)
_AWS_ACCESS_KEY = re.compile(
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)

_DEFAULT_PATTERNS: Dict[str, Pattern[str]] = {
    "email": _EMAIL,
    "us_ssn": _US_SSN,
    "credit_card": _CREDIT_CARD_CANDIDATE,
    "phone_us": _PHONE_US,
    "api_key_like": _API_KEY_LIKE,
    "aws_access_key": _AWS_ACCESS_KEY,
    "private_key_block": _PRIVATE_KEY_BLOCK,
}


def _luhn_ok(digits: str) -> bool:
    """Validate credit-card candidate with Luhn checksum."""
    d = [int(c) for c in digits if c.isdigit()]
    if len(d) < 13 or len(d) > 19:
        return False
    checksum = 0
    for i, n in enumerate(reversed(d)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIIMatch:
    """A single detected PII occurrence."""
    category: str
    field_path: str        # dotted path into kwargs (e.g. "user.email")
    matched_text: str      # redacted preview (first/last 2 chars only)
    severity: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "field_path": self.field_path,
            "matched_text": self.matched_text,
            "severity": self.severity,
        }


@dataclass
class PIIRule:
    """
    Per-action PII detection rule.

    Attributes:
        categories: Which PII categories to scan for. Defaults to all built-ins.
        extra_patterns: Caller-supplied regex map to add to detection.
        allow_categories: Categories to explicitly allow (overrides deny).
        redact_preview: If True, redact matched text to first/last 2 chars.
    """
    categories: Iterable[str] = field(default_factory=lambda: tuple(_DEFAULT_PATTERNS.keys()))
    extra_patterns: Dict[str, str] = field(default_factory=dict)
    allow_categories: Iterable[str] = field(default_factory=tuple)
    redact_preview: bool = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _redact(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def detect_pii(text: str, rule: Optional[PIIRule] = None) -> List[PIIMatch]:
    """
    Detect PII in a single string.

    Returns a list of PIIMatch. Uses default categories unless `rule` overrides.
    `field_path` will be empty when called directly on a string.
    """
    if not isinstance(text, str) or not text:
        return []

    rule = rule or PIIRule()
    allow = set(rule.allow_categories)
    categories = [c for c in rule.categories if c not in allow]

    patterns: List[Tuple[str, Pattern[str]]] = [
        (c, _DEFAULT_PATTERNS[c]) for c in categories if c in _DEFAULT_PATTERNS
    ]
    for name, raw in rule.extra_patterns.items():
        if name in allow:
            continue
        try:
            patterns.append((name, re.compile(raw)))
        except re.error:
            # Silently skip invalid user regex — never crash agent execution.
            continue

    matches: List[PIIMatch] = []
    for category, pattern in patterns:
        for m in pattern.finditer(text):
            matched = m.group(0)
            if category == "credit_card" and not _luhn_ok(matched):
                continue
            matches.append(
                PIIMatch(
                    category=category,
                    field_path="",
                    matched_text=_redact(matched, rule.redact_preview),
                )
            )
    return matches


class PIIGuard:
    """
    PII guardrail for action kwargs.

    Walks a kwargs dict recursively and returns all PII matches, annotated
    with their field path inside the structure.
    """

    @staticmethod
    def scan_kwargs(
        kwargs: Optional[Dict[str, Any]],
        rule: Optional[PIIRule] = None,
    ) -> List[PIIMatch]:
        if not kwargs:
            return []
        rule = rule or PIIRule()
        results: List[PIIMatch] = []
        PIIGuard._walk("", kwargs, rule, results)
        return results

    @staticmethod
    def _walk(
        prefix: str,
        value: Any,
        rule: PIIRule,
        sink: List[PIIMatch],
    ) -> None:
        if isinstance(value, str):
            for m in detect_pii(value, rule):
                sink.append(
                    PIIMatch(
                        category=m.category,
                        field_path=prefix or "<root>",
                        matched_text=m.matched_text,
                        severity=m.severity,
                    )
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                PIIGuard._walk(child, v, rule, sink)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{prefix}[{i}]"
                PIIGuard._walk(child, v, rule, sink)
        # Non-string scalars (int/bool/float/None) are ignored.
