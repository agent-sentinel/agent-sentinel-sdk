"""
Built-in runtime guardrails.

Provides first-class primitives that previously required users to hand-write
constraints: PII detection, content moderation, and loop protection.

Each guardrail:
- Is pure Python / stdlib-only (no new dependencies)
- Integrates with the existing PolicyEngine.check_action pipeline
- Emits structured PolicyViolationError with a machine-parseable reason_code
- Is individually toggleable via PolicyConfig fields

See:
- pii.py         -> PII detection (email, SSN, credit card, phone, API keys)
- moderation.py  -> Content moderation (pluggable moderator, keyword default)
- loop_detector.py -> Tight-loop detection on repeated (action, args) calls
- idempotency.py -> Deduplication for idempotent actions by caller-supplied key
"""
from __future__ import annotations

from .pii import PIIGuard, PIIMatch, PIIRule, detect_pii
from .moderation import (
    ModerationGuard,
    ModerationResult,
    ModerationRule,
    Moderator,
    KeywordModerator,
)
from .loop_detector import LoopGuard, LoopRule, LoopDetection
from .idempotency import IdempotencyCache, IdempotencyHit

__all__ = [
    "PIIGuard",
    "PIIMatch",
    "PIIRule",
    "detect_pii",
    "ModerationGuard",
    "ModerationResult",
    "ModerationRule",
    "Moderator",
    "KeywordModerator",
    "LoopGuard",
    "LoopRule",
    "LoopDetection",
    "IdempotencyCache",
    "IdempotencyHit",
]
