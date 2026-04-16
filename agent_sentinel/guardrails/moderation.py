"""
Content moderation guardrail.

Provides a pluggable moderator interface. The default implementation is a
fully-offline keyword-based moderator that requires no external services
and no new dependencies. Callers can swap in richer moderators (e.g. one
that wraps the OpenAI moderation endpoint) by implementing the Moderator
Protocol.

The guardrail integrates with PolicyEngine.check_action the same way as
PIIGuard: it is called before execution with the action kwargs, walks the
structure extracting strings, and returns a structured ModerationResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModerationResult:
    """Outcome of moderating a single string."""
    flagged: bool
    categories: List[str]
    field_path: str = ""
    snippet: str = ""          # truncated excerpt for debugging
    severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flagged": self.flagged,
            "categories": self.categories,
            "field_path": self.field_path,
            "snippet": self.snippet,
            "severity": self.severity,
        }


@dataclass
class ModerationRule:
    """
    Per-action moderation rule.

    Attributes:
        strictness: "strict" | "balanced" | "permissive". Permissive only
            flags explicit category matches; strict escalates any match
            to high severity.
        block_categories: Categories that trigger a block. Others are warnings.
        moderator: Pluggable moderator instance. Defaults to KeywordModerator.
    """
    strictness: str = "balanced"
    block_categories: Iterable[str] = field(
        default_factory=lambda: (
            "violence",
            "self_harm",
            "sexual_minors",
            "hate_severe",
        )
    )
    moderator: Optional["Moderator"] = None


# ---------------------------------------------------------------------------
# Moderator protocol + default
# ---------------------------------------------------------------------------


class Moderator(Protocol):
    """Protocol for content moderators."""

    def moderate(self, text: str) -> ModerationResult:  # pragma: no cover - protocol
        ...


# Keyword-based default moderator. Intentionally minimal; users should
# configure a richer moderator (OpenAI moderation, Azure Content Safety, etc.)
# for production. The lists below are deliberately short and conservative —
# long, explicit lexicons are out of scope for this file.
_DEFAULT_KEYWORDS: Dict[str, List[str]] = {
    "violence": [
        "kill yourself",
        "how to make a bomb",
        "how to build a bomb",
        "mass shooting plan",
    ],
    "self_harm": [
        "ways to self-harm",
        "how to cut myself",
    ],
    "hate_severe": [
        # placeholder slot — operators should supply site-specific terms
    ],
    "sexual_minors": [
        # placeholder slot — never populate with actual content
    ],
}


class KeywordModerator:
    """
    Offline keyword moderator.

    Matches are case-insensitive substring checks. Designed as a safe default
    for environments without outbound network access; not a substitute for
    a real moderation service.
    """

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None) -> None:
        base = {k: list(v) for k, v in _DEFAULT_KEYWORDS.items()}
        if keywords:
            for cat, words in keywords.items():
                base.setdefault(cat, []).extend(words)
        # Precompute lowercased for speed
        self._lc: Dict[str, List[str]] = {
            k: [w.lower() for w in v if w] for k, v in base.items()
        }

    def moderate(self, text: str) -> ModerationResult:
        if not isinstance(text, str) or not text:
            return ModerationResult(flagged=False, categories=[])
        hay = text.lower()
        hit: List[str] = []
        for cat, words in self._lc.items():
            for w in words:
                if w and w in hay:
                    hit.append(cat)
                    break
        if not hit:
            return ModerationResult(flagged=False, categories=[])
        return ModerationResult(
            flagged=True,
            categories=hit,
            snippet=text[:120],
            severity="high",
        )


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class ModerationGuard:
    """Scans kwargs recursively using a configured Moderator."""

    @staticmethod
    def scan_kwargs(
        kwargs: Optional[Dict[str, Any]],
        rule: Optional[ModerationRule] = None,
    ) -> List[ModerationResult]:
        if not kwargs:
            return []
        rule = rule or ModerationRule()
        moderator = rule.moderator or KeywordModerator()
        results: List[ModerationResult] = []
        ModerationGuard._walk("", kwargs, moderator, results)
        return results

    @staticmethod
    def should_block(
        results: List[ModerationResult],
        rule: ModerationRule,
    ) -> List[ModerationResult]:
        """Filter to only the results that should cause a block under the rule."""
        block_cats = set(rule.block_categories)
        blocking: List[ModerationResult] = []
        for r in results:
            if not r.flagged:
                continue
            if rule.strictness == "strict":
                blocking.append(r)
                continue
            if any(c in block_cats for c in r.categories):
                blocking.append(r)
        return blocking

    @staticmethod
    def _walk(
        prefix: str,
        value: Any,
        moderator: Moderator,
        sink: List[ModerationResult],
    ) -> None:
        if isinstance(value, str):
            res = moderator.moderate(value)
            if res.flagged:
                sink.append(
                    ModerationResult(
                        flagged=True,
                        categories=res.categories,
                        field_path=prefix or "<root>",
                        snippet=res.snippet,
                        severity=res.severity,
                    )
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                ModerationGuard._walk(child, v, moderator, sink)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{prefix}[{i}]"
                ModerationGuard._walk(child, v, moderator, sink)
