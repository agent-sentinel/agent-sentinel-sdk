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
from typing import Any, Protocol
from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModerationResult:
    """Outcome of moderating a single string."""
    flagged: bool
    categories: list[str]
    field_path: str = ""
    snippet: str = ""          # truncated excerpt for debugging
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
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
    moderator: Moderator | None = None


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
_DEFAULT_KEYWORDS: dict[str, list[str]] = {
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

    def __init__(self, keywords: dict[str, list[str]] | None = None) -> None:
        base = {k: list(v) for k, v in _DEFAULT_KEYWORDS.items()}
        if keywords:
            for cat, words in keywords.items():
                base.setdefault(cat, []).extend(words)
        # Precompute lowercased for speed
        self._lc: dict[str, list[str]] = {
            k: [w.lower() for w in v if w] for k, v in base.items()
        }

    def moderate(self, text: str) -> ModerationResult:
        if not isinstance(text, str) or not text:
            return ModerationResult(flagged=False, categories=[])
        hay = text.lower()
        hit: list[str] = []
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
# Gemini-backed moderator
# ---------------------------------------------------------------------------

_GEMINI_MODERATION_PROMPT = """\
You are a content safety classifier. Given a text string, classify it into
zero or more of these categories if they apply:

  violence, self_harm, hate_severe, sexual_minors,
  prompt_injection, data_exfiltration, policy_evasion

Return a JSON object with exactly two keys:
  {"flagged": true/false, "categories": ["<cat1>", ...]}

Output only the JSON object, no markdown fences. If nothing is concerning,
return {"flagged": false, "categories": []}.
"""

_GEMINI_VALID_CATEGORIES = frozenset({
    "violence", "self_harm", "hate_severe", "sexual_minors",
    "prompt_injection", "data_exfiltration", "policy_evasion",
})


class GeminiModerator:
    """
    LLM-backed moderator using Gemini.

    Detects intent-level violations that keyword matching misses, including
    prompt injection, novel phrasing, and policy evasion attempts.

    Requires the ``google-genai`` package and a GEMINI_API_KEY (or
    GOOGLE_API_KEY) environment variable. Falls back to returning
    ``flagged=False`` with a warning if unavailable — pair with
    ``KeywordModerator`` as a defence-in-depth fallback if needed.

    Usage::

        from agent_sentinel.guardrails.moderation import GeminiModerator, ModerationRule
        rule = ModerationRule(moderator=GeminiModerator())
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        import os
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._model = model

    def moderate(self, text: str) -> ModerationResult:
        if not isinstance(text, str) or not text.strip():
            return ModerationResult(flagged=False, categories=[])

        if not self._api_key:
            import logging
            logging.getLogger("agent_sentinel").warning(
                "GeminiModerator: no API key found; returning unflagged. "
                "Set GEMINI_API_KEY to enable semantic moderation."
            )
            return ModerationResult(flagged=False, categories=[])

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            import logging
            logging.getLogger("agent_sentinel").warning(
                "GeminiModerator: google-genai not installed; "
                "install with: pip install google-genai"
            )
            return ModerationResult(flagged=False, categories=[])

        import json as _json
        import time as _time

        client = genai.Client(api_key=self._api_key)
        config = genai_types.GenerateContentConfig(
            system_instruction=_GEMINI_MODERATION_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=text[:2000],  # cap to avoid token bloat
                    config=config,
                )
                break
            except Exception as exc:
                msg = str(exc)
                transient = any(c in msg for c in ("503", "429", "500", "UNAVAILABLE"))
                if not transient or attempt == 2:
                    import logging
                    logging.getLogger("agent_sentinel").warning(
                        "GeminiModerator: API error: %s", exc
                    )
                    return ModerationResult(flagged=False, categories=[])
                _time.sleep(1.0 * (attempt + 1))

        if not response:
            return ModerationResult(flagged=False, categories=[])

        raw = (response.text or "").strip()
        if raw.startswith("```"):
            lines = [ln for ln in raw.splitlines() if not ln.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            return ModerationResult(flagged=False, categories=[])

        flagged = bool(parsed.get("flagged", False))
        categories = [
            c for c in parsed.get("categories", [])
            if c in _GEMINI_VALID_CATEGORIES
        ]
        return ModerationResult(
            flagged=flagged,
            categories=categories,
            snippet=text[:120],
            severity="high" if flagged else "medium",
        )


# ---------------------------------------------------------------------------


class ModerationGuard:
    """Scans kwargs recursively using a configured Moderator."""

    @staticmethod
    def scan_kwargs(
        kwargs: dict[str, Any] | None,
        rule: ModerationRule | None = None,
    ) -> list[ModerationResult]:
        if not kwargs:
            return []
        rule = rule or ModerationRule()
        moderator = rule.moderator or KeywordModerator()
        results: list[ModerationResult] = []
        ModerationGuard._walk("", kwargs, moderator, results)
        return results

    @staticmethod
    def should_block(
        results: list[ModerationResult],
        rule: ModerationRule,
    ) -> list[ModerationResult]:
        """Filter to only the results that should cause a block under the rule."""
        block_cats = set(rule.block_categories)
        blocking: list[ModerationResult] = []
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
        sink: list[ModerationResult],
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
