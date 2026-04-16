"""
Evidence Tracking Module: Tracks evidence produced by agent actions.

This module is the core of the "action correctness" enforcement system.
It records when evidence-producing actions (e.g., lookup_order) complete
successfully, and allows commit actions (e.g., issue_refund) to verify
that required prerequisites have been satisfied.

Design:
- Thread-safe singleton (same pattern as CostTracker)
- Scoped to run (cleared on reset_run) and session (persists across runs)
- Evidence records include timestamps for freshness checks
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_sentinel")


@dataclass
class EvidenceRecord:
    """
    Record of evidence produced by an action.

    Captures the action name, when it ran, and optionally a hash
    of its arguments and a summary of its result.
    """
    action_name: str
    timestamp: float = field(default_factory=time.time)
    args_hash: Optional[str] = None
    result_summary: Any = None
    raw_kwargs: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "action_name": self.action_name,
            "timestamp": self.timestamp,
            "args_hash": self.args_hash,
            "result_summary": self.result_summary,
            "raw_kwargs": self.raw_kwargs,
        }
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d


def _hash_args(kwargs: Optional[Dict[str, Any]]) -> Optional[str]:
    """Create a stable hash of action arguments for matching."""
    if not kwargs:
        return None
    try:
        serialized = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return None


def _get_nested_value(data: Any, dotted_path: str) -> Any:
    """Get a value from a nested dict using dot-notation path."""
    if data is None:
        return None
    parts = dotted_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


class EvidenceTracker:
    """
    Thread-safe singleton for tracking evidence produced during agent runs.

    Usage:
        # Record evidence after a lookup
        EvidenceTracker.record_evidence("lookup_order", kwargs={"order_id": "123"})

        # Check if evidence exists
        if EvidenceTracker.has_evidence("lookup_order"):
            ...

        # Check all requirements for a commit action
        all_met, missing = EvidenceTracker.check_requirements(
            ["lookup_order", "verify_identity"],
            max_age_seconds=300
        )

        # Reset between runs
        EvidenceTracker.reset_run()
    """

    _lock = threading.Lock()
    _run_evidence: List[EvidenceRecord] = []
    _session_evidence: List[EvidenceRecord] = []

    @classmethod
    def record_evidence(
        cls,
        action_name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        result: Any = None,
    ) -> None:
        """
        Record that an evidence-producing action completed successfully.

        Args:
            action_name: Name of the action that produced evidence
            kwargs: Arguments passed to the action (hashed for matching)
            result: Summary of the action's result
        """
        # Get run_id from ExecutionContext if available
        run_id = None
        try:
            from .context import ExecutionContext
            run_id = ExecutionContext.get_run_id()
        except Exception:
            pass

        record = EvidenceRecord(
            action_name=action_name,
            timestamp=time.time(),
            args_hash=_hash_args(kwargs),
            result_summary=result,
            raw_kwargs=dict(kwargs) if kwargs else None,
            run_id=run_id,
        )

        with cls._lock:
            cls._run_evidence.append(record)
            cls._session_evidence.append(record)

        logger.debug(f"Evidence recorded for action '{action_name}'")

    @classmethod
    def has_evidence(
        cls,
        action_name: str,
        max_age_seconds: Optional[int] = None,
    ) -> bool:
        """
        Check if evidence exists for a given action.

        Args:
            action_name: Action to check for
            max_age_seconds: If set, only consider evidence newer than this

        Returns:
            True if valid evidence exists
        """
        with cls._lock:
            now = time.time()
            for record in reversed(cls._run_evidence):
                if record.action_name == action_name:
                    if max_age_seconds is not None:
                        age = now - record.timestamp
                        if age > max_age_seconds:
                            continue
                    return True
            return False

    @classmethod
    def get_evidence(
        cls,
        action_name: str,
        max_age_seconds: Optional[int] = None,
    ) -> List[EvidenceRecord]:
        """
        Get all evidence records for a given action.

        Args:
            action_name: Action to get evidence for
            max_age_seconds: If set, only return evidence newer than this

        Returns:
            List of matching evidence records (most recent first)
        """
        with cls._lock:
            now = time.time()
            results = []
            for record in reversed(cls._run_evidence):
                if record.action_name == action_name:
                    if max_age_seconds is not None:
                        age = now - record.timestamp
                        if age > max_age_seconds:
                            continue
                    results.append(record)
            return results

    @classmethod
    def check_requirements(
        cls,
        requires: List[str],
        max_age_seconds: Optional[int] = None,
    ) -> tuple[bool, List[str], List[str]]:
        """
        Check if all required evidence has been produced.

        Args:
            requires: List of action names that must have evidence
            max_age_seconds: If set, evidence must be newer than this

        Returns:
            Tuple of (all_satisfied, missing_actions, stale_actions)
            - all_satisfied: True if all requirements are met
            - missing_actions: Actions with no evidence at all
            - stale_actions: Actions with evidence that has expired
        """
        missing = []
        stale = []

        with cls._lock:
            now = time.time()
            for required_action in requires:
                found = False
                found_but_stale = False
                for record in reversed(cls._run_evidence):
                    if record.action_name == required_action:
                        if max_age_seconds is not None:
                            age = now - record.timestamp
                            if age > max_age_seconds:
                                found_but_stale = True
                                continue
                        found = True
                        break

                if not found:
                    if found_but_stale:
                        stale.append(required_action)
                    else:
                        missing.append(required_action)

        all_satisfied = len(missing) == 0 and len(stale) == 0
        return all_satisfied, missing, stale

    @classmethod
    def check_groundedness(
        cls,
        action_kwargs: Dict[str, Any],
        grounding_rules: Dict[str, Dict[str, str]],
        max_age_seconds: Optional[int] = None,
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Verify that action arguments are grounded in prior evidence.

        Each grounding rule maps an argument field to a source action and field:
            {
                "order_id": {
                    "source_action": "lookup_order",
                    "source_field": "order_id",  # dot-notation for nested: "order.id"
                    "source_from": "result",      # "result" (default) or "kwargs"
                },
            }

        Args:
            action_kwargs: The commit action's keyword arguments
            grounding_rules: Map of field_name -> source rule
            max_age_seconds: If set, only consider evidence newer than this

        Returns:
            Tuple of (all_grounded, ungrounded_details)
        """
        if not grounding_rules:
            return True, []

        ungrounded = []

        with cls._lock:
            now = time.time()
            for field_name, rule in grounding_rules.items():
                source_action = rule["source_action"]
                source_field = rule["source_field"]
                source_from = rule.get("source_from", "result")

                actual_value = action_kwargs.get(field_name)
                if actual_value is None:
                    continue

                found = False
                for record in reversed(cls._run_evidence):
                    if record.action_name != source_action:
                        continue
                    if max_age_seconds is not None:
                        if (now - record.timestamp) > max_age_seconds:
                            continue

                    if source_from == "kwargs":
                        source_data = record.raw_kwargs
                    else:
                        source_data = record.result_summary

                    if source_data is None:
                        continue

                    evidence_value = _get_nested_value(source_data, source_field)
                    if evidence_value is not None and evidence_value == actual_value:
                        found = True
                        break

                if not found:
                    ungrounded.append({
                        "field": field_name,
                        "expected_source": f"{source_action}.{source_field}",
                        "actual_value": actual_value,
                    })

        return len(ungrounded) == 0, ungrounded

    @classmethod
    def reset_run(cls) -> None:
        """Clear run-level evidence. Session evidence is preserved."""
        with cls._lock:
            cls._run_evidence.clear()
        logger.debug("Run evidence cleared")

    @classmethod
    def reset_session(cls) -> None:
        """Clear all evidence (both run and session)."""
        with cls._lock:
            cls._run_evidence.clear()
            cls._session_evidence.clear()
        logger.debug("Session evidence cleared")

    @classmethod
    def get_run_evidence_count(cls) -> int:
        """Get count of evidence records in the current run."""
        with cls._lock:
            return len(cls._run_evidence)

    @classmethod
    def get_session_evidence_count(cls) -> int:
        """Get count of evidence records in the current session."""
        with cls._lock:
            return len(cls._session_evidence)
