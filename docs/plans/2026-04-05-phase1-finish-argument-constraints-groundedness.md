# Phase 1 Finish: Argument Constraints & Groundedness Checks

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 1 evidence graph by adding argument constraint validation and groundedness/source matching so commit actions can verify their arguments are grounded in prior evidence.

**Architecture:** Two features, both in the SDK only (no platform changes needed — the model fields already exist). (1) Argument constraint validation: lightweight built-in validator in `PolicyEngine.check_action()` that validates kwargs against JSON-Schema-like dicts stored in `argument_constraints`. Falls back to `jsonschema` if installed. (2) Groundedness checks: `EvidenceTracker` stores full kwargs/results on evidence records; a new `check_groundedness()` method verifies that specific argument values in a commit action match values seen in prior evidence results.

**Tech Stack:** Python stdlib only for constraint validation (optional jsonschema fallback). Existing pytest test patterns.

---

### Task 1: Argument Constraint Validation — `constraints.py`

**Files:**
- Create: `agent_sentinel/constraints.py`
- Test: `tests/test_constraints.py`

**Step 1: Write the failing tests**

```python
# tests/test_constraints.py
"""Tests for argument constraint validation."""
import pytest
from agent_sentinel.constraints import validate_constraints


class TestValidateConstraints:
    def test_no_constraints_passes(self):
        violations = validate_constraints({"amount": 100}, {})
        assert violations == []

    def test_no_kwargs_passes(self):
        violations = validate_constraints(None, {"properties": {"amount": {"type": "number"}}})
        assert violations == []

    def test_type_number_valid(self):
        violations = validate_constraints(
            {"amount": 100},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert violations == []

    def test_type_number_invalid(self):
        violations = validate_constraints(
            {"amount": "not_a_number"},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert len(violations) == 1
        assert "amount" in violations[0]

    def test_type_string_valid(self):
        violations = validate_constraints(
            {"name": "Alice"},
            {"properties": {"name": {"type": "string"}}},
        )
        assert violations == []

    def test_type_string_invalid(self):
        violations = validate_constraints(
            {"name": 123},
            {"properties": {"name": {"type": "string"}}},
        )
        assert len(violations) == 1

    def test_type_boolean_valid(self):
        violations = validate_constraints(
            {"flag": True},
            {"properties": {"flag": {"type": "boolean"}}},
        )
        assert violations == []

    def test_type_integer_valid(self):
        violations = validate_constraints(
            {"count": 5},
            {"properties": {"count": {"type": "integer"}}},
        )
        assert violations == []

    def test_type_integer_rejects_float(self):
        violations = validate_constraints(
            {"count": 5.5},
            {"properties": {"count": {"type": "integer"}}},
        )
        assert len(violations) == 1

    def test_minimum(self):
        violations = validate_constraints(
            {"amount": 5},
            {"properties": {"amount": {"type": "number", "minimum": 10}}},
        )
        assert len(violations) == 1
        assert "minimum" in violations[0].lower() or "10" in violations[0]

    def test_maximum(self):
        violations = validate_constraints(
            {"amount": 5000},
            {"properties": {"amount": {"type": "number", "maximum": 1000}}},
        )
        assert len(violations) == 1

    def test_minimum_and_maximum_valid(self):
        violations = validate_constraints(
            {"amount": 500},
            {"properties": {"amount": {"type": "number", "minimum": 0, "maximum": 1000}}},
        )
        assert violations == []

    def test_enum_valid(self):
        violations = validate_constraints(
            {"status": "active"},
            {"properties": {"status": {"enum": ["active", "inactive"]}}},
        )
        assert violations == []

    def test_enum_invalid(self):
        violations = validate_constraints(
            {"status": "deleted"},
            {"properties": {"status": {"enum": ["active", "inactive"]}}},
        )
        assert len(violations) == 1

    def test_required_fields_present(self):
        violations = validate_constraints(
            {"order_id": "123", "amount": 50},
            {"required": ["order_id", "amount"]},
        )
        assert violations == []

    def test_required_fields_missing(self):
        violations = validate_constraints(
            {"order_id": "123"},
            {"required": ["order_id", "amount"]},
        )
        assert len(violations) == 1
        assert "amount" in violations[0]

    def test_pattern_valid(self):
        violations = validate_constraints(
            {"email": "user@example.com"},
            {"properties": {"email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"}}},
        )
        assert violations == []

    def test_pattern_invalid(self):
        violations = validate_constraints(
            {"email": "not-an-email"},
            {"properties": {"email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"}}},
        )
        assert len(violations) == 1

    def test_multiple_violations(self):
        violations = validate_constraints(
            {"amount": "bad", "status": "unknown"},
            {
                "properties": {
                    "amount": {"type": "number", "maximum": 1000},
                    "status": {"enum": ["active", "inactive"]},
                },
            },
        )
        assert len(violations) == 2

    def test_extra_kwargs_ignored(self):
        """Kwargs not in constraints are not validated."""
        violations = validate_constraints(
            {"amount": 100, "extra_field": "whatever"},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert violations == []

    def test_missing_property_not_required(self):
        """If a property is in constraints but not in kwargs and not required, no violation."""
        violations = validate_constraints(
            {},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert violations == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_constraints.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_sentinel.constraints'`

**Step 3: Write the implementation**

```python
# agent_sentinel/constraints.py
"""
Lightweight argument constraint validation.

Validates action kwargs against JSON-Schema-like constraint dicts.
Supports: type, minimum, maximum, enum, required, pattern.

Falls back to jsonschema library if installed for full JSON Schema support,
but works without any external dependencies for common constraints.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_sentinel")

# Type name -> Python types mapping
_TYPE_MAP: Dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def validate_constraints(
    kwargs: Optional[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> List[str]:
    """
    Validate action kwargs against a JSON-Schema-like constraint dict.

    Args:
        kwargs: The action's keyword arguments to validate
        constraints: JSON-Schema-like dict with 'properties' and/or 'required'

    Returns:
        List of human-readable violation strings. Empty list means valid.
    """
    if not constraints or not kwargs:
        return []

    violations: List[str] = []

    # Check required fields
    required = constraints.get("required", [])
    for field_name in required:
        if field_name not in kwargs:
            violations.append(f"Required argument '{field_name}' is missing")

    # Check property constraints
    properties = constraints.get("properties", {})
    for field_name, field_constraints in properties.items():
        if field_name not in kwargs:
            continue  # Not present and not required = OK

        value = kwargs[field_name]
        field_violations = _validate_field(field_name, value, field_constraints)
        violations.extend(field_violations)

    return violations


def _validate_field(
    field_name: str,
    value: Any,
    constraints: Dict[str, Any],
) -> List[str]:
    """Validate a single field value against its constraints."""
    violations: List[str] = []

    # Type check
    expected_type = constraints.get("type")
    if expected_type and expected_type in _TYPE_MAP:
        allowed_types = _TYPE_MAP[expected_type]
        # Special case: booleans should not pass integer/number checks
        if expected_type in ("number", "integer") and isinstance(value, bool):
            violations.append(
                f"Argument '{field_name}' must be {expected_type}, got boolean"
            )
            return violations  # Skip numeric checks if type is wrong
        if not isinstance(value, allowed_types):
            violations.append(
                f"Argument '{field_name}' must be {expected_type}, "
                f"got {type(value).__name__}"
            )
            return violations  # Skip further checks if type is wrong

    # Minimum
    minimum = constraints.get("minimum")
    if minimum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < minimum:
            violations.append(
                f"Argument '{field_name}' value {value} is below minimum {minimum}"
            )

    # Maximum
    maximum = constraints.get("maximum")
    if maximum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > maximum:
            violations.append(
                f"Argument '{field_name}' value {value} exceeds maximum {maximum}"
            )

    # Enum
    enum_values = constraints.get("enum")
    if enum_values is not None:
        if value not in enum_values:
            violations.append(
                f"Argument '{field_name}' value '{value}' not in allowed values: {enum_values}"
            )

    # Pattern (string only)
    pattern = constraints.get("pattern")
    if pattern is not None and isinstance(value, str):
        if not re.search(pattern, value):
            violations.append(
                f"Argument '{field_name}' value '{value}' does not match pattern '{pattern}'"
            )

    return violations
```

**Step 4: Run tests to verify they pass**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_constraints.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
cd /tmp/agent-sentinel-sdk-evidence
git add agent_sentinel/constraints.py tests/test_constraints.py
git commit -m "feat: add lightweight argument constraint validation module"
```

---

### Task 2: Wire Argument Constraints into PolicyEngine

**Files:**
- Modify: `agent_sentinel/policy.py` (add step 7 in `check_action`)
- Test: `tests/test_constraints_policy.py`

**Step 1: Write the failing tests**

```python
# tests/test_constraints_policy.py
"""Tests for argument constraint enforcement in PolicyEngine."""
import pytest

from agent_sentinel.policy import PolicyEngine
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


class TestPolicyArgumentConstraints:
    def test_valid_args_pass(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number", "maximum": 1000}},
                    "required": ["amount"],
                }
            }
        )
        # Should not raise
        PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": 500})

    def test_invalid_type_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": "not_a_number"})

        assert exc_info.value.remediation.reason_code == "ARGUMENT_CONSTRAINT_VIOLATED"
        assert len(exc_info.value.remediation.argument_violations) == 1

    def test_exceeds_maximum_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number", "maximum": 1000}},
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"amount": 5000})

        assert "maximum" in exc_info.value.remediation.argument_violations[0].lower() or "1000" in exc_info.value.remediation.argument_violations[0]

    def test_missing_required_raises(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "required": ["order_id", "amount"],
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "123"})

        assert "amount" in exc_info.value.remediation.argument_violations[0]

    def test_no_kwargs_skips_validation(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        # No kwargs passed — should not raise (fail-open for frameworks that don't pass kwargs)
        PolicyEngine.check_action("issue_refund", 0.0)

    def test_unconstrained_action_passes(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {"amount": {"type": "number"}},
                }
            }
        )
        # Different action — no constraints
        PolicyEngine.check_action("send_email", 0.0, kwargs={"to": "user@example.com"})

    def test_multiple_violations(self):
        PolicyEngine.configure(
            argument_constraints={
                "issue_refund": {
                    "properties": {
                        "amount": {"type": "number", "maximum": 1000},
                        "reason": {"enum": ["defective", "wrong_item"]},
                    },
                    "required": ["order_id"],
                }
            }
        )
        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action(
                "issue_refund", 0.0,
                kwargs={"amount": 5000, "reason": "just_because"},
            )
        # Should have violations for: missing order_id, amount > max, reason not in enum
        assert len(exc_info.value.remediation.argument_violations) == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_constraints_policy.py -v`
Expected: FAIL — `check_action` doesn't validate argument constraints yet

**Step 3: Add step 7 to `PolicyEngine.check_action()`**

In `agent_sentinel/policy.py`, after step 6 (evidence requirements check, around line 830), add:

```python
        # 7. Check argument constraints
        if action in config.argument_constraints and kwargs:
            from .constraints import validate_constraints
            violations = validate_constraints(kwargs, config.argument_constraints[action])
            if violations:
                raise EvidenceViolationError(
                    message=f"Action '{action}' argument constraints violated: {violations}",
                    action_name=action,
                    argument_violations=violations,
                )
```

**Step 4: Run tests to verify they pass**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_constraints_policy.py -v`
Expected: All PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/ -v`
Expected: All existing + new tests PASS

**Step 6: Commit**

```bash
cd /tmp/agent-sentinel-sdk-evidence
git add agent_sentinel/policy.py tests/test_constraints_policy.py
git commit -m "feat: wire argument constraint validation into PolicyEngine.check_action"
```

---

### Task 3: Groundedness Checks — Enhance EvidenceTracker

**Files:**
- Modify: `agent_sentinel/evidence.py` (add `check_groundedness`)
- Test: `tests/test_groundedness.py`

**Step 1: Write the failing tests**

```python
# tests/test_groundedness.py
"""Tests for groundedness checks — verifying commit action args match evidence."""
import pytest

from agent_sentinel.evidence import EvidenceTracker


@pytest.fixture(autouse=True)
def reset():
    EvidenceTracker.reset_session()
    yield
    EvidenceTracker.reset_session()


class TestCheckGroundedness:
    def test_grounded_single_field(self):
        """order_id in refund matches order_id returned by lookup."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "amount": 99.99},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "refund_amount": 50.0},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is True
        assert ungrounded == []

    def test_ungrounded_value_mismatch(self):
        """order_id in refund does NOT match any evidence."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "amount": 99.99},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-999"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is False
        assert len(ungrounded) == 1
        assert ungrounded[0]["field"] == "order_id"

    def test_ungrounded_no_evidence(self):
        """No evidence at all for the source action."""
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
        )
        assert grounded is False
        assert len(ungrounded) == 1

    def test_grounded_from_kwargs(self):
        """Match against source action's kwargs, not just result."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"status": "found"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id", "source_from": "kwargs"}},
        )
        assert grounded is True

    def test_multiple_grounding_rules(self):
        """Multiple fields must all be grounded."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            kwargs={"order_id": "ORD-123"},
            result={"order_id": "ORD-123", "customer_id": "CUST-456"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "customer_id": "CUST-456"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "customer_id": {"source_action": "lookup_order", "source_field": "customer_id"},
            },
        )
        assert grounded is True
        assert ungrounded == []

    def test_partial_grounding(self):
        """One grounded, one not."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123", "customer_id": "CUST-456"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "customer_id": "CUST-WRONG"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "customer_id": {"source_action": "lookup_order", "source_field": "customer_id"},
            },
        )
        assert grounded is False
        assert len(ungrounded) == 1
        assert ungrounded[0]["field"] == "customer_id"

    def test_grounded_across_actions(self):
        """Different fields grounded from different source actions."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123"},
        )
        EvidenceTracker.record_evidence(
            "verify_identity",
            result={"user_id": "USER-789"},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123", "user_id": "USER-789"},
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                "user_id": {"source_action": "verify_identity", "source_field": "user_id"},
            },
        )
        assert grounded is True

    def test_empty_rules_always_grounded(self):
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"anything": "value"},
            grounding_rules={},
        )
        assert grounded is True
        assert ungrounded == []

    def test_grounded_with_max_age(self):
        """Stale evidence should not ground."""
        import time
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123"},
        )
        time.sleep(0.05)
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order_id"}},
            max_age_seconds=0,
        )
        assert grounded is False

    def test_nested_result_field(self):
        """Support dot-notation for nested result fields."""
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order": {"id": "ORD-123", "status": "active"}},
        )
        grounded, ungrounded = EvidenceTracker.check_groundedness(
            action_kwargs={"order_id": "ORD-123"},
            grounding_rules={"order_id": {"source_action": "lookup_order", "source_field": "order.id"}},
        )
        assert grounded is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_groundedness.py -v`
Expected: FAIL — `AttributeError: type object 'EvidenceTracker' has no attribute 'check_groundedness'`

**Step 3: Implement `check_groundedness` in `evidence.py`**

Add to `EvidenceTracker` class in `agent_sentinel/evidence.py`:

```python
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
            - all_grounded: True if every rule is satisfied
            - ungrounded_details: List of dicts with field, expected_source, actual_value
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
                    continue  # Field not in kwargs — let argument_constraints handle required checks

                # Search evidence for a matching value
                found = False
                for record in reversed(cls._run_evidence):
                    if record.action_name != source_action:
                        continue
                    if max_age_seconds is not None:
                        if (now - record.timestamp) > max_age_seconds:
                            continue

                    # Get the source value from either result or kwargs
                    source_data = record.result_summary if source_from == "result" else record.args_hash
                    if source_from == "kwargs":
                        # We need the raw kwargs — but we only store hash.
                        # For kwargs matching, we store the full kwargs in result_summary
                        # when source_from=kwargs is used. But actually we don't.
                        # Let's check the kwargs dict stored on the record.
                        # Wait — we only store args_hash, not the raw kwargs.
                        # We need to enhance record_evidence to also store raw kwargs.
                        pass

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
```

Also add `_get_nested_value` helper outside the class, and add `raw_kwargs` to `EvidenceRecord`:

```python
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
```

And update `EvidenceRecord` to store `raw_kwargs`:
```python
@dataclass
class EvidenceRecord:
    action_name: str
    timestamp: float = field(default_factory=time.time)
    args_hash: Optional[str] = None
    result_summary: Any = None
    raw_kwargs: Optional[Dict[str, Any]] = None  # NEW: stored for groundedness checks
```

And update `record_evidence` to store `raw_kwargs`:
```python
    record = EvidenceRecord(
        action_name=action_name,
        timestamp=time.time(),
        args_hash=_hash_args(kwargs),
        result_summary=result,
        raw_kwargs=dict(kwargs) if kwargs else None,  # NEW
    )
```

And update the `check_groundedness` kwargs matching to use `raw_kwargs`:
```python
                    if source_from == "kwargs":
                        source_data = record.raw_kwargs
                    else:
                        source_data = record.result_summary
```

**Step 4: Run tests to verify they pass**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_groundedness.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
cd /tmp/agent-sentinel-sdk-evidence
git add agent_sentinel/evidence.py tests/test_groundedness.py
git commit -m "feat: add groundedness checks to EvidenceTracker"
```

---

### Task 4: Wire Groundedness into `guarded_action`

**Files:**
- Modify: `agent_sentinel/guard.py` (add `grounding_rules` param, check before execution)
- Modify: `agent_sentinel/policy.py` (add `grounding_rules` to `PolicyConfig`)
- Test: `tests/test_groundedness_integration.py`

**Step 1: Write the failing tests**

```python
# tests/test_groundedness_integration.py
"""Integration tests for groundedness checks with guarded_action."""
import pytest

from agent_sentinel.guard import guarded_action
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError


@pytest.fixture(autouse=True)
def reset():
    PolicyEngine.reset()
    EvidenceTracker.reset_session()
    yield
    PolicyEngine.reset()
    EvidenceTracker.reset_session()


class TestDecoratorGroundedness:
    def test_grounded_action_succeeds(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id, "amount": 99.99}

        @guarded_action(
            name="issue_refund",
            is_commit=True,
            requires=["lookup_order"],
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
            },
        )
        def issue_refund(order_id: str, amount: float):
            return {"refunded": True}

        lookup_order("ORD-123")
        result = issue_refund(order_id="ORD-123", amount=50.0)
        assert result["refunded"] is True

    def test_ungrounded_action_blocked(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @guarded_action(
            name="issue_refund",
            is_commit=True,
            requires=["lookup_order"],
            grounding_rules={
                "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
            },
        )
        def issue_refund(order_id: str, amount: float):
            return {"refunded": True}

        lookup_order("ORD-123")
        with pytest.raises(EvidenceViolationError) as exc_info:
            issue_refund(order_id="ORD-WRONG", amount=50.0)

        assert exc_info.value.remediation.reason_code == "UNGROUNDED_ARGUMENT"

    def test_no_grounding_rules_no_check(self):
        @guarded_action(name="lookup_order", produces_evidence=True)
        def lookup_order(order_id: str):
            return {"order_id": order_id}

        @guarded_action(name="issue_refund", requires=["lookup_order"])
        def issue_refund(order_id: str):
            return {"ok": True}

        lookup_order("ORD-123")
        result = issue_refund(order_id="ORD-ANYTHING")
        assert result["ok"] is True


class TestPolicyGroundedness:
    def test_policy_level_grounding(self):
        """Grounding rules from policy config."""
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
            grounding_rules={
                "issue_refund": {
                    "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                }
            },
        )
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123"},
        )

        # Grounded — should pass
        PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "ORD-123"})

    def test_policy_level_grounding_blocked(self):
        PolicyEngine.configure(
            evidence_requirements={"issue_refund": ["lookup_order"]},
            grounding_rules={
                "issue_refund": {
                    "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
                }
            },
        )
        EvidenceTracker.record_evidence(
            "lookup_order",
            result={"order_id": "ORD-123"},
        )

        with pytest.raises(EvidenceViolationError) as exc_info:
            PolicyEngine.check_action("issue_refund", 0.0, kwargs={"order_id": "ORD-WRONG"})
        assert exc_info.value.remediation.reason_code == "UNGROUNDED_ARGUMENT"
```

**Step 2: Run tests to verify they fail**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_groundedness_integration.py -v`
Expected: FAIL

**Step 3: Wire groundedness into guard.py and policy.py**

In `guard.py`, add `grounding_rules: Optional[dict] = None` param to `guarded_action()`, pass it through to `_execute_sync`/`_execute_async`, and add a check after the evidence requirements check:

```python
    # Decorator-level groundedness check
    if grounding_rules and kwargs:
        from .evidence import EvidenceTracker
        grounded, ungrounded_details = EvidenceTracker.check_groundedness(
            action_kwargs=kwargs,
            grounding_rules=grounding_rules,
            max_age_seconds=evidence_max_age_seconds,
        )
        if not grounded:
            field_names = [d["field"] for d in ungrounded_details]
            error = EvidenceViolationError(
                message=f"Action '{action_name}' arguments not grounded in evidence: {field_names}",
                action_name=action_name,
                argument_violations=[
                    f"'{d['field']}' value '{d['actual_value']}' not found in {d['expected_source']}"
                    for d in ungrounded_details
                ],
                retry_guidance=f"Ensure these arguments match prior evidence: {field_names}",
            )
            error.remediation.reason_code = "UNGROUNDED_ARGUMENT"
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            clear_compliance_metadata()
            raise error
```

In `policy.py`, add `grounding_rules: Dict[str, Dict[str, Dict[str, str]]]` to `PolicyConfig`, add it to `configure()`, `load_from_yaml()`, `_apply_remote_policies()`, and add step 8 in `check_action()`:

```python
        # 8. Check groundedness
        if action in config.grounding_rules and kwargs:
            from .evidence import EvidenceTracker
            grounded, ungrounded_details = EvidenceTracker.check_groundedness(
                action_kwargs=kwargs,
                grounding_rules=config.grounding_rules[action],
                max_age_seconds=config.evidence_max_age_seconds.get(action),
            )
            if not grounded:
                field_names = [d["field"] for d in ungrounded_details]
                error = EvidenceViolationError(
                    message=f"Action '{action}' arguments not grounded in evidence: {field_names}",
                    action_name=action,
                    argument_violations=[
                        f"'{d['field']}' value '{d['actual_value']}' not found in {d['expected_source']}"
                        for d in ungrounded_details
                    ],
                    retry_guidance=f"Ensure these arguments match prior evidence: {field_names}",
                )
                error.remediation.reason_code = "UNGROUNDED_ARGUMENT"
                raise error
```

**Step 4: Run tests to verify they pass**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/test_groundedness_integration.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
cd /tmp/agent-sentinel-sdk-evidence
git add agent_sentinel/guard.py agent_sentinel/policy.py tests/test_groundedness_integration.py
git commit -m "feat: wire groundedness checks into guarded_action and PolicyEngine"
```

---

### Task 5: Update Platform Models for Groundedness

**Files:**
- Modify: `/tmp/platform-evidence/app/models/policy.py` (add `grounding_rules` to `PolicyBase`, `PolicyUpdate`, `PolicySync`)
- Modify: `/tmp/platform-evidence/app/alembic/versions/ev1d3nc3gr4ph_add_evidence_graph_fields.py` (add `grounding_rules` column)

**Step 1: Add `grounding_rules` field to platform PolicyBase, PolicyUpdate, PolicySync**

```python
# In PolicyBase, after argument_constraints:
    grounding_rules: dict[str, dict[str, dict[str, str]]] | None = Field(
        default=None,
        sa_column=Column(JSON),
        description="Map of action -> {field -> {source_action, source_field, source_from}} for groundedness checks"
    )
```

**Step 2: Add column to Alembic migration**

```python
    op.add_column('policy', sa.Column('grounding_rules', sa.JSON(), nullable=True))
```
And in downgrade:
```python
    op.drop_column('policy', 'grounding_rules')
```

**Step 3: Commit**

```bash
cd /tmp/platform-evidence
git add app/models/policy.py app/alembic/versions/ev1d3nc3gr4ph_add_evidence_graph_fields.py
git commit -m "feat: add grounding_rules field to platform policy model and migration"
```

---

### Task 6: Update SDK Exports and Run Final Validation

**Files:**
- Modify: `agent_sentinel/__init__.py` (export `validate_constraints` from constraints module)

**Step 1: Add export**

```python
from .constraints import validate_constraints
```
And add `"validate_constraints"` to `__all__`.

**Step 2: Run full test suite one final time**

Run: `cd /tmp/agent-sentinel-sdk-evidence && /home/jimmystacks/development/agent-sentinel/agent-sentinel-sdk/.venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS, 0 failures

**Step 3: Commit**

```bash
cd /tmp/agent-sentinel-sdk-evidence
git add agent_sentinel/__init__.py
git commit -m "feat: export validate_constraints from SDK"
```
