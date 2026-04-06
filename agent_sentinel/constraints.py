"""
Lightweight argument constraint validation.

Validates action kwargs against JSON-Schema-like constraint dicts.
Supports: type, minimum, maximum, enum, required, pattern.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent_sentinel")

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
            continue

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
        # Booleans should not pass integer/number checks
        if expected_type in ("number", "integer") and isinstance(value, bool):
            violations.append(
                f"Argument '{field_name}' must be {expected_type}, got boolean"
            )
            return violations
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
