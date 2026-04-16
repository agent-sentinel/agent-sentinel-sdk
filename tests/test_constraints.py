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
        violations = validate_constraints(
            {"amount": 100, "extra_field": "whatever"},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert violations == []

    def test_missing_property_not_required(self):
        violations = validate_constraints(
            {},
            {"properties": {"amount": {"type": "number"}}},
        )
        assert violations == []
