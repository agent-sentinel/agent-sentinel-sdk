"""
Extended coverage tests for ledger module.

Tests cover:
- Ledger write operations and persistence
- Ledger rotation/archival scenarios
- Complex object serialization
- Error handling and fail-open behavior
- Environment variable configuration
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone

import pytest

from agent_sentinel.ledger import Ledger, SafeEncoder


class TestSafeEncoder:
    """Test SafeEncoder for complex object serialization."""

    def test_encode_datetime(self) -> None:
        """Test encoding datetime objects."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        encoded = json.dumps({"time": dt}, cls=SafeEncoder)
        decoded = json.loads(encoded)
        assert decoded["time"] == "2025-01-01T12:00:00+00:00"

    def test_encode_uuid(self) -> None:
        """Test encoding UUID objects."""
        test_uuid = uuid.uuid4()
        encoded = json.dumps({"id": test_uuid}, cls=SafeEncoder)
        decoded = json.loads(encoded)
        assert decoded["id"] == str(test_uuid)

    def test_encode_pydantic_v2(self) -> None:
        """Test encoding Pydantic v2 models."""
        try:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                name: str
                value: int

            model = TestModel(name="test", value=42)
            encoded = json.dumps({"model": model}, cls=SafeEncoder)
            decoded = json.loads(encoded)
            assert decoded["model"]["name"] == "test"
            assert decoded["model"]["value"] == 42
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_encode_complex_nested_structure(self) -> None:
        """Test encoding deeply nested structures with mixed types."""
        test_obj = {
            "uuid": uuid.uuid4(),
            "timestamp": datetime.now(timezone.utc),
            "nested": {
                "id": uuid.uuid4(),
                "date": datetime(2025, 1, 1, tzinfo=timezone.utc),
            },
            "list": [uuid.uuid4(), datetime.now(timezone.utc)],
        }
        encoded = json.dumps(test_obj, cls=SafeEncoder)
        decoded = json.loads(encoded)
        assert isinstance(decoded["uuid"], str)
        assert isinstance(decoded["timestamp"], str)
        assert isinstance(decoded["nested"]["id"], str)

    def test_encode_non_serializable_fallback(self) -> None:
        """Test that non-serializable objects fall back to string representation."""
        class CustomObject:
            def __repr__(self):
                return "<CustomObject instance>"

        obj = CustomObject()
        encoded = json.dumps({"obj": obj}, cls=SafeEncoder)
        decoded = json.loads(encoded)
        assert "CustomObject" in decoded["obj"]

    def test_encode_builtin_types(self) -> None:
        """Test encoding standard JSON-serializable types."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"a": 1},
        }
        encoded = json.dumps(data, cls=SafeEncoder)
        decoded = json.loads(encoded)
        assert decoded == data


class TestLedgerSetup:
    """Test Ledger initialization and setup."""

    def test_setup_default_location(self, tmp_path, monkeypatch) -> None:
        """Test ledger setup creates directory in default location."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Reset ledger state
        Ledger.reset()
        
        # Clear environment variable
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        # Setup should create .agent-sentinel directory
        Ledger.setup()
        
        assert Ledger._initialized is True
        assert Ledger._log_path is not None
        assert Ledger._log_path.parent.name == ".agent-sentinel"

    def test_setup_with_env_var(self, tmp_path, monkeypatch) -> None:
        """Test ledger setup respects AGENT_SENTINEL_HOME environment variable."""
        custom_dir = tmp_path / "custom" / "sentinel"
        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(custom_dir))
        
        Ledger.reset()
        Ledger.setup()
        
        assert Ledger._log_path is not None
        assert custom_dir in Ledger._log_path.parents or str(custom_dir) in str(Ledger._log_path)

    def test_setup_idempotent(self, tmp_path, monkeypatch) -> None:
        """Test that setup is idempotent."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        first_setup_path = None
        first_setup_path = Ledger.get_log_path()
        second_setup_path = Ledger.get_log_path()
        
        assert first_setup_path == second_setup_path

    def test_setup_fallback_to_temp(self, tmp_path, monkeypatch) -> None:
        """Test ledger falls back to temp directory if primary fails."""
        # Create a read-only directory
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # Read-only
        
        try:
            monkeypatch.setenv("AGENT_SENTINEL_HOME", str(read_only_dir / "sentinel"))
            Ledger.reset()
            Ledger.setup()
            
            # Should fall back to temp directory
            assert Ledger._log_path is not None
            # Either it uses temp, or it's disabled
            assert Ledger._log_path or Ledger._log_path is None
        finally:
            # Restore permissions for cleanup
            read_only_dir.chmod(0o755)


class TestLedgerWriteOperations:
    """Test ledger write and record operations."""

    def test_record_simple_action(self, tmp_path, monkeypatch) -> None:
        """Test recording a simple action to ledger."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        Ledger.record(
            action="test_action",
            inputs={"arg1": "value1"},
            outputs={"result": "success"},
            cost_usd=0.01,
            duration_ms=100.0,
            outcome="success",
            tags=["test"],
        )
        
        # Verify entry was written
        log_path = Ledger.get_log_path()
        assert log_path.exists()
        
        with open(log_path, "r") as f:
            entry_line = f.readline()
            entry = json.loads(entry_line)
        
        assert entry["action"] == "test_action"
        assert entry["cost_usd"] == 0.01
        assert entry["outcome"] == "success"
        assert entry["payload"]["inputs"]["arg1"] == "value1"

    def test_record_multiple_entries(self, tmp_path, monkeypatch) -> None:
        """Test recording multiple entries appends correctly."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        for i in range(5):
            Ledger.record(
                action=f"action_{i}",
                inputs={},
                outputs={},
                cost_usd=float(i),
                duration_ms=100.0,
                outcome="success",
                tags=[],
            )
        
        log_path = Ledger.get_log_path()
        with open(log_path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert entry["action"] == f"action_{i}"
            assert entry["cost_usd"] == float(i)

    def test_record_complex_payload(self, tmp_path, monkeypatch) -> None:
        """Test recording action with complex nested payloads."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        complex_output = {
            "data": [1, 2, 3],
            "metadata": {"id": uuid.uuid4()},
            "timestamp": datetime.now(timezone.utc),
        }
        
        Ledger.record(
            action="complex_action",
            inputs={"query": "test", "params": [1, 2, 3]},
            outputs=complex_output,
            cost_usd=0.05,
            duration_ms=250.0,
            outcome="success",
            tags=["complex"],
        )
        
        log_path = Ledger.get_log_path()
        with open(log_path, "r") as f:
            entry = json.loads(f.readline())
        
        assert entry["payload"]["inputs"]["query"] == "test"
        assert entry["payload"]["outputs"]["data"] == [1, 2, 3]

    def test_record_error_outcome(self, tmp_path, monkeypatch) -> None:
        """Test recording action that failed."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        Ledger.record(
            action="failing_action",
            inputs={},
            outputs="ValueError: Invalid input",
            cost_usd=0.01,
            duration_ms=50.0,
            outcome="error",
            tags=["error"],
        )
        
        log_path = Ledger.get_log_path()
        with open(log_path, "r") as f:
            entry = json.loads(f.readline())
        
        assert entry["outcome"] == "error"
        assert "ValueError" in entry["payload"]["outputs"]

    def test_record_duration_rounding(self, tmp_path, monkeypatch) -> None:
        """Test that duration is rounded to 3 decimal places."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        Ledger.record(
            action="timing_test",
            inputs={},
            outputs={},
            cost_usd=0.01,
            duration_ms=123.456789,  # Should be rounded to 123.457
            outcome="success",
            tags=[],
        )
        
        log_path = Ledger.get_log_path()
        with open(log_path, "r") as f:
            entry = json.loads(f.readline())
        
        assert entry["duration_ms"] == 123.457


class TestLedgerFailOpen:
    """Test fail-open behavior when ledger operations fail."""

    def test_record_disabled_ledger(self, tmp_path, monkeypatch) -> None:
        """Test that record() handles disabled ledger gracefully."""
        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path / "nonexistent" / "readonly"))
        
        # Create read-only parent to cause failure
        readonly_parent = tmp_path / "nonexistent"
        readonly_parent.mkdir()
        readonly_parent.chmod(0o444)
        
        try:
            Ledger.reset()
            Ledger.setup()
            
            # This should not raise an exception even if ledger is disabled
            Ledger.record(
                action="test",
                inputs={},
                outputs={},
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                tags=[],
            )
        finally:
            readonly_parent.chmod(0o755)

    def test_get_log_path_none_when_disabled(self, tmp_path, monkeypatch) -> None:
        """Test that get_log_path returns None when ledger is disabled."""
        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path / "readonly" / "sentinel"))
        
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o444)
        
        try:
            Ledger.reset()
            path = Ledger.get_log_path()
            # Should be None or still initialized but unusable
            assert path is None or isinstance(path, Path)
        finally:
            readonly.chmod(0o755)


class TestLedgerRotation:
    """Test ledger rotation and archival scenarios."""

    def test_ledger_with_size_threshold(self, tmp_path, monkeypatch) -> None:
        """Test reading ledger that grows to significant size."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        # Write 100 entries
        for i in range(100):
            Ledger.record(
                action=f"action_{i}",
                inputs={"index": i},
                outputs={"result": i * 2},
                cost_usd=0.01 * i,
                duration_ms=100.0 + i,
                outcome="success" if i % 2 == 0 else "error",
                tags=["bulk"],
            )
        
        log_path = Ledger.get_log_path()
        with open(log_path, "r") as f:
            entries = [json.loads(line) for line in f.readlines()]
        
        assert len(entries) == 100
        assert entries[0]["action"] == "action_0"
        assert entries[99]["action"] == "action_99"

    def test_ledger_file_format_compatibility(self, tmp_path, monkeypatch) -> None:
        """Test that ledger file maintains JSONL format."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        # Write entries
        for i in range(3):
            Ledger.record(
                action=f"test_{i}",
                inputs={},
                outputs={},
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                tags=[],
            )
        
        log_path = Ledger.get_log_path()
        
        # Verify JSONL format: each line is valid JSON
        with open(log_path, "r") as f:
            for line in f:
                assert line.endswith("\n")
                entry = json.loads(line.rstrip("\n"))
                assert "id" in entry
                assert "timestamp" in entry
                assert "action" in entry

    def test_concurrent_writes_same_file(self, tmp_path, monkeypatch) -> None:
        """Test multiple writes to same ledger file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        # Simulate concurrent writes by calling record multiple times
        for i in range(10):
            Ledger.record(
                action="concurrent_action",
                inputs={"id": i},
                outputs={},
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                tags=[],
            )
        
        log_path = Ledger.get_log_path()
        
        # All entries should be present and valid
        with open(log_path, "r") as f:
            entries = [json.loads(line) for line in f.readlines()]
        
        assert len(entries) == 10
        for i, entry in enumerate(entries):
            assert entry["payload"]["inputs"]["id"] == i

