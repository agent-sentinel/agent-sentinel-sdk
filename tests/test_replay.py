"""
Tests for replay mode functionality.

Phase 4: Tests for ReplayMode, replay context manager, and integration with guarded_action.
"""
from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent_sentinel import (
    guarded_action,
    ReplayMode,
    ReplayEntry,
    replay_mode,
    ReplayDivergenceError,
)
from agent_sentinel.ledger import Ledger


class TestReplayEntry:
    """Test the ReplayEntry dataclass."""
    
    def test_replay_entry_creation(self):
        """Test that ReplayEntry can be created with all fields."""
        entry = ReplayEntry(
            id="test-id",
            action="test_action",
            inputs={"args": (), "kwargs": {"x": 1}},
            outputs={"result": 42},
            cost_usd=0.01,
            duration_ms=100.0,
            outcome="success",
            timestamp="2025-01-01T00:00:00Z",
            tags=["test"],
        )
        
        assert entry.id == "test-id"
        assert entry.action == "test_action"
        assert entry.inputs == {"args": (), "kwargs": {"x": 1}}
        assert entry.outputs == {"result": 42}
        assert entry.cost_usd == 0.01
        assert entry.duration_ms == 100.0
        assert entry.outcome == "success"
        assert entry.timestamp == "2025-01-01T00:00:00Z"
        assert entry.tags == ["test"]


class TestReplayModeBasics:
    """Test basic ReplayMode functionality."""
    
    def test_replay_mode_initialization(self):
        """Test that ReplayMode can be initialized with entries."""
        entries = [
            ReplayEntry(
                id="1",
                action="action1",
                inputs={"args": (), "kwargs": {}},
                outputs="result1",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=True, warn_on_divergence=True)
        
        assert replay.entries == entries
        assert replay.strict is True
        assert replay.warn_on_divergence is True
        assert replay.current_index == 0
        assert replay.divergences == []
    
    def test_replay_mode_context_manager(self):
        """Test that ReplayMode works as a context manager."""
        entries = [
            ReplayEntry(
                id="1",
                action="action1",
                inputs={"args": (), "kwargs": {}},
                outputs="result1",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries)
        
        assert not ReplayMode.is_active()
        
        with replay:
            assert ReplayMode.is_active()
            assert ReplayMode.get_active() == replay
        
        assert not ReplayMode.is_active()
        assert ReplayMode.get_active() is None
    
    def test_replay_mode_progress(self):
        """Test progress tracking."""
        entries = [
            ReplayEntry(
                id=str(i),
                action=f"action{i}",
                inputs={"args": (), "kwargs": {}},
                outputs=f"result{i}",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
            for i in range(3)
        ]
        
        replay = ReplayMode(entries)
        
        assert replay.get_progress() == (0, 3)
        
        replay.get_next_output("action0", {"args": (), "kwargs": {}})
        assert replay.get_progress() == (1, 3)
        
        replay.get_next_output("action1", {"args": (), "kwargs": {}})
        assert replay.get_progress() == (2, 3)
        
        replay.get_next_output("action2", {"args": (), "kwargs": {}})
        assert replay.get_progress() == (3, 3)
    
    def test_replay_mode_reset(self):
        """Test resetting replay progress."""
        entries = [
            ReplayEntry(
                id="1",
                action="action1",
                inputs={"args": (), "kwargs": {}},
                outputs="result1",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries)
        
        replay.get_next_output("action1", {"args": (), "kwargs": {}})
        assert replay.current_index == 1
        
        replay.reset()
        assert replay.current_index == 0
        assert replay.divergences == []


class TestReplayModeOutputRetrieval:
    """Test output retrieval and matching logic."""
    
    def test_get_next_output_success(self):
        """Test getting next output with matching inputs."""
        entries = [
            ReplayEntry(
                id="1",
                action="test_action",
                inputs={"args": (), "kwargs": {"x": 1}},
                outputs="success_result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=True)
        
        output, inputs_match = replay.get_next_output(
            "test_action",
            {"args": (), "kwargs": {"x": 1}}
        )
        
        assert output == "success_result"
        assert inputs_match is True
    
    def test_get_next_output_action_mismatch_strict(self):
        """Test action mismatch in strict mode raises error."""
        entries = [
            ReplayEntry(
                id="1",
                action="expected_action",
                inputs={"args": (), "kwargs": {}},
                outputs="result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=True)
        
        with pytest.raises(ReplayDivergenceError) as exc_info:
            replay.get_next_output("different_action", {"args": (), "kwargs": {}})
        
        assert "Action mismatch" in str(exc_info.value)
        assert "expected_action" in str(exc_info.value)
        assert "different_action" in str(exc_info.value)
    
    def test_get_next_output_action_mismatch_lenient(self):
        """Test action mismatch in lenient mode logs but continues."""
        entries = [
            ReplayEntry(
                id="1",
                action="expected_action",
                inputs={"args": (), "kwargs": {}},
                outputs="result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=False)
        
        output, inputs_match = replay.get_next_output(
            "different_action",
            {"args": (), "kwargs": {}}
        )
        
        assert output == "result"
        assert len(replay.divergences) == 1
        assert replay.divergences[0]["type"] == "action_mismatch"
    
    def test_get_next_output_input_mismatch_strict(self):
        """Test input mismatch in strict mode raises error."""
        entries = [
            ReplayEntry(
                id="1",
                action="test_action",
                inputs={"args": (), "kwargs": {"x": 1}},
                outputs="result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=True)
        
        with pytest.raises(ReplayDivergenceError) as exc_info:
            replay.get_next_output(
                "test_action",
                {"args": (), "kwargs": {"x": 2}}  # Different input
            )
        
        assert "Input mismatch" in str(exc_info.value)
    
    def test_get_next_output_input_mismatch_lenient(self):
        """Test input mismatch in lenient mode logs but continues."""
        entries = [
            ReplayEntry(
                id="1",
                action="test_action",
                inputs={"args": (), "kwargs": {"x": 1}},
                outputs="result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=False)
        
        output, inputs_match = replay.get_next_output(
            "test_action",
            {"args": (), "kwargs": {"x": 2}}
        )
        
        assert output == "result"
        assert inputs_match is False
        assert len(replay.divergences) == 1
        assert replay.divergences[0]["type"] == "input_mismatch"
    
    def test_get_next_output_exhausted(self):
        """Test that exhausting entries raises IndexError."""
        entries = [
            ReplayEntry(
                id="1",
                action="test_action",
                inputs={"args": (), "kwargs": {}},
                outputs="result",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries)
        
        # First call succeeds
        replay.get_next_output("test_action", {"args": (), "kwargs": {}})
        
        # Second call should fail
        with pytest.raises(IndexError) as exc_info:
            replay.get_next_output("test_action", {"args": (), "kwargs": {}})
        
        assert "No more replay entries available" in str(exc_info.value)


class TestReplayModeInputMatching:
    """Test input matching logic."""
    
    def test_inputs_match_identical(self):
        """Test that identical inputs match."""
        recorded = {"args": (1, 2, 3), "kwargs": {"x": "test", "y": 42}}
        current = {"args": (1, 2, 3), "kwargs": {"x": "test", "y": 42}}
        
        assert ReplayMode._inputs_match(recorded, current) is True
    
    def test_inputs_match_different_key_order(self):
        """Test that different key order still matches."""
        recorded = {"kwargs": {"x": 1, "y": 2}, "args": ()}
        current = {"args": (), "kwargs": {"y": 2, "x": 1}}
        
        assert ReplayMode._inputs_match(recorded, current) is True
    
    def test_inputs_match_different_values(self):
        """Test that different values don't match."""
        recorded = {"args": (), "kwargs": {"x": 1}}
        current = {"args": (), "kwargs": {"x": 2}}
        
        assert ReplayMode._inputs_match(recorded, current) is False
    
    def test_inputs_match_missing_key(self):
        """Test that missing keys don't match."""
        recorded = {"args": (), "kwargs": {"x": 1, "y": 2}}
        current = {"args": (), "kwargs": {"x": 1}}
        
        assert ReplayMode._inputs_match(recorded, current) is False
    
    def test_inputs_match_nested_structures(self):
        """Test matching nested dictionaries and lists."""
        recorded = {
            "args": ([1, 2, 3],),
            "kwargs": {"data": {"nested": {"value": 42}}}
        }
        current = {
            "args": ([1, 2, 3],),
            "kwargs": {"data": {"nested": {"value": 42}}}
        }
        
        assert ReplayMode._inputs_match(recorded, current) is True


class TestReplayModeFromLedger:
    """Test loading replay data from ledger files."""
    
    def test_load_entries_from_ledger(self):
        """Test loading entries from a ledger file."""
        # Create a temporary ledger file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            # Write some test entries
            entry1 = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "action1",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": ["test"],
                "payload": {
                    "inputs": {"args": (1,), "kwargs": {"x": 1}},
                    "outputs": "result1"
                }
            }
            entry2 = {
                "id": "entry-2",
                "timestamp": "2025-01-01T00:01:00Z",
                "action": "action2",
                "cost_usd": 0.02,
                "duration_ms": 200.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "result2"
                }
            }
            
            f.write(json.dumps(entry1) + "\n")
            f.write(json.dumps(entry2) + "\n")
        
        try:
            # Load entries
            entries = ReplayMode._load_entries(ledger_path, run_id=None)
            
            assert len(entries) == 2
            assert entries[0].id == "entry-1"
            assert entries[0].action == "action1"
            assert entries[0].outputs == "result1"
            assert entries[1].id == "entry-2"
            assert entries[1].action == "action2"
            assert entries[1].outputs == "result2"
        finally:
            # Clean up
            ledger_path.unlink()
    
    def test_load_entries_empty_lines_ignored(self):
        """Test that empty lines in ledger are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "action1",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "result1"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
            f.write("\n")  # Empty line
            f.write("\n")  # Another empty line
        
        try:
            entries = ReplayMode._load_entries(ledger_path, run_id=None)
            assert len(entries) == 1
        finally:
            ledger_path.unlink()
    
    def test_load_entries_invalid_json_skipped(self):
        """Test that invalid JSON lines are skipped with warning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            valid_entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "action1",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "result1"
                }
            }
            
            f.write(json.dumps(valid_entry) + "\n")
            f.write("{ invalid json }\n")  # Invalid JSON
            f.write(json.dumps(valid_entry) + "\n")
        
        try:
            entries = ReplayMode._load_entries(ledger_path, run_id=None)
            # Should load 2 valid entries, skip 1 invalid
            assert len(entries) == 2
        finally:
            ledger_path.unlink()
    
    def test_from_ledger_file(self):
        """Test creating ReplayMode from ledger file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "action1",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "result1"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
        
        try:
            replay = ReplayMode.from_ledger_file(ledger_path, strict=True)
            
            assert len(replay.entries) == 1
            assert replay.strict is True
        finally:
            ledger_path.unlink()
    
    def test_from_ledger_file_not_found(self):
        """Test that missing ledger file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ReplayMode.from_ledger_file(Path("/nonexistent/ledger.jsonl"))
    
    def test_from_ledger_file_empty(self):
        """Test that empty ledger raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ReplayMode.from_ledger_file(ledger_path)
            
            assert "No entries found" in str(exc_info.value)
        finally:
            ledger_path.unlink()


class TestReplayModeIntegration:
    """Test integration with guarded_action decorator."""
    
    def test_replay_with_guarded_action(self):
        """Test that guarded_action uses replay mode when active."""
        # Create a temporary ledger
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "test_func",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (5,), "kwargs": {}},
                    "outputs": "replayed_result"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
        
        try:
            # Define a function that should NOT be called during replay
            call_count = [0]
            
            @guarded_action(name="test_func", cost_usd=0.01)
            def test_func(x):
                call_count[0] += 1
                return f"real_result_{x}"
            
            # Test normal execution (not in replay mode)
            result = test_func(5)
            assert result == "real_result_5"
            assert call_count[0] == 1
            
            # Test replay mode
            replay = ReplayMode.from_ledger_file(ledger_path, strict=True)
            
            with replay:
                result = test_func(5)
                # Should return replayed result without calling the function
                assert result == "replayed_result"
                assert call_count[0] == 1  # Should NOT increment
        finally:
            ledger_path.unlink()
    
    def test_replay_with_divergence(self):
        """Test that divergence is detected during replay."""
        # Create a temporary ledger
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "test_func",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (5,), "kwargs": {}},
                    "outputs": "recorded_result"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
        
        try:
            @guarded_action(name="test_func", cost_usd=0.01)
            def test_func(x):
                return f"result_{x}"
            
            replay = ReplayMode.from_ledger_file(ledger_path, strict=True)
            
            with replay:
                # Call with different input - should raise divergence error
                with pytest.raises(ReplayDivergenceError):
                    test_func(10)  # Different from recorded input (5)
        finally:
            ledger_path.unlink()
    
    def test_replay_context_manager(self):
        """Test the replay_mode context manager helper."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "test_func",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "replayed"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
        
        try:
            @guarded_action(name="test_func", cost_usd=0.01)
            def test_func():
                return "real"
            
            # Use the helper context manager
            with replay_mode(ledger_path=ledger_path) as replay:
                result = test_func()
                assert result == "replayed"
                assert replay.get_progress() == (1, 1)
        finally:
            ledger_path.unlink()


class TestReplayModeDivergenceTracking:
    """Test divergence tracking and reporting."""
    
    def test_get_divergences(self):
        """Test retrieving list of divergences."""
        entries = [
            ReplayEntry(
                id="1",
                action="action1",
                inputs={"args": (), "kwargs": {"x": 1}},
                outputs="result1",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=False)
        
        # Cause an input mismatch
        replay.get_next_output("action1", {"args": (), "kwargs": {"x": 2}})
        
        divergences = replay.get_divergences()
        assert len(divergences) == 1
        assert divergences[0]["type"] == "input_mismatch"
        assert divergences[0]["action"] == "action1"
    
    def test_multiple_divergences(self):
        """Test tracking multiple divergences."""
        entries = [
            ReplayEntry(
                id="1",
                action="action1",
                inputs={"args": (), "kwargs": {}},
                outputs="result1",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:00:00Z",
                tags=[],
            ),
            ReplayEntry(
                id="2",
                action="action2",
                inputs={"args": (), "kwargs": {}},
                outputs="result2",
                cost_usd=0.01,
                duration_ms=100.0,
                outcome="success",
                timestamp="2025-01-01T00:01:00Z",
                tags=[],
            )
        ]
        
        replay = ReplayMode(entries, strict=False)
        
        # Cause action name mismatch
        replay.get_next_output("wrong_action", {"args": (), "kwargs": {}})
        
        # Cause input mismatch
        replay.get_next_output("action2", {"args": (), "kwargs": {"x": 1}})
        
        divergences = replay.get_divergences()
        assert len(divergences) == 2
        assert divergences[0]["type"] == "action_mismatch"
        assert divergences[1]["type"] == "input_mismatch"


class TestReplayModeAsync:
    """Test replay mode with async functions."""
    
    @pytest.mark.asyncio
    async def test_replay_async_function(self):
        """Test that async functions work with replay mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            ledger_path = Path(f.name)
            
            entry = {
                "id": "entry-1",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": "async_func",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": [],
                "payload": {
                    "inputs": {"args": (), "kwargs": {}},
                    "outputs": "async_replayed"
                }
            }
            
            f.write(json.dumps(entry) + "\n")
        
        try:
            @guarded_action(name="async_func", cost_usd=0.01)
            async def async_func():
                return "async_real"
            
            # Test replay
            replay = ReplayMode.from_ledger_file(ledger_path, strict=True)
            
            with replay:
                result = await async_func()
                assert result == "async_replayed"
        finally:
            ledger_path.unlink()

