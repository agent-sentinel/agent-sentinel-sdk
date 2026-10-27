"""Tests for ReplayMode.from_json() — loading a console-downloaded run bundle."""
import pytest
from agent_sentinel.replay import ReplayMode


SAMPLE_BUNDLE = {
    "run_id": "run-abc-123",
    "agent_id": "agent-x",
    "entries": [
        {
            "id": "entry-1",
            "action": "fetch_data",
            "inputs": {"url": "https://api.example.com/data"},
            "outputs": {"status": 200, "body": "hello"},
            "cost_usd": 0.001,
            "duration_ms": 123.4,
            "outcome": "completed",
            "timestamp": "2026-06-05T10:00:00Z",
            "tags": ["network"],
            "run_id": "run-abc-123",
        },
        {
            "id": "entry-2",
            "action": "write_file",
            "inputs": {"path": "/tmp/out.txt", "content": "hello"},
            "outputs": None,
            "cost_usd": 0.0,
            "duration_ms": 5.0,
            "outcome": "completed",
            "timestamp": "2026-06-05T10:00:01Z",
            "tags": [],
            "run_id": "run-abc-123",
        },
    ],
}


def test_from_json_loads_correct_entry_count():
    replay = ReplayMode.from_json(SAMPLE_BUNDLE)
    assert len(replay.entries) == 2


def test_from_json_entry_fields_are_correct():
    replay = ReplayMode.from_json(SAMPLE_BUNDLE)
    entry = replay.entries[0]
    assert entry.action == "fetch_data"
    assert entry.inputs == {"url": "https://api.example.com/data"}
    assert entry.outputs == {"status": 200, "body": "hello"}
    assert entry.cost_usd == 0.001
    assert entry.run_id == "run-abc-123"


def test_from_json_raises_on_missing_entries():
    with pytest.raises(ValueError, match="non-empty"):
        ReplayMode.from_json({"run_id": "x"})


def test_from_json_respects_strict_flag():
    replay = ReplayMode.from_json(SAMPLE_BUNDLE, strict=False)
    assert replay.strict is False


def test_from_json_with_empty_entries_raises():
    with pytest.raises(ValueError):
        ReplayMode.from_json({"run_id": "x", "entries": []})
