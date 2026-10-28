"""
Tests for the redaction module.

Covers:
- Key-based redaction (sensitive key names)
- Value-based redaction (common secret shapes)
- Truncation of oversized values
- The AGENT_SENTINEL_LEDGER_INPUTS=none off switch
- Guard/ledger integration (inputs_redacted flag on the ledger entry)
"""
from __future__ import annotations

import json
import os

import pytest

from agent_sentinel.redaction import (
    LEDGER_INPUTS_ENV,
    ledger_input_capture_enabled,
    redact_inputs,
)
from agent_sentinel.ledger import Ledger
from agent_sentinel.guard import guarded_action


# Secret-shaped fixtures are assembled at runtime (never as a single
# contiguous literal) so this file has nothing for a git-history secret
# scanner to flag while still exercising the real value-shape patterns in
# agent_sentinel.redaction.
def _fake_openai_key() -> str:
    return "sk-" + "a" * 48


def _fake_stripe_key() -> str:
    return "sk_live_" + "5" + "1H8" + "a" * 20


def _fake_aws_key() -> str:
    return "AKIA" + "B" * 16


def _fake_github_token() -> str:
    return "ghp_" + "a" * 36


def _fake_slack_token(kind: str) -> str:
    return f"xox{kind}" + "-1234567890-abcdefghij"


def _fake_pem_header(inner: str) -> str:
    return "-----BEGIN " + inner + "-----"


def _fake_jwt() -> str:
    part_a = "eyJhbGciOiJIUzI1NiJ9"
    part_b = "eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    part_c = "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PYb4T_Snyo3g"
    return ".".join([part_a, part_b, part_c])


OPENAI_KEY = _fake_openai_key()
STRIPE_KEY = _fake_stripe_key()
AWS_KEY = _fake_aws_key()
GITHUB_TOKEN = _fake_github_token()
SLACK_TOKEN_B = _fake_slack_token("b")
SLACK_TOKEN_P = _fake_slack_token("p")
SLACK_TOKEN_A = _fake_slack_token("a")
PEM_RSA_HEADER = _fake_pem_header("RSA PRIVATE KEY")
PEM_PLAIN_HEADER = _fake_pem_header("PRIVATE KEY")
JWT = _fake_jwt()


class TestKeyBasedRedaction:
    """Values are redacted based on a sensitive key name, regardless of shape."""

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "api-key",
            "apikey",
            "secret",
            "client_secret",
            "token",
            "auth_token",
            "password",
            "passwd",
            "authorization",
            "credential",
            "credentials",
            "private_key",
            "private-key",
        ],
    )
    def test_sensitive_key_redacted(self, key: str) -> None:
        _, kwargs, redacted = redact_inputs((), {key: "totally-plain-value"})
        assert redacted is True
        assert kwargs[key] == "[redacted:key]"

    def test_normal_key_not_redacted(self) -> None:
        _, kwargs, redacted = redact_inputs((), {"username": "alice", "email": "a@b.com"})
        assert redacted is False
        assert kwargs == {"username": "alice", "email": "a@b.com"}

    def test_sensitive_key_nested_dict_fully_replaced(self) -> None:
        _, kwargs, redacted = redact_inputs(
            (), {"credentials": {"user": "alice", "pass": "hunter2"}}
        )
        assert redacted is True
        assert kwargs["credentials"] == "[redacted:key]"

    def test_sensitive_key_with_none_value_preserved(self) -> None:
        _, kwargs, redacted = redact_inputs((), {"api_key": None})
        assert redacted is False
        assert kwargs["api_key"] is None


class TestValueBasedRedaction:
    """Values matching common secret shapes are redacted even under an innocuous key."""

    @pytest.mark.parametrize(
        "value,kind",
        [
            (OPENAI_KEY, "openai_key"),
            (STRIPE_KEY, "stripe_key"),
            (AWS_KEY, "aws_key"),
            (GITHUB_TOKEN, "github_token"),
            (SLACK_TOKEN_B, "slack_token"),
            (SLACK_TOKEN_P, "slack_token"),
            (SLACK_TOKEN_A, "slack_token"),
            (PEM_RSA_HEADER, "private_key"),
            (PEM_PLAIN_HEADER, "private_key"),
            (JWT, "jwt"),
        ],
    )
    def test_secret_shaped_value_redacted(self, value: str, kind: str) -> None:
        _, kwargs, redacted = redact_inputs((), {"notes": value})
        assert redacted is True
        assert kwargs["notes"] == f"[redacted:{kind}]"

    def test_plain_string_not_redacted(self) -> None:
        _, kwargs, redacted = redact_inputs((), {"notes": "just a normal comment"})
        assert redacted is False
        assert kwargs["notes"] == "just a normal comment"

    def test_secret_in_nested_dict(self) -> None:
        _, kwargs, redacted = redact_inputs(
            (), {"config": {"upstream": OPENAI_KEY}}
        )
        assert redacted is True
        assert kwargs["config"]["upstream"] == "[redacted:openai_key]"

    def test_secret_in_list(self) -> None:
        _, kwargs, redacted = redact_inputs((), {"notes": [OPENAI_KEY, "safe"]})
        assert redacted is True
        assert kwargs["notes"][0] == "[redacted:openai_key]"
        assert kwargs["notes"][1] == "safe"

    def test_secret_in_positional_args(self) -> None:
        args, _, redacted = redact_inputs((OPENAI_KEY,), {})
        assert redacted is True
        assert args[0] == "[redacted:openai_key]"

    def test_non_string_values_untouched(self) -> None:
        _, kwargs, redacted = redact_inputs((), {"count": 5, "enabled": True, "ratio": 1.5})
        assert redacted is False
        assert kwargs == {"count": 5, "enabled": True, "ratio": 1.5}


class TestTruncation:
    """Oversized values are truncated rather than redacted."""

    def test_value_under_limit_untouched(self) -> None:
        value = "a" * 100
        _, kwargs, redacted = redact_inputs((), {"note": value}, max_bytes=8192)
        assert redacted is False
        assert kwargs["note"] == value

    def test_value_over_limit_truncated(self) -> None:
        value = "a" * 100
        _, kwargs, redacted = redact_inputs((), {"note": value}, max_bytes=10)
        assert redacted is True
        assert kwargs["note"].startswith("a" * 10)
        assert "...[truncated 90 bytes]" in kwargs["note"]

    def test_default_max_bytes_is_8192(self) -> None:
        value = "a" * 8193
        _, kwargs, redacted = redact_inputs((), {"note": value})
        assert redacted is True
        assert "...[truncated 1 bytes]" in kwargs["note"]


class TestOffSwitch:
    """AGENT_SENTINEL_LEDGER_INPUTS=none disables ledger input capture."""

    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LEDGER_INPUTS_ENV, raising=False)
        assert ledger_input_capture_enabled() is True

    @pytest.mark.parametrize("value", ["none", "None", "NONE"])
    def test_disabled_when_set_to_none(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(LEDGER_INPUTS_ENV, value)
        assert ledger_input_capture_enabled() is False

    def test_other_values_leave_capture_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LEDGER_INPUTS_ENV, "all")
        assert ledger_input_capture_enabled() is True


class TestNoMutation:
    def test_original_kwargs_not_mutated(self) -> None:
        original = {"api_key": OPENAI_KEY}
        redact_inputs((), original)
        assert original == {"api_key": OPENAI_KEY}


class TestLedgerIntegration:
    """The ledger entry carries inputs_redacted: true only when something was replaced."""

    def _read_last_entry(self, tmp_path) -> dict:
        log_path = tmp_path / ".agent-sentinel" / "ledger.jsonl"
        with open(log_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        return json.loads(lines[-1])

    def test_flag_absent_when_nothing_redacted(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        Ledger.reset()
        Ledger.record(
            action="test_action",
            inputs={"args": (), "kwargs": {"username": "alice"}},
            outputs="ok",
            cost_usd=0.0,
            duration_ms=1.0,
            outcome="success",
            tags=[],
        )
        entry = self._read_last_entry(tmp_path)
        assert "inputs_redacted" not in entry

    def test_flag_true_when_redacted(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        Ledger.reset()
        Ledger.record(
            action="test_action",
            inputs={"args": (), "kwargs": {"api_key": "[redacted:key]"}},
            outputs="ok",
            cost_usd=0.0,
            duration_ms=1.0,
            outcome="success",
            tags=[],
            inputs_redacted=True,
        )
        entry = self._read_last_entry(tmp_path)
        assert entry["inputs_redacted"] is True


class TestGuardIntegration:
    """End-to-end: guarded_action redacts secrets before they reach the ledger."""

    def _read_entries(self, tmp_path) -> list:
        log_path = Ledger.get_log_path()
        with open(log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_secret_kwarg_redacted_before_ledger_write(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        monkeypatch.delenv(LEDGER_INPUTS_ENV, raising=False)
        Ledger.reset()

        @guarded_action(name="call_with_secret", cost_usd=0.0)
        def call_with_secret(api_key: str) -> str:
            return "ok"

        call_with_secret(api_key=OPENAI_KEY)

        entry = self._read_entries(tmp_path)[-1]
        assert entry["inputs_redacted"] is True
        assert entry["payload"]["inputs"]["kwargs"]["api_key"] == "[redacted:key]"
        assert OPENAI_KEY not in json.dumps(entry)

    def test_ledger_inputs_none_disables_capture(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        monkeypatch.setenv(LEDGER_INPUTS_ENV, "none")
        Ledger.reset()

        @guarded_action(name="call_with_secret_off", cost_usd=0.0)
        def call_with_secret_off(api_key: str) -> str:
            return "ok"

        call_with_secret_off(api_key=OPENAI_KEY)

        entry = self._read_entries(tmp_path)[-1]
        assert entry["payload"]["inputs"] == {}
        assert "inputs_redacted" not in entry
