"""
Tests for fail_closed coverage: unreachable platform, kill-switch errors,
and policy evaluation exceptions.

fail_closed already blocked one case before this change: an action checked
while no policy is loaded at all (PolicyNotConfiguredError, see
test_enforcement_and_remediation.py). This file covers the paths that used
to stay fail-open regardless of enforcement mode:

1. A remote policy sync that has never succeeded and has no local cache --
   the engine never becomes "configured", so this is actually the same
   PolicyNotConfiguredError path, exercised through enable_remote_sync().
2. A kill-switch check that cannot complete (platform unreachable, non-200,
   or httpx missing) -- previously always returned False (fail-open); now
   fail_closed raises KillSwitchUnavailableError.
3. An unexpected exception raised from inside policy evaluation itself
   (a bug in a guardrail, the rate limiter, or CostTracker) -- previously
   propagated raw in every mode; now fail_closed wraps it in
   PolicyViolationError(reason_code="EVALUATION_ERROR").

Also covers the public parameters (configure(), guarded_action(),
PolicyEngine.configure()) and precedence, and that a pinned enforcement mode
survives PolicyEngine.reset() unless clear_pin=True.

t16 adds "fail_closed_writes" as the actual SDK default (see the
PolicyEngine._enforcement_mode class attribute and _handle_unconfigured):
the same three trigger conditions above now depend on the action's declared
impact (guarded_action(impact=...), default "medium"). "medium"/"high" block
with NoPolicyForWriteError (reason_code NO_POLICY_FOR_WRITE); "read"/"low"
proceed with a warning, the same as fail_open. PolicyEngine.reset() itself
now reverts an unpinned mode to that same "fail_closed_writes" default (see
its docstring), so conftest.py's autouse fixture pins "fail_open" explicitly
after every reset() -- that pin, not reset()'s own contract, is what keeps
the rest of the suite on "fail_open". This is the file that exercises
"fail_closed_writes" itself -- see TestImpactAwareFailClosedWritesDefault
below, whose tests call set_enforcement_mode("fail_closed_writes") directly.
"""
from __future__ import annotations

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from agent_sentinel.cost import CostTracker
from agent_sentinel.errors import (
    AgentKilledError,
    KillSwitchUnavailableError,
    NoPolicyForWriteError,
    PolicyNotConfiguredError,
    PolicyViolationError,
)
from agent_sentinel.guard import _reason_code_to_intervention, guarded_action
from agent_sentinel.kill_switch import KillSwitchClient
from agent_sentinel.policy import PolicyCache, PolicyEngine


@pytest.fixture(autouse=True)
def _clean_kill_switch():
    """conftest.py resets PolicyEngine/CostTracker; KillSwitchClient needs the same treatment here."""
    KillSwitchClient.reset()
    yield
    KillSwitchClient.reset()


def _raise(*_args, **_kwargs):
    raise RuntimeError("policy evaluation blew up")


# ---------------------------------------------------------------------------
# 1. Remote policy sync that has never succeeded and has no cache.
# ---------------------------------------------------------------------------


class TestFailClosedUnreachablePlatform:
    @patch("httpx.get")
    def test_fail_closed_blocks_when_sync_never_succeeded_no_cache(self, mock_get, tmp_path):
        mock_get.side_effect = Exception("network error")
        cache = PolicyCache(tmp_path)

        PolicyEngine.set_enforcement_mode("fail_closed")
        with patch("agent_sentinel.policy.PolicyCache", return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999,
            )

        assert not PolicyEngine.is_configured()
        with pytest.raises(PolicyNotConfiguredError):
            PolicyEngine.check_action("do_thing", cost=0.0)

    @patch("httpx.get")
    def test_fail_open_still_permits_when_sync_never_succeeded_no_cache(self, mock_get, tmp_path):
        """Default behaviour, unchanged: sync failure without a policy must not block."""
        mock_get.side_effect = Exception("network error")
        cache = PolicyCache(tmp_path)

        with patch("agent_sentinel.policy.PolicyCache", return_value=cache):
            PolicyEngine.enable_remote_sync(
                platform_url="https://api.test.com",
                api_token="test-token",
                refresh_interval=9999,
            )

        assert not PolicyEngine.is_configured()
        PolicyEngine.check_action("do_thing", cost=0.0)  # must not raise


# ---------------------------------------------------------------------------
# 2. Kill-switch checks that cannot complete.
# ---------------------------------------------------------------------------


class TestFailClosedKillSwitch:
    def _configure_kill_switch(self):
        KillSwitchClient.configure(platform_url="https://api.test.com", api_token="tok")

    @patch("httpx.Client")
    def test_fail_open_network_error_returns_false(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client_cls.side_effect = RuntimeError("connection refused")

        assert KillSwitchClient.is_killed("agent", "a-1") is False

    @patch("httpx.Client")
    def test_fail_closed_network_error_raises_kill_switch_unavailable(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client_cls.side_effect = RuntimeError("connection refused")

        with pytest.raises(KillSwitchUnavailableError) as exc:
            KillSwitchClient.is_killed("agent", "a-1", fail_closed=True)

        assert isinstance(exc.value, AgentKilledError)
        assert exc.value.details["reason_code"] == "KILL_SWITCH_UNAVAILABLE"
        assert exc.value.details["target_type"] == "agent"
        assert exc.value.details["target_id"] == "a-1"

    @patch("httpx.Client")
    def test_fail_closed_non_200_response_raises(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client = mock_client_cls.return_value
        mock_client.__enter__.return_value.get.return_value = Mock(status_code=503)

        with pytest.raises(KillSwitchUnavailableError):
            KillSwitchClient.is_killed("run", "r-1", fail_closed=True)

    @patch("httpx.Client")
    def test_fail_open_non_200_response_returns_false(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client = mock_client_cls.return_value
        mock_client.__enter__.return_value.get.return_value = Mock(status_code=503)

        assert KillSwitchClient.is_killed("run", "r-1") is False

    def test_fail_closed_httpx_missing_raises(self, monkeypatch):
        self._configure_kill_switch()
        monkeypatch.setattr("agent_sentinel.kill_switch.HTTPX_AVAILABLE", False)

        with pytest.raises(KillSwitchUnavailableError):
            KillSwitchClient.is_killed("agent", "a-2", fail_closed=True)

    def test_fail_open_httpx_missing_returns_false(self, monkeypatch):
        self._configure_kill_switch()
        monkeypatch.setattr("agent_sentinel.kill_switch.HTTPX_AVAILABLE", False)

        assert KillSwitchClient.is_killed("agent", "a-2") is False

    @patch("httpx.Client")
    def test_check_or_raise_fail_closed_propagates_unavailable(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client_cls.side_effect = RuntimeError("down")

        with pytest.raises(KillSwitchUnavailableError):
            KillSwitchClient.check_or_raise(agent_id="a-1", fail_closed=True)

    @patch("httpx.Client")
    def test_check_or_raise_fail_open_does_not_raise(self, mock_client_cls):
        self._configure_kill_switch()
        mock_client_cls.side_effect = RuntimeError("down")

        KillSwitchClient.check_or_raise(agent_id="a-1")  # must not raise

    def test_confirmed_kill_still_raises_agent_killed_regardless_of_mode(self):
        """fail_closed must not change behaviour for an actual, confirmed kill."""
        self._configure_kill_switch()
        with patch.object(KillSwitchClient, "is_killed", return_value=True):
            with pytest.raises(AgentKilledError):
                KillSwitchClient.check_or_raise(agent_id="a-1", fail_closed=True)
            with pytest.raises(AgentKilledError):
                KillSwitchClient.check_or_raise(agent_id="a-1")


class TestGuardedActionKillSwitch:
    """guarded_action(fail_closed=...) must thread through to the kill switch check."""

    def test_guarded_action_fail_closed_blocks_on_kill_switch_error(self):
        KillSwitchClient.configure(platform_url="https://api.test.com", api_token="tok")

        @guarded_action(name="risky", fail_closed=True, agent_id="a-1")
        def risky():
            return "ran"

        with patch("httpx.Client", side_effect=RuntimeError("down")):
            with pytest.raises(KillSwitchUnavailableError):
                risky()

    def test_guarded_action_default_permits_on_kill_switch_error(self):
        KillSwitchClient.configure(platform_url="https://api.test.com", api_token="tok")
        PolicyEngine.configure(run_budget=10.0)

        @guarded_action(name="risky2", agent_id="a-1")
        def risky2():
            return "ran"

        with patch("httpx.Client", side_effect=RuntimeError("down")):
            assert risky2() == "ran"


# ---------------------------------------------------------------------------
# 3. Policy evaluation exceptions.
# ---------------------------------------------------------------------------


class TestFailClosedEvaluationError:
    def test_fail_closed_wraps_unexpected_exception_in_check_action(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("x", cost=1.0, fail_closed=True)

        assert exc.value.remediation.reason_code == "EVALUATION_ERROR"
        assert exc.value.recoverable is False
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_fail_open_lets_unexpected_exception_propagate_in_check_action(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(RuntimeError, match="policy evaluation blew up"):
            PolicyEngine.check_action("x", cost=1.0)

    def test_fail_closed_wraps_unexpected_exception_in_check_and_reserve(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_and_reserve("x", cost=1.0, fail_closed=True)

        assert exc.value.remediation.reason_code == "EVALUATION_ERROR"
        assert CostTracker.get_run_total() == 0.0, "a bug in evaluation must not reserve budget"

    def test_fail_open_lets_unexpected_exception_propagate_in_check_and_reserve(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(RuntimeError, match="policy evaluation blew up"):
            PolicyEngine.check_and_reserve("x", cost=1.0)

    def test_fail_closed_wraps_try_reserve_error(self, monkeypatch):
        """Mirrors check_action's safety net around the budget reservation itself."""
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(CostTracker, "try_reserve", classmethod(lambda cls, *a, **kw: _raise()))

        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_and_reserve("x", cost=1.0, fail_closed=True)

        assert exc.value.remediation.reason_code == "EVALUATION_ERROR"

    def test_fail_open_lets_try_reserve_error_propagate(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(CostTracker, "try_reserve", classmethod(lambda cls, *a, **kw: _raise()))

        with pytest.raises(RuntimeError, match="policy evaluation blew up"):
            PolicyEngine.check_and_reserve("x", cost=1.0)

    def test_reason_code_is_mapped_to_hard_block_intervention(self):
        assert _reason_code_to_intervention()["EVALUATION_ERROR"].value == "hard_block"

    def test_expected_policy_exceptions_pass_through_unwrapped_under_fail_closed(self):
        """A real deny-list block is not an 'evaluation error' -- it must stay a normal block."""
        PolicyEngine.configure(denied_actions=["nope"])

        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("nope", cost=0.0, fail_closed=True)

        assert exc.value.remediation.reason_code == "DENIED_ACTION"


class TestGuardedActionEvaluationError:
    def test_guarded_action_fail_closed_blocks_on_evaluation_bug(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        @guarded_action(name="buggy", fail_closed=True)
        def buggy():
            return "ran"

        with pytest.raises(PolicyViolationError) as exc:
            buggy()
        assert exc.value.remediation.reason_code == "EVALUATION_ERROR"

    def test_guarded_action_default_propagates_evaluation_bug(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        @guarded_action(name="buggy2")
        def buggy2():
            return "ran"

        with pytest.raises(RuntimeError, match="policy evaluation blew up"):
            buggy2()


# ---------------------------------------------------------------------------
# Public parameters and precedence.
# ---------------------------------------------------------------------------


class TestPublicParametersAndPrecedence:
    def test_resolve_enforcement_mode_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_closed")
        assert PolicyEngine.resolve_enforcement_mode(False) == "fail_open"
        assert PolicyEngine.resolve_enforcement_mode(True) == "fail_closed"

    def test_resolve_enforcement_mode_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_closed")
        assert PolicyEngine.resolve_enforcement_mode(None) == "fail_closed"

    def test_resolve_enforcement_mode_defaults_fail_open(self):
        assert PolicyEngine.resolve_enforcement_mode(None) == "fail_open"

    def test_guarded_action_fail_closed_true_overrides_global_fail_open(self):
        @guarded_action(name="strict", fail_closed=True)
        def strict():
            return "ran"

        with pytest.raises(PolicyNotConfiguredError):
            strict()

    def test_guarded_action_fail_closed_false_overrides_global_pin(self):
        PolicyEngine.set_enforcement_mode("fail_closed", pin=True)

        @guarded_action(name="lenient", fail_closed=False)
        def lenient():
            return "ran"

        assert lenient() == "ran"

    def test_guarded_action_none_defers_to_global_mode(self):
        PolicyEngine.set_enforcement_mode("fail_closed")

        @guarded_action(name="deferred")
        def deferred():
            return "ran"

        with pytest.raises(PolicyNotConfiguredError):
            deferred()


class TestConfigureDoesNotInstallPermissivePolicy:
    """Linear feedback: configure(fail_closed=True) alone must not become a
    permissive, "configured" empty policy."""

    def test_policy_engine_configure_fail_closed_alone_stays_unconfigured(self):
        PolicyEngine.configure(fail_closed=True)

        assert not PolicyEngine.is_configured()
        assert PolicyEngine.get_enforcement_mode() == "fail_closed"
        with pytest.raises(PolicyNotConfiguredError):
            PolicyEngine.check_action("anything", cost=0.0)

    def test_policy_engine_configure_fail_closed_with_budget_still_configures(self):
        PolicyEngine.configure(fail_closed=True, run_budget=5.0)

        assert PolicyEngine.is_configured()
        PolicyEngine.check_action("cheap", cost=1.0)  # permitted: under budget

    def test_policy_engine_configure_fail_open_alone_also_installs_nothing(self):
        """The guard applies symmetrically -- not just to the fail_closed=True case."""
        PolicyEngine.configure(fail_closed=False)

        assert not PolicyEngine.is_configured()
        assert PolicyEngine.get_enforcement_mode() == "fail_open"

    def test_top_level_configure_fail_closed_alone_pins_mode_only(self, monkeypatch, tmp_path):
        """
        Ledger/background sync is not what this test is about, and a real
        BackgroundSync thread that outlives the test can pollute unrelated
        tests (it keeps calling Ledger.get_log_path(), which other tests
        patch process-wide). Stub it out entirely.
        """
        import agent_sentinel

        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path))
        with patch("agent_sentinel.sync.enable_remote_sync"), \
             patch("httpx.get", side_effect=Exception("network error")):
            try:
                agent_sentinel.configure(
                    api_key="test-key",
                    platform_url="https://api.test.com",
                    fail_closed=True,
                )
                assert PolicyEngine.get_enforcement_mode() == "fail_closed"
                assert not PolicyEngine.is_configured()
            finally:
                # enable_remote_sync is stubbed out above, so there is no
                # real background sync to flush/stop -- just undo the
                # ExecutionContext.__enter__() that configure() leaves open.
                from agent_sentinel.context import _current_context

                _current_context.set(None)


class TestEndToEndTopLevelConfigureFailClosed:
    """
    Linear feedback: agent_sentinel.configure(api_key, platform_url,
    fail_closed=True) with the HTTP client patched to raise and an empty
    policy cache -> PolicyEngine.is_configured() is False and a
    guarded/checked action raises PolicyNotConfiguredError.
    """

    def test_top_level_configure_blocks_actions_when_sync_fails_and_no_cache(self, monkeypatch, tmp_path):
        """See test_top_level_configure_fail_closed_alone_pins_mode_only for
        why ledger/background sync is stubbed out rather than exercised."""
        import agent_sentinel

        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path))
        with patch("agent_sentinel.sync.enable_remote_sync"), \
             patch("httpx.get", side_effect=Exception("network error")):
            try:
                agent_sentinel.configure(
                    api_key="test-key",
                    platform_url="https://api.test.com",
                    fail_closed=True,
                )

                assert not PolicyEngine.is_configured()

                @guarded_action(name="blocked_by_default")
                def blocked_by_default():
                    return "ran"

                with pytest.raises(PolicyNotConfiguredError):
                    blocked_by_default()
            finally:
                # enable_remote_sync is stubbed out above, so there is no
                # real background sync to flush/stop -- just undo the
                # ExecutionContext.__enter__() that configure() leaves open.
                from agent_sentinel.context import _current_context

                _current_context.set(None)


# ---------------------------------------------------------------------------
# reset() / pin lifecycle.
# ---------------------------------------------------------------------------


class TestResetPreservesPin:
    def test_reset_preserves_pinned_mode_by_default(self):
        PolicyEngine.configure(run_budget=1.0, fail_closed=True)
        PolicyEngine.reset()

        assert PolicyEngine.get_enforcement_mode() == "fail_closed"
        assert not PolicyEngine.is_configured()
        with pytest.raises(PolicyNotConfiguredError):
            PolicyEngine.check_action("x", cost=0.0)

    def test_reset_clear_pin_reverts_to_sdk_default(self):
        PolicyEngine.configure(run_budget=1.0, fail_closed=True)
        PolicyEngine.reset(clear_pin=True)

        assert PolicyEngine.get_enforcement_mode() == "fail_closed_writes"
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_action("x", cost=0.0)

    def test_unpinned_set_enforcement_mode_still_cleared_by_plain_reset(self):
        """An unpinned set_enforcement_mode() call must not survive reset()
        -- only a pin does. It reverts to the SDK default
        ("fail_closed_writes"), not "fail_open" (see reset()'s docstring)."""
        PolicyEngine.set_enforcement_mode("fail_closed")
        PolicyEngine.reset()

        assert PolicyEngine.get_enforcement_mode() == "fail_closed_writes"

    def test_set_enforcement_mode_pin_true_survives_bare_reset(self):
        PolicyEngine.set_enforcement_mode("fail_closed", pin=True)
        PolicyEngine.reset()

        assert PolicyEngine.get_enforcement_mode() == "fail_closed"

    def test_reset_on_never_pinned_engine_defaults_to_fail_closed_writes(self):
        """Regression: a process that never pins or sets an enforcement mode
        is the ordinary case, and reset() must not silently downgrade the
        new secure default to "fail_open" for it -- only an explicit
        AGENT_SENTINEL_ENFORCEMENT=fail_open opt-in should do that (AC2)."""
        assert not PolicyEngine._enforcement_mode_pinned

        PolicyEngine.reset()

        assert PolicyEngine.get_enforcement_mode() == "fail_closed_writes"
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_action("x", cost=0.0)


# ---------------------------------------------------------------------------
# Exports.
# ---------------------------------------------------------------------------


class TestExports:
    def test_kill_switch_unavailable_error_exported_from_package_root(self):
        import agent_sentinel

        assert agent_sentinel.KillSwitchUnavailableError is KillSwitchUnavailableError
        assert "KillSwitchUnavailableError" in agent_sentinel.__all__

    def test_kill_switch_unavailable_error_is_agent_sentinel_error(self):
        from agent_sentinel.errors import AgentSentinelError

        err = KillSwitchUnavailableError(target_type="agent", target_id="a-1")
        assert isinstance(err, AgentKilledError)
        assert isinstance(err, AgentSentinelError)
        assert err.error_code == "AGENT_KILLED"
        assert err.recoverable is False

    def test_no_policy_for_write_error_exported_from_package_root(self):
        import agent_sentinel

        assert agent_sentinel.NoPolicyForWriteError is NoPolicyForWriteError
        assert "NoPolicyForWriteError" in agent_sentinel.__all__

    def test_no_policy_for_write_error_is_policy_violation_error(self):
        err = NoPolicyForWriteError(action_name="ship_it", impact="high")
        assert isinstance(err, PolicyViolationError)
        assert err.error_code == "NO_POLICY_FOR_WRITE"
        assert err.recoverable is False
        assert err.remediation.reason_code == "NO_POLICY_FOR_WRITE"
        assert err.details["impact"] == "high"
        assert err.details["enforcement_mode"] == "fail_closed_writes"


# ---------------------------------------------------------------------------
# t16: "fail_closed_writes" as the actual SDK default.
#
# conftest.py pins the whole suite to "fail_open" so ~30 other test files
# written against the pre-0.4 assumption stay green. These tests unpin it
# explicitly -- this is the file that documents the new default.
# ---------------------------------------------------------------------------


class TestImpactAwareFailClosedWritesDefault:
    def _use_fail_closed_writes(self):
        PolicyEngine.set_enforcement_mode("fail_closed_writes")

    def test_valid_enforcement_modes_include_fail_closed_writes(self):
        assert PolicyEngine.VALID_ENFORCEMENT_MODES == (
            "fail_open", "fail_closed_writes", "fail_closed",
        )

    def test_fresh_process_defaults_to_fail_closed_writes(self):
        """
        The true SDK default for a process that never calls reset() or
        set_enforcement_mode() -- which every real deployment is, since
        reset() is a test/ops utility applications don't call at startup.
        Run in a subprocess: within this suite, conftest's autouse fixture
        already calls PolicyEngine.reset() before this test body even
        starts, and then pins the mode to "fail_open" directly (see
        conftest.py's docstring) -- reset() itself now reverts to this same
        "fail_closed_writes" default, not "fail_open" -- so the bare class
        attribute default isn't observable in-process here.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agent_sentinel.policy import PolicyEngine; "
                "print(PolicyEngine.get_enforcement_mode())",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "fail_closed_writes"

    def test_unconfigured_default_impact_medium_blocks(self):
        self._use_fail_closed_writes()
        with pytest.raises(NoPolicyForWriteError) as exc:
            PolicyEngine.check_action("write_row", cost=0.0)
        assert exc.value.remediation.reason_code == "NO_POLICY_FOR_WRITE"
        assert exc.value.recoverable is False

    def test_unconfigured_high_impact_blocks(self):
        self._use_fail_closed_writes()
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_action("delete_row", cost=0.0, impact="high")

    def test_unconfigured_read_impact_permits_with_warning(self, caplog):
        self._use_fail_closed_writes()
        caplog.set_level("WARNING")
        PolicyEngine.check_action("read_row", cost=0.0, impact="read")  # must not raise
        assert any("fail_closed_writes" in r.message for r in caplog.records)

    def test_unconfigured_low_impact_permits(self):
        self._use_fail_closed_writes()
        PolicyEngine.check_action("list_rows", cost=0.0, impact="low")  # must not raise

    def test_unrecognised_impact_treated_as_write(self):
        """Conservative default: an unrecognised impact string blocks, the
        same as the "medium" default, rather than silently permitting."""
        self._use_fail_closed_writes()
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_action("mystery", cost=0.0, impact="urgent")

    def test_check_and_reserve_unconfigured_write_blocked_records_no_cost(self):
        self._use_fail_closed_writes()
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_and_reserve("write_row", cost=5.0, impact="medium")
        assert CostTracker.get_run_total() == 0.0

    def test_check_and_reserve_unconfigured_read_permits_and_counts_cost(self):
        self._use_fail_closed_writes()
        PolicyEngine.check_and_reserve("read_row", cost=0.02, impact="read")
        assert CostTracker.get_run_total() == pytest.approx(0.02)

    def test_env_fail_open_restores_old_behaviour_regardless_of_impact(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_open")
        PolicyEngine.check_action("write_row", cost=0.0, impact="high")  # must not raise

    def test_env_fail_closed_blocks_everything_regardless_of_impact(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_closed")
        with pytest.raises(PolicyNotConfiguredError):
            PolicyEngine.check_action("read_row", cost=0.0, impact="read")

    def test_evaluation_error_write_impact_wraps_as_no_policy_for_write(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        self._use_fail_closed_writes()
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(NoPolicyForWriteError) as exc:
            PolicyEngine.check_action("x", cost=1.0, impact="medium")
        assert exc.value.remediation.reason_code == "NO_POLICY_FOR_WRITE"
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_evaluation_error_read_impact_propagates_raw(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        self._use_fail_closed_writes()
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(RuntimeError, match="policy evaluation blew up"):
            PolicyEngine.check_action("x", cost=1.0, impact="read")

    def test_evaluation_error_in_check_and_reserve_write_impact_blocks(self, monkeypatch):
        PolicyEngine.configure(run_budget=100.0)
        self._use_fail_closed_writes()
        monkeypatch.setattr(PolicyEngine, "_check_evidence_and_args", staticmethod(_raise))

        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_and_reserve("x", cost=1.0, impact="medium")
        assert CostTracker.get_run_total() == 0.0

    def test_guarded_action_default_impact_blocks_when_unconfigured(self):
        self._use_fail_closed_writes()

        @guarded_action(name="writer")
        def writer():
            return "ran"

        with pytest.raises(NoPolicyForWriteError):
            writer()

    def test_guarded_action_read_impact_permits_when_unconfigured(self):
        self._use_fail_closed_writes()

        @guarded_action(name="reader", impact="read")
        def reader():
            return "ran"

        assert reader() == "ran"

    async def test_guarded_action_async_default_impact_blocks_when_unconfigured(self):
        self._use_fail_closed_writes()

        @guarded_action(name="async_writer")
        async def async_writer():
            return "ran"

        with pytest.raises(NoPolicyForWriteError):
            await async_writer()

    async def test_guarded_action_async_low_impact_permits_when_unconfigured(self):
        self._use_fail_closed_writes()

        @guarded_action(name="async_reader", impact="low")
        async def async_reader():
            return "ran"

        assert await async_reader() == "ran"

    def test_reset_without_clear_pin_preserves_fail_closed_writes_pin(self):
        PolicyEngine.set_enforcement_mode("fail_closed_writes", pin=True)
        PolicyEngine.reset()

        assert PolicyEngine.get_enforcement_mode() == "fail_closed_writes"
        with pytest.raises(NoPolicyForWriteError):
            PolicyEngine.check_action("x", cost=0.0)


class TestImpactReachesLedgerIngestEntry:
    """Acceptance criterion 1: impact is sent in the ingest entry (the ledger
    entry sync.py uploads to POST /api/v1/ingest verbatim) and used locally."""

    def test_ledger_entry_carries_declared_impact(self, monkeypatch, tmp_path):
        import json

        from agent_sentinel.ledger import Ledger

        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path))
        Ledger.reset()
        PolicyEngine.configure(run_budget=10.0)

        @guarded_action(name="write_thing", impact="high")
        def write_thing():
            return "done"

        write_thing()

        log_path = Ledger.get_log_path()
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        entry = json.loads(lines[-1])
        assert entry["impact"] == "high"

    def test_ledger_entry_defaults_to_medium(self, monkeypatch, tmp_path):
        import json

        from agent_sentinel.ledger import Ledger

        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path))
        Ledger.reset()
        PolicyEngine.configure(run_budget=10.0)

        @guarded_action(name="default_impact_action")
        def default_impact_action():
            return "done"

        default_impact_action()

        log_path = Ledger.get_log_path()
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        entry = json.loads(lines[-1])
        assert entry["impact"] == "medium"

    def test_blocked_action_ledger_entry_also_carries_impact(self, monkeypatch, tmp_path):
        import json

        from agent_sentinel.ledger import Ledger

        monkeypatch.setenv("AGENT_SENTINEL_HOME", str(tmp_path))
        Ledger.reset()
        PolicyEngine.set_enforcement_mode("fail_closed_writes")

        @guarded_action(name="blocked_write", impact="high")
        def blocked_write():
            return "ran"

        with pytest.raises(NoPolicyForWriteError):
            blocked_write()

        log_path = Ledger.get_log_path()
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        entry = json.loads(lines[-1])
        assert entry["impact"] == "high"
        assert entry["outcome"] == "blocked"
