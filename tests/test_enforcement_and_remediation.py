"""
Tests for enforcement mode and remediation payloads.

Two behaviours under test:

1. Enforcement mode. The default is fail-open — an unconfigured engine permits
   everything — and that default MUST be preserved, because flipping it would
   start blocking for every existing installation whose config fails to load.
   Deployments that need a real control opt into fail_closed.

2. Remediation. Every block type must hand the agent something it can act on.
   A bare "denied" causes the model to retry the identical call, which cannot
   succeed.
"""
from __future__ import annotations

import pytest

from agent_sentinel.cost import CostTracker
from agent_sentinel.cost_backend import InMemoryCostBackend
from agent_sentinel.errors import (
    BudgetExceededError,
    PolicyNotConfiguredError,
    PolicyViolationError,
)
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.integrations.tool_executor import SentinelToolExecutor


@pytest.fixture(autouse=True)
def clean_state():
    """Reset engine and counters around every test."""
    CostTracker.set_backend(InMemoryCostBackend())
    PolicyEngine._config = None
    PolicyEngine._initialized = False
    PolicyEngine._enforcement_mode = "fail_open"
    PolicyEngine._enforcement_mode_explicit = False
    PolicyEngine._env_enforcement_checked = False
    PolicyEngine._unconfigured_warning_emitted = False
    yield
    PolicyEngine._config = None
    PolicyEngine._initialized = False
    PolicyEngine._enforcement_mode = "fail_open"
    PolicyEngine._enforcement_mode_explicit = False
    PolicyEngine._env_enforcement_checked = False
    CostTracker.set_backend(None)


def noop(**kwargs):
    return "ok"


class TestEnforcementModeDefault:
    def test_default_is_fail_open(self):
        """Back-compat: existing installs must not start blocking."""
        assert PolicyEngine.get_enforcement_mode() == "fail_open"

    def test_unconfigured_permits_under_fail_open(self):
        PolicyEngine.check_action("anything", cost=99.0)  # must not raise

    def test_unconfigured_warns_once_not_per_call(self, caplog):
        """
        An agent making thousands of tool calls would drown the log, and a
        warning that appears 5000 times is a warning nobody reads.
        """
        caplog.set_level("WARNING")
        for _ in range(50):
            PolicyEngine.check_action("anything", cost=0.0)
        warnings = [r for r in caplog.records if "not configured" in r.message]
        assert len(warnings) == 1


class TestEnforcementModeFailClosed:
    def test_unconfigured_blocks_under_fail_closed(self):
        PolicyEngine.set_enforcement_mode("fail_closed")
        with pytest.raises(PolicyNotConfiguredError) as exc:
            PolicyEngine.check_action("transfer_funds", cost=0.0)
        assert exc.value.error_code == "POLICY_NOT_CONFIGURED"
        assert exc.value.details["action_name"] == "transfer_funds"
        assert exc.value.details["enforcement_mode"] == "fail_closed"

    def test_configured_engine_unaffected_by_fail_closed(self):
        """fail_closed governs the unconfigured case only, not normal policy."""
        PolicyEngine.set_enforcement_mode("fail_closed")
        PolicyEngine.configure(denied_actions=["drop_table"])
        PolicyEngine.check_action("safe_action", cost=0.0)  # permitted
        with pytest.raises(PolicyViolationError):
            PolicyEngine.check_action("drop_table", cost=0.0)

    def test_not_recoverable(self):
        """No prompt-level retry fixes a missing policy; it is an operator bug."""
        PolicyEngine.set_enforcement_mode("fail_closed")
        with pytest.raises(PolicyNotConfiguredError) as exc:
            PolicyEngine.check_action("x", cost=0.0)
        assert exc.value.recoverable is False


class TestEnforcementModeConfiguration:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown enforcement mode"):
            PolicyEngine.set_enforcement_mode("sort-of-closed")

    def test_mode_from_environment(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_closed")
        PolicyEngine._env_enforcement_checked = False
        PolicyEngine._enforcement_mode_explicit = False
        assert PolicyEngine.get_enforcement_mode() == "fail_closed"

    def test_explicit_call_beats_environment(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "fail_closed")
        PolicyEngine.set_enforcement_mode("fail_open")
        assert PolicyEngine.get_enforcement_mode() == "fail_open"

    def test_invalid_env_raises_rather_than_failing_open(self, monkeypatch):
        """
        An operator who typed this was turning enforcement ON. Falling back to
        fail_open would land permissive, which is the wrong direction to fail
        for a security control. Matches backend_from_env's contract.
        """
        monkeypatch.setenv("AGENT_SENTINEL_ENFORCEMENT", "closed-ish")
        PolicyEngine._env_enforcement_checked = False
        PolicyEngine._enforcement_mode_explicit = False
        with pytest.raises(ValueError, match="Invalid AGENT_SENTINEL_ENFORCEMENT"):
            PolicyEngine.get_enforcement_mode()


class TestRemediationOnPolicyBlocks:
    def test_denied_action_says_do_not_retry(self):
        PolicyEngine.configure(denied_actions=["drop_table"])
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("drop_table", cost=0.0)
        rem = exc.value.remediation
        assert rem.reason_code == "DENIED_ACTION"
        assert "not retry" in rem.retry_guidance.lower()
        assert exc.value.recoverable is False

    def test_allowlist_miss_names_the_alternatives(self):
        """The single most actionable thing a blocked agent can be told."""
        PolicyEngine.configure(allowed_actions=["read_file", "list_dir"])
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("send_email", cost=0.0)
        rem = exc.value.remediation
        assert rem.reason_code == "NOT_IN_ALLOWLIST"
        assert rem.safe_alternatives == ["read_file", "list_dir"]
        assert "read_file" in rem.retry_guidance

    def test_rate_limit_is_recoverable_and_carries_wait(self):
        """Unlike a deny list, this one clears on its own."""
        PolicyEngine.configure(rate_limits={"ping": {"max_count": 1, "window_seconds": 60}})
        PolicyEngine.check_action("ping", cost=0.0)
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("ping", cost=0.0)
        rem = exc.value.remediation
        assert rem.reason_code == "RATE_LIMITED"
        assert exc.value.recoverable is True
        assert rem.retry_after_seconds is not None
        assert 0 <= rem.retry_after_seconds <= 60

    def test_budget_block_carries_remediation(self):
        PolicyEngine.configure(run_budget=0.01)
        with pytest.raises(BudgetExceededError) as exc:
            PolicyEngine.check_action("spendy", cost=5.0)
        rem = exc.value.remediation
        assert rem.reason_code == "BUDGET_EXCEEDED"
        assert "cheaper" in rem.retry_guidance.lower()
        assert exc.value.recoverable is False

    def test_remediation_serialized_into_details(self):
        """Callers reading .details (ledger, sync) get the payload too."""
        PolicyEngine.configure(denied_actions=["x"])
        with pytest.raises(PolicyViolationError) as exc:
            PolicyEngine.check_action("x", cost=0.0)
        assert exc.value.details["remediation"]["reason_code"] == "DENIED_ACTION"


class TestBackwardCompatibleConstruction:
    def test_bare_message_still_works(self):
        err = PolicyViolationError("something went wrong")
        assert err.error_code == "POLICY_VIOLATION"
        assert err.recoverable is False
        assert err.remediation.reason_code == "POLICY_VIOLATION"

    def test_legacy_kwargs_still_work(self):
        err = PolicyViolationError("nope", policy_rule="rule_a", action_name="act")
        assert err.details["policy_rule"] == "rule_a"
        assert err.details["action_name"] == "act"

    def test_budget_error_legacy_fields_intact(self):
        err = BudgetExceededError("over", spent=2.0, limit=1.0, budget_type="run")
        assert err.spent == 2.0
        assert err.limit == 1.0
        assert err.budget_type == "run"
        assert err.details["spent"] == 2.0


class TestToolExecutorSurfacesRemediation:
    """
    The end-to-end path: a block becomes a ToolResult carrying guidance, which
    the OpenAI/Anthropic integrations hand back to the model as a tool result.
    Previously only evidence violations reached here with a payload.
    """

    @pytest.mark.parametrize(
        "configure_kwargs,tool,expected_reason",
        [
            ({"denied_actions": ["drop_table"]}, "drop_table", "DENIED_ACTION"),
            ({"allowed_actions": ["read_file"]}, "send_email", "NOT_IN_ALLOWLIST"),
            ({"run_budget": 0.001}, "spendy", "BUDGET_EXCEEDED"),
        ],
    )
    def test_block_types_reach_the_agent_with_guidance(
        self, configure_kwargs, tool, expected_reason
    ):
        PolicyEngine.configure(**configure_kwargs)
        cost = {"cost_usd": 5.0} if expected_reason == "BUDGET_EXCEEDED" else {}
        executor = SentinelToolExecutor({tool: (noop, cost)})

        result = executor.execute_tool(tool, {}, tool_call_id="tc_1")

        assert result.blocked is True
        assert result.remediation is not None
        assert result.remediation["reason_code"] == expected_reason
        assert result.remediation["retry_guidance"]

    def test_successful_call_is_not_blocked(self):
        PolicyEngine.configure(run_budget=10.0)
        executor = SentinelToolExecutor({"ok_tool": (noop, {})})
        result = executor.execute_tool("ok_tool", {}, tool_call_id="tc_2")
        assert result.blocked is False
        assert result.remediation is None
        assert result.output == "ok"

    def test_unknown_tool_blocked_without_remediation(self):
        executor = SentinelToolExecutor({})
        result = executor.execute_tool("ghost", {}, tool_call_id="tc_3")
        assert result.blocked is True
        assert "Unknown tool" in result.error


class TestReviewRegressions:
    """Regressions caught by the pre-landing review."""

    def test_reset_clears_enforcement_mode(self):
        """
        reset() promises a pristine engine. Leaving fail_closed set meant a
        test that opted in poisoned every later test in the process. An
        unpinned mode reverts to the SDK's own default ("fail_closed_writes"),
        not "fail_open" -- see PolicyEngine.reset()'s docstring.
        """
        from agent_sentinel.policy import PolicyEngine as PE
        from agent_sentinel.errors import NoPolicyForWriteError

        PE.set_enforcement_mode("fail_closed")
        PE.reset()
        assert PE.get_enforcement_mode() == "fail_closed_writes"
        with pytest.raises(NoPolicyForWriteError):
            PE.check_action("anything", cost=0.0)

    def test_reset_reenables_the_fail_open_warning(self):
        """The warning is documented as once-per-configuration, not per-process."""
        from agent_sentinel.policy import PolicyEngine as PE

        PE.check_action("x", cost=0.0)
        assert PE._unconfigured_warning_emitted is True
        PE.reset()
        assert PE._unconfigured_warning_emitted is False

    @pytest.mark.parametrize(
        "reason_code,expected",
        [
            ("RATE_LIMITED", "rate_limited"),
            ("BUDGET_EXCEEDED", "budget_exceeded"),
            ("DENIED_ACTION", "hard_block"),
            ("NOT_IN_ALLOWLIST", "hard_block"),
        ],
    )
    def test_intervention_type_comes_from_reason_code(self, reason_code, expected):
        """
        Classification used to sniff the error message for "rate limit".
        Rewording the message silently reclassified every rate-limit block.
        """
        from agent_sentinel.guard import _reason_code_to_intervention

        assert _reason_code_to_intervention()[reason_code].value == expected

    def test_reworded_message_still_classifies_as_rate_limited(self):
        from agent_sentinel.guard import _reason_code_to_intervention

        err = PolicyViolationError("wording changed entirely", reason_code="RATE_LIMITED")
        mapping = _reason_code_to_intervention()
        assert mapping[err.remediation.reason_code].value == "rate_limited"

    def test_evidence_payload_shape_unchanged_when_no_retry_after(self):
        """
        retry_after_seconds is only meaningful for time-bounded blocks. Emitting
        it unconditionally would change every persisted evidence payload.
        """
        from agent_sentinel.errors import EvidenceViolationError

        err = EvidenceViolationError("x", action_name="a", missing_requirements=["b"])
        assert "retry_after_seconds" not in err.remediation.to_dict()

    def test_rate_limit_payload_carries_retry_after(self):
        err = PolicyViolationError(
            "x", reason_code="RATE_LIMITED", retry_after_seconds=12.0
        )
        assert err.remediation.to_dict()["retry_after_seconds"] == 12.0

    def test_budget_error_tolerates_non_numeric_limit(self):
        """
        A None limit used to raise TypeError from inside the exception
        constructor, masking the original budget error.
        """
        err = BudgetExceededError("over", spent=None, limit=None, budget_type="run")
        assert err.remediation.reason_code == "BUDGET_EXCEEDED"
        assert err.remediation.retry_guidance


class TestInterventionRecordsCarryRemediation:
    """
    The new reason codes must reach the platform, not just the agent.
    Previously only evidence blocks forwarded a payload.
    """

    @pytest.mark.parametrize(
        "config,tool,cost,expected_type,expected_reason",
        [
            ({"denied_actions": ["drop_table"]}, "drop_table", 0.0, "hard_block", "DENIED_ACTION"),
            ({"allowed_actions": ["read_only"]}, "send_email", 0.0, "hard_block", "NOT_IN_ALLOWLIST"),
            ({"run_budget": 0.01}, "spendy", 5.0, "budget_exceeded", "BUDGET_EXCEEDED"),
        ],
    )
    def test_policy_block_records_payload(
        self, config, tool, cost, expected_type, expected_reason
    ):
        from agent_sentinel import guarded_action
        from agent_sentinel.intervention import InterventionTracker

        InterventionTracker.clear()
        PolicyEngine.configure(**config)

        @guarded_action(name=tool, cost_usd=cost)
        def blocked_tool():
            return "x"

        with pytest.raises((PolicyViolationError, BudgetExceededError)):
            blocked_tool()

        records = InterventionTracker.get_interventions()
        assert records, "no intervention recorded"
        record = records[-1]
        assert record.intervention_type.value == expected_type
        assert record.remediation_payload is not None
        assert record.remediation_payload["reason_code"] == expected_reason
        assert record.remediation_payload["retry_guidance"]


class TestAsyncExecutorSurfacesRemediation:
    """async_execute_tool got the same change and had no coverage at all."""

    async def test_async_block_surfaces_remediation(self):
        PolicyEngine.configure(denied_actions=["drop_table"])

        async def atool(**kwargs):
            return "ok"

        executor = SentinelToolExecutor({"drop_table": (atool, {})})
        result = await executor.async_execute_tool("drop_table", {}, tool_call_id="tc_a")
        assert result.blocked is True
        assert result.remediation["reason_code"] == "DENIED_ACTION"

    async def test_async_success_path(self):
        PolicyEngine.configure(run_budget=10.0)

        async def atool(**kwargs):
            return "ok"

        executor = SentinelToolExecutor({"ok_tool": (atool, {})})
        result = await executor.async_execute_tool("ok_tool", {}, tool_call_id="tc_b")
        assert result.blocked is False
        assert result.output == "ok"


class TestDecoratorEnforcesTheCap:
    """
    The property users actually get. check_and_reserve existing is not the
    same as @guarded_action using it — it shipped unwired once already.
    """

    def test_guarded_action_holds_the_run_budget_under_threads(self):
        import threading
        import time as _time

        from agent_sentinel import guarded_action

        PolicyEngine.configure(run_budget=1.00)

        @guarded_action(name="t", cost_usd=0.50)
        def tool():
            _time.sleep(0.05)  # work between check and completion
            return "x"

        permitted, blocked = [0], [0]

        def worker():
            try:
                tool()
                permitted[0] += 1
            except BudgetExceededError:
                blocked[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert CostTracker.get_run_total() <= 1.0 + 1e-9, (
            f"cap blown: ${CostTracker.get_run_total():.2f} against $1.00"
        )
        assert permitted[0] == 2 and blocked[0] == 4

    def test_no_double_counting(self):
        """The reservation is the record; a second add_cost would double it."""
        from agent_sentinel import guarded_action

        PolicyEngine.configure(run_budget=10.0)

        @guarded_action(name="t", cost_usd=0.50)
        def tool():
            return "x"

        tool()
        assert CostTracker.get_run_total() == pytest.approx(0.50)

    def test_cost_still_recorded_when_action_raises(self):
        """Parity with the old finally-block: a failed action still costs."""
        from agent_sentinel import guarded_action

        PolicyEngine.configure(run_budget=10.0)

        @guarded_action(name="boom", cost_usd=0.25)
        def boom():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            boom()
        assert CostTracker.get_run_total() == pytest.approx(0.25)

    def test_blocked_action_consumes_no_budget(self):
        """A rejection must not reserve. Deny list fires before the reserve."""
        from agent_sentinel import guarded_action

        PolicyEngine.configure(denied_actions=["nope"], run_budget=10.0)

        @guarded_action(name="nope", cost_usd=5.0)
        def nope():
            return "x"

        with pytest.raises(PolicyViolationError):
            nope()
        assert CostTracker.get_run_total() == 0.0
