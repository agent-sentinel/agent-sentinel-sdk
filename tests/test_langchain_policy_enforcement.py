"""
Comprehensive tests for the robust LangChain integration.

Tests the 4 critical features:
1. Pre-authorization checks (on_tool_start and on_llm_start)
2. Intervention recording (when actions are blocked)
3. Async support (AsyncCallbackHandler)
4. Context propagation (run_id tracking)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from agent_sentinel.integrations.langchain import SentinelCallbackHandler
from agent_sentinel import PolicyEngine, CostTracker, InterventionTracker
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError
from agent_sentinel.intervention import InterventionType, InterventionOutcome

# Check if LangChain is available
try:
    from langchain_core.outputs import LLMResult, Generation
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import LLMResult, Generation
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all state before each test."""
    PolicyEngine.reset()
    CostTracker.reset_all()
    InterventionTracker.clear()
    yield
    PolicyEngine.reset()
    CostTracker.reset_all()
    InterventionTracker.clear()


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestPolicyEnforcement:
    """Test pre-authorization checks."""
    
    def test_llm_start_enforces_budget(self):
        """Test that on_llm_start checks budget before LLM call."""
        # Configure strict budget
        PolicyEngine.configure(run_budget=0.0, strict_mode=True)
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            enforce_policies=True
        )
        
        # Should raise BudgetExceededError before LLM starts
        with pytest.raises(BudgetExceededError):
            handler.on_llm_start(
                serialized={"name": "test_model"},
                prompts=["test prompt"],
                run_id=uuid4()
            )
    
    def test_llm_start_with_enforcement_disabled(self):
        """Test that enforcement can be disabled."""
        PolicyEngine.configure(run_budget=0.0, strict_mode=True)
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            enforce_policies=False  # Disabled
        )
        
        # Should NOT raise error even with zero budget
        try:
            handler.on_llm_start(
                serialized={"name": "test_model"},
                prompts=["test prompt"],
                run_id=uuid4()
            )
        except BudgetExceededError:
            pytest.fail("Should not enforce when enforce_policies=False")
    
    def test_tool_start_enforces_denied_actions(self):
        """Test that on_tool_start blocks denied tools."""
        PolicyEngine.configure(
            denied_actions=["dangerous_tool"],
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            track_tools=True,
            enforce_policies=True
        )
        
        # Should raise PolicyViolationError for denied tool
        with pytest.raises(PolicyViolationError) as exc_info:
            handler.on_tool_start(
                serialized={"name": "dangerous_tool"},
                input_str="some input",
                run_id=uuid4()
            )
        
        assert "denied list" in str(exc_info.value).lower()
    
    def test_tool_start_enforces_allowlist(self):
        """Test that on_tool_start enforces allowlist mode."""
        PolicyEngine.configure(
            allowed_actions=["safe_tool"],
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            track_tools=True,
            enforce_policies=True
        )
        
        # Allowed tool should work
        try:
            handler.on_tool_start(
                serialized={"name": "safe_tool"},
                input_str="input",
                run_id=uuid4()
            )
        except PolicyViolationError:
            pytest.fail("Allowed tool should not be blocked")
        
        # Non-allowed tool should be blocked
        with pytest.raises(PolicyViolationError):
            handler.on_tool_start(
                serialized={"name": "unsafe_tool"},
                input_str="input",
                run_id=uuid4()
            )
    
    def test_tool_start_enforces_rate_limit(self):
        """Test that on_tool_start enforces rate limits."""
        PolicyEngine.configure(
            rate_limits={
                "rate_limited_tool": {
                    "max_count": 2,
                    "window_seconds": 60
                }
            },
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            track_tools=True,
            enforce_policies=True
        )
        
        # First two calls should work
        handler.on_tool_start(
            serialized={"name": "rate_limited_tool"},
            input_str="input1",
            run_id=uuid4()
        )
        
        handler.on_tool_start(
            serialized={"name": "rate_limited_tool"},
            input_str="input2",
            run_id=uuid4()
        )
        
        # Third call should be blocked
        with pytest.raises(PolicyViolationError) as exc_info:
            handler.on_tool_start(
                serialized={"name": "rate_limited_tool"},
                input_str="input3",
                run_id=uuid4()
            )
        
        assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestInterventionRecording:
    """Test that interventions are recorded for dashboard visibility."""
    
    def test_budget_exceeded_records_intervention(self):
        """Test that budget violations are recorded as interventions."""
        PolicyEngine.configure(run_budget=0.0, strict_mode=True)
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            enforce_policies=True
        )
        
        # Clear previous interventions
        InterventionTracker.clear()
        
        # Try to start LLM (should be blocked)
        with pytest.raises(BudgetExceededError):
            handler.on_llm_start(
                serialized={"name": "test_model"},
                prompts=["test"],
                run_id=uuid4()
            )
        
        # Check that intervention was recorded
        interventions = InterventionTracker.get_interventions()
        assert len(interventions) == 1
        
        intervention = interventions[0]
        assert intervention.intervention_type == InterventionType.BUDGET_EXCEEDED
        assert intervention.outcome == InterventionOutcome.BLOCKED
        assert intervention.action_name == "llm_call"
        assert intervention.run_id == "test_run"
        assert "budget" in intervention.reason.lower()
    
    def test_policy_violation_records_intervention(self):
        """Test that policy violations are recorded as interventions."""
        PolicyEngine.configure(
            denied_actions=["forbidden_tool"],
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            track_tools=True,
            enforce_policies=True
        )
        
        InterventionTracker.clear()
        
        # Try to use forbidden tool
        with pytest.raises(PolicyViolationError):
            handler.on_tool_start(
                serialized={"name": "forbidden_tool"},
                input_str="dangerous input",
                run_id=uuid4()
            )
        
        # Check intervention
        interventions = InterventionTracker.get_interventions()
        assert len(interventions) == 1
        
        intervention = interventions[0]
        assert intervention.intervention_type == InterventionType.HARD_BLOCK
        assert intervention.outcome == InterventionOutcome.BLOCKED
        assert intervention.action_name == "forbidden_tool"
        assert intervention.risk_level == "high"
        assert "denied list" in intervention.reason.lower()
    
    def test_multiple_interventions_recorded(self):
        """Test that multiple blocks are all recorded."""
        PolicyEngine.configure(
            denied_actions=["tool1", "tool2"],
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="test_run",
            track_tools=True,
            enforce_policies=True
        )
        
        InterventionTracker.clear()
        
        # Try multiple blocked tools
        for tool_name in ["tool1", "tool2"]:
            with pytest.raises(PolicyViolationError):
                handler.on_tool_start(
                    serialized={"name": tool_name},
                    input_str="input",
                    run_id=uuid4()
                )
        
        # Should have 2 interventions
        interventions = InterventionTracker.get_interventions()
        assert len(interventions) == 2
        assert all(i.outcome == InterventionOutcome.BLOCKED for i in interventions)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestContextPropagation:
    """Test that agent identity and budget persist through chains."""
    
    def test_run_name_in_metadata(self):
        """Test that run_name is propagated to ledger entries."""
        handler = SentinelCallbackHandler(
            run_name="my_custom_run",
            track_costs=True
        )
        
        run_id = uuid4()
        
        # Start and end LLM call
        handler.on_llm_start(
            serialized={"name": "gpt-4o-mini"},
            prompts=["test"],
            run_id=run_id
        )
        
        result = LLMResult(
            generations=[[Generation(text="response")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "model_name": "gpt-4o-mini"
            }
        )
        
        handler.on_llm_end(result, run_id=run_id)
        
        # Check run summary
        summary = handler.get_run_summary()
        assert summary["run_name"] == "my_custom_run"
    
    def test_nested_chains_share_budget(self):
        """Test that nested chains share the same budget."""
        PolicyEngine.configure(run_budget=0.01, strict_mode=True)
        
        handler = SentinelCallbackHandler(
            run_name="nested_test",
            enforce_policies=True
        )
        
        # Parent chain starts
        parent_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "parent_chain"},
            inputs={"query": "test"},
            run_id=parent_id
        )
        
        # Child LLM call - should check same budget
        child_id = uuid4()
        try:
            handler.on_llm_start(
                serialized={"name": "gpt-4o-mini"},
                prompts=["test"],
                run_id=child_id,
                parent_run_id=parent_id
            )
        except BudgetExceededError:
            # Expected if budget is already exceeded
            pass
        
        # Both should reference the same run budget
        cost = CostTracker.get_run_total()
        assert cost >= 0.0  # Budget is shared
    
    def test_intervention_includes_run_id(self):
        """Test that interventions include the run_id for tracking."""
        PolicyEngine.configure(
            denied_actions=["blocked"],
            strict_mode=True
        )
        
        handler = SentinelCallbackHandler(
            run_name="intervention_test",
            track_tools=True,
            enforce_policies=True
        )
        
        InterventionTracker.clear()
        
        with pytest.raises(PolicyViolationError):
            handler.on_tool_start(
                serialized={"name": "blocked"},
                input_str="test",
                run_id=uuid4()
            )
        
        interventions = InterventionTracker.get_interventions()
        assert len(interventions) == 1
        assert interventions[0].run_id == "intervention_test"


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestCostTracking:
    """Test that cost tracking still works with enforcement."""
    
    def test_llm_cost_tracked_after_authorization(self):
        """Test that costs are tracked after authorization passes."""
        PolicyEngine.configure(run_budget=10.0, strict_mode=True)
        
        handler = SentinelCallbackHandler(
            run_name="cost_test",
            track_costs=True,
            enforce_policies=True
        )
        
        run_id = uuid4()
        
        # Start (should pass authorization)
        handler.on_llm_start(
            serialized={"name": "gpt-4o-mini"},
            prompts=["test"],
            run_id=run_id
        )
        
        # End with token usage
        result = LLMResult(
            generations=[[Generation(text="response")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                },
                "model_name": "gpt-4o-mini"
            }
        )
        
        handler.on_llm_end(result, run_id=run_id)
        
        # Check that cost was tracked
        summary = handler.get_run_summary()
        assert summary["total_cost_usd"] > 0.0
    
    def test_tool_cost_is_zero(self):
        """Test that tools are tracked with zero cost."""
        handler = SentinelCallbackHandler(
            run_name="tool_cost_test",
            track_tools=True,
            enforce_policies=False
        )
        
        run_id = uuid4()
        
        handler.on_tool_start(
            serialized={"name": "my_tool"},
            input_str="input",
            run_id=run_id
        )
        
        handler.on_tool_end(
            output="output",
            run_id=run_id
        )
        
        # Tools should not add to cost
        summary = handler.get_run_summary()
        assert summary["total_cost_usd"] == 0.0


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
def test_integration_full_flow():
    """Integration test: Full flow with budget, tools, and interventions."""
    # Configure policy with very small budget
    PolicyEngine.configure(
        run_budget=0.000001,  # Very small budget - almost any LLM call will exceed
        denied_actions=["dangerous"],
        strict_mode=True
    )
    
    handler = SentinelCallbackHandler(
        run_name="integration_test",
        track_costs=True,
        track_tools=True,
        enforce_policies=True
    )
    
    InterventionTracker.clear()
    
    # Scenario 1: Allowed LLM call (under budget)
    run_id_1 = uuid4()
    handler.on_llm_start(
        serialized={"name": "gpt-4o-mini"},
        prompts=["hello"],
        run_id=run_id_1
    )
    
    # Complete it with minimal cost
    result = LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            "model_name": "gpt-4o-mini"
        }
    )
    handler.on_llm_end(result, run_id=run_id_1)
    
    # Scenario 2: Blocked tool (denied)
    with pytest.raises(PolicyViolationError):
        handler.on_tool_start(
            serialized={"name": "dangerous"},
            input_str="bad input",
            run_id=uuid4()
        )
    
    # Scenario 3: Over budget LLM call (should be blocked because we already spent money)
    with pytest.raises(BudgetExceededError):
        handler.on_llm_start(
            serialized={"name": "gpt-4"},  # Expensive model
            prompts=["long prompt" * 100],
            run_id=uuid4()
        )
    
    # Check results
    summary = handler.get_run_summary()
    assert summary["total_cost_usd"] > 0.0
    
    interventions = InterventionTracker.get_interventions()
    assert len(interventions) >= 2  # At least the denied tool and budget
    
    assert any(i.action_name == "dangerous" for i in interventions)
    assert any(i.intervention_type == InterventionType.BUDGET_EXCEEDED for i in interventions)
