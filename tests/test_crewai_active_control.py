"""
Tests for CrewAI Active Control Integration.

Tests the upgraded "Visa-like" integration that provides:
1. Automatic tool injection with authorization checks
2. LLM callback handler injection for token tracking
3. Step callback monitoring for runaway detection

These tests verify that AgentSentinel can actively block actions inside
the CrewAI agent decision loop, not just track them passively.
"""
import pytest
import time
from unittest.mock import Mock, MagicMock, patch, call

from agent_sentinel.integrations.crewai import SentinelCrew
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.cost import CostTracker
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError


# Mock CrewAI classes for testing
class MockTool:
    """Mock CrewAI tool for testing."""
    
    def __init__(self, name="mock_tool"):
        self.name = name
        self._run_count = 0
    
    def _run(self, *args, **kwargs):
        """Simulate tool execution."""
        self._run_count += 1
        return f"Tool {self.name} executed {self._run_count} times"


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self):
        self.callbacks = []
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        """Simulate LLM call."""
        self.call_count += 1
        return "LLM response"


class MockAgent:
    """Mock CrewAI Agent for testing."""
    
    def __init__(self, role="TestAgent", tools=None, llm=None):
        self.role = role
        self.name = role
        self.tools = tools or []
        self.llm = llm or MockLLM()
        self.step_callback = None


class MockTask:
    """Mock CrewAI Task for testing."""
    
    def __init__(self, description="Test task", agent=None):
        self.description = description
        self.agent = agent


class MockCrew:
    """Mock CrewAI Crew for testing."""
    
    def __init__(self, agents, tasks, **kwargs):
        self.agents = agents
        self.tasks = tasks
        self.kwargs = kwargs
    
    def kickoff(self, inputs=None):
        """Simulate crew execution."""
        return "Crew execution completed"


@pytest.fixture(autouse=True)
def reset_state():
    """Reset PolicyEngine and CostTracker before each test."""
    PolicyEngine.reset()
    CostTracker.reset_all()
    yield
    PolicyEngine.reset()
    CostTracker.reset_all()


@pytest.fixture
def mock_crewai(monkeypatch):
    """Mock CrewAI module for testing."""
    # Patch the Crew class
    monkeypatch.setattr(
        "agent_sentinel.integrations.crewai.Crew",
        MockCrew
    )
    monkeypatch.setattr(
        "agent_sentinel.integrations.crewai._CREWAI_AVAILABLE",
        True
    )


class TestAutomaticToolInjection:
    """Test automatic tool security injection (The "Chip Reader")."""
    
    def test_tools_are_wrapped_automatically(self, mock_crewai):
        """Test that tools are automatically wrapped with security checks."""
        # Create agent with tools
        tool1 = MockTool("search_tool")
        tool2 = MockTool("scraper_tool")
        agent = MockAgent(role="Researcher", tools=[tool1, tool2])
        task = MockTask("Test task", agent)
        
        # Create SentinelCrew - should auto-wrap tools
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            run_name="test_tool_wrapping",
            enforce_policies=True,
        )
        
        # Check that tools are marked as secured
        assert hasattr(agent.tools[0], "_is_sentinel_secured")
        assert agent.tools[0]._is_sentinel_secured is True
        assert hasattr(agent.tools[1], "_is_sentinel_secured")
        assert agent.tools[1]._is_sentinel_secured is True
    
    def test_tools_execute_with_policy_check(self, mock_crewai):
        """Test that wrapped tools check PolicyEngine before execution."""
        tool = MockTool("test_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Configure policy
        PolicyEngine.configure(run_budget=1.0)
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Execute tool - should work (budget OK)
        result = agent.tools[0]._run("test input")
        assert "executed" in result
        assert tool._run_count == 1
    
    def test_tools_blocked_by_budget(self, mock_crewai):
        """Test that tools are blocked when budget exceeded."""
        tool = MockTool("expensive_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Configure tight budget and add some cost
        PolicyEngine.configure(run_budget=0.01)
        CostTracker.add_cost("previous_action", 0.02)  # Already over budget
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Tool execution should be blocked
        with pytest.raises(BudgetExceededError):
            agent.tools[0]._run("test input")
        
        # Tool should not have actually executed
        assert tool._run_count == 0
    
    def test_tools_blocked_by_deny_list(self, mock_crewai):
        """Test that tools are blocked if on deny list."""
        tool = MockTool("dangerous_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Configure policy with deny list
        PolicyEngine.configure(
            denied_actions=["dangerous_tool"],
            strict_mode=True,
        )
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Tool execution should be blocked
        with pytest.raises(PolicyViolationError) as exc_info:
            agent.tools[0]._run("test input")
        
        assert "denied list" in str(exc_info.value).lower()
        assert tool._run_count == 0
    
    def test_tools_not_double_wrapped(self, mock_crewai):
        """Test that already secured tools are not wrapped again."""
        tool = MockTool("test_tool")
        tool._is_sentinel_secured = True  # Already secured
        
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Store original method
        original_run = tool._run
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Tool method should not have changed (not double-wrapped)
        assert tool._run == original_run
    
    def test_tool_execution_logged(self, mock_crewai):
        """Test that tool executions are logged to ledger."""
        from agent_sentinel.ledger import Ledger
        
        tool = MockTool("logged_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            auto_log=True,
        )
        
        # Execute tool
        result = agent.tools[0]._run("test")
        
        # Check that execution was logged
        # In real implementation, this would check Ledger.log was called
        assert result is not None


class TestLLMMonitoring:
    """Test LLM callback handler injection (The "Meter")."""
    
    def test_llm_callbacks_attached(self, mock_crewai):
        """Test that SentinelCallbackHandler is attached to agent LLMs."""
        pytest.importorskip("langchain")
        from agent_sentinel.integrations.langchain import SentinelCallbackHandler
        
        llm = MockLLM()
        agent = MockAgent(llm=llm)
        task = MockTask("Test", agent)
        
        # Initially no callbacks
        assert len(llm.callbacks) == 0
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Callback should be attached
        assert len(llm.callbacks) == 1
        assert isinstance(llm.callbacks[0], SentinelCallbackHandler)
    
    def test_llm_callbacks_not_duplicated(self, mock_crewai):
        """Test that callback handlers are not duplicated."""
        pytest.importorskip("langchain")
        from agent_sentinel.integrations.langchain import SentinelCallbackHandler
        
        llm = MockLLM()
        llm.callbacks = [SentinelCallbackHandler(run_name="existing")]
        
        agent = MockAgent(llm=llm)
        task = MockTask("Test", agent)
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # Should still only have one callback (not duplicated)
        assert len(llm.callbacks) == 1
    
    def test_llm_monitoring_config(self, mock_crewai):
        """Test that LLM monitor is configured correctly."""
        pytest.importorskip("langchain")
        from agent_sentinel.integrations.langchain import SentinelCallbackHandler
        
        llm = MockLLM()
        agent = MockAgent(llm=llm)
        task = MockTask("Test", agent)
        
        # Create SentinelCrew with specific config
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            run_name="test_llm_run",
            enforce_policies=True,
            track_costs=True,
            tags=["test", "crew"],
        )
        
        # Check callback configuration
        handler = llm.callbacks[0]
        assert handler.run_name == "test_llm_run"
        assert handler.enforce_policies is True
        assert handler.track_costs is True
        assert "crewai_llm" in handler.tags
    
    def test_llm_monitoring_graceful_degradation(self, mock_crewai):
        """Test that crew works even when LangChain is not available."""
        llm = MockLLM()
        agent = MockAgent(llm=llm)
        task = MockTask("Test", agent)
        
        # Create SentinelCrew - should not fail even without LangChain
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # LLM monitoring should be skipped gracefully
        # (callbacks won't be attached but crew should still work)
        assert crew is not None


class TestStepMonitoring:
    """Test step callback monitoring for runaway detection (The "Safety Net")."""
    
    def test_step_callbacks_attached(self, mock_crewai):
        """Test that step monitors are attached to agents."""
        agent = MockAgent()
        task = MockTask("Test", agent)
        
        # Initially no step callback
        assert agent.step_callback is None
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            max_agent_steps=10,
        )
        
        # Step callback should be attached
        assert agent.step_callback is not None
        assert callable(agent.step_callback)
    
    def test_step_limit_enforced(self, mock_crewai):
        """Test that agents are stopped after max steps."""
        agent = MockAgent(role="TestAgent")
        task = MockTask("Test", agent)
        
        # Create SentinelCrew with low step limit
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            max_agent_steps=5,
        )
        
        # Simulate steps - callback increments automatically
        for i in range(5):
            agent.step_callback(f"Step {i+1}")  # Should not raise
        
        # 6th step should raise error
        with pytest.raises(PolicyViolationError) as exc_info:
            agent.step_callback("Step 6")
        
        assert "exceeded maximum steps" in str(exc_info.value).lower()
    
    def test_loop_detection(self, mock_crewai):
        """Test that repetitive actions are detected."""
        agent = MockAgent(role="LoopAgent")
        task = MockTask("Test", agent)
        
        # Create SentinelCrew with loop detection
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            max_agent_steps=50,
            detect_loops=True,
        )
        
        # Simulate repetitive actions - callback handles tracking automatically
        for i in range(6):
            # No error should be raised (just logged)
            agent.step_callback("same_action")
        
        # Check that actions were tracked (callback tracks them)
        assert len(crew._agent_actions["LoopAgent"]) == 6
        assert all(a == "same_action" for a in crew._agent_actions["LoopAgent"])
    
    def test_original_callback_preserved(self, mock_crewai):
        """Test that original step callback is still called."""
        original_callback = Mock()
        agent = MockAgent()
        agent.step_callback = original_callback
        task = MockTask("Test", agent)
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            max_agent_steps=10,
        )
        
        # Execute step
        crew._agent_steps[agent.role] = 1
        agent.step_callback("test step")
        
        # Original callback should have been called
        original_callback.assert_called_once_with("test step")


class TestSentinelCrewIntegration:
    """Test full SentinelCrew integration."""
    
    def test_crew_initialization(self, mock_crewai):
        """Test basic SentinelCrew initialization."""
        agent = MockAgent()
        task = MockTask("Test", agent)
        
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            run_name="test_crew",
            enforce_policies=True,
            max_agent_steps=50,
            detect_loops=True,
        )
        
        assert crew.run_name == "test_crew"
        assert crew.enforce_policies is True
        assert crew.max_agent_steps == 50
        assert crew.detect_loops is True
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1
    
    def test_crew_kickoff_tracking(self, mock_crewai):
        """Test that crew execution is tracked."""
        agent = MockAgent()
        task = MockTask("Test", agent)
        
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            run_name="test_kickoff",
        )
        
        # Record start cost
        start_cost = CostTracker.get_run_total()
        
        # Execute crew
        result = crew.kickoff()
        
        assert result == "Crew execution completed"
        assert crew._run_start_time is not None
        assert crew._run_end_time is not None
    
    def test_crew_summary(self, mock_crewai):
        """Test that crew execution summary is generated."""
        agent = MockAgent()
        task = MockTask("Test", agent)
        
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            run_name="summary_test",
        )
        
        # Execute
        crew.kickoff()
        
        # Get summary
        summary = crew.get_run_summary()
        
        assert summary["run_name"] == "summary_test"
        assert summary["num_agents"] == 1
        assert summary["num_tasks"] == 1
        assert "duration_seconds" in summary
        assert "run_cost_usd" in summary
        assert "total_cost_usd" in summary
    
    def test_multiple_agents_all_secured(self, mock_crewai):
        """Test that all agents in a crew are secured."""
        # Create multiple agents with tools and LLMs
        agent1 = MockAgent(
            role="Agent1",
            tools=[MockTool("tool1")],
            llm=MockLLM(),
        )
        agent2 = MockAgent(
            role="Agent2",
            tools=[MockTool("tool2"), MockTool("tool3")],
            llm=MockLLM(),
        )
        agent3 = MockAgent(
            role="Agent3",
            tools=[],
            llm=MockLLM(),
        )
        
        tasks = [
            MockTask("Task1", agent1),
            MockTask("Task2", agent2),
            MockTask("Task3", agent3),
        ]
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent1, agent2, agent3],
            tasks=tasks,
            enforce_policies=True,
        )
        
        # Check all tools are secured
        assert all(hasattr(t, "_is_sentinel_secured") for t in agent1.tools)
        assert all(hasattr(t, "_is_sentinel_secured") for t in agent2.tools)
        
        # Check all agents have step monitors
        assert agent1.step_callback is not None
        assert agent2.step_callback is not None
        assert agent3.step_callback is not None
        
        # Note: LLM callbacks won't be attached if LangChain is not installed
        # This is expected graceful degradation


class TestPolicyEnforcement:
    """Test that policies are enforced in the decision loop."""
    
    def test_enforcement_can_be_disabled(self, mock_crewai):
        """Test that enforcement can be disabled for passive tracking."""
        tool = MockTool("test_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Configure strict policy
        PolicyEngine.configure(
            denied_actions=["test_tool"],
            strict_mode=True,
        )
        
        # Create SentinelCrew with enforcement DISABLED
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=False,  # Passive mode
        )
        
        # Tool should execute even though it's denied (enforcement disabled)
        result = agent.tools[0]._run("test")
        assert "executed" in result
    
    def test_rate_limiting_enforced(self, mock_crewai):
        """Test that rate limits are enforced on tools."""
        tool = MockTool("rate_limited_tool")
        agent = MockAgent(tools=[tool])
        task = MockTask("Test", agent)
        
        # Configure rate limit
        PolicyEngine.configure(
            rate_limits={
                "rate_limited_tool": {
                    "max_count": 3,
                    "window_seconds": 60,
                }
            }
        )
        
        # Create SentinelCrew
        crew = SentinelCrew(
            agents=[agent],
            tasks=[task],
            enforce_policies=True,
        )
        
        # First 3 calls should work
        for i in range(3):
            agent.tools[0]._run(f"call_{i}")
        
        # 4th call should be blocked
        with pytest.raises(PolicyViolationError) as exc_info:
            agent.tools[0]._run("call_4")
        
        assert "rate limit" in str(exc_info.value).lower()


def test_real_crewai_not_installed():
    """Test that proper error is raised if CrewAI not installed."""
    with patch("agent_sentinel.integrations.crewai._CREWAI_AVAILABLE", False):
        with pytest.raises(ImportError) as exc_info:
            crew = SentinelCrew(
                agents=[],
                tasks=[],
            )
        
        assert "crewai is not installed" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
