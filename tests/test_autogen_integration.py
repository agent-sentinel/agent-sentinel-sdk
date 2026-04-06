"""
Tests for AutoGen integration.

Note: These tests require AutoGen to be installed.
Run: pip install pyautogen
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from agent_sentinel.integrations.autogen import (
    SentinelInspector,
    create_sentinel_agents,
    _AUTOGEN_AVAILABLE,
)
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.cost import CostTracker
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError


# Skip all tests if AutoGen not available
pytestmark = pytest.mark.skipif(
    not _AUTOGEN_AVAILABLE,
    reason="AutoGen not installed"
)


@pytest.fixture
def reset_sentinel():
    """Reset Sentinel state before each test."""
    CostTracker.reset_all()
    PolicyEngine.reset()
    yield
    CostTracker.reset_all()
    PolicyEngine.reset()


@pytest.fixture
def mock_autogen_agent():
    """Create a mock AutoGen ConversableAgent."""
    if not _AUTOGEN_AVAILABLE:
        return None
    
    from autogen import ConversableAgent
    
    # Create a real agent but with mocked LLM
    agent = ConversableAgent(
        name="test_agent",
        llm_config={"model": "gpt-4", "api_key": "test-key"},
        human_input_mode="NEVER",
    )
    
    return agent


class TestSentinelInspector:
    """Test the SentinelInspector class."""
    
    def test_initialization(self, reset_sentinel):
        """Test inspector initialization."""
        inspector = SentinelInspector(
            run_name="test_run",
            enforce_policies=True,
            track_costs=True,
            track_messages=True,
        )
        
        assert inspector.run_name == "test_run"
        assert inspector.enforce_policies is True
        assert inspector.track_costs is True
        assert inspector.track_messages is True
        assert inspector.tags == ["autogen"]
        assert inspector._message_count == 0
        assert inspector._llm_call_count == 0
    
    def test_register_agent(self, reset_sentinel, mock_autogen_agent):
        """Test registering an agent."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        inspector = SentinelInspector(run_name="test_run")
        agent = mock_autogen_agent
        
        # Register the agent
        inspector.register(agent)
        
        # Check that agent was registered
        assert "test_agent" in inspector._registered_agents
        
        # Check that reply handler was added
        # AutoGen stores reply functions in _reply_func_list
        assert hasattr(agent, "_reply_func_list")
    
    def test_authorization_hook_allows_normal_flow(self, reset_sentinel, mock_autogen_agent):
        """Test that authorization hook allows normal flow when no policies violated."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        inspector = SentinelInspector(
            run_name="test_run",
            enforce_policies=False,  # Disable enforcement for this test
        )
        
        agent = mock_autogen_agent
        inspector.register(agent)
        
        # Create authorization hook
        hook = inspector._create_authorization_hook("test_agent")
        
        # Call with test message
        messages = [{"content": "Hello, world!", "role": "user"}]
        result = hook(
            recipient=agent,
            messages=messages,
            sender=None,
            config=None,
        )
        
        # Should return None to allow normal processing
        assert result is None
        
        # Message count should be incremented
        assert inspector._message_count == 1
        assert inspector._agent_message_counts["test_agent"] == 1
    
    def test_authorization_hook_blocks_on_budget_exceeded(self, reset_sentinel, mock_autogen_agent):
        """Test that authorization hook blocks when budget is exceeded."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        # Set very low budget
        PolicyEngine.configure(run_budget=0.001)
        
        # Add some cost to exceed budget
        CostTracker.add_cost("test_action", 0.002)
        
        inspector = SentinelInspector(
            run_name="test_run",
            enforce_policies=True,  # Enable enforcement
        )
        
        agent = mock_autogen_agent
        inspector.register(agent)
        
        # Create authorization hook
        hook = inspector._create_authorization_hook("test_agent")
        
        # Call with test message
        messages = [{"content": "Hello, world!", "role": "user"}]
        result = hook(
            recipient=agent,
            messages=messages,
            sender=None,
            config=None,
        )
        
        # Should return blocking message (not None)
        assert result is not None
        assert isinstance(result, dict)
        assert "BLOCKED" in result["content"]
        assert "AgentSentinel" in result["content"]
    
    def test_run_lifecycle(self, reset_sentinel):
        """Test run start and end tracking."""
        inspector = SentinelInspector(run_name="test_run")
        
        # Start run
        inspector.start_run()
        
        assert inspector._run_start_time is not None
        assert inspector._run_start_cost == 0.0
        
        # Add some cost
        CostTracker.add_cost("test_action", 0.05)
        
        # Simulate some work
        time.sleep(0.1)
        
        # End run
        inspector.end_run()
        
        # Verify run tracking
        summary = inspector.get_run_summary()
        assert summary["run_name"] == "test_run"
        assert summary["duration_seconds"] >= 0.1
        assert summary["run_cost_usd"] >= 0.05
    
    def test_get_run_summary(self, reset_sentinel):
        """Test getting run summary."""
        inspector = SentinelInspector(run_name="test_run")
        
        inspector.start_run()
        
        # Simulate some activity
        inspector._message_count = 5
        inspector._llm_call_count = 3
        inspector._agent_message_counts["agent1"] = 3
        inspector._agent_message_counts["agent2"] = 2
        
        CostTracker.add_cost("llm_call:gpt-4", 0.08)
        
        time.sleep(0.1)
        inspector.end_run()
        
        # Get summary
        summary = inspector.get_run_summary()
        
        assert summary["run_name"] == "test_run"
        assert summary["message_count"] == 5
        assert summary["llm_call_count"] == 3
        assert summary["duration_seconds"] >= 0.1
        assert summary["run_cost_usd"] >= 0.08
        assert summary["agent_message_counts"]["agent1"] == 3
        assert summary["agent_message_counts"]["agent2"] == 2
        assert "llm_call:gpt-4" in summary["action_costs"]
    
    def test_multiple_agents(self, reset_sentinel, mock_autogen_agent):
        """Test registering multiple agents."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        inspector = SentinelInspector(run_name="test_run")
        
        # Create multiple mock agents
        from autogen import ConversableAgent
        
        agent1 = ConversableAgent(
            name="agent1",
            llm_config=None,
            human_input_mode="NEVER",
        )
        
        agent2 = ConversableAgent(
            name="agent2",
            llm_config=None,
            human_input_mode="NEVER",
        )
        
        # Register both
        inspector.register(agent1)
        inspector.register(agent2)
        
        # Check both registered
        assert "agent1" in inspector._registered_agents
        assert "agent2" in inspector._registered_agents
        assert len(inspector._registered_agents) == 2


class TestCreateSentinelAgents:
    """Test the convenience function for creating secured agents."""
    
    def test_create_sentinel_agents(self, reset_sentinel):
        """Test creating agents with sentinel in one step."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        from autogen import AssistantAgent, UserProxyAgent
        
        agents, sentinel = create_sentinel_agents(
            agent_configs=[
                {
                    "agent_class": AssistantAgent,
                    "name": "assistant",
                    "llm_config": {"model": "gpt-4", "api_key": "test-key"},
                },
                {
                    "agent_class": UserProxyAgent,
                    "name": "user_proxy",
                    "human_input_mode": "NEVER",
                }
            ],
            run_name="test_run",
            enforce_policies=True,
        )
        
        # Check agents created
        assert len(agents) == 2
        assert agents[0].name == "assistant"
        assert agents[1].name == "user_proxy"
        
        # Check sentinel created
        assert sentinel.run_name == "test_run"
        assert sentinel.enforce_policies is True
        
        # Check agents registered
        assert "assistant" in sentinel._registered_agents
        assert "user_proxy" in sentinel._registered_agents


class TestPolicyEnforcement:
    """Test policy enforcement with AutoGen."""
    
    def test_budget_enforcement(self, reset_sentinel, mock_autogen_agent):
        """Test that budget limits are enforced."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        # Set low budget
        PolicyEngine.configure(run_budget=0.01)
        
        inspector = SentinelInspector(
            run_name="test_run",
            enforce_policies=True,
        )
        
        agent = mock_autogen_agent
        inspector.register(agent)
        
        # Exceed budget
        CostTracker.add_cost("test_action", 0.02)
        
        # Try to authorize an action - should fail
        hook = inspector._create_authorization_hook("test_agent")
        messages = [{"content": "Test", "role": "user"}]
        
        result = hook(
            recipient=agent,
            messages=messages,
            sender=None,
            config=None,
        )
        
        # Should be blocked
        assert result is not None
        assert isinstance(result, dict)
        assert "BLOCKED" in result["content"]
    
    def test_denied_actions(self, reset_sentinel, mock_autogen_agent):
        """Test that denied actions are blocked."""
        if not _AUTOGEN_AVAILABLE:
            pytest.skip("AutoGen not available")
        
        # Deny specific action
        PolicyEngine.configure(denied_actions=["agent_reply:test_agent"])
        
        inspector = SentinelInspector(
            run_name="test_run",
            enforce_policies=True,
        )
        
        agent = mock_autogen_agent
        inspector.register(agent)
        
        # Try to perform denied action
        hook = inspector._create_authorization_hook("test_agent")
        messages = [{"content": "Test", "role": "user"}]
        
        result = hook(
            recipient=agent,
            messages=messages,
            sender=None,
            config=None,
        )
        
        # Should be blocked
        assert result is not None
        assert isinstance(result, dict)
        assert "BLOCKED" in result["content"]


class TestIntegration:
    """Integration tests with real AutoGen agents (mocked LLM)."""
    
    @pytest.mark.skipif(not _AUTOGEN_AVAILABLE, reason="AutoGen not available")
    def test_full_conversation_flow(self, reset_sentinel):
        """Test a full conversation with mocked LLM responses."""
        from autogen import ConversableAgent
        
        # Create inspector
        inspector = SentinelInspector(
            run_name="integration_test",
            enforce_policies=False,  # Disable for integration test
            track_costs=True,
            track_messages=True,
        )
        
        # Create agents with no LLM (for testing)
        agent1 = ConversableAgent(
            name="agent1",
            llm_config=False,  # No LLM
            human_input_mode="NEVER",
        )
        
        agent2 = ConversableAgent(
            name="agent2",
            llm_config=False,  # No LLM
            human_input_mode="NEVER",
        )
        
        # Register agents
        inspector.register(agent1)
        inspector.register(agent2)
        
        # Start run
        inspector.start_run()
        
        # Simulate message exchange by calling hooks directly
        hook1 = inspector._create_authorization_hook("agent1")
        hook2 = inspector._create_authorization_hook("agent2")
        
        # Agent2 sends to Agent1
        messages = [{"content": "Hello Agent1!", "role": "user"}]
        result = hook1(recipient=agent1, messages=messages, sender=agent2, config=None)
        assert result is None  # Should allow
        
        # Agent1 replies to Agent2
        messages = [{"content": "Hello Agent2!", "role": "assistant"}]
        result = hook2(recipient=agent2, messages=messages, sender=agent1, config=None)
        assert result is None  # Should allow
        
        # End run
        inspector.end_run()
        
        # Check summary
        summary = inspector.get_run_summary()
        assert summary["message_count"] == 2
        assert summary["agent_message_counts"]["agent1"] == 1
        assert summary["agent_message_counts"]["agent2"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
