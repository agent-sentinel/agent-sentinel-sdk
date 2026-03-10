"""
Example: LangChain Integration with Active Policy Enforcement

This example demonstrates the robust LangChain integration that includes:
1. Pre-authorization checks (like a credit card terminal checking funds)
2. Intervention recording (for dashboard visibility)
3. Async support (for production FastAPI/Streamlit agents)
4. Context propagation (agent identity persists through chain)

Requirements:
    pip install agent-sentinel[remote] langchain langchain-openai openai pyyaml
"""
import os
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agent_sentinel import PolicyEngine, CostTracker
from agent_sentinel.integrations.langchain import SentinelCallbackHandler
from agent_sentinel.intervention import InterventionTracker
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError


def example_basic_enforcement():
    """Example 1: Basic budget enforcement - blocks before LLM call."""
    print("\n=== Example 1: Budget Enforcement ===\n")
    
    # Configure a strict budget
    PolicyEngine.configure(
        run_budget=0.01,  # Only $0.01 allowed
        strict_mode=True
    )
    
    # Create handler with enforcement enabled
    sentinel = SentinelCallbackHandler(
        run_name="budget_enforcement_demo",
        track_costs=True,
        enforce_policies=True  # This enables the "Visa check"
    )
    
    # Create LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    try:
        # First call should work (under budget)
        print("Making first LLM call (should succeed)...")
        response1 = llm.predict("Say 'Hello' in one word.")
        print(f"✓ Response: {response1}\n")
        
        # Second call should be BLOCKED (over budget)
        print("Making second LLM call (should be blocked)...")
        response2 = llm.predict("Say 'World' in one word.")
        print(f"✓ Response: {response2}\n")
        
    except BudgetExceededError as e:
        print(f"✗ BLOCKED: {e}\n")
        print("This is working correctly! The second call was blocked.")
        
        # Check interventions
        interventions = InterventionTracker.get_interventions(limit=5)
        print(f"\nRecorded {len(interventions)} intervention(s):")
        for intervention in interventions:
            print(f"  - Type: {intervention.intervention_type.value}")
            print(f"  - Outcome: {intervention.outcome.value}")
            print(f"  - Action: {intervention.action_name}")
            print(f"  - Reason: {intervention.reason}\n")
    
    finally:
        # Cleanup
        PolicyEngine.reset()
        CostTracker.reset_run()


def example_denied_tools():
    """Example 2: Block specific tools (like deny lists)."""
    print("\n=== Example 2: Denied Tools ===\n")
    
    # Configure policy to block dangerous tools
    PolicyEngine.configure(
        denied_actions=["delete_database", "send_email"],
        strict_mode=True
    )
    
    # Define some tools
    def safe_search(query: str) -> str:
        """Safe search tool."""
        return f"Search results for: {query}"
    
    def delete_database(confirmation: str) -> str:
        """Dangerous tool that deletes database."""
        return "Database deleted!"
    
    tools = [
        Tool(
            name="search",
            func=safe_search,
            description="Search for information"
        ),
        Tool(
            name="delete_database",
            func=delete_database,
            description="Delete the entire database"
        ),
    ]
    
    # Create handler with enforcement
    sentinel = SentinelCallbackHandler(
        run_name="denied_tools_demo",
        track_costs=True,
        track_tools=True,
        enforce_policies=True
    )
    
    # Create agent
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[sentinel],
        verbose=True,
    )
    
    try:
        # This should work (safe tool)
        print("Asking agent to search (should work)...")
        result = agent.run("Search for Python tutorials")
        print(f"✓ Result: {result}\n")
        
    except Exception as e:
        print(f"Result: {e}\n")
    
    try:
        # This should be BLOCKED (denied tool)
        print("Asking agent to delete database (should be blocked)...")
        result = agent.run("Delete the database")
        print(f"✓ Result: {result}\n")
        
    except PolicyViolationError as e:
        print(f"✗ BLOCKED: {e}\n")
        print("This is working correctly! The dangerous tool was blocked.")
    
    finally:
        # Cleanup
        PolicyEngine.reset()
        CostTracker.reset_run()


def example_rate_limiting():
    """Example 3: Rate limiting - max 2 calls per minute."""
    print("\n=== Example 3: Rate Limiting ===\n")
    
    # Configure rate limit
    PolicyEngine.configure(
        rate_limits={
            "expensive_search": {
                "max_count": 2,
                "window_seconds": 60
            }
        },
        strict_mode=True
    )
    
    # Define tool
    def expensive_search(query: str) -> str:
        """Expensive search that costs money."""
        return f"Expensive results for: {query}"
    
    tools = [
        Tool(
            name="expensive_search",
            func=expensive_search,
            description="Run expensive search"
        ),
    ]
    
    sentinel = SentinelCallbackHandler(
        run_name="rate_limit_demo",
        track_tools=True,
        enforce_policies=True
    )
    
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[sentinel],
        verbose=False,
    )
    
    try:
        # First two calls should work
        print("Making call 1/3...")
        agent.run("Use expensive_search for 'query1'")
        print("✓ Call 1 succeeded\n")
        
        print("Making call 2/3...")
        agent.run("Use expensive_search for 'query2'")
        print("✓ Call 2 succeeded\n")
        
        # Third call should be BLOCKED (rate limit)
        print("Making call 3/3 (should be blocked)...")
        agent.run("Use expensive_search for 'query3'")
        print("✓ Call 3 succeeded\n")
        
    except PolicyViolationError as e:
        print(f"✗ BLOCKED: {e}\n")
        print("This is working correctly! Rate limit enforced.")
    
    finally:
        # Cleanup
        PolicyEngine.reset()
        CostTracker.reset_run()


def example_allowlist_mode():
    """Example 4: Allowlist mode - only specific tools permitted."""
    print("\n=== Example 4: Allowlist Mode ===\n")
    
    # Configure allowlist (only these tools are permitted)
    PolicyEngine.configure(
        allowed_actions=["read_file", "list_directory"],
        strict_mode=True
    )
    
    def read_file(path: str) -> str:
        return f"Contents of {path}"
    
    def write_file(path: str, content: str) -> str:
        return f"Wrote to {path}"
    
    def list_directory(path: str) -> str:
        return f"Listing {path}"
    
    tools = [
        Tool(name="read_file", func=read_file, description="Read a file"),
        Tool(name="write_file", func=write_file, description="Write a file"),
        Tool(name="list_directory", func=list_directory, description="List directory"),
    ]
    
    sentinel = SentinelCallbackHandler(
        run_name="allowlist_demo",
        track_tools=True,
        enforce_policies=True
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", callbacks=[sentinel])
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[sentinel], verbose=False
    )
    
    try:
        # Allowed tool should work
        print("Using allowed tool 'read_file'...")
        agent.run("Use read_file on 'test.txt'")
        print("✓ Allowed tool succeeded\n")
        
    except Exception as e:
        print(f"Result: {e}\n")
    
    try:
        # Non-allowed tool should be blocked
        print("Using non-allowed tool 'write_file'...")
        agent.run("Use write_file to save 'test.txt'")
        print("✓ Completed\n")
        
    except PolicyViolationError as e:
        print(f"✗ BLOCKED: {e}\n")
        print("This is working correctly! Only allowlist tools permitted.")
    
    finally:
        PolicyEngine.reset()
        CostTracker.reset_run()


async def example_async_agent():
    """Example 5: Async agent with policy enforcement."""
    print("\n=== Example 5: Async Agent ===\n")
    
    # Configure budget
    PolicyEngine.configure(run_budget=0.05, strict_mode=True)
    
    sentinel = SentinelCallbackHandler(
        run_name="async_demo",
        track_costs=True,
        enforce_policies=True
    )
    
    # Create async-compatible LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        callbacks=[sentinel]
    )
    
    # In a real async scenario, you'd use async langchain components
    # For now, we demonstrate that the callbacks support async
    try:
        print("Running async LLM call...")
        # Simulate async call (in real code, use apredict or async chains)
        response = llm.predict("What is 2+2?")
        print(f"✓ Response: {response}\n")
        
    except BudgetExceededError as e:
        print(f"✗ BLOCKED: {e}\n")
    
    finally:
        PolicyEngine.reset()
        CostTracker.reset_run()


def example_yaml_policy():
    """Example 6: Load policies from YAML file."""
    print("\n=== Example 6: YAML Policy Configuration ===\n")
    
    # Create a temporary policy file
    import tempfile
    import yaml
    
    policy = {
        "budgets": {
            "run": 0.02,
            "actions": {
                "expensive_model": 0.01
            }
        },
        "denied_actions": ["drop_table", "rm_rf"],
        "rate_limits": {
            "api_call": {
                "max_count": 5,
                "window_seconds": 60
            }
        },
        "strict_mode": True
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(policy, f)
        policy_path = f.name
    
    try:
        # Load policy from YAML
        PolicyEngine.load_from_yaml(policy_path)
        print(f"✓ Loaded policy from {policy_path}")
        
        config = PolicyEngine.get_config()
        print(f"  - Run budget: ${config.run_budget}")
        print(f"  - Denied actions: {config.denied_actions}")
        print(f"  - Rate limits: {config.rate_limits}\n")
        
        # Now use with LangChain
        sentinel = SentinelCallbackHandler(
            run_name="yaml_policy_demo",
            enforce_policies=True
        )
        
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", callbacks=[sentinel])
        
        print("Running LLM with YAML policy...")
        response = llm.predict("Hello!")
        print(f"✓ Response: {response}\n")
        
    finally:
        import os as os_module
        os_module.unlink(policy_path)
        PolicyEngine.reset()
        CostTracker.reset_run()


def example_intervention_visibility():
    """Example 7: View all interventions for dashboard."""
    print("\n=== Example 7: Intervention Visibility ===\n")
    
    # Clear previous interventions
    InterventionTracker.clear()
    
    # Configure restrictive policy
    PolicyEngine.configure(
        run_budget=0.005,  # Very small budget
        denied_actions=["dangerous_tool"],
        strict_mode=True
    )
    
    sentinel = SentinelCallbackHandler(
        run_name="intervention_demo",
        enforce_policies=True
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", callbacks=[sentinel])
    
    # Try multiple calls to trigger interventions
    for i in range(3):
        try:
            print(f"Attempt {i+1}...")
            llm.predict(f"Say the number {i}")
            print(f"  ✓ Succeeded\n")
        except BudgetExceededError as e:
            print(f"  ✗ Blocked: Budget exceeded\n")
    
    # View all interventions
    interventions = InterventionTracker.get_interventions()
    print(f"\n=== Dashboard View: {len(interventions)} Intervention(s) ===")
    
    for idx, intervention in enumerate(interventions, 1):
        print(f"\nIntervention #{idx}:")
        print(f"  Type: {intervention.intervention_type.value}")
        print(f"  Outcome: {intervention.outcome.value}")
        print(f"  Action: {intervention.action_name}")
        print(f"  Estimated Cost: ${intervention.estimated_cost:.6f}")
        print(f"  Risk Level: {intervention.risk_level}")
        print(f"  Reason: {intervention.reason}")
        print(f"  Timestamp: {intervention.timestamp}")
    
    print("\n✓ These interventions would appear in your AgentSentinel dashboard!")
    
    PolicyEngine.reset()
    CostTracker.reset_run()


if __name__ == "__main__":
    # Make sure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        exit(1)
    
    print("=" * 70)
    print("LangChain + AgentSentinel: Active Policy Enforcement Demo")
    print("=" * 70)
    
    # Run examples (comment out ones you don't want to run)
    try:
        example_basic_enforcement()
        # example_denied_tools()
        # example_rate_limiting()
        # example_allowlist_mode()
        # asyncio.run(example_async_agent())
        # example_yaml_policy()
        # example_intervention_visibility()
        
        print("\n" + "=" * 70)
        print("✓ Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running example: {e}")
        import traceback
        traceback.print_exc()
