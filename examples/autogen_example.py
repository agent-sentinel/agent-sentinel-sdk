"""
Example: AutoGen Integration with Agent Sentinel

This example demonstrates how to use Agent Sentinel with Microsoft's AutoGen
framework to monitor and control multi-agent conversations.

AutoGen is particularly popular in enterprise environments (Microsoft shops)
and provides powerful multi-agent orchestration capabilities.

Key Features Demonstrated:
1. Policy enforcement (budget limits)
2. Cost tracking across multiple agents
3. Message flow monitoring
4. Active blocking of policy violations
"""

# Import AutoGen
try:
    from autogen import AssistantAgent, UserProxyAgent
except ImportError:
    print("Please install AutoGen: pip install pyautogen")
    exit(1)

# Import Agent Sentinel
from agent_sentinel.integrations.autogen import SentinelInspector
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.cost import CostTracker

# ============================================================================
# Step 1: Configure Policies
# ============================================================================

print("=" * 80)
print("AutoGen + Agent Sentinel Example")
print("=" * 80)

# Set a budget limit
PolicyEngine.configure(
    run_budget=0.50,  # Max $0.50 for this run
    session_budget=1.0,  # Max $1.00 for entire session
    denied_actions=[],  # No specific actions denied
    strict_mode=True,
)

print("\n✅ Policy configured: Run budget = $0.50, Session budget = $1.00")

# ============================================================================
# Step 2: Create AutoGen Agents
# ============================================================================

# Standard AutoGen configuration
llm_config = {
    "model": "gpt-4",
    "api_key": "your-api-key-here",  # Replace with actual key
    "temperature": 0.7,
}

# Create an assistant agent (uses LLM)
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config=llm_config,
)

# Create a user proxy agent (represents human, but automated here)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Fully automated
    max_consecutive_auto_reply=10,
    code_execution_config=False,
)

print("\n✅ Created AutoGen agents: assistant and user_proxy")

# ============================================================================
# Step 3: Secure Agents with Sentinel
# ============================================================================

# Create the Sentinel inspector
sentinel = SentinelInspector(
    run_name="autogen_demo",
    enforce_policies=True,  # Enable active blocking
    track_costs=True,       # Track LLM costs
    track_messages=True,    # Track message flow
)

# Register each agent with Sentinel (one line per agent!)
sentinel.register(assistant)
sentinel.register(user_proxy)

print("\n✅ Sentinel registered with both agents")
print("   - Policy enforcement: ENABLED")
print("   - Cost tracking: ENABLED")
print("   - Message tracking: ENABLED")

# ============================================================================
# Step 4: Run the Multi-Agent Conversation
# ============================================================================

print("\n" + "=" * 80)
print("Starting Agent Conversation")
print("=" * 80)

# Mark the start of the run
sentinel.start_run()

try:
    # Initiate the chat
    # The user_proxy will send a message to the assistant
    user_proxy.initiate_chat(
        assistant,
        message="Please explain what AutoGen is and how it works. Keep it brief.",
    )
    
    # Mark successful completion
    sentinel.end_run(outcome="completed")
    print("\n✅ Conversation completed successfully")

except Exception as e:
    # Mark failed completion
    sentinel.end_run(outcome="failed")
    print(f"\n❌ Conversation failed: {e}")

# ============================================================================
# Step 5: Review Results
# ============================================================================

print("\n" + "=" * 80)
print("Run Summary")
print("=" * 80)

# Get the run summary from Sentinel
summary = sentinel.get_run_summary()

print(f"\nRun Name: {summary['run_name']}")
print(f"Duration: {summary['duration_seconds']:.2f} seconds")
print(f"Cost: ${summary['run_cost_usd']:.6f}")
print(f"Messages Exchanged: {summary['message_count']}")
print(f"LLM Calls: {summary['llm_call_count']}")

print("\nAgent Message Counts:")
for agent_name, count in summary['agent_message_counts'].items():
    print(f"  - {agent_name}: {count} messages")

print("\nAction Costs:")
for action, cost in summary['action_costs'].items():
    count = summary['action_counts'].get(action, 0)
    print(f"  - {action}: ${cost:.6f} ({count} calls)")

# Show remaining budget
session_total = CostTracker.get_session_total()
run_total = CostTracker.get_run_total()
policy = PolicyEngine.get_config()

if policy:
    run_remaining = policy.run_budget - run_total if policy.run_budget else float('inf')
    session_remaining = policy.session_budget - session_total if policy.session_budget else float('inf')
    
    print(f"\nBudget Status:")
    print(f"  - Run: ${run_total:.6f} / ${policy.run_budget or 'unlimited':.2f} "
          f"(${run_remaining:.6f} remaining)")
    print(f"  - Session: ${session_total:.6f} / ${policy.session_budget or 'unlimited':.2f} "
          f"(${session_remaining:.6f} remaining)")

# ============================================================================
# Step 6: Demonstrate Policy Violation
# ============================================================================

print("\n" + "=" * 80)
print("Demonstrating Policy Enforcement")
print("=" * 80)

# Set a very low budget to trigger a block
PolicyEngine.configure(run_budget=0.001)  # $0.001 - will definitely exceed
CostTracker.reset_run()  # Reset for new run

print("\n⚠️  Set run budget to $0.001 to demonstrate blocking")

# Create new inspector for the second run
sentinel2 = SentinelInspector(
    run_name="autogen_demo_blocked",
    enforce_policies=True,
)

# Register agents again
sentinel2.register(assistant)
sentinel2.register(user_proxy)

print("✅ Starting conversation that will exceed budget...")

sentinel2.start_run()

try:
    user_proxy.initiate_chat(
        assistant,
        message="Please write a very long essay about AI.",  # This will cost more than $0.001
    )
    sentinel2.end_run(outcome="completed")

except Exception as e:
    sentinel2.end_run(outcome="failed")
    print(f"\n🛑 BLOCKED! Sentinel prevented action: {e}")
    print("   This demonstrates active policy enforcement!")

print("\n" + "=" * 80)
print("Example Complete")
print("=" * 80)
