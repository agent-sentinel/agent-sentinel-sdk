"""
AgentSentinel MCP Demo

This example demonstrates how to use AgentSentinel's Model Context Protocol (MCP)
support to enable LLMs and agents to interact with AgentSentinel programmatically.

Features demonstrated:
1. MCP client initialization
2. Tool discovery
3. Policy management via MCP
4. Run queries via MCP
5. Approval workflows via MCP
6. Statistics and analytics
7. Prompt execution
8. Resource access

Requirements:
    pip install agent-sentinel[remote]

Usage:
    export AGENT_SENTINEL_PLATFORM_URL="http://localhost:8000"
    export AGENT_SENTINEL_API_TOKEN="your-jwt-token"
    python examples/mcp_demo.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_sentinel.mcp import MCPClient, MCPTool


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


async def demo_tool_discovery(client: MCPClient):
    """Demonstrate MCP tool discovery"""
    print_section("1. Tool Discovery")
    
    # List all available tools
    tools = await client.list_tools()
    print(f"📋 Found {len(tools)} MCP tools\n")
    
    # Group tools by type
    tools_by_type = {}
    for tool in tools:
        if tool.type not in tools_by_type:
            tools_by_type[tool.type] = []
        tools_by_type[tool.type].append(tool)
    
    # Display tools by category
    for tool_type, type_tools in sorted(tools_by_type.items()):
        print(f"\n{tool_type.upper()} Tools ({len(type_tools)}):")
        for tool in sorted(type_tools, key=lambda t: t.name):
            print(f"  • {tool.name}: {tool.description}")


async def demo_resource_discovery(client: MCPClient):
    """Demonstrate MCP resource discovery"""
    print_section("2. Resource Discovery")
    
    # List all available resources
    resources = await client.list_resources()
    print(f"📦 Found {len(resources)} MCP resources\n")
    
    for resource in resources:
        print(f"  • {resource.uri}")
        print(f"    {resource.description}")


async def demo_prompt_discovery(client: MCPClient):
    """Demonstrate MCP prompt discovery"""
    print_section("3. Prompt Discovery")
    
    # List all available prompts
    prompts = await client.list_prompts()
    print(f"💡 Found {len(prompts)} MCP prompts\n")
    
    for prompt in prompts:
        print(f"  • {prompt.name}")
        print(f"    {prompt.description}")
        if prompt.arguments:
            args = ", ".join(arg["name"] for arg in prompt.arguments)
            print(f"    Arguments: {args}")


async def demo_policy_management(client: MCPClient):
    """Demonstrate policy management via MCP"""
    print_section("4. Policy Management")
    
    # Create a policy
    print("Creating a new policy...")
    result = await client.call_tool("create_policy", {
        "name": "MCP Demo Policy",
        "description": "Policy created via MCP for demonstration",
        "enabled": True,
        "run_budget": 10.0,
        "session_budget": 100.0,
        "denied_actions": ["dangerous_operation"]
    })
    
    if result.success:
        policy_id = result.data["id"]
        print(f"✅ Policy created: {policy_id}\n")
        
        # List policies
        print("Listing all policies...")
        result = await client.call_tool("list_policies", {"limit": 10})
        
        if result.success:
            policies = result.data
            print(f"Found {len(policies)} policies:\n")
            for policy in policies:
                enabled = "✓" if policy["enabled"] else "✗"
                print(f"  [{enabled}] {policy['name']}")
                if policy.get("run_budget"):
                    print(f"      Run budget: ${policy['run_budget']}")
        
        return policy_id
    else:
        print(f"❌ Error creating policy: {result.error}")
        return None


async def demo_run_queries(client: MCPClient):
    """Demonstrate run queries via MCP"""
    print_section("5. Run Queries")
    
    # Get latest runs
    print("Fetching latest runs...")
    result = await client.call_tool("get_latest_runs", {"count": 5})
    
    if result.success:
        runs = result.data
        print(f"Latest {len(runs)} runs:\n")
        for run in runs:
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "running": "🔄"
            }.get(run["status"], "❓")
            
            print(f"  {status_emoji} {run['id']}")
            print(f"     Agent: {run.get('agent_id', 'N/A')}")
            print(f"     Status: {run['status']}")
            print(f"     Cost: ${run.get('total_cost', 0.0):.4f}")
            print()
    else:
        print(f"❌ Error fetching runs: {result.error}")


async def demo_statistics(client: MCPClient):
    """Demonstrate statistics access via MCP"""
    print_section("6. Statistics & Analytics")
    
    # Get agent statistics
    print("Fetching agent statistics (last 7 days)...")
    result = await client.call_tool("get_agent_stats", {"days": 7})
    
    if result.success:
        stats = result.data
        print("\n📊 Agent Statistics:\n")
        print(f"  Period: Last {stats['period_days']} days")
        print(f"  Total Runs: {stats['total_runs']}")
        print(f"  Successful: {stats['successful_runs']}")
        print(f"  Failed: {stats['failed_runs']}")
        print(f"  Success Rate: {stats['successful_runs'] / stats['total_runs'] * 100:.1f}%" if stats['total_runs'] > 0 else "  Success Rate: N/A")
        print(f"  Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"  Avg Cost/Run: ${stats['avg_cost_per_run']:.4f}")
        print(f"  Total Actions: {stats['total_actions']}")
        print(f"  Avg Duration: {stats['avg_duration_ms']:.2f}ms")
    else:
        print(f"❌ Error fetching stats: {result.error}")


async def demo_resource_access(client: MCPClient):
    """Demonstrate resource access via MCP"""
    print_section("7. Resource Access")
    
    # Access dashboard stats resource
    print("Accessing dashboard stats resource...")
    data = await client.get_resource("agentsentinel://stats/dashboard")
    
    if data:
        print("\n📈 Dashboard Stats:\n")
        print(f"  Period: {data['period']}")
        print(f"  Total Runs: {data['total_runs']}")
        print(f"  Successful: {data['successful_runs']}")
        print(f"  Failed: {data['failed_runs']}")
        print(f"  Total Cost: ${data['total_cost_usd']:.4f}")
    else:
        print("❌ Error accessing resource")
    
    # Access active policies resource
    print("\nAccessing active policies resource...")
    data = await client.get_resource("agentsentinel://policies/active")
    
    if data:
        print(f"\n✅ Found {data['count']} active policies")
        for policy in data.get("active_policies", [])[:3]:
            print(f"  • {policy['name']}: ${policy.get('run_budget', 'N/A')} per run")
    else:
        print("❌ Error accessing resource")


async def demo_prompt_execution(client: MCPClient):
    """Demonstrate prompt execution via MCP"""
    print_section("8. Prompt Execution")
    
    # Execute budget policy creation prompt
    print("Executing 'create_budget_policy' prompt...")
    result = await client.execute_prompt("create_budget_policy", {
        "use_case": "customer support",
        "risk_level": "medium"
    })
    
    if result:
        print("\n💡 Budget Recommendations:\n")
        rec = result["recommendation"]
        print(f"  Use Case: {rec['use_case']}")
        print(f"  Risk Level: {rec['risk_level']}")
        print(f"  Recommended Run Budget: ${rec['recommended_run_budget']}")
        print(f"  Recommended Session Budget: ${rec['recommended_session_budget']}")
        print(f"  Rationale: {rec['rationale']}")
        
        print("\n  Next Steps:")
        for step in result["next_steps"]:
            print(f"    • {step}")
    else:
        print("❌ Error executing prompt")


async def demo_approval_workflow(client: MCPClient):
    """Demonstrate approval workflow via MCP"""
    print_section("9. Approval Workflow")
    
    # Get pending approvals
    print("Checking for pending approvals...")
    data = await client.get_pending_approvals()
    
    if data:
        count = data.get("count", 0)
        print(f"\n⏳ Found {count} pending approvals")
        
        if count > 0:
            print("\nPending actions:")
            for approval in data.get("pending_approvals", [])[:5]:
                print(f"  • {approval['action_name']}")
                print(f"    ID: {approval['action_id']}")
                print(f"    Cost: ${approval['cost_usd']:.4f}")
                print(f"    Run: {approval['run_id']}")
        else:
            print("  All caught up! No pending approvals.")
    else:
        print("❌ Error fetching approvals")


async def demo_convenience_methods(client: MCPClient):
    """Demonstrate convenience methods"""
    print_section("10. Convenience Methods")
    
    # Use convenience methods instead of call_tool
    print("Using convenience methods for common operations...\n")
    
    # Create policy (convenience method)
    print("1. Creating policy with convenience method...")
    policy = await client.create_policy(
        name="Convenience Method Policy",
        run_budget=5.0,
        session_budget=50.0
    )
    
    if policy:
        print(f"   ✅ Created: {policy['id']}")
    else:
        print("   ❌ Failed to create policy")
    
    # List runs (convenience method)
    print("\n2. Listing runs with convenience method...")
    runs = await client.list_runs(limit=5, status="completed")
    
    if runs:
        print(f"   ✅ Found {len(runs)} completed runs")
    else:
        print("   ⚠️  No completed runs found")
    
    # Get stats (convenience method)
    print("\n3. Getting stats with convenience method...")
    stats = await client.get_agent_stats(days=7)
    
    if stats:
        print(f"   ✅ Total cost (7 days): ${stats['total_cost_usd']:.4f}")
    else:
        print("   ❌ Failed to get stats")


async def main():
    """Main demo function"""
    # Get configuration from environment
    platform_url = os.getenv("AGENT_SENTINEL_PLATFORM_URL", "http://localhost:8000")
    api_token = os.getenv("AGENT_SENTINEL_API_TOKEN")
    
    if not api_token:
        print("❌ Error: AGENT_SENTINEL_API_TOKEN environment variable not set")
        print("\nPlease set your JWT token:")
        print("  export AGENT_SENTINEL_API_TOKEN='your-token-here'")
        print("\nGet a token by logging in:")
        print(f"  curl -X POST {platform_url}/api/v1/login/access-token \\")
        print("    -d 'username=your-email' \\")
        print("    -d 'password=your-password'")
        return
    
    print("🚀 AgentSentinel MCP Demo")
    print(f"📡 Platform: {platform_url}")
    print("🔑 Token: ***" + api_token[-8:])
    
    try:
        # Initialize MCP client
        async with MCPClient(
            platform_url=platform_url,
            api_token=api_token,
            timeout=30.0
        ) as client:
            
            # Run all demos
            await demo_tool_discovery(client)
            await demo_resource_discovery(client)
            await demo_prompt_discovery(client)
            
            policy_id = await demo_policy_management(client)
            
            await demo_run_queries(client)
            await demo_statistics(client)
            await demo_resource_access(client)
            await demo_prompt_execution(client)
            await demo_approval_workflow(client)
            await demo_convenience_methods(client)
            
            # Cleanup - delete demo policy
            if policy_id:
                print_section("Cleanup")
                print("Deleting demo policy...")
                result = await client.call_tool("delete_policy", {"policy_id": policy_id})
                if result.success:
                    print("✅ Demo policy deleted")
                else:
                    print(f"⚠️  Could not delete demo policy: {result.error}")
            
            print_section("Demo Complete!")
            print("✅ All MCP features demonstrated successfully!")
            print("\nNext steps:")
            print("  • Review the code in examples/mcp_demo.py")
            print("  • Read the MCP Guide: MCP_GUIDE.md")
            print("  • Integrate MCP with your LLM/agent")
            print()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure the platform is running (docker compose up)")
        print("  2. Verify your API token is valid")
        print("  3. Check the platform URL is correct")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


