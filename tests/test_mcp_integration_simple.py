"""
Simple MCP Integration Test

This test verifies that the MCP server and client work together correctly.
It's a lightweight test that can be run to validate the integration.

Usage:
    export AGENT_SENTINEL_PLATFORM_URL="http://localhost:8000"
    export AGENT_SENTINEL_API_TOKEN="your-jwt-token"
    python tests/test_mcp_integration_simple.py
"""

import asyncio
import os
import sys

# Add parent directory for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_sentinel.mcp import MCPClient


async def test_mcp_integration():
    """Test basic MCP functionality"""
    
    platform_url = os.getenv("AGENT_SENTINEL_PLATFORM_URL", "http://localhost:8000")
    api_token = os.getenv("AGENT_SENTINEL_API_TOKEN")
    
    if not api_token:
        print("❌ AGENT_SENTINEL_API_TOKEN not set")
        return False
    
    print("🧪 Testing MCP Integration")
    print(f"📡 Platform: {platform_url}")
    
    try:
        async with MCPClient(
            platform_url=platform_url,
            api_token=api_token,
            timeout=10.0
        ) as client:
            
            # Test 1: Tool Discovery
            print("\n1. Testing tool discovery...")
            tools = await client.list_tools()
            assert len(tools) > 0, "No tools found"
            print(f"   ✅ Found {len(tools)} tools")
            
            # Test 2: Resource Discovery
            print("\n2. Testing resource discovery...")
            resources = await client.list_resources()
            assert len(resources) > 0, "No resources found"
            print(f"   ✅ Found {len(resources)} resources")
            
            # Test 3: Prompt Discovery
            print("\n3. Testing prompt discovery...")
            prompts = await client.list_prompts()
            assert len(prompts) > 0, "No prompts found"
            print(f"   ✅ Found {len(prompts)} prompts")
            
            # Test 4: Tool Execution (list_policies)
            print("\n4. Testing tool execution (list_policies)...")
            result = await client.call_tool("list_policies", {"limit": 10})
            assert result.success, f"Tool call failed: {result.error}"
            print(f"   ✅ Successfully executed tool")
            
            # Test 5: Convenience Method
            print("\n5. Testing convenience method (list_runs)...")
            runs = await client.list_runs(limit=5)
            # Runs might be None if no runs exist, that's okay
            print(f"   ✅ Convenience method executed")
            
            # Test 6: Resource Access
            print("\n6. Testing resource access (stats/dashboard)...")
            data = await client.get_resource("agentsentinel://stats/dashboard")
            # Data might be minimal if no activity, that's okay
            print(f"   ✅ Resource accessed")
            
            # Test 7: Tool Execution (get_agent_stats)
            print("\n7. Testing statistics tool...")
            result = await client.call_tool("get_agent_stats", {"days": 7})
            assert result.success, f"Stats tool failed: {result.error}"
            print(f"   ✅ Statistics retrieved")
            
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - MCP Integration Working!")
            print("=" * 60)
            return True
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_mcp_integration())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


