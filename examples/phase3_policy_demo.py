"""
Phase 3 Demo: Budget & Policy Engine with Remote Sync

This example demonstrates:
1. Remote policy synchronization from platform
2. Local policy caching
3. Pre-execution budget and policy checks
4. Rate limiting
5. Automatic policy refresh
6. Fail-open behavior on network issues

Setup:
1. Set environment variables:
   export AGENT_SENTINEL_PLATFORM_URL="http://localhost:8000"
   export AGENT_SENTINEL_API_TOKEN="your-jwt-token"

2. Create policies on the platform:
   - Budget limits per action/run/session
   - Action deny/allow lists
   - Rate limits

3. Run this demo:
   python examples/phase3_policy_demo.py
"""
from __future__ import annotations

import os
import time
import asyncio
from agent_sentinel import (
    guarded_action,
    PolicyEngine,
    enable_remote_sync,
    flush_and_stop,
    BudgetExceededError,
    PolicyViolationError,
)


# Configuration from environment
PLATFORM_URL = os.getenv("AGENT_SENTINEL_PLATFORM_URL", "http://localhost:8000")
API_TOKEN = os.getenv("AGENT_SENTINEL_API_TOKEN")


def setup_remote_policies():
    """
    Setup remote policy sync from platform.
    
    This will:
    1. Download policies from platform
    2. Cache them locally
    3. Start background refresh every 5 minutes
    """
    if not API_TOKEN:
        print("⚠️  No API token found. Skipping remote policy sync.")
        print("   Set AGENT_SENTINEL_API_TOKEN to enable remote policies.")
        return False
    
    print("📋 Enabling remote policy sync...")
    
    try:
        PolicyEngine.enable_remote_sync(
            platform_url=PLATFORM_URL,
            api_token=API_TOKEN,
            agent_id="demo-agent",  # Optional: get agent-specific policies
            refresh_interval=300.0,  # Refresh every 5 minutes
            cache_ttl=600.0,  # Cache valid for 10 minutes
        )
        print("✅ Remote policy sync enabled")
        return True
    except Exception as e:
        print(f"⚠️  Failed to enable remote sync: {e}")
        return False


def setup_local_policies():
    """
    Setup local policies as fallback or for testing.
    
    This demonstrates manual policy configuration.
    """
    print("📋 Configuring local policies...")
    
    PolicyEngine.configure(
        # Budget limits
        session_budget=10.0,  # Max $10 per session
        run_budget=2.0,       # Max $2 per run
        action_budgets={
            "expensive_api": 0.50,  # Max $0.50 total for this action
            "moderate_api": 0.20,
        },
        
        # Action control
        denied_actions=[
            "dangerous_operation",
            "delete_database",
        ],
        
        # Rate limits
        rate_limits={
            "api_call": {
                "max_count": 10,      # Max 10 calls
                "window_seconds": 60, # Per 60 seconds
            },
            "expensive_api": {
                "max_count": 3,
                "window_seconds": 300,  # Max 3 calls per 5 minutes
            },
        },
    )
    
    print("✅ Local policies configured")


# Define some example actions

@guarded_action(name="cheap_api", cost_usd=0.001, tags=["api", "cheap"])
def cheap_api_call(query: str):
    """A cheap API call."""
    print(f"  → Calling cheap API with: {query}")
    time.sleep(0.1)
    return {"result": f"Processed: {query}"}


@guarded_action(name="moderate_api", cost_usd=0.05, tags=["api", "moderate"])
def moderate_api_call(data: dict):
    """A moderately priced API call."""
    print(f"  → Calling moderate API with: {data}")
    time.sleep(0.2)
    return {"status": "success"}


@guarded_action(name="expensive_api", cost_usd=0.25, tags=["api", "expensive"])
def expensive_api_call(params: dict):
    """An expensive API call."""
    print(f"  → Calling expensive API with: {params}")
    time.sleep(0.3)
    return {"data": "expensive_result"}


@guarded_action(name="dangerous_operation", cost_usd=0.0, tags=["dangerous"])
def dangerous_operation():
    """This action should be blocked by policy."""
    print("  → This should never execute!")
    return {"error": "This should not happen"}


@guarded_action(name="api_call", cost_usd=0.01, tags=["api"])
def rate_limited_call(endpoint: str):
    """A rate-limited API call."""
    print(f"  → Calling rate-limited API: {endpoint}")
    time.sleep(0.05)
    return {"endpoint": endpoint, "status": "ok"}


def demo_basic_actions():
    """Demo basic actions that should succeed."""
    print("\n" + "="*60)
    print("Demo 1: Basic Actions (Should Succeed)")
    print("="*60)
    
    try:
        result = cheap_api_call("test query")
        print(f"✅ Cheap API succeeded: {result}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def demo_budget_limits():
    """Demo budget limit enforcement."""
    print("\n" + "="*60)
    print("Demo 2: Budget Limit Enforcement")
    print("="*60)
    
    # Call expensive API multiple times
    print("\n📊 Testing action-specific budget limits...")
    
    try:
        # First call (0.25 USD) - should succeed
        print("\n1st expensive call ($0.25):")
        expensive_api_call({"test": 1})
        print("✅ Within budget")
        
        # Second call (0.50 USD total) - should succeed if limit is >= 0.50
        print("\n2nd expensive call ($0.50 total):")
        expensive_api_call({"test": 2})
        print("✅ Still within budget")
        
        # Third call (0.75 USD total) - likely to exceed budget
        print("\n3rd expensive call ($0.75 total):")
        expensive_api_call({"test": 3})
        print("✅ Still within budget (high limit)")
        
    except BudgetExceededError as e:
        print(f"🚫 Budget exceeded (as expected): {e}")


def demo_denied_actions():
    """Demo denied action enforcement."""
    print("\n" + "="*60)
    print("Demo 3: Denied Action Enforcement")
    print("="*60)
    
    print("\n🚫 Attempting to call denied action...")
    
    try:
        dangerous_operation()
        print("❌ ERROR: Dangerous operation was not blocked!")
    except PolicyViolationError as e:
        print(f"✅ Blocked (as expected): {e}")


def demo_rate_limiting():
    """Demo rate limiting enforcement."""
    print("\n" + "="*60)
    print("Demo 4: Rate Limiting")
    print("="*60)
    
    print("\n⏱️  Testing rate limits (max 10 calls per 60s)...")
    
    for i in range(12):
        try:
            rate_limited_call(f"/endpoint/{i+1}")
            print(f"  ✅ Call {i+1}/12 succeeded")
        except PolicyViolationError as e:
            print(f"  🚫 Call {i+1}/12 blocked: {e}")
            break
        
        # Small delay between calls
        time.sleep(0.1)


def demo_run_budget():
    """Demo run budget enforcement."""
    print("\n" + "="*60)
    print("Demo 5: Run Budget Enforcement")
    print("="*60)
    
    print("\n💰 Making multiple moderate API calls to approach run budget...")
    
    # Run budget is $2.00, moderate calls are $0.05 each
    # So we can make up to 40 calls before hitting the limit
    
    call_count = 0
    try:
        for i in range(50):  # Try more than the limit
            moderate_api_call({"call": i+1})
            call_count += 1
            
            if (i+1) % 10 == 0:
                print(f"  Made {i+1} calls so far (${(i+1)*0.05:.2f})...")
    
    except BudgetExceededError as e:
        print(f"\n🚫 Run budget exceeded after {call_count} calls: {e}")


def demo_policy_status():
    """Display current policy configuration."""
    print("\n" + "="*60)
    print("Current Policy Configuration")
    print("="*60)
    
    if not PolicyEngine.is_configured():
        print("❌ No policies configured")
        return
    
    config = PolicyEngine.get_config()
    
    print(f"\n📊 Budgets:")
    print(f"  Session Budget: ${config.session_budget}" if config.session_budget else "  Session Budget: Unlimited")
    print(f"  Run Budget: ${config.run_budget}" if config.run_budget else "  Run Budget: Unlimited")
    
    if config.action_budgets:
        print(f"\n💵 Action-Specific Budgets:")
        for action, budget in config.action_budgets.items():
            print(f"  {action}: ${budget}")
    
    if config.denied_actions:
        print(f"\n🚫 Denied Actions:")
        for action in config.denied_actions:
            print(f"  - {action}")
    
    if config.allowed_actions:
        print(f"\n✅ Allowed Actions (allowlist mode):")
        for action in config.allowed_actions:
            print(f"  - {action}")
    
    if config.rate_limits:
        print(f"\n⏱️  Rate Limits:")
        for action, limits in config.rate_limits.items():
            max_count = limits.get("max_count", "N/A")
            window = limits.get("window_seconds", "N/A")
            print(f"  {action}: {max_count} calls per {window}s")


def main():
    """Run the Phase 3 demo."""
    print("🚀 Agent Sentinel - Phase 3 Demo")
    print("=" * 60)
    print("\nThis demo shows budget enforcement and policy management.")
    print("Policies can be configured locally or synced from the platform.")
    
    # Setup policies (remote if available, local as fallback)
    remote_enabled = setup_remote_policies()
    
    if not remote_enabled:
        print("\n💡 Using local policies for demo...")
        setup_local_policies()
    
    # Setup remote sync for telemetry
    if API_TOKEN:
        print("\n📡 Enabling telemetry sync to platform...")
        try:
            enable_remote_sync(
                platform_url=PLATFORM_URL,
                api_token=API_TOKEN,
                flush_interval=10.0,
            )
            print("✅ Telemetry sync enabled")
        except Exception as e:
            print(f"⚠️  Telemetry sync failed: {e}")
    
    # Display current policy configuration
    demo_policy_status()
    
    # Run demos
    demo_basic_actions()
    demo_denied_actions()
    demo_rate_limiting()
    demo_budget_limits()
    demo_run_budget()
    
    # Summary
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\n📊 Summary:")
    print("  ✅ Pre-execution policy checks prevent violations")
    print("  ✅ Budget limits enforced at action/run/session levels")
    print("  ✅ Rate limiting prevents abuse")
    print("  ✅ Remote policies synced and cached locally")
    print("  ✅ Agent remains operational even if platform is down")
    
    # Flush telemetry
    if API_TOKEN:
        print("\n📤 Flushing telemetry to platform...")
        flush_and_stop()
        print("✅ Telemetry flushed")
    
    # Cleanup
    PolicyEngine.stop_remote_sync()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        if API_TOKEN:
            flush_and_stop()
        PolicyEngine.stop_remote_sync()
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

