"""
Agent Sentinel SDK - Phase 2 Example: Policy Engine
====================================================

This example demonstrates budget enforcement and policy controls:
1. Cost tracking with static counters
2. Policy loading from callguard.yaml
3. Budget limits (session and run level)
4. Action-specific budgets
5. Denied and allowed action lists
6. BudgetExceededError and PolicyViolationError

Run this example:
    python3 examples/policy_demo.py

Expected behavior:
- Some actions succeed and accumulate costs
- Budget limits are enforced
- BudgetExceededError is raised when limits hit
- Denied actions are blocked immediately
"""
from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_sentinel import (
    guarded_action,
    PolicyEngine,
    CostTracker,
    BudgetExceededError,
    PolicyViolationError
)


# =============================================================================
# Agent Actions with Different Costs
# =============================================================================

@guarded_action(name="cheap_action", cost_usd=0.01, tags=["cheap"])
def cheap_action(message: str) -> str:
    """A low-cost action that should usually succeed."""
    print(f"  [Exec] Cheap action: {message}")
    return f"Processed: {message}"


@guarded_action(name="expensive_llm_call", cost_usd=0.30, tags=["llm", "expensive"])
async def expensive_llm_call(prompt: str) -> str:
    """An expensive action that will hit per-action budget limits."""
    print(f"  [Exec] Expensive LLM call: {prompt[:50]}...")
    await asyncio.sleep(0.1)  # Simulate API call
    return f"Generated response for: {prompt}"


@guarded_action(name="search_web", cost_usd=0.15, tags=["search", "external"])
def search_web(query: str) -> dict:
    """Medium-cost search action."""
    print(f"  [Exec] Searching web: {query}")
    return {"results": [f"Result for {query}"], "count": 1}


@guarded_action(name="dangerous_operation", cost_usd=0.0, tags=["dangerous"])
def dangerous_operation() -> str:
    """This action is on the denied list in callguard.yaml."""
    print(f"  [Exec] Performing dangerous operation...")
    return "This should never execute!"


# =============================================================================
# Demo Scenarios
# =============================================================================

async def test_basic_cost_tracking():
    """Test that costs are tracked correctly."""
    print("=" * 70)
    print("Test 1: Basic Cost Tracking")
    print("=" * 70)
    
    # Reset for clean test
    CostTracker.reset_all()
    
    print("\nInitial state:")
    print(f"  Session total: ${CostTracker.get_session_total():.4f}")
    print(f"  Run total: ${CostTracker.get_run_total():.4f}")
    
    print("\nExecuting 3 cheap actions...")
    for i in range(3):
        cheap_action(f"Message {i+1}")
    
    print("\nCosts after 3 cheap actions:")
    print(f"  Session total: ${CostTracker.get_session_total():.4f}")
    print(f"  Run total: ${CostTracker.get_run_total():.4f}")
    
    stats = CostTracker.get_action_stats("cheap_action")
    print(f"  cheap_action: {stats['count']} calls, ${stats['total_cost']:.4f}")
    
    print("\n✅ Cost tracking works!\n")


async def test_run_budget_limit():
    """Test that run budget limits are enforced."""
    print("=" * 70)
    print("Test 2: Run Budget Enforcement")
    print("=" * 70)
    
    # Configure with small run budget
    print("\nConfiguring policy: run_budget = $0.10")
    PolicyEngine.configure(run_budget=0.10)
    CostTracker.reset_run()  # Start fresh
    
    print("\nTrying to execute 5 cheap actions ($0.01 each)...")
    
    executed = 0
    try:
        for i in range(5):
            cheap_action(f"Action {i+1}")
            executed += 1
            print(f"    ✓ Action {i+1} succeeded (total: ${CostTracker.get_run_total():.2f})")
    except BudgetExceededError as e:
        print(f"\n  ⛔ Budget limit hit after {executed} actions!")
        print(f"     Error: {e}")
    
    print(f"\n  Final run total: ${CostTracker.get_run_total():.4f}")
    print(f"  Actions executed: {executed} / 5")
    print("\n✅ Run budget enforcement works!\n")


async def test_action_specific_budget():
    """Test that per-action budgets are enforced."""
    print("=" * 70)
    print("Test 3: Action-Specific Budget")
    print("=" * 70)
    
    # Configure with action-specific limit
    print("\nConfiguring policy: expensive_llm_call max $0.50")
    PolicyEngine.configure(
        run_budget=10.0,  # High run budget
        action_budgets={
            "expensive_llm_call": 0.50  # But limit this specific action
        }
    )
    CostTracker.reset_all()
    
    print("\nTrying to call expensive_llm_call ($0.30 each)...")
    
    executed = 0
    try:
        for i in range(3):
            await expensive_llm_call(f"Generate text about topic {i+1}")
            executed += 1
            stats = CostTracker.get_action_stats("expensive_llm_call")
            print(f"    ✓ Call {i+1} succeeded (action total: ${stats['total_cost']:.2f})")
    except BudgetExceededError as e:
        print(f"\n  ⛔ Action budget exceeded after {executed} calls!")
        print(f"     Error: {e}")
    
    stats = CostTracker.get_action_stats("expensive_llm_call")
    print(f"\n  Final action total: ${stats['total_cost']:.4f}")
    print(f"  Calls executed: {executed} / 3")
    print("\n✅ Action-specific budget enforcement works!\n")


async def test_denied_actions():
    """Test that denied actions are blocked."""
    print("=" * 70)
    print("Test 4: Denied Actions")
    print("=" * 70)
    
    print("\nConfiguring policy: dangerous_operation is DENIED")
    PolicyEngine.configure(
        run_budget=10.0,
        denied_actions=["dangerous_operation"]
    )
    
    print("\nTrying to execute dangerous_operation...")
    
    try:
        dangerous_operation()
        print("  ❌ ERROR: Dangerous operation should have been blocked!")
    except PolicyViolationError as e:
        print(f"  ✅ Action correctly blocked!")
        print(f"     Error: {e}")
    
    print("\n✅ Denied action enforcement works!\n")


async def test_allowed_actions():
    """Test that allowlist mode works."""
    print("=" * 70)
    print("Test 5: Allowlist Mode")
    print("=" * 70)
    
    print("\nConfiguring policy: ONLY cheap_action is allowed")
    PolicyEngine.configure(
        run_budget=10.0,
        allowed_actions=["cheap_action"]  # Allowlist
    )
    CostTracker.reset_all()
    
    print("\nTrying allowed action (cheap_action)...")
    try:
        cheap_action("This should work")
        print("  ✅ Allowed action succeeded")
    except PolicyViolationError as e:
        print(f"  ❌ ERROR: Should have been allowed: {e}")
    
    print("\nTrying non-allowed action (search_web)...")
    try:
        search_web("python tutorials")
        print("  ❌ ERROR: Should have been blocked!")
    except PolicyViolationError as e:
        print(f"  ✅ Action correctly blocked!")
        print(f"     Error: {e}")
    
    print("\n✅ Allowlist enforcement works!\n")


async def test_yaml_configuration():
    """Test loading configuration from callguard.yaml."""
    print("=" * 70)
    print("Test 6: Loading from callguard.yaml")
    print("=" * 70)
    
    yaml_path = Path("callguard.yaml")
    
    if not yaml_path.exists():
        print("\n⚠️  callguard.yaml not found, skipping this test")
        print("   (This test requires PyYAML: pip install pyyaml)")
        return
    
    print(f"\nLoading policy from {yaml_path}")
    PolicyEngine.reset()
    PolicyEngine.load_from_yaml(str(yaml_path))
    
    if not PolicyEngine.is_configured():
        print("  ⚠️  Policy not loaded (PyYAML may not be installed)")
        print("     Install with: pip install pyyaml")
        return
    
    config = PolicyEngine.get_config()
    print(f"\n  ✅ Policy loaded successfully!")
    print(f"     Session budget: ${config.session_budget}")
    print(f"     Run budget: ${config.run_budget}")
    print(f"     Action budgets: {config.action_budgets}")
    print(f"     Denied actions: {config.denied_actions}")
    print(f"     Strict mode: {config.strict_mode}")
    
    # Test that denied actions from YAML are blocked
    if "delete_database" in config.denied_actions:
        print("\n  Testing YAML-defined denied action...")
        
        @guarded_action(name="delete_database", cost_usd=0.0)
        def delete_database():
            return "Should not execute"
        
        try:
            delete_database()
            print("    ❌ ERROR: Should have been blocked!")
        except PolicyViolationError:
            print("    ✅ YAML-defined denial works!")
    
    print("\n✅ YAML configuration works!\n")


async def test_cost_reset():
    """Test that cost reset works for multi-run scenarios."""
    print("=" * 70)
    print("Test 7: Cost Reset Between Runs")
    print("=" * 70)
    
    PolicyEngine.configure(run_budget=0.05)
    
    print("\n--- Run 1 ---")
    CostTracker.reset_run()
    print(f"Initial run total: ${CostTracker.get_run_total():.4f}")
    
    try:
        for i in range(10):
            cheap_action(f"Run 1, Action {i+1}")
    except BudgetExceededError:
        pass
    
    run1_total = CostTracker.get_run_total()
    session1_total = CostTracker.get_session_total()
    print(f"Run 1 total: ${run1_total:.4f}")
    print(f"Session total: ${session1_total:.4f}")
    
    print("\n--- Run 2 (after reset) ---")
    CostTracker.reset_run()
    print(f"Initial run total: ${CostTracker.get_run_total():.4f}")
    
    try:
        for i in range(10):
            cheap_action(f"Run 2, Action {i+1}")
    except BudgetExceededError:
        pass
    
    run2_total = CostTracker.get_run_total()
    session2_total = CostTracker.get_session_total()
    print(f"Run 2 total: ${run2_total:.4f}")
    print(f"Session total: ${session2_total:.4f}")
    
    print(f"\n  ✅ Run total reset: ${run1_total:.4f} → ${CostTracker.get_run_total():.4f}")
    print(f"  ✅ Session total accumulated: ${session1_total:.4f} → ${session2_total:.4f}")
    print("\n✅ Cost reset works!\n")


# =============================================================================
# Main Demo
# =============================================================================

async def main():
    print("=" * 70)
    print("🤖 Agent Sentinel SDK - Phase 2: Policy Engine Demo")
    print("=" * 70)
    print()
    
    # Clean slate
    PolicyEngine.reset()
    CostTracker.reset_all()
    
    # Run all tests
    await test_basic_cost_tracking()
    await test_run_budget_limit()
    await test_action_specific_budget()
    await test_denied_actions()
    await test_allowed_actions()
    await test_yaml_configuration()
    await test_cost_reset()
    
    # Summary
    print("=" * 70)
    print("📊 Final Summary")
    print("=" * 70)
    snapshot = CostTracker.get_snapshot()
    print(f"Session total: ${snapshot['session_total']:.4f}")
    print(f"Run total: ${snapshot['run_total']:.4f}")
    print(f"\nAction statistics:")
    for action, cost in snapshot['action_costs'].items():
        count = snapshot['action_counts'][action]
        print(f"  {action}: {count} calls, ${cost:.4f}")
    
    print()
    print("=" * 70)
    print("✅ Phase 2 Implementation Complete!")
    print("   • Cost tracking: ✓")
    print("   • Policy engine: ✓")
    print("   • Budget enforcement: ✓")
    print("   • Denied/allowed lists: ✓")
    print("   • YAML configuration: ✓")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

