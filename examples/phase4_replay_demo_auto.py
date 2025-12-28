#!/usr/bin/env python3
"""
Phase 4: Replay Mode Demo (Non-Interactive)

This is a non-interactive version that runs all demos automatically.
Use this for testing or automated demonstrations.
"""
from __future__ import annotations

import time
import random
from agent_sentinel import (
    guarded_action,
    replay_mode,
    ReplayMode,
    ReplayDivergenceError,
    Ledger,
)


# ============================================================================
# Part 1: Record actions (normal execution)
# ============================================================================

@guarded_action(name="fetch_weather", cost_usd=0.01, tags=["api", "weather"])
def fetch_weather(city: str) -> dict:
    """Simulate fetching weather data from an API."""
    print(f"  📡 Fetching weather for {city}...")
    time.sleep(0.1)  # Simulate API call delay
    
    # Simulate API response
    return {
        "city": city,
        "temperature": random.randint(60, 80),
        "condition": random.choice(["sunny", "cloudy", "rainy"]),
        "timestamp": time.time(),
    }


@guarded_action(name="process_weather", cost_usd=0.005, tags=["processing"])
def process_weather(weather_data: dict) -> str:
    """Process weather data and generate a report."""
    print(f"  🔄 Processing weather data for {weather_data['city']}...")
    time.sleep(0.05)
    
    city = weather_data["city"]
    temp = weather_data["temperature"]
    condition = weather_data["condition"]
    
    return f"Weather in {city}: {temp}°F and {condition}"


@guarded_action(name="send_notification", cost_usd=0.02, tags=["notification"])
def send_notification(message: str) -> bool:
    """Simulate sending a notification."""
    print(f"  📬 Sending notification: {message}")
    time.sleep(0.1)
    return True


def run_weather_agent(cities: list[str]):
    """Run the weather agent for a list of cities."""
    print(f"\n🤖 Running weather agent for {len(cities)} cities...\n")
    
    for city in cities:
        # Fetch weather
        weather = fetch_weather(city)
        
        # Process weather
        report = process_weather(weather)
        
        # Send notification
        send_notification(report)
    
    print("\n✅ Weather agent completed!\n")


# ============================================================================
# Demos
# ============================================================================

def demo_replay_exact():
    """Demo: Replay with exact same inputs (no divergence)."""
    print("=" * 70)
    print("DEMO 1: Replay with Exact Same Inputs")
    print("=" * 70)
    
    cities = ["San Francisco", "New York"]
    
    # Clear the ledger to start fresh
    ledger_path = Ledger.get_log_path()
    if ledger_path and ledger_path.exists():
        print("\n🗑️  Clearing old ledger entries...\n")
        ledger_path.unlink()
    
    # Step 1: Record actions
    print("\n📝 STEP 1: Recording actions (normal execution)...\n")
    run_weather_agent(cities)
    
    ledger_path = Ledger.get_log_path()
    print(f"✅ Actions recorded to: {ledger_path}\n")
    
    # Step 2: Replay with same inputs
    print("🔁 STEP 2: Replaying with same inputs...\n")
    
    with replay_mode(ledger_path=ledger_path, strict=True) as replay:
        # This will use recorded outputs WITHOUT calling the APIs
        run_weather_agent(cities)
        
        progress = replay.get_progress()
        divergences = replay.get_divergences()
        
        print(f"✅ Replay completed: {progress[0]}/{progress[1]} actions replayed")
        print(f"✅ Divergences detected: {len(divergences)}")
    
    print("\n💡 Notice: Functions returned recorded outputs without executing!")
    print("   No API calls were made, no costs incurred.\n")


def demo_replay_divergence_strict():
    """Demo: Replay with different inputs in strict mode (raises error)."""
    print("=" * 70)
    print("DEMO 2: Replay with Different Inputs (Strict Mode)")
    print("=" * 70)
    
    cities_original = ["Los Angeles"]
    cities_different = ["Chicago"]  # Different city!
    
    # Clear the ledger to start fresh
    ledger_path = Ledger.get_log_path()
    if ledger_path and ledger_path.exists():
        ledger_path.unlink()
    
    # Step 1: Record actions
    print("\n📝 STEP 1: Recording actions with original inputs...\n")
    run_weather_agent(cities_original)
    
    ledger_path = Ledger.get_log_path()
    
    # Step 2: Try to replay with different inputs
    print("🔁 STEP 2: Attempting replay with DIFFERENT inputs (strict mode)...\n")
    
    try:
        with replay_mode(ledger_path=ledger_path, strict=True) as replay:
            run_weather_agent(cities_different)
    except ReplayDivergenceError as e:
        print(f"❌ Divergence detected (as expected)!")
        print(f"   Error: {e}\n")
        print("💡 Strict mode raises an error when inputs don't match.")
        print("   This helps detect non-deterministic behavior!\n")


def demo_replay_divergence_lenient():
    """Demo: Replay with different inputs in lenient mode (logs warnings)."""
    print("=" * 70)
    print("DEMO 3: Replay with Different Inputs (Lenient Mode)")
    print("=" * 70)
    
    cities_original = ["Seattle"]
    cities_different = ["Boston"]
    
    # Clear the ledger to start fresh
    ledger_path = Ledger.get_log_path()
    if ledger_path and ledger_path.exists():
        ledger_path.unlink()
    
    # Step 1: Record actions
    print("\n📝 STEP 1: Recording actions with original inputs...\n")
    run_weather_agent(cities_original)
    
    ledger_path = Ledger.get_log_path()
    
    # Step 2: Replay with different inputs in lenient mode
    print("🔁 STEP 2: Replaying with DIFFERENT inputs (lenient mode)...\n")
    
    with replay_mode(ledger_path=ledger_path, strict=False) as replay:
        run_weather_agent(cities_different)
        
        progress = replay.get_progress()
        divergences = replay.get_divergences()
        
        print(f"⚠️  Replay completed with warnings")
        print(f"   Progress: {progress[0]}/{progress[1]} actions")
        print(f"   Divergences: {len(divergences)}")
        
        if divergences:
            print("\n📊 Divergence details:")
            for i, div in enumerate(divergences, 1):
                print(f"   {i}. Type: {div['type']}")
                if div['type'] == 'input_mismatch':
                    print(f"      Action: {div['action']}")
    
    print("\n💡 Lenient mode logs warnings but continues execution.")
    print("   Useful for debugging when you expect some variance.\n")


def demo_testing_without_costs():
    """Demo: Use replay for cost-free testing."""
    print("=" * 70)
    print("DEMO 4: Testing Without API Costs")
    print("=" * 70)
    
    print("\n💡 Scenario: Test agent logic without API costs.\n")
    
    # Clear the ledger to start fresh
    ledger_path = Ledger.get_log_path()
    if ledger_path and ledger_path.exists():
        ledger_path.unlink()
    
    # Step 1: Record once
    print("📝 STEP 1: Record actions ONCE (incur costs)...\n")
    
    @guarded_action(name="expensive_api", cost_usd=0.50, tags=["expensive"])
    def expensive_api(query: str) -> dict:
        print(f"  💰 Calling expensive API: ${0.50}")
        time.sleep(0.2)
        return {"query": query, "result": f"data for {query}"}
    
    result1 = expensive_api("test query")
    print(f"  Result: {result1}\n")
    
    ledger_path = Ledger.get_log_path()
    
    # Step 2: Test multiple times with replay (no costs!)
    print("🔁 STEP 2: Test 10 times with replay (NO costs)...\n")
    
    for i in range(10):
        with replay_mode(ledger_path=ledger_path, strict=True):
            result = expensive_api("test query")
            print(f"  Test {i+1}: Got result (cost: $0.00)")
    
    print("\n✅ Ran 10 tests with ZERO API costs!")
    print("   Original cost: $0.50 × 10 = $5.00")
    print("   Replay cost: $0.00\n")
    print("   💰 Savings: $4.50\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos automatically."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║         AGENT SENTINEL - PHASE 4 REPLAY DEMO (AUTO MODE)          ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    demos = [
        demo_replay_exact,
        demo_replay_divergence_strict,
        demo_replay_divergence_lenient,
        demo_testing_without_costs,
    ]
    
    for i, demo in enumerate(demos, 1):
        demo()
        
        if i < len(demos):
            print("\n" + "—" * 70 + "\n")
            time.sleep(1)  # Brief pause between demos
    
    print("=" * 70)
    print("ALL DEMOS COMPLETED!")
    print("=" * 70)
    print("\n✅ Phase 4 replay functionality demonstrated successfully!")
    print("\n📖 Key Takeaways:")
    print("   • Replay mode returns recorded outputs without execution")
    print("   • Strict mode detects input divergence and raises errors")
    print("   • Lenient mode logs warnings but continues execution")
    print("   • Great for debugging, testing, and cost savings")
    print("\n💡 Next Steps:")
    print("   • Try replay mode in your own agent code")
    print("   • Use it to debug non-deterministic behavior")
    print("   • Save costs by testing with recorded data")
    print("\n")


if __name__ == "__main__":
    main()

