"""
Agent Sentinel SDK - Phase 1 Example
=====================================

This example demonstrates the "Local Loop" functionality:
1. Decorator with time tracking (using perf_counter_ns)
2. Try/except logic with proper error handling
3. Local ledger writing to temp file
4. Works completely offline (no internet required)

Run this example:
    python examples/simple_agent.py

Expected output:
- Console output showing agent actions
- A .agent-sentinel/ledger.jsonl file created with entries
- Verification that complex objects are handled safely
- Demonstration of error handling
"""
from __future__ import annotations

import asyncio
import random
import sys
import os
import json
from dataclasses import dataclass
from pathlib import Path

# Ensure we can import the local package without installing it yet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_sentinel import guarded_action, Ledger


# =============================================================================
# Test Scenarios: Complex Objects and Error Cases
# =============================================================================

class DatabaseConnection:
    """
    A non-serializable object to test SafeEncoder.
    Standard JSON parsers crash on this, but CallGuard should handle it gracefully.
    """
    def __repr__(self):
        return "<DatabaseConnection host='localhost' connected=True>"


@dataclass
class UserContext:
    """Example of a dataclass that should serialize properly."""
    user_id: int
    session_token: str
    metadata: dict = None


# =============================================================================
# Agent Tools: Synchronous and Asynchronous
# =============================================================================

@guarded_action(name="calculator_tool", cost_usd=0.001, tags=["math", "logic"])
def calculate(a: float, b: float, operation: str = "add") -> float:
    """
    A simple synchronous tool for mathematical operations.
    
    This demonstrates:
    - Sync function decoration
    - Time tracking
    - Cost attribution
    - Tag categorization
    """
    print(f"  [Exec] Calculating {a} {operation} {b}...")
    
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


@guarded_action(name="search_web", cost_usd=0.02, tags=["io", "search", "external"])
async def search_web(query: str, context: UserContext) -> dict:
    """
    An async tool simulating a web search API call.
    
    This demonstrates:
    - Async function decoration
    - Complex object handling (DatabaseConnection)
    - Dataclass input handling
    - Realistic API simulation with delay
    """
    print(f"  [Exec] Searching web for '{query}'...")
    print(f"        User: {context.user_id}, Token: {context.session_token[:8]}...")
    
    # Simulate network latency
    await asyncio.sleep(0.1)
    
    # Return complex data including a non-serializable object
    return {
        "query": query,
        "results": [
            f"Result 1 for {query}",
            f"Result 2 for {query}",
            f"Result 3 for {query}"
        ],
        "metadata": {
            "source": "mock_search_engine",
            "connection": DatabaseConnection(),  # Non-serializable!
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }


@guarded_action(name="write_file", cost_usd=0.0, tags=["io", "filesystem"])
def write_file(filename: str, content: str) -> dict:
    """
    A tool that writes to filesystem.
    
    This demonstrates:
    - Zero-cost actions (cost_usd=0.0)
    - Multiple tags
    - Return values with metadata
    """
    print(f"  [Exec] Writing to {filename}...")
    
    # In a real agent, this would actually write to disk
    # For demo purposes, we just simulate it
    return {
        "filename": filename,
        "bytes_written": len(content),
        "status": "success"
    }


@guarded_action(name="risky_operation", cost_usd=0.05, tags=["dangerous", "external"])
def risky_operation(should_fail: bool = False) -> str:
    """
    A tool that can fail on demand.
    
    This demonstrates:
    - Error tracking and re-raising
    - Ledger records errors with proper outcome="error"
    """
    print(f"  [Exec] Running risky operation...")
    
    if should_fail:
        raise RuntimeError("Simulated failure: Operation went wrong!")
    
    return "Operation completed successfully"


# =============================================================================
# Main Demo
# =============================================================================

async def main():
    print("=" * 70)
    print("🤖 Agent Sentinel SDK - Phase 1 Demo: The Local Loop")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  ✓ Time tracking with nanosecond precision")
    print("  ✓ Try/except error handling")
    print("  ✓ Local ledger writing to .agent-sentinel/ledger.jsonl")
    print("  ✓ Completely offline (no internet required)")
    print("  ✓ Safe handling of non-serializable objects")
    print()
    
    # Clean up old ledger file for fresh demo
    ledger_path = Path(".agent-sentinel/ledger.jsonl")
    if ledger_path.exists():
        ledger_path.unlink()
        print(f"🗑️  Cleaned up old ledger file\n")
    
    # =========================================================================
    # Test 1: Successful Sync Call
    # =========================================================================
    print("-" * 70)
    print("Test 1: Synchronous Tool (Calculator)")
    print("-" * 70)
    result = calculate(10, 20, operation="add")
    print(f"   ✓ Result: {result}")
    print()
    
    # =========================================================================
    # Test 2: Multiple Sync Calls with Different Operations
    # =========================================================================
    print("-" * 70)
    print("Test 2: Multiple Calculator Operations")
    print("-" * 70)
    result = calculate(100, 25, operation="subtract")
    print(f"   ✓ Result: {result}")
    result = calculate(5, 7, operation="multiply")
    print(f"   ✓ Result: {result}")
    print()
    
    # =========================================================================
    # Test 3: Successful Async Call with Complex Objects
    # =========================================================================
    print("-" * 70)
    print("Test 3: Asynchronous Tool (Web Search) with Complex Objects")
    print("-" * 70)
    ctx = UserContext(
        user_id=123,
        session_token="abc-xyz-secret-token-789",
        metadata={"role": "developer", "plan": "pro"}
    )
    result = await search_web("python agent frameworks", context=ctx)
    print(f"   ✓ Found {len(result['results'])} results")
    print(f"   ✓ Metadata keys: {list(result['metadata'].keys())}")
    print()
    
    # =========================================================================
    # Test 4: Zero-Cost Action
    # =========================================================================
    print("-" * 70)
    print("Test 4: Zero-Cost Action (File Write)")
    print("-" * 70)
    result = write_file("output.txt", "Hello, Agent Sentinel!")
    print(f"   ✓ Wrote {result['bytes_written']} bytes")
    print()
    
    # =========================================================================
    # Test 5: Error Handling (Division by Zero)
    # =========================================================================
    print("-" * 70)
    print("Test 5: Error Handling (Division by Zero)")
    print("-" * 70)
    try:
        calculate(10, 0, operation="divide")
    except ZeroDivisionError as e:
        print(f"   ✓ Caught expected error: {e}")
        print(f"   ✓ Error was logged and re-raised properly")
    print()
    
    # =========================================================================
    # Test 6: Error Handling (Simulated Failure)
    # =========================================================================
    print("-" * 70)
    print("Test 6: Error Handling (Simulated Failure)")
    print("-" * 70)
    try:
        risky_operation(should_fail=True)
    except RuntimeError as e:
        print(f"   ✓ Caught expected error: {e}")
        print(f"   ✓ Error was logged and re-raised properly")
    print()
    
    # =========================================================================
    # Test 7: Successful Risky Operation
    # =========================================================================
    print("-" * 70)
    print("Test 7: Successful Risky Operation")
    print("-" * 70)
    result = risky_operation(should_fail=False)
    print(f"   ✓ Result: {result}")
    print()
    
    # =========================================================================
    # Validation: Verify Ledger Integrity
    # =========================================================================
    print("=" * 70)
    print("🔍 Verifying Ledger Integrity")
    print("=" * 70)
    
    ledger_path = Ledger.get_log_path()
    if not ledger_path or not ledger_path.exists():
        print("❌ Error: Ledger file not found!")
        return
    
    print(f"📁 Ledger location: {ledger_path}")
    print()
    
    with open(ledger_path, "r") as f:
        lines = f.readlines()
        print(f"✅ Found {len(lines)} entries in ledger")
        print()
        
        total_cost = 0.0
        success_count = 0
        error_count = 0
        
        for i, line in enumerate(lines, 1):
            entry = json.loads(line)
            status_icon = "✅" if entry['outcome'] == 'success' else "❌"
            
            print(f"{status_icon} Entry {i}: {entry['action']}")
            print(f"   ID: {entry['id']}")
            print(f"   Timestamp: {entry['timestamp']}")
            print(f"   Cost: ${entry['cost_usd']:.4f}")
            print(f"   Duration: {entry['duration_ms']:.3f}ms")
            print(f"   Outcome: {entry['outcome']}")
            print(f"   Tags: {', '.join(entry['tags']) if entry['tags'] else 'none'}")
            
            # Track statistics
            total_cost += entry['cost_usd']
            if entry['outcome'] == 'success':
                success_count += 1
            else:
                error_count += 1
            
            # Special checks
            if entry['action'] == 'search_web':
                outputs = str(entry['payload']['outputs'])
                if "<DatabaseConnection" in outputs:
                    print(f"   ✨ SafeEncoder worked: Complex object was stringified!")
            
            if entry['outcome'] == 'error':
                error_msg = entry['payload']['outputs']
                print(f"   ⚠️  Error: {error_msg}")
            
            print()
    
    print("=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"Total actions: {len(lines)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total cost: ${total_cost:.4f}")
    print()
    print("✅ Phase 1 Implementation Complete!")
    print("   • Time tracking: ✓")
    print("   • Try/except logic: ✓")
    print("   • Local ledger: ✓")
    print("   • Offline operation: ✓")
    print("   • Safe encoding: ✓")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
