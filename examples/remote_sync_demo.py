"""
Agent Sentinel SDK - Phase 3 Example: Cloud Connect
===================================================

This example demonstrates remote sync to the platform:
1. Background flusher that uploads logs periodically
2. Batch uploads with retry logic
3. Graceful shutdown and final flush
4. Works alongside local ledger (fail-open)

Prerequisites:
1. Platform must be running (docker compose up)
2. You need a JWT token or API key for authentication
3. Install with remote support: pip install agent-sentinel[remote]

Run this example:
    python3 examples/remote_sync_demo.py

Expected behavior:
- Actions are logged locally to .agent-sentinel/ledger.jsonl
- Background thread uploads logs to platform every 10 seconds
- On exit, remaining logs are flushed
- If platform is unavailable, logs remain local only (fail-open)
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
from pathlib import Path

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_sentinel import (
    guarded_action,
    enable_remote_sync,
    flush_and_stop,
    CostTracker,
)


# =============================================================================
# Configuration
# =============================================================================

# Platform configuration
# Update these for your environment
PLATFORM_URL = os.getenv("AGENT_SENTINEL_PLATFORM_URL", "http://localhost:8000")
API_TOKEN = os.getenv("AGENT_SENTINEL_API_TOKEN", "")

# If no token provided, skip remote sync (local-only mode)
ENABLE_REMOTE = bool(API_TOKEN)


# =============================================================================
# Agent Actions
# =============================================================================

@guarded_action(name="process_document", cost_usd=0.002, tags=["processing"])
def process_document(doc_id: int) -> dict:
    """Simulate document processing."""
    print(f"  [Exec] Processing document {doc_id}...")
    time.sleep(0.1)  # Simulate work
    return {"doc_id": doc_id, "pages": 42, "status": "processed"}


@guarded_action(name="call_llm", cost_usd=0.05, tags=["llm", "expensive"])
async def call_llm(prompt: str) -> str:
    """Simulate LLM API call."""
    print(f"  [Exec] Calling LLM with prompt: {prompt[:50]}...")
    await asyncio.sleep(0.2)  # Simulate API latency
    return f"Generated response for: {prompt}"


@guarded_action(name="search_database", cost_usd=0.001, tags=["db"])
def search_database(query: str) -> list:
    """Simulate database search."""
    print(f"  [Exec] Searching database: {query}...")
    return [{"id": 1, "title": "Result 1"}, {"id": 2, "title": "Result 2"}]


# =============================================================================
# Main Demo
# =============================================================================

async def main():
    print("=" * 70)
    print("🤖 Agent Sentinel SDK - Phase 3: Cloud Connect Demo")
    print("=" * 70)
    print()
    
    # Check if remote sync is enabled
    if ENABLE_REMOTE:
        print(f"🌐 Remote sync enabled")
        print(f"   Platform: {PLATFORM_URL}")
        print(f"   Token: {API_TOKEN[:10]}..." if len(API_TOKEN) > 10 else "   Token: (hidden)")
        print()
        
        # Start background sync
        sync = enable_remote_sync(
            platform_url=PLATFORM_URL,
            api_token=API_TOKEN,
            flush_interval=5.0,  # Flush every 5 seconds for demo
        )
        print("✅ Background sync started (5 second intervals)")
    else:
        print("⚠️  Remote sync disabled (no API token)")
        print("   Set AGENT_SENTINEL_API_TOKEN environment variable to enable")
        print("   Running in local-only mode")
    
    print()
    print("=" * 70)
    print("Running Agent Tasks...")
    print("=" * 70)
    print()
    
    # Reset cost tracker for clean demo
    CostTracker.reset_all()
    
    # Simulate agent workflow
    try:
        # Task 1: Process multiple documents
        print("📄 Task 1: Processing documents")
        for i in range(5):
            process_document(i + 1)
        print()
        
        # Task 2: LLM calls
        print("🤖 Task 2: LLM interactions")
        await call_llm("Summarize the quarterly report")
        await call_llm("Generate a follow-up email")
        print()
        
        # Task 3: Database searches
        print("🔍 Task 3: Database searches")
        search_database("customer feedback")
        search_database("sales data Q4")
        print()
        
        # Wait a bit for background sync to happen
        if ENABLE_REMOTE:
            print("⏳ Waiting for background sync...")
            print("   (Logs are being uploaded in the background)")
            await asyncio.sleep(6)  # Let one flush cycle complete
            print()
        
        # More work
        print("📊 Task 4: Additional processing")
        for i in range(3):
            process_document(i + 10)
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    # Summary
    print("=" * 70)
    print("📊 Summary")
    print("=" * 70)
    snapshot = CostTracker.get_snapshot()
    print(f"Total actions: {sum(snapshot['action_counts'].values())}")
    print(f"Total cost: ${snapshot['session_total']:.4f}")
    print()
    
    for action, count in snapshot['action_counts'].items():
        cost = snapshot['action_costs'][action]
        print(f"  {action}: {count} calls, ${cost:.4f}")
    print()
    
    # Check local ledger
    ledger_path = Path(".agent-sentinel/ledger.jsonl")
    if ledger_path.exists():
        with open(ledger_path, "r") as f:
            lines = len(f.readlines())
        print(f"📁 Local ledger: {lines} entries in {ledger_path}")
    print()
    
    # Graceful shutdown
    if ENABLE_REMOTE:
        print("🔄 Flushing remaining logs to platform...")
        flush_and_stop()
        print("✅ Final sync complete")
    else:
        print("💾 Logs saved locally only")
    
    print()
    print("=" * 70)
    print("✅ Phase 3 Demo Complete!")
    print()
    print("What happened:")
    print("  1. Actions were logged to local ledger (.agent-sentinel/ledger.jsonl)")
    if ENABLE_REMOTE:
        print("  2. Background thread uploaded logs to platform every 5 seconds")
        print("  3. Final flush ensured all logs were synced before exit")
        print("  4. You can view these logs in the platform dashboard")
    else:
        print("  2. Logs remained local (remote sync was disabled)")
    print()
    print("Try these next:")
    if not ENABLE_REMOTE:
        print("  • Set AGENT_SENTINEL_API_TOKEN and run again")
    print("  • Check platform database for ingested runs and actions")
    print("  • Simulate network failure (stop platform) - agent keeps working!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

