"""
Anthropic Tool Executor with Evidence Enforcement

Demonstrates AgentSentinel's correctness enforcement for Anthropic tool-use agents.
Shows evidence chains, argument constraints, and groundedness checks.

Usage:
    export ANTHROPIC_API_KEY='your-key'
    python examples/anthropic_tool_executor.py
"""
from agent_sentinel.integrations.anthropic_tools import AnthropicSentinelTools
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.policy import PolicyEngine


# ── Define tools ─────────────────────────────────────────────────────

def verify_identity(user_id: str) -> dict:
    """Verify a user's identity."""
    return {"user_id": user_id, "verified": True, "name": "Alice Johnson"}


def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    """Transfer funds between accounts."""
    return {"transfer_id": "TXN-456", "amount": amount, "status": "completed"}


# ── Configure PolicyEngine for argument constraint enforcement ───────

PolicyEngine.configure(
    argument_constraints={
        "transfer_funds": {
            "properties": {
                "amount": {"type": "number", "minimum": 0.01, "maximum": 10000},
            },
            "required": ["from_account", "to_account", "amount"],
        },
    },
)


# ── Create executor with evidence config ─────────────────────────────

tools = AnthropicSentinelTools({
    "verify_identity": (verify_identity, {
        "produces_evidence": True,
    }),
    "transfer_funds": (transfer_funds, {
        "is_commit": True,
        "requires": ["verify_identity"],
    }),
})


# ── Simulate the agent loop ──────────────────────────────────────────

def simulate_agent_loop():
    from types import SimpleNamespace

    print("=" * 60)
    print("AgentSentinel Anthropic Tool Executor Demo")
    print("=" * 60)

    EvidenceTracker.reset_session()

    # Step 1: Try transfer without identity verification
    print("\n[BLOCKED] Step 1: Transfer without identity verification...")
    block = SimpleNamespace(
        id="toolu_001", type="tool_use", name="transfer_funds",
        input={"from_account": "ACC-001", "to_account": "ACC-002", "amount": 500.0},
    )
    result = tools.execute(block)
    resp = tools.to_tool_result_block(result)
    print(f"   Blocked: {result.blocked}")
    print(f"   is_error: {resp.get('is_error', False)}")

    # Step 2: Verify identity first
    print("\n[OK] Step 2: Verify identity...")
    block = SimpleNamespace(
        id="toolu_002", type="tool_use", name="verify_identity",
        input={"user_id": "USER-789"},
    )
    result = tools.execute(block)
    resp = tools.to_tool_result_block(result)
    print(f"   Output: {result.output}")
    print(f"   Evidence: {EvidenceTracker.has_evidence('verify_identity')}")

    # Step 3: Transfer succeeds
    print("\n[OK] Step 3: Transfer with evidence present...")
    block = SimpleNamespace(
        id="toolu_003", type="tool_use", name="transfer_funds",
        input={"from_account": "ACC-001", "to_account": "ACC-002", "amount": 500.0},
    )
    result = tools.execute(block)
    print(f"   Blocked: {result.blocked}")
    print(f"   Output: {result.output}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    simulate_agent_loop()
