"""
OpenAI Tool Executor with Evidence Enforcement

Demonstrates AgentSentinel's correctness enforcement for OpenAI tool-calling agents.
The refund tool CANNOT execute unless the order lookup has been performed first.

Usage:
    export OPENAI_API_KEY='your-key'
    python examples/openai_tool_executor.py
"""
from agent_sentinel.integrations.openai import OpenAISentinelTools
from agent_sentinel.integrations.tools import sentinel_tool
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.policy import PolicyEngine


# ── Define tools with evidence requirements ──────────────────────────

def lookup_order(order_id: str) -> dict:
    """Simulate looking up an order from the database."""
    orders = {
        "ORD-001": {"order_id": "ORD-001", "customer": "Alice", "amount": 99.99, "status": "delivered"},
        "ORD-002": {"order_id": "ORD-002", "customer": "Bob", "amount": 49.50, "status": "shipped"},
    }
    order = orders.get(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    return order


def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Simulate issuing a refund."""
    return {"refund_id": "REF-123", "order_id": order_id, "amount": amount, "status": "processed"}


# ── Configure PolicyEngine for argument constraint enforcement ───────

PolicyEngine.configure(
    argument_constraints={
        "issue_refund": {
            "properties": {
                "amount": {"type": "number", "minimum": 0, "maximum": 1000},
                "reason": {"enum": ["defective", "wrong_item", "not_received", "changed_mind"]},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
)


# ── Create the tool executor with evidence config ────────────────────

tools = OpenAISentinelTools({
    "lookup_order": (lookup_order, {
        "produces_evidence": True,
    }),
    "issue_refund": (issue_refund, {
        "is_commit": True,
        "requires": ["lookup_order"],
        "grounding_rules": {
            "order_id": {"source_action": "lookup_order", "source_field": "order_id"},
        },
    }),
})


# ── Simulate the agent loop ──────────────────────────────────────────

def simulate_agent_loop():
    """Simulate what happens in an OpenAI tool-calling agent loop."""
    from types import SimpleNamespace
    import json

    print("=" * 60)
    print("AgentSentinel OpenAI Tool Executor Demo")
    print("=" * 60)

    # Reset evidence for clean demo
    EvidenceTracker.reset_session()

    # Step 1: Agent tries to refund WITHOUT looking up the order first
    print("\n[BLOCKED] Step 1: Agent tries to issue refund without lookup...")
    tc = SimpleNamespace(
        id="call_001",
        function=SimpleNamespace(
            name="issue_refund",
            arguments=json.dumps({"order_id": "ORD-001", "amount": 50.0, "reason": "defective"}),
        ),
    )
    result = tools.execute(tc)
    msg = tools.to_tool_message(result)
    print(f"   Blocked: {result.blocked}")
    print(f"   Remediation: {result.remediation}")
    print(f"   Tool message content: {msg['content'][:200]}")

    # Step 2: Agent looks up the order (produces evidence)
    print("\n[OK] Step 2: Agent looks up the order first...")
    tc = SimpleNamespace(
        id="call_002",
        function=SimpleNamespace(
            name="lookup_order",
            arguments=json.dumps({"order_id": "ORD-001"}),
        ),
    )
    result = tools.execute(tc)
    msg = tools.to_tool_message(result)
    print(f"   Output: {result.output}")
    print(f"   Evidence recorded: {EvidenceTracker.has_evidence('lookup_order')}")

    # Step 3: Now the refund succeeds
    print("\n[OK] Step 3: Agent retries refund (evidence now present)...")
    tc = SimpleNamespace(
        id="call_003",
        function=SimpleNamespace(
            name="issue_refund",
            arguments=json.dumps({"order_id": "ORD-001", "amount": 50.0, "reason": "defective"}),
        ),
    )
    result = tools.execute(tc)
    msg = tools.to_tool_message(result)
    print(f"   Blocked: {result.blocked}")
    print(f"   Output: {result.output}")

    # Step 4: Agent tries with WRONG order_id (groundedness check)
    print("\n[BLOCKED] Step 4: Agent tries refund with wrong order_id (groundedness)...")
    tc = SimpleNamespace(
        id="call_004",
        function=SimpleNamespace(
            name="issue_refund",
            arguments=json.dumps({"order_id": "ORD-FAKE", "amount": 50.0, "reason": "defective"}),
        ),
    )
    result = tools.execute(tc)
    print(f"   Blocked: {result.blocked}")
    print(f"   Reason: {result.remediation.get('reason_code', 'N/A') if result.remediation else 'N/A'}")

    # Step 5: Agent tries with invalid arguments
    print("\n[BLOCKED] Step 5: Agent tries refund with amount > $1000...")
    tc = SimpleNamespace(
        id="call_005",
        function=SimpleNamespace(
            name="issue_refund",
            arguments=json.dumps({"order_id": "ORD-001", "amount": 5000.0, "reason": "defective"}),
        ),
    )
    result = tools.execute(tc)
    print(f"   Blocked: {result.blocked}")
    print(f"   Violations: {result.remediation.get('argument_violations', []) if result.remediation else []}")

    print("\n" + "=" * 60)
    print("Demo complete. Evidence enforcement prevents incorrect actions.")
    print("=" * 60)


if __name__ == "__main__":
    simulate_agent_loop()
