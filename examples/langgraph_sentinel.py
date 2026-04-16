"""
LangGraph + AgentSentinel end-to-end example.

Demonstrates a 3-node agent workflow where every tool call is enforced by
AgentSentinel: policies, evidence graph, groundedness, and structured
block feedback for self-repair.

Scenario: A customer-support agent that:
  1. Looks up an order (produces evidence)
  2. Validates refund eligibility against evidence
  3. Issues a refund (a commit action requiring prior evidence)

If the agent attempts to call `issue_refund` before `lookup_order`, the
guardrail returns a structured remediation payload telling the LLM to
call the lookup tool first — the structured self-repair pattern.

Run:
    pip install langchain-core langgraph
    python examples/langgraph_sentinel.py
"""
from __future__ import annotations

from typing import Any, Dict

from agent_sentinel import (
    InterventionTracker,
    PolicyConfig,
    PolicyEngine,
)
from agent_sentinel.integrations.langgraph import SentinelToolNode


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def lookup_order(order_id: str) -> Dict[str, Any]:
    """Read tool: fetches order data and produces evidence."""
    # In real life: call your DB/API.
    db = {
        "ord-123": {"order_id": "ord-123", "amount": 49.99, "status": "delivered"},
        "ord-456": {"order_id": "ord-456", "amount": 10.00, "status": "cancelled"},
    }
    return db.get(order_id, {"order_id": order_id, "error": "not_found"})


def validate_refund(order_id: str, reason: str) -> Dict[str, Any]:
    """Read tool: decides eligibility based on prior evidence."""
    return {"order_id": order_id, "reason": reason, "eligible": True}


def issue_refund(order_id: str, amount: float) -> Dict[str, Any]:
    """Commit tool: performs the irreversible action."""
    return {"order_id": order_id, "amount": amount, "refunded": True}


# ---------------------------------------------------------------------------
# Policy configuration
# ---------------------------------------------------------------------------


PolicyEngine.reset()
PolicyEngine._config = PolicyConfig(
    # issue_refund requires prior lookup_order evidence
    evidence_requirements={"issue_refund": ["lookup_order"]},
    # amount must match the looked-up order amount
    grounding_rules={
        "issue_refund": {
            "amount": {"source_action": "lookup_order", "source_field": "amount"},
        }
    },
    commit_actions=["issue_refund"],
    evidence_actions=["lookup_order", "validate_refund"],
)
PolicyEngine._initialized = True


# ---------------------------------------------------------------------------
# LangGraph tool node
# ---------------------------------------------------------------------------


tools = SentinelToolNode(
    tools={
        "lookup_order": (lookup_order, {"produces_evidence": True}),
        "validate_refund": (validate_refund, {"produces_evidence": True}),
        "issue_refund": (
            issue_refund,
            {
                "is_commit": True,
                "requires": ["lookup_order"],
                "grounding_rules": {
                    "amount": {"source_action": "lookup_order", "source_field": "amount"},
                },
            },
        ),
    }
)


# ---------------------------------------------------------------------------
# Demo: run two scenarios through the node directly
# ---------------------------------------------------------------------------


def _tool_call(name: str, args: Dict[str, Any], tc_id: str) -> Dict[str, Any]:
    return {
        "tool_calls": [{"name": name, "args": args, "id": tc_id}],
    }


def _state(name: str, args: Dict[str, Any], tc_id: str) -> Dict[str, Any]:
    return {"messages": [_tool_call(name, args, tc_id)]}


def main() -> None:
    print("=" * 72)
    print("Scenario A: agent tries issue_refund WITHOUT looking up the order")
    print("=" * 72)
    state = _state("issue_refund", {"order_id": "ord-123", "amount": 49.99}, "tc-1")
    result = tools(state)
    for msg in result["messages"]:
        content = msg.content if hasattr(msg, "content") else msg["content"]
        print("TOOL RESPONSE:", content)

    print()
    print("=" * 72)
    print("Scenario B: proper sequence — lookup, then refund")
    print("=" * 72)
    # Step 1: lookup
    print("STEP 1 (lookup):", tools(_state("lookup_order", {"order_id": "ord-123"}, "tc-2"))["messages"][0])

    # Step 2: refund with matching amount
    print("STEP 2 (refund):", tools(_state("issue_refund", {"order_id": "ord-123", "amount": 49.99}, "tc-3"))["messages"][0])

    # Step 3: attempt refund with wrong amount — groundedness guardrail blocks
    print("STEP 3 (ungrounded amount):", tools(_state("issue_refund", {"order_id": "ord-123", "amount": 999.00}, "tc-4"))["messages"][0])

    print()
    print("=" * 72)
    print("Interventions recorded:")
    print("=" * 72)
    for iv in InterventionTracker.get_interventions(limit=10):
        print(f"  - {iv.intervention_type.value} ({iv.outcome.value}): {iv.reason}")


if __name__ == "__main__":
    main()
