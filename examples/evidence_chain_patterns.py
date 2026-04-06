"""
Evidence Chain Patterns

Demonstrates the @sentinel_tool decorator for building correct agent tool chains.
No framework dependency -- works with any Python agent.

Usage:
    python examples/evidence_chain_patterns.py
"""
from agent_sentinel.integrations.tools import sentinel_tool
from agent_sentinel.evidence import EvidenceTracker
from agent_sentinel.errors import EvidenceViolationError


# ===================================================================
# Pattern 1: Simple prerequisite chain
# ===================================================================

@sentinel_tool(name="lookup_order", produces_evidence=True)
def lookup_order(order_id: str) -> dict:
    """Read-only lookup -- produces evidence that order exists."""
    return {"order_id": order_id, "amount": 99.99, "status": "delivered"}


@sentinel_tool(
    name="issue_refund",
    is_commit=True,
    requires=["lookup_order"],
)
def issue_refund(order_id: str, amount: float) -> dict:
    """Commit action -- requires prior lookup evidence."""
    return {"refund_id": "REF-001", "order_id": order_id, "amount": amount}


# ===================================================================
# Pattern 2: Multi-step prerequisite chain
# ===================================================================

@sentinel_tool(name="verify_identity", produces_evidence=True)
def verify_identity(user_id: str) -> dict:
    return {"user_id": user_id, "verified": True}


@sentinel_tool(name="check_balance", produces_evidence=True)
def check_balance(account_id: str) -> dict:
    return {"account_id": account_id, "balance": 5000.00}


@sentinel_tool(
    name="transfer_funds",
    is_commit=True,
    requires=["verify_identity", "check_balance"],
)
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    return {"transfer_id": "TXN-001", "status": "completed"}


# ===================================================================
# Pattern 3: Groundedness -- commit args must match evidence
# ===================================================================

@sentinel_tool(name="search_product", produces_evidence=True)
def search_product(query: str) -> dict:
    return {"product_id": "PROD-42", "name": "Widget", "price": 29.99}


@sentinel_tool(
    name="add_to_cart",
    is_commit=True,
    requires=["search_product"],
    grounding_rules={
        "product_id": {"source_action": "search_product", "source_field": "product_id"},
    },
)
def add_to_cart(product_id: str, quantity: int) -> dict:
    return {"cart_item": product_id, "quantity": quantity}


# ===================================================================
# Demo
# ===================================================================

def demo():
    print("=" * 60)
    print("Evidence Chain Patterns Demo")
    print("=" * 60)

    # -- Pattern 1: Simple chain ------------------------------------
    print("\n-- Pattern 1: Simple prerequisite chain --")
    EvidenceTracker.reset_session()

    print("  Attempting refund without lookup...")
    try:
        issue_refund(order_id="ORD-001", amount=50.0)
        print("  ERROR: Should have been blocked!")
    except EvidenceViolationError as e:
        print(f"  Blocked: {e.remediation.reason_code}")
        print(f"  Missing: {e.remediation.missing_requirements}")
        print(f"  Guidance: {e.remediation.retry_guidance}")

    print("  Looking up order...")
    result = lookup_order(order_id="ORD-001")
    print(f"  Evidence: {result}")

    print("  Retrying refund...")
    result = issue_refund(order_id="ORD-001", amount=50.0)
    print(f"  Success: {result}")

    # -- Pattern 2: Multi-step chain --------------------------------
    print("\n-- Pattern 2: Multi-step prerequisite chain --")
    EvidenceTracker.reset_session()

    print("  Attempting transfer without prerequisites...")
    try:
        transfer_funds(from_account="ACC-1", to_account="ACC-2", amount=100.0)
    except EvidenceViolationError as e:
        print(f"  Blocked -- missing: {e.remediation.missing_requirements}")

    print("  Verifying identity...")
    verify_identity(user_id="USER-1")

    print("  Attempting transfer (still missing balance check)...")
    try:
        transfer_funds(from_account="ACC-1", to_account="ACC-2", amount=100.0)
    except EvidenceViolationError as e:
        print(f"  Blocked -- missing: {e.remediation.missing_requirements}")

    print("  Checking balance...")
    check_balance(account_id="ACC-1")

    print("  Retrying transfer (all prerequisites met)...")
    result = transfer_funds(from_account="ACC-1", to_account="ACC-2", amount=100.0)
    print(f"  Success: {result}")

    # -- Pattern 3: Groundedness ------------------------------------
    print("\n-- Pattern 3: Groundedness (args match evidence) --")
    EvidenceTracker.reset_session()

    print("  Searching for product...")
    search_product(query="widget")

    print("  Adding correct product to cart...")
    result = add_to_cart(product_id="PROD-42", quantity=1)
    print(f"  Success: {result}")

    print("  Trying to add WRONG product (not in evidence)...")
    try:
        add_to_cart(product_id="PROD-FAKE", quantity=1)
    except EvidenceViolationError as e:
        print(f"  Blocked: {e.remediation.reason_code}")
        print(f"  Violations: {e.remediation.argument_violations}")

    print("\n" + "=" * 60)
    print("All patterns demonstrated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
