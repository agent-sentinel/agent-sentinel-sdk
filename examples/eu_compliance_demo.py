"""
EU Compliance Demo - Agent Sentinel

Demonstrates Enterprise Tier compliance features for EU AI Act:
- Human-in-the-loop approval workflow (Article 14)
- Decision rationale tracking
- Data lineage documentation
- Compliance metadata collection

This is a foundational implementation showing how to use the compliance features.
"""
from __future__ import annotations

import asyncio
from agent_sentinel import (
    guarded_action,
    HumanApprovalHandler,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    add_data_lineage,
    set_decision_rationale,
    set_model_card,
    start_run,
    flush_and_stop,
)


# ============================================================================
# Setup Human Approval Handler
# ============================================================================

def simple_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
    """
    Simple CLI-based approval handler.
    
    In production, this would:
    - Send request to AgentSentinel platform
    - Show notification in web UI
    - Wait for human approval via webhook
    - Return response with approver details
    """
    print("\n" + "=" * 60)
    print("🚨 HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"Action: {request.action_name}")
    print(f"Description: {request.action_description}")
    print(f"Estimated Cost: ${request.estimated_cost:.4f}")
    print(f"Risk Level: {request.risk_level.value}")
    print(f"Inputs: {request.inputs}")
    print("=" * 60)
    
    # Simulate approval decision
    user_input = input("Approve this action? (y/n): ").strip().lower()
    
    if user_input == 'y':
        notes = input("Optional approval notes: ").strip()
        return ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.APPROVED,
            approver_id="demo-user-123",
            approver_email="compliance@company.com",
            notes=notes if notes else "Approved via CLI demo"
        )
    else:
        reason = input("Rejection reason: ").strip()
        return ApprovalResponse(
            request_id=request.request_id,
            status=ApprovalStatus.REJECTED,
            approver_id="demo-user-123",
            approver_email="compliance@company.com",
            notes=reason or "Rejected by human oversight"
        )


# Register the approval handler
HumanApprovalHandler.set_approval_handler(simple_approval_handler)


# ============================================================================
# Example Actions with Compliance Tracking
# ============================================================================

@guarded_action(
    name="check_account_balance",
    cost_usd=0.001,
    tags=["finance", "read-only"]
)
def check_account_balance(account_id: str) -> dict:
    """
    Check account balance - low risk, no approval needed.
    Demonstrates data lineage tracking.
    """
    # Track data lineage for compliance
    add_data_lineage(
        source="PostgreSQL",
        version="v14.5",
        table="accounts",
        query_hash="sha256:abc123",
        environment="production"
    )
    
    # Track model usage
    set_model_card(
        model_name="gpt-4-turbo",
        version="2024-04-09",
        provider="OpenAI",
        system_prompt_hash="sha256:def456"
    )
    
    # Track decision rationale
    set_decision_rationale(
        f"User requested balance check for account {account_id}. "
        f"Verified user has permission. Queried production database.",
        confidence=0.98
    )
    
    # Simulate balance check
    balance = 4200.50
    return {
        "account_id": account_id,
        "balance": balance,
        "currency": "USD",
        "last_updated": "2024-01-15T10:30:00Z"
    }


@guarded_action(
    name="transfer_funds",
    cost_usd=0.10,
    tags=["finance", "high-risk"],
    requires_human_approval=True,
    approval_description="Transfer funds between accounts (high-risk operation)"
)
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    """
    Transfer funds - HIGH RISK, requires human approval (EU AI Act Article 14).
    
    This will pause execution and wait for human approval before proceeding.
    """
    # Track data sources
    add_data_lineage(
        source="PostgreSQL",
        version="v14.5",
        table="accounts",
        query="SELECT balance FROM accounts WHERE id IN (...)",
        environment="production"
    )
    
    add_data_lineage(
        source="Redis Cache",
        version="v7.0",
        key_pattern="fraud_check:*",
        environment="production"
    )
    
    # Track decision process
    set_decision_rationale(
        f"Initiating transfer of ${amount} from {from_account} to {to_account}. "
        f"Verified both accounts exist and have sufficient funds. "
        f"Fraud check passed. Awaiting human approval per EU AI Act Article 14.",
        confidence=0.95
    )
    
    # Track model
    set_model_card(
        model_name="gpt-4-turbo",
        version="2024-04-09",
        provider="OpenAI",
        temperature=0.0,
        system_prompt_hash="sha256:transfer_prompt_v2"
    )
    
    # Simulate transfer
    print(f"✅ Transfer approved and executed: ${amount} from {from_account} to {to_account}")
    
    return {
        "transaction_id": "txn_demo_123",
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "status": "completed",
        "timestamp": "2024-01-15T10:35:00Z",
        "approved_by": "compliance@company.com"
    }


@guarded_action(
    name="generate_financial_report",
    cost_usd=0.25,
    tags=["finance", "reporting", "medium-risk"],
    requires_human_approval=True,
    approval_description="Generate financial report with sensitive data"
)
def generate_financial_report(user_id: str, report_type: str) -> dict:
    """
    Generate financial report - requires approval for sensitive data access.
    """
    # Track comprehensive data lineage
    add_data_lineage(
        source="PostgreSQL",
        version="v14.5",
        table="transactions",
        row_count=1523,
        time_range="2024-01-01 to 2024-12-31"
    )
    
    add_data_lineage(
        source="BigQuery DataWarehouse",
        version="prod-2024-01",
        dataset="analytics.financial_summary",
        query_cost_usd=0.05
    )
    
    # Decision rationale
    set_decision_rationale(
        f"Generating {report_type} report for user {user_id}. "
        f"Data aggregated from 1,523 transactions across 2024. "
        f"Includes PII and financial details, requires human oversight.",
        confidence=0.92
    )
    
    # Model card
    set_model_card(
        model_name="claude-3-opus",
        version="20240229",
        provider="Anthropic",
        system_prompt_hash="sha256:report_gen_v3"
    )
    
    print(f"📊 Report generated: {report_type} for user {user_id}")
    
    return {
        "report_id": "rpt_demo_456",
        "report_type": report_type,
        "user_id": user_id,
        "generated_at": "2024-01-15T10:40:00Z",
        "pages": 15,
        "status": "completed"
    }


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run the EU compliance demo"""
    print("\n" + "=" * 60)
    print("EU AI Act Compliance Demo - Agent Sentinel")
    print("Enterprise Tier Features")
    print("=" * 60 + "\n")
    
    # Start a run for compliance tracking
    run_id = start_run()
    print(f"Started compliance-tracked run: {run_id}\n")
    
    try:
        # Example 1: Low-risk action (no approval needed)
        print("Example 1: Low-risk action (automatic data lineage tracking)")
        print("-" * 60)
        result1 = check_account_balance("ACC-123")
        print(f"✅ Balance check result: ${result1['balance']}\n")
        
        # Example 2: High-risk action (requires approval)
        print("\nExample 2: High-risk action (requires human approval)")
        print("-" * 60)
        try:
            result2 = transfer_funds("ACC-123", "ACC-456", 500.00)
            print(f"✅ Transfer result: {result2['status']}\n")
        except Exception as e:
            print(f"❌ Transfer blocked: {e}\n")
        
        # Example 3: Another high-risk action
        print("\nExample 3: Sensitive report generation (requires approval)")
        print("-" * 60)
        try:
            result3 = generate_financial_report("USER-789", "annual_summary")
            print(f"✅ Report generated: {result3['report_id']}\n")
        except Exception as e:
            print(f"❌ Report generation blocked: {e}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    
    finally:
        # Flush logs to platform
        print("\n" + "=" * 60)
        print("Flushing compliance logs to platform...")
        flush_and_stop()
        print("✅ Demo complete! Check AgentSentinel console for compliance report.")
        print("=" * 60 + "\n")
        
        print("📋 Compliance Features Demonstrated:")
        print("  ✅ Human-in-the-loop approval workflow (Article 14)")
        print("  ✅ Decision rationale tracking")
        print("  ✅ Data lineage documentation")
        print("  ✅ Model card tracking")
        print("  ✅ Compliance metadata collection")
        print("\n💡 Next Steps:")
        print("  1. View compliance report: /api/v1/compliance/export")
        print("  2. Check oversight log: /api/v1/compliance/oversight")
        print("  3. Verify compliance health: /api/v1/compliance/health")
        print()


if __name__ == "__main__":
    main()

