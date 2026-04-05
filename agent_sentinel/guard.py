"""
Guard Module: Decorator for wrapping agent actions with telemetry and cost tracking.

Phase 1 Implementation:
- Time tracking using time.perf_counter_ns() for high precision
- Try/except logic for proper error handling and fail-open behavior
- Local ledger writing without any network dependencies

Phase 2 Implementation:
- Cost tracking integration
- Policy engine checks before execution
- Budget enforcement with BudgetExceededError

Phase 4 Implementation:
- Replay mode support for mocking function execution
- Returns recorded outputs when in replay mode
- Detects input divergence during replay
"""
from __future__ import annotations

import functools
import inspect
import time
import logging
from typing import Optional, Any, Callable

from .ledger import Ledger
from .cost import CostTracker
from .policy import PolicyEngine
from .context import ExecutionContext
from .errors import BudgetExceededError, PolicyViolationError, EvidenceViolationError

logger = logging.getLogger("agent_sentinel")


def _record_policy_intervention(
    action_name: str,
    cost: float,
    error: Exception,
    args: tuple,
    kwargs: dict
) -> None:
    """
    Record an intervention when a policy blocks an action.
    
    This is the core value proposition - tracking where Sentinel said "no".
    """
    from .intervention import InterventionTracker, InterventionType, InterventionOutcome

    # Determine intervention type
    if isinstance(error, BudgetExceededError):
        intervention_type = InterventionType.BUDGET_EXCEEDED
        reason = str(error)
        blast_radius = {
            "budget_protected": error.limit,
            "cost_prevented": cost,
        }
    else:
        # PolicyViolationError - could be deny list, rate limit, etc.
        if "rate limit" in str(error).lower():
            intervention_type = InterventionType.RATE_LIMITED
        else:
            intervention_type = InterventionType.HARD_BLOCK
        reason = str(error)
        blast_radius = {"cost_prevented": cost}

    # Get current context for attribution
    ctx = ExecutionContext.current()

    # Record the intervention
    InterventionTracker.record(
        intervention_type=intervention_type,
        outcome=InterventionOutcome.BLOCKED,
        action_name=action_name,
        estimated_cost=cost,
        reason=reason,
        blast_radius=blast_radius,
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="high" if cost > 1.0 else "medium",
    )


def _record_approval_intervention(
    action_name: str,
    cost: float,
    approval_response: Any,
    args: tuple,
    kwargs: dict
) -> None:
    """
    Record an intervention when an action requires approval.
    
    This tracks escalations - where Sentinel paused and asked a human.
    """
    from .intervention import InterventionTracker, InterventionType, InterventionOutcome
    from .compliance import ApprovalStatus

    # Get current context for attribution
    ctx = ExecutionContext.current()

    # Determine outcome based on approval status
    if approval_response.status == ApprovalStatus.APPROVED:
        outcome = InterventionOutcome.APPROVED_AFTER_REVIEW
        actual_cost = cost  # Action was executed
        reason = f"Action required approval and was approved by {approval_response.approver_email or 'human reviewer'}"
    elif approval_response.status == ApprovalStatus.REJECTED:
        outcome = InterventionOutcome.REJECTED_AFTER_REVIEW
        actual_cost = 0.0
        reason = f"Action required approval and was rejected by {approval_response.approver_email or 'human reviewer'}"
    else:
        outcome = InterventionOutcome.ESCALATED
        actual_cost = 0.0
        reason = f"Action required approval, status: {approval_response.status}"

    # Record the intervention
    InterventionTracker.record(
        intervention_type=InterventionType.APPROVAL_REQUIRED,
        outcome=outcome,
        action_name=action_name,
        estimated_cost=cost,
        actual_cost=actual_cost,
        reason=reason,
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="medium",
        context={
            "approver_email": approval_response.approver_email,
            "approved_at": str(approval_response.approved_at) if approval_response.approved_at else None,
            "notes": approval_response.notes,
        }
    )


def _record_evidence_intervention(
    action_name: str,
    cost: float,
    error: EvidenceViolationError,
    args: tuple,
    kwargs: dict,
) -> None:
    """Record an intervention when an action is blocked due to missing evidence."""
    from .intervention import InterventionTracker, InterventionType, InterventionOutcome

    ctx = ExecutionContext.current()

    InterventionTracker.record(
        intervention_type=InterventionType.MISSING_EVIDENCE,
        outcome=InterventionOutcome.BLOCKED,
        action_name=action_name,
        estimated_cost=cost,
        reason=str(error),
        blast_radius={"cost_prevented": cost},
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="medium",
        remediation_payload=error.remediation.to_dict(),
    )


def guarded_action(
    name: Optional[str] = None,
    cost_usd: float = 0.0,
    tags: Optional[list[str]] = None,
    requires_human_approval: bool = False,
    approval_description: Optional[str] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    produces_evidence: bool = False,
    is_commit: bool = False,
    requires: Optional[list[str]] = None,
    argument_constraints: Optional[dict] = None,
    evidence_max_age_seconds: Optional[int] = None,
    grounding_rules: Optional[dict] = None,
):
    """
    Decorator to wrap an agent action (tool call, API request) with
    telemetry, cost tracking, and policy enforcement.
    
    Phase 1: Local Loop
    - Records timing information (start/end using perf_counter_ns)
    - Captures exceptions and re-raises them (fail-open behavior)
    - Writes to local ledger file
    - Works completely offline (no internet required)
    
    Phase 5: EU Compliance Foundation
    - Human-in-the-loop approval workflow (requires_human_approval=True)
    - Compliance metadata tracking for EU AI Act Article 14
    - Enterprise Tier feature foundation
    
    Usage:
        @guarded_action(name="send_email", cost_usd=0.005, tags=["email"])
        def send_email(to, body):
            # Your implementation
            pass
            
        @guarded_action(name="search_api", cost_usd=0.02)
        async def search_api(query):
            # Your async implementation
            pass
        
        # EU Compliance: Require human approval (Enterprise Tier)
        @guarded_action(
            name="transfer_funds",
            cost_usd=0.10,
            requires_human_approval=True,
            approval_description="Transfer funds from user account"
        )
        def transfer_funds(from_account, to_account, amount):
            # This will pause and wait for human approval
            pass
    
    Args:
        name: Optional custom name for the action (defaults to function name)
        cost_usd: Cost in USD for this action (default 0.0)
        tags: Optional list of tags for categorization
        requires_human_approval: If True, pauses execution and requests human approval (Enterprise)
        approval_description: Human-readable description shown in approval request
        
    Returns:
        Decorated function that records telemetry to local ledger
    """
    def decorator(func: Callable[..., Any]):
        action_name = name or func.__name__
        description = approval_description or f"Execute {action_name}"
        
        # Detect if the user's function is async (coroutine) or sync
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_async(
                func, action_name, cost_usd, tags,
                requires_human_approval, description,
                agent_id, task_id, mission_id,
                produces_evidence, is_commit, requires,
                argument_constraints, evidence_max_age_seconds,
                grounding_rules,
                *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_sync(
                func, action_name, cost_usd, tags,
                requires_human_approval, description,
                agent_id, task_id, mission_id,
                produces_evidence, is_commit, requires,
                argument_constraints, evidence_max_age_seconds,
                grounding_rules,
                *args, **kwargs
            )

        return async_wrapper if is_async else sync_wrapper
    
    return decorator


async def _execute_async(
    func: Callable,
    action_name: str,
    cost: float,
    tags: Optional[list[str]],
    requires_approval: bool,
    approval_description: str,
    agent_id: Optional[str],
    task_id: Optional[str],
    mission_id: Optional[str],
    produces_evidence: bool,
    is_commit: bool,
    requires: Optional[list[str]],
    argument_constraints: Optional[dict],
    evidence_max_age_seconds: Optional[int],
    grounding_rules: Optional[dict],
    *args,
    **kwargs
):
    """
    Execute an async function with timing and error tracking.

    This handles:
    1. Pre-execution: Human approval (if required) and policy checks
    2. Execution: Await the async function with timing (or use replay)
    3. Post-execution: Record cost and telemetry (even on failure)
    """
    # Resolve attribution from ExecutionContext if not set explicitly
    agent_id = agent_id or ExecutionContext.get_agent_id()
    task_id = task_id or ExecutionContext.get_task_id()
    mission_id = mission_id or ExecutionContext.get_mission_id()

    # Phase 4: Check if replay mode is active
    from .replay import ReplayMode
    
    if ReplayMode.is_active():
        # In replay mode, return recorded output instead of executing
        replay = ReplayMode.get_active()
        if replay is None:
            raise RuntimeError("Replay mode is active but no replay instance found")
        inputs = {"args": args, "kwargs": kwargs}

        try:
            recorded_output, _inputs_match = replay.get_next_output(action_name, inputs)
            
            # Record that we replayed this action
            _safe_log(
                action_name, args, kwargs, recorded_output, None,
                cost, 0.0, "replayed", tags or [], None, agent_id, task_id, mission_id
            )
            
            return recorded_output
            
        except Exception as e:
            # If replay fails, log and re-raise
            logger.error(f"Replay failed for '{action_name}': {e}")
            raise
    
    # Phase 5: Human-in-the-loop approval (EU AI Act Article 14)
    from .compliance import (
        HumanApprovalHandler, 
        ComplianceMetadata, 
        ApprovalStatus,
        set_compliance_metadata,
        clear_compliance_metadata
    )
    
    compliance_metadata = ComplianceMetadata()
    approval_response = None
    
    if requires_approval:
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING
        
        try:
            # Request human approval
            approval_response = await HumanApprovalHandler.request_approval_async(
                action_name=action_name,
                action_description=approval_description,
                inputs={"args": args, "kwargs": kwargs},
                cost=cost
            )
            
            # Update compliance metadata with approval result
            compliance_metadata.approval_status = approval_response.status
            compliance_metadata.human_in_the_loop_id = approval_response.approver_id
            compliance_metadata.human_in_the_loop_email = approval_response.approver_email
            compliance_metadata.approval_timestamp = approval_response.approved_at
            compliance_metadata.approval_notes = approval_response.notes
            
            # Record the approval intervention
            _record_approval_intervention(action_name, cost, approval_response, args, kwargs)
            
            # Block execution if not approved
            if approval_response.status != ApprovalStatus.APPROVED:
                logger.warning(f"Action '{action_name}' not approved: {approval_response.status}")
                raise PolicyViolationError(
                    f"Human approval required but status is: {approval_response.status}"
                )
                
        except RuntimeError as e:
            # No approval handler configured
            logger.error(f"Human approval required but no handler configured for '{action_name}'")
            raise PolicyViolationError(str(e))
    
    # Set compliance metadata for the action context
    set_compliance_metadata(compliance_metadata)

    # Kill switch check (fail-open: only blocks when kill is confirmed)
    from .kill_switch import KillSwitchClient
    if KillSwitchClient.is_configured():
        KillSwitchClient.check_or_raise(
            agent_id=agent_id, mission_id=mission_id
        )

    # Phase 2: Check policy BEFORE execution (includes evidence requirements from policy)
    # This may raise BudgetExceededError, PolicyViolationError, or EvidenceViolationError
    try:
        PolicyEngine.check_action(action_name, cost, kwargs=kwargs)
    except EvidenceViolationError as e:
        logger.warning(f"Evidence requirement blocked action '{action_name}': {e}")
        _record_evidence_intervention(action_name, cost, e, args, kwargs)
        clear_compliance_metadata()
        raise
    except (BudgetExceededError, PolicyViolationError) as e:
        logger.warning(f"Policy blocked action '{action_name}': {e}")
        _record_policy_intervention(action_name, cost, e, args, kwargs)
        clear_compliance_metadata()
        raise

    # Decorator-level evidence requirement check
    if requires:
        from .evidence import EvidenceTracker
        all_met, missing, stale = EvidenceTracker.check_requirements(requires, evidence_max_age_seconds)
        if not all_met:
            error = EvidenceViolationError(
                message=f"Action '{action_name}' requires prior execution of: {missing + stale}",
                action_name=action_name,
                missing_requirements=missing,
                required_prior_actions=requires,
                stale_evidence=stale,
            )
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            clear_compliance_metadata()
            raise error

    # Decorator-level groundedness check
    if grounding_rules and kwargs:
        from .evidence import EvidenceTracker
        grounded, ungrounded_details = EvidenceTracker.check_groundedness(
            action_kwargs=kwargs,
            grounding_rules=grounding_rules,
            max_age_seconds=evidence_max_age_seconds,
        )
        if not grounded:
            field_names = [d["field"] for d in ungrounded_details]
            error = EvidenceViolationError(
                message=f"Action '{action_name}' arguments not grounded in evidence: {field_names}",
                action_name=action_name,
                argument_violations=[
                    f"'{d['field']}' value '{d['actual_value']}' not found in {d['expected_source']}"
                    for d in ungrounded_details
                ],
                retry_guidance=f"Ensure these arguments match prior evidence: {field_names}",
            )
            error.remediation.reason_code = "UNGROUNDED_ARGUMENT"
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            clear_compliance_metadata()
            raise error

    start_ns = time.perf_counter_ns()
    outcome = "success"
    error_message = None
    result = None

    try:
        # Execute the actual async function
        result = await func(*args, **kwargs)
        return result

    except Exception as e:
        # Capture the error details
        outcome = "error"
        error_message = f"{type(e).__name__}: {str(e)}"
        # Re-raise to preserve user's error handling
        raise

    finally:
        # Always record telemetry and cost, even if function failed
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        # Phase 2: Record cost to tracker
        CostTracker.add_cost(action_name, cost)

        # Record evidence if this action produces it and succeeded
        if produces_evidence and outcome == "success":
            from .evidence import EvidenceTracker
            EvidenceTracker.record_evidence(action_name, kwargs=kwargs, result=result)

        # Phase 5: Include compliance metadata in logs if present
        from .compliance import get_compliance_metadata, clear_compliance_metadata
        compliance_meta = get_compliance_metadata()
        compliance_dict = compliance_meta.to_dict() if compliance_meta else None

        _safe_log(
            action_name, args, kwargs, result, error_message,
            cost,
            duration_ms,
            outcome,
            tags,
            compliance_dict,
            agent_id,
            task_id,
            mission_id,
        )

        # Clear compliance metadata after logging
        clear_compliance_metadata()


def _execute_sync(
    func: Callable,
    action_name: str,
    cost: float,
    tags: Optional[list[str]],
    requires_approval: bool,
    approval_description: str,
    agent_id: Optional[str],
    task_id: Optional[str],
    mission_id: Optional[str],
    produces_evidence: bool,
    is_commit: bool,
    requires: Optional[list[str]],
    argument_constraints: Optional[dict],
    evidence_max_age_seconds: Optional[int],
    grounding_rules: Optional[dict],
    *args,
    **kwargs
):
    """
    Execute a sync function with timing and error tracking.

    This handles:
    1. Pre-execution: Human approval (if required) and policy checks
    2. Execution: Call the sync function with timing (or use replay)
    3. Post-execution: Record cost and telemetry (even on failure)
    """
    # Resolve attribution from ExecutionContext if not set explicitly
    agent_id = agent_id or ExecutionContext.get_agent_id()
    task_id = task_id or ExecutionContext.get_task_id()
    mission_id = mission_id or ExecutionContext.get_mission_id()

    # Phase 4: Check if replay mode is active
    from .replay import ReplayMode
    
    if ReplayMode.is_active():
        # In replay mode, return recorded output instead of executing
        replay = ReplayMode.get_active()
        if replay is None:
            raise RuntimeError("Replay mode is active but no replay instance found")
        inputs = {"args": args, "kwargs": kwargs}

        try:
            recorded_output, _inputs_match = replay.get_next_output(action_name, inputs)
            
            # Record that we replayed this action
            _safe_log(
                action_name, args, kwargs, recorded_output, None,
                cost, 0.0, "replayed", tags or [], None, agent_id, task_id, mission_id
            )
            
            return recorded_output
            
        except Exception as e:
            # If replay fails, log and re-raise
            logger.error(f"Replay failed for '{action_name}': {e}")
            raise
    
    # Phase 5: Human-in-the-loop approval (EU AI Act Article 14)
    from .compliance import (
        HumanApprovalHandler, 
        ComplianceMetadata, 
        ApprovalStatus,
        set_compliance_metadata,
        clear_compliance_metadata
    )
    
    compliance_metadata = ComplianceMetadata()
    approval_response = None
    
    if requires_approval:
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING
        
        try:
            # Request human approval
            approval_response = HumanApprovalHandler.request_approval_sync(
                action_name=action_name,
                action_description=approval_description,
                inputs={"args": args, "kwargs": kwargs},
                cost=cost
            )
            
            # Update compliance metadata with approval result
            compliance_metadata.approval_status = approval_response.status
            compliance_metadata.human_in_the_loop_id = approval_response.approver_id
            compliance_metadata.human_in_the_loop_email = approval_response.approver_email
            compliance_metadata.approval_timestamp = approval_response.approved_at
            compliance_metadata.approval_notes = approval_response.notes
            
            # Record the approval intervention
            _record_approval_intervention(action_name, cost, approval_response, args, kwargs)
            
            # Block execution if not approved
            if approval_response.status != ApprovalStatus.APPROVED:
                logger.warning(f"Action '{action_name}' not approved: {approval_response.status}")
                raise PolicyViolationError(
                    f"Human approval required but status is: {approval_response.status}"
                )
                
        except RuntimeError as e:
            # No approval handler configured
            logger.error(f"Human approval required but no handler configured for '{action_name}'")
            raise PolicyViolationError(str(e))
    
    # Set compliance metadata for the action context
    set_compliance_metadata(compliance_metadata)

    # Kill switch check (fail-open: only blocks when kill is confirmed)
    from .kill_switch import KillSwitchClient
    if KillSwitchClient.is_configured():
        KillSwitchClient.check_or_raise(
            agent_id=agent_id, mission_id=mission_id
        )

    # Phase 2: Check policy BEFORE execution (includes evidence requirements from policy)
    try:
        PolicyEngine.check_action(action_name, cost, kwargs=kwargs)
    except EvidenceViolationError as e:
        logger.warning(f"Evidence requirement blocked action '{action_name}': {e}")
        _record_evidence_intervention(action_name, cost, e, args, kwargs)
        clear_compliance_metadata()
        raise
    except (BudgetExceededError, PolicyViolationError) as e:
        logger.warning(f"Policy blocked action '{action_name}': {e}")
        _record_policy_intervention(action_name, cost, e, args, kwargs)
        clear_compliance_metadata()
        raise

    # Decorator-level evidence requirement check
    if requires:
        from .evidence import EvidenceTracker
        all_met, missing, stale = EvidenceTracker.check_requirements(requires, evidence_max_age_seconds)
        if not all_met:
            error = EvidenceViolationError(
                message=f"Action '{action_name}' requires prior execution of: {missing + stale}",
                action_name=action_name,
                missing_requirements=missing,
                required_prior_actions=requires,
                stale_evidence=stale,
            )
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            clear_compliance_metadata()
            raise error

    # Decorator-level groundedness check
    if grounding_rules and kwargs:
        from .evidence import EvidenceTracker
        grounded, ungrounded_details = EvidenceTracker.check_groundedness(
            action_kwargs=kwargs,
            grounding_rules=grounding_rules,
            max_age_seconds=evidence_max_age_seconds,
        )
        if not grounded:
            field_names = [d["field"] for d in ungrounded_details]
            error = EvidenceViolationError(
                message=f"Action '{action_name}' arguments not grounded in evidence: {field_names}",
                action_name=action_name,
                argument_violations=[
                    f"'{d['field']}' value '{d['actual_value']}' not found in {d['expected_source']}"
                    for d in ungrounded_details
                ],
                retry_guidance=f"Ensure these arguments match prior evidence: {field_names}",
            )
            error.remediation.reason_code = "UNGROUNDED_ARGUMENT"
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            clear_compliance_metadata()
            raise error

    start_ns = time.perf_counter_ns()
    outcome = "success"
    error_message = None
    result = None

    try:
        # Execute the actual sync function
        result = func(*args, **kwargs)
        return result

    except Exception as e:
        # Capture the error details
        outcome = "error"
        error_message = f"{type(e).__name__}: {str(e)}"
        # Re-raise to preserve user's error handling
        raise

    finally:
        # Always record telemetry and cost, even if function failed
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        # Phase 2: Record cost to tracker
        CostTracker.add_cost(action_name, cost)

        # Record evidence if this action produces it and succeeded
        if produces_evidence and outcome == "success":
            from .evidence import EvidenceTracker
            EvidenceTracker.record_evidence(action_name, kwargs=kwargs, result=result)

        # Phase 5: Include compliance metadata in logs if present
        from .compliance import get_compliance_metadata, clear_compliance_metadata
        compliance_meta = get_compliance_metadata()
        compliance_dict = compliance_meta.to_dict() if compliance_meta else None

        _safe_log(
            action_name, args, kwargs, result, error_message,
            cost,
            duration_ms,
            outcome,
            tags,
            compliance_dict,
            agent_id,
            task_id,
            mission_id,
        )

        # Clear compliance metadata after logging
        clear_compliance_metadata()


def _safe_log(
    action: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    error: Optional[str],
    cost: float,
    duration: float,
    outcome: str,
    tags: Optional[list[str]],
    compliance_metadata: Optional[dict] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    mission_id: Optional[str] = None,
):
    """
    Isolate the logging logic to ensure fail-open behavior.
    
    This is the critical safety mechanism: if the ledger write fails,
    we log the error but NEVER crash the user's agent.
    
    Args:
        action: Name of the action being recorded
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        result: The return value (if successful)
        error: Error message (if failed)
        cost: Cost in USD
        duration: Duration in milliseconds
        outcome: "success" or "error"
        tags: Optional tags for categorization
        compliance_metadata: Optional EU compliance metadata dict (Enterprise Tier)
    """
    try:
        Ledger.record(
            action=action,
            inputs={"args": args, "kwargs": kwargs},
            outputs=result if outcome == "success" else error,
            cost_usd=cost,
            duration_ms=duration,
            outcome=outcome,
            tags=tags or [],
            compliance_metadata=compliance_metadata,
            agent_id=agent_id,
            task_id=task_id,
            mission_id=mission_id,
        )
    except Exception as e:
        # FAIL-OPEN: Log to stderr but never crash the agent
        logger.error(f"Agent Sentinel Ledger Failed: {e}")
        # In production, you might want to send this to a fallback monitoring service
