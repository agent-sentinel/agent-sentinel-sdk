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
from datetime import datetime
from typing import Optional, Any, Callable, Literal

from .ledger import Ledger
from .cost import CostTracker
from .policy import PolicyEngine
from .context import ExecutionContext
from .errors import BudgetExceededError, PolicyViolationError, EvidenceViolationError
from .approval import ApprovalClient
from .redaction import redact_inputs, ledger_input_capture_enabled

logger = logging.getLogger("agent_sentinel")


class _PlatformApprovalAdapter:
    """Adapts approval.ApprovalResponse to the interface expected by _record_approval_intervention."""

    def __init__(self, platform_resp):
        from .compliance import ApprovalStatus as ComplianceApprovalStatus
        # Map status
        status_str = platform_resp.status.value if hasattr(platform_resp.status, "value") else str(platform_resp.status)
        try:
            self.status = ComplianceApprovalStatus(status_str)
        except ValueError:
            self.status = ComplianceApprovalStatus.PENDING
        self.approver_email = platform_resp.decided_by_email
        self.approver_id = platform_resp.decided_by_email
        self.approved_at = platform_resp.decided_at
        self.notes = platform_resp.decision_notes


def _reason_code_to_intervention() -> dict:
    """
    Map a RemediationPayload reason_code to an InterventionType.

    Built lazily so intervention.py stays imported inside the recording
    functions rather than becoming a load-time import cycle.
    """
    from .intervention import InterventionType

    return {
        "RATE_LIMITED": InterventionType.RATE_LIMITED,
        "BUDGET_EXCEEDED": InterventionType.BUDGET_EXCEEDED,
        "DENIED_ACTION": InterventionType.HARD_BLOCK,
        "NOT_IN_ALLOWLIST": InterventionType.HARD_BLOCK,
        "POLICY_NOT_CONFIGURED": InterventionType.HARD_BLOCK,
        "EVALUATION_ERROR": InterventionType.HARD_BLOCK,
        "MISSING_EVIDENCE": InterventionType.MISSING_EVIDENCE,
        "STALE_EVIDENCE": InterventionType.MISSING_EVIDENCE,
        "ARGUMENT_CONSTRAINT_VIOLATED": InterventionType.MISSING_EVIDENCE,
        "UNGROUNDED_ARGUMENT": InterventionType.MISSING_EVIDENCE,
        # Built-in guardrails.
        "PII_DETECTED": InterventionType.PII_BLOCKED,
        "CONTENT_MODERATED": InterventionType.CONTENT_BLOCKED,
        "LOOP_DETECTED": InterventionType.LOOP_DETECTED,
    }


def _reason_code_of(error) -> "str | None":
    """
    Read the structured reason code wherever it lives.

    Guardrails put it at details["reason_code"]; policy errors also expose it
    on .remediation. Checking both keeps one dispatch path for all of them.
    """
    remediation = getattr(error, "remediation", None)
    code = getattr(remediation, "reason_code", None)
    if code:
        return code
    details = getattr(error, "details", None)
    return details.get("reason_code") if isinstance(details, dict) else None


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
            "funds_protected": error.limit,
            "cost_prevented": cost,
        }
    else:
        # PolicyViolationError - could be deny list, rate limit, PII/moderation/loop, etc.
        # Dispatch on the structured reason_code rather than sniffing the
        # human-readable message: rewording a message must not silently
        # reclassify every rate-limit block as a hard block.
        intervention_type = _reason_code_to_intervention().get(
            _reason_code_of(error), InterventionType.HARD_BLOCK
        )
        reason = str(error)
        blast_radius = {"funds_protected": cost}

    # Build structured context for guardrail-type-specific UI rendering.
    # Versioned payload so the console can dispatch to a typed detail panel
    # instead of rendering generic intervention text.
    context: Optional[dict[str, Any]] = None
    if isinstance(error, PolicyViolationError) and isinstance(error.details, dict):
        details = error.details
        reason_code = details.get("reason_code")
        if reason_code == "PII_DETECTED":
            context = {
                "guardrail_schema_version": 1,
                "guardrail": "pii",
                "matches": details.get("matches", []),
                "categories": details.get("categories", []),
                "fields": details.get("fields", []),
            }
        elif reason_code == "CONTENT_MODERATED":
            context = {
                "guardrail_schema_version": 1,
                "guardrail": "moderation",
                "categories": details.get("categories", []),
                "fields": details.get("fields", []),
                "results": details.get("results", []),
            }
        elif reason_code == "LOOP_DETECTED":
            detection = details.get("detection", {}) or {}
            context = {
                "guardrail_schema_version": 1,
                "guardrail": "loop",
                "arg_hash": detection.get("arg_hash"),
                "occurrences": detection.get("count") or details.get("prior_attempts"),
                "window_seconds": detection.get("window_seconds"),
                "break_out_hint": detection.get("break_out_hint") or details.get("retry_guidance"),
            }

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
        run_id=ctx.run_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="high" if cost > 1.0 else "medium",
        context=context,
        # Forward the structured payload so deny-list, allowlist, rate-limit
        # and budget blocks carry a reason_code and guidance into the platform,
        # the same way evidence blocks already do.
        remediation_payload=(
            error.remediation.to_dict()
            if getattr(error, "remediation", None) is not None
            else None
        ),
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

    # Calculate blast radius - cost prevented if rejected
    blast_radius = {"funds_protected": cost if outcome == InterventionOutcome.REJECTED_AFTER_REVIEW else 0.0}

    # Record the intervention
    InterventionTracker.record(
        intervention_type=InterventionType.APPROVAL_REQUIRED,
        outcome=outcome,
        action_name=action_name,
        estimated_cost=cost,
        actual_cost=actual_cost,
        reason=reason,
        blast_radius=blast_radius,
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        run_id=ctx.run_id if ctx else None,
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
        blast_radius={"funds_protected": cost},
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        run_id=ctx.run_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="medium",
        remediation_payload=error.remediation.to_dict(),
    )


def _resolve_idempotency_key(
    key: Any,
    args: tuple,
    kwargs: dict,
) -> Optional[str]:
    """
    Resolve an idempotency key argument on @guarded_action.

    Accepts:
      - None -> returns None (idempotency disabled)
      - str  -> returned as-is (static key)
      - callable(*args, **kwargs) -> invoked; returned value stringified

    Failures in the callable are swallowed (we never break user execution
    because of a bad key factory); None is returned in that case.
    """
    if key is None:
        return None
    if isinstance(key, str):
        return key or None
    if callable(key):
        try:
            resolved = key(*args, **kwargs)
        except Exception:
            return None
        return str(resolved) if resolved else None
    # Anything else: try to stringify.
    return str(key) or None


def _record_idempotency_intervention(
    action_name: str,
    key: str,
    cost: float,
    args: tuple,
    kwargs: dict,
    first_seen_at: datetime,
    ttl_seconds: float,
) -> None:
    """Record an intervention when a cached idempotency result is returned."""
    from .intervention import InterventionTracker, InterventionType, InterventionOutcome
    ctx = ExecutionContext.current()
    InterventionTracker.record(
        intervention_type=InterventionType.IDEMPOTENT_REPLAY,
        outcome=InterventionOutcome.MODIFIED,  # action was replaced with cached result
        action_name=action_name,
        estimated_cost=cost,
        actual_cost=0.0,
        reason=f"Idempotency key {key!r} was seen previously; returning cached result",
        blast_radius={"funds_protected": cost},
        original_inputs={"args": args, "kwargs": kwargs},
        agent_id=ctx.agent_id if ctx else None,
        run_id=ctx.run_id if ctx else None,
        task_id=ctx.task_id if ctx else None,
        mission_id=ctx.mission_id if ctx else None,
        risk_level="low",
        context={
            "guardrail_schema_version": 1,
            "guardrail": "idempotency",
            "key": key,
            "first_seen_at": first_seen_at.isoformat(),
            "ttl_seconds": ttl_seconds,
            "cost_saved_usd": cost,
        },
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
    risk_level: Optional[str] = None,
    idempotency_key: Optional[Any] = None,
    idempotency_ttl_seconds: Optional[float] = None,
    fail_closed: Optional[bool] = None,
    impact: Literal["read", "low", "medium", "high"] = "medium",
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
        fail_closed: Per-action enforcement override, forwarded to the kill
            switch check and PolicyEngine.check_and_reserve(). True forces
            "fail_closed" for this action; False forces "fail_open"; None
            (default) defers to PolicyEngine.get_enforcement_mode()
            (set_enforcement_mode() / AGENT_SENTINEL_ENFORCEMENT / default,
            which is "fail_closed_writes" as of 0.4.0).
        impact: Declared impact of this action: "read", "low", "medium"
            (default), or "high". Sent to the platform in the ledger/ingest
            entry, and consulted locally by PolicyEngine whenever it cannot
            vouch for the action (unconfigured, unreachable with no cache,
            or an evaluation error) under the default "fail_closed_writes"
            mode: "medium"/"high" block with NoPolicyForWriteError,
            "read"/"low" proceed with a warning. Declare read-only actions
            with impact="read" or "low" so they keep working without a
            policy loaded.

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
                grounding_rules, risk_level,
                idempotency_key, idempotency_ttl_seconds,
                fail_closed, impact,
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
                grounding_rules, risk_level,
                idempotency_key, idempotency_ttl_seconds,
                fail_closed, impact,
                *args, **kwargs
            )

        wrapper = async_wrapper if is_async else sync_wrapper
        # Tag for discovery by sentinel_tool and auto_register_tools
        wrapper._sentinel_guarded = True  # type: ignore[attr-defined]
        wrapper._sentinel_config = {  # type: ignore[attr-defined]
            "name": action_name,
            "cost_usd": cost_usd,
            "tags": tags,
            "produces_evidence": produces_evidence,
            "is_commit": is_commit,
            "requires": requires,
            "argument_constraints": argument_constraints,
            "evidence_max_age_seconds": evidence_max_age_seconds,
            "grounding_rules": grounding_rules,
            "impact": impact,
        }
        return wrapper

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
    risk_level: Optional[str] = None,
    idempotency_key: Optional[Any] = None,
    idempotency_ttl_seconds: Optional[float] = None,
    fail_closed: Optional[bool] = None,
    impact: str = "medium",
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
                cost, 0.0, "replayed", tags or [], None, agent_id, task_id, mission_id,
                impact=impact,
            )
            
            return recorded_output
            
        except Exception as e:
            # If replay fails, log and re-raise
            logger.error(f"Replay failed for '{action_name}': {e}")
            raise

    # Phase 7 A3: Idempotency lookup before policy/approval checks.
    # If the caller supplied an idempotency_key and the cache has a live
    # entry, return the cached result — skipping approval, policy, budget,
    # and execution entirely (idempotency primitive).
    _idem_key = _resolve_idempotency_key(idempotency_key, args, kwargs)
    if _idem_key is not None:
        from .guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE, IdempotencyHit
        ctx = ExecutionContext.current()
        run_id = ctx.run_id if ctx else None
        try:
            DEFAULT_IDEMPOTENCY_CACHE.lookup(run_id, _idem_key, action_name)
        except IdempotencyHit as hit:
            _record_idempotency_intervention(
                action_name,
                _idem_key,
                cost,
                args,
                kwargs,
                first_seen_at=hit.first_seen_at,
                ttl_seconds=hit.ttl_seconds,
            )
            _safe_log(
                action_name, args, kwargs, hit.value, None,
                cost, 0.0, "idempotent_replay", tags or [],
                None, agent_id, task_id, mission_id, impact=impact,
            )
            return hit.value

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

    # Check decorator-level approval (legacy HumanApprovalHandler)
    if requires_approval:
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING

        try:
            approval_response = await HumanApprovalHandler._class_request_approval_async(
                action_name=action_name,
                action_description=approval_description,
                inputs={"args": args, "kwargs": kwargs},
                cost=cost
            )

            compliance_metadata.approval_status = approval_response.status
            compliance_metadata.human_in_the_loop_id = approval_response.approver_id
            compliance_metadata.human_in_the_loop_email = approval_response.approver_email
            compliance_metadata.approval_timestamp = approval_response.approved_at
            compliance_metadata.approval_notes = approval_response.notes
            _record_approval_intervention(action_name, cost, approval_response, args, kwargs)

            if approval_response.status != ApprovalStatus.APPROVED:
                logger.warning(f"Action '{action_name}' not approved: {approval_response.status}")
                status_str = approval_response.status.value if hasattr(approval_response.status, "value") else str(approval_response.status)
                _safe_log(
                    action_name, args, kwargs, None,
                    f"Approval {status_str}: {approval_response.notes or approval_response.status}",
                    cost, 0.0, "rejected" if status_str == "rejected" else "blocked", tags,
                    compliance_metadata.to_dict() if compliance_metadata else None,
                    agent_id, task_id, mission_id, impact=impact,
                )
                clear_compliance_metadata()
                raise PolicyViolationError(
                    f"Human approval required but status is: {approval_response.status}"
                )

        except RuntimeError as e:
            logger.error(f"Human approval required but no handler configured for '{action_name}'")
            raise PolicyViolationError(str(e))

    # Check platform policy-based approval (ApprovalClient)
    elif PolicyEngine.requires_approval(action_name, cost=cost, tags=tags, risk_level=risk_level):
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING

        if not ApprovalClient.is_configured():
            raise PolicyViolationError(
                f"Platform policy requires approval for '{action_name}' but ApprovalClient is not configured. "
                "Call ApprovalClient.configure() with platform_url and api_token."
            )

        ctx = ExecutionContext.current()
        platform_resp = await ApprovalClient.request_approval_async(
            action_name=action_name,
            action_description=approval_description,
            agent_id=agent_id or (ctx.agent_id if ctx else None),
            run_id=getattr(ctx, "run_id", None) if ctx else None,
            task_id=task_id or (ctx.task_id if ctx else None),
            mission_id=mission_id or (ctx.mission_id if ctx else None),
            estimated_cost=cost,
            timeout_seconds=PolicyEngine.get_approval_timeout(),
            action_inputs={"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
            context={"tags": tags or [], "risk_level": risk_level},
        )

        status_str = platform_resp.status.value if hasattr(platform_resp.status, "value") else str(platform_resp.status)
        try:
            compliance_metadata.approval_status = ApprovalStatus(status_str)
        except ValueError:
            compliance_metadata.approval_status = ApprovalStatus.PENDING
        compliance_metadata.human_in_the_loop_email = platform_resp.decided_by_email
        compliance_metadata.approval_timestamp = platform_resp.decided_at
        compliance_metadata.approval_notes = platform_resp.decision_notes

        approval_response = _PlatformApprovalAdapter(platform_resp)
        _record_approval_intervention(action_name, cost, approval_response, args, kwargs)

        if status_str != "approved":
            logger.warning(f"Action '{action_name}' not approved by platform: {platform_resp.status}")
            outcome = "rejected" if status_str == "rejected" else "blocked"
            _safe_log(
                action_name, args, kwargs, None,
                f"Approval {status_str}: {platform_resp.decision_notes or platform_resp.status}",
                cost, 0.0, outcome, tags,
                compliance_metadata.to_dict() if compliance_metadata else None,
                agent_id, task_id, mission_id, impact=impact,
            )
            clear_compliance_metadata()
            raise PolicyViolationError(
                f"Platform approval required but status is: {platform_resp.status}"
            )

    # Set compliance metadata for the action context
    set_compliance_metadata(compliance_metadata)

    # Kill switch check (fail-open by default: only blocks when kill is
    # confirmed; fail_closed also blocks when the check itself cannot be
    # completed — see KillSwitchClient.is_killed).
    from .kill_switch import KillSwitchClient
    if KillSwitchClient.is_configured():
        KillSwitchClient.check_or_raise(
            agent_id=agent_id, run_id=ExecutionContext.get_run_id(), mission_id=mission_id,
            fail_closed=PolicyEngine.resolve_enforcement_mode(fail_closed) == "fail_closed",
        )

    # Phase 2: Check policy BEFORE execution (includes evidence requirements from policy)
    # This may raise BudgetExceededError, PolicyViolationError, or EvidenceViolationError
    try:
        PolicyEngine.check_and_reserve(
            action_name, cost, kwargs=kwargs, fail_closed=fail_closed, impact=impact
        )
    except EvidenceViolationError as e:
        logger.warning(f"Evidence requirement blocked action '{action_name}': {e}")
        _record_evidence_intervention(action_name, cost, e, args, kwargs)
        _safe_log(
            action_name, args, kwargs, None, str(e),
            cost, 0.0, "blocked", tags,
            compliance_metadata.to_dict() if compliance_metadata else None,
            agent_id, task_id, mission_id, impact=impact,
        )
        clear_compliance_metadata()
        raise
    except (BudgetExceededError, PolicyViolationError) as e:
        logger.warning(f"Policy blocked action '{action_name}': {e}")
        _record_policy_intervention(action_name, cost, e, args, kwargs)
        _safe_log(
            action_name, args, kwargs, None, str(e),
            cost, 0.0, "blocked", tags,
            compliance_metadata.to_dict() if compliance_metadata else None,
            agent_id, task_id, mission_id, impact=impact,
        )
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
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags,
                compliance_metadata.to_dict() if compliance_metadata else None,
                agent_id, task_id, mission_id, impact=impact,
            )
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
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags,
                compliance_metadata.to_dict() if compliance_metadata else None,
                agent_id, task_id, mission_id, impact=impact,
            )
            clear_compliance_metadata()
            raise error

    # Decorator-level argument constraint check
    if argument_constraints and kwargs:
        from .constraints import validate_constraints
        violations = validate_constraints(kwargs, argument_constraints)
        if violations:
            error = EvidenceViolationError(
                message=f"Action '{action_name}' argument constraints violated: {violations}",
                action_name=action_name,
                argument_violations=violations,
                retry_guidance="Fix the argument values to satisfy constraints",
            )
            error.remediation.reason_code = "ARGUMENT_CONSTRAINT_VIOLATED"
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags,
                compliance_metadata.to_dict() if compliance_metadata else None,
                agent_id, task_id, mission_id, impact=impact,
            )
            clear_compliance_metadata()
            raise error

    start_ns = time.perf_counter_ns()
    outcome = "success"
    error_message = None
    result = None

    try:
        # Execute the actual async function
        result = await func(*args, **kwargs)
        # Phase 7 A3: Store successful result in idempotency cache for future replays.
        if _idem_key is not None:
            from .guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE
            _ctx_store = ExecutionContext.current()
            DEFAULT_IDEMPOTENCY_CACHE.store(
                _ctx_store.run_id if _ctx_store else None,
                _idem_key,
                result,
                idempotency_ttl_seconds,
            )
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

        # Cost was already recorded by check_and_reserve above — reserving is
        # what makes the budget a real cap, and it records as it reserves.
        # Adding here too would double-count every spend.

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
            impact=impact,
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
    risk_level: Optional[str] = None,
    idempotency_key: Optional[Any] = None,
    idempotency_ttl_seconds: Optional[float] = None,
    fail_closed: Optional[bool] = None,
    impact: str = "medium",
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
                cost, 0.0, "replayed", tags or [], None, agent_id, task_id, mission_id,
                impact=impact,
            )
            
            return recorded_output
            
        except Exception as e:
            # If replay fails, log and re-raise
            logger.error(f"Replay failed for '{action_name}': {e}")
            raise

    # Phase 7 A3: Idempotency lookup (sync path) — see _execute_async for docs.
    _idem_key = _resolve_idempotency_key(idempotency_key, args, kwargs)
    if _idem_key is not None:
        from .guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE, IdempotencyHit
        ctx = ExecutionContext.current()
        run_id = ctx.run_id if ctx else None
        try:
            DEFAULT_IDEMPOTENCY_CACHE.lookup(run_id, _idem_key, action_name)
        except IdempotencyHit as hit:
            _record_idempotency_intervention(
                action_name,
                _idem_key,
                cost,
                args,
                kwargs,
                first_seen_at=hit.first_seen_at,
                ttl_seconds=hit.ttl_seconds,
            )
            _safe_log(
                action_name, args, kwargs, hit.value, None,
                cost, 0.0, "idempotent_replay", tags or [],
                None, agent_id, task_id, mission_id, impact=impact,
            )
            return hit.value

    from .compliance import (
        HumanApprovalHandler,
        ComplianceMetadata,
        ApprovalStatus,
        set_compliance_metadata,
        clear_compliance_metadata
    )

    compliance_metadata = ComplianceMetadata()
    approval_response = None

    # Kill switch check (fail-open by default: only blocks when kill is
    # confirmed; fail_closed also blocks when the check itself cannot be
    # completed — see KillSwitchClient.is_killed).
    from .kill_switch import KillSwitchClient
    if KillSwitchClient.is_configured():
        KillSwitchClient.check_or_raise(
            agent_id=agent_id, run_id=ExecutionContext.get_run_id(), mission_id=mission_id,
            fail_closed=PolicyEngine.resolve_enforcement_mode(fail_closed) == "fail_closed",
        )

    # Hard blocks FIRST — deny lists, budgets, evidence, constraints, groundedness.
    # These take priority over approval (no point approving a blocked action).
    try:
        PolicyEngine.check_and_reserve(
            action_name, cost, kwargs=kwargs, fail_closed=fail_closed, impact=impact
        )
    except EvidenceViolationError as e:
        logger.warning(f"Evidence requirement blocked action '{action_name}': {e}")
        _record_evidence_intervention(action_name, cost, e, args, kwargs)
        _safe_log(
            action_name, args, kwargs, None, str(e),
            cost, 0.0, "blocked", tags or [],
            None, agent_id, task_id, mission_id, impact=impact,
        )
        raise
    except (BudgetExceededError, PolicyViolationError) as e:
        logger.warning(f"Policy blocked action '{action_name}': {e}")
        _record_policy_intervention(action_name, cost, e, args, kwargs)
        _safe_log(
            action_name, args, kwargs, None, str(e),
            cost, 0.0, "blocked", tags or [],
            None, agent_id, task_id, mission_id, impact=impact,
        )
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
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags or [],
                None, agent_id, task_id, mission_id, impact=impact,
            )
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
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags or [],
                None, agent_id, task_id, mission_id, impact=impact,
            )
            raise error

    # Decorator-level argument constraint check
    if argument_constraints and kwargs:
        from .constraints import validate_constraints
        violations = validate_constraints(kwargs, argument_constraints)
        if violations:
            error = EvidenceViolationError(
                message=f"Action '{action_name}' argument constraints violated: {violations}",
                action_name=action_name,
                argument_violations=violations,
                retry_guidance="Fix the argument values to satisfy constraints",
            )
            error.remediation.reason_code = "ARGUMENT_CONSTRAINT_VIOLATED"
            _record_evidence_intervention(action_name, cost, error, args, kwargs)
            _safe_log(
                action_name, args, kwargs, None, str(error),
                cost, 0.0, "blocked", tags,
                None, agent_id, task_id, mission_id, impact=impact,
            )
            raise error

    # Approval check — only reached if all hard blocks passed
    if requires_approval:
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING

        try:
            approval_response = HumanApprovalHandler._class_request_approval_sync(
                action_name=action_name,
                action_description=approval_description,
                inputs={"args": args, "kwargs": kwargs},
                cost=cost
            )

            compliance_metadata.approval_status = approval_response.status
            compliance_metadata.human_in_the_loop_id = approval_response.approver_id
            compliance_metadata.human_in_the_loop_email = approval_response.approver_email
            compliance_metadata.approval_timestamp = approval_response.approved_at
            compliance_metadata.approval_notes = approval_response.notes
            _record_approval_intervention(action_name, cost, approval_response, args, kwargs)

            if approval_response.status != ApprovalStatus.APPROVED:
                logger.warning(f"Action '{action_name}' not approved: {approval_response.status}")
                raise PolicyViolationError(
                    f"Human approval required but status is: {approval_response.status}"
                )

        except RuntimeError as e:
            logger.error(f"Human approval required but no handler configured for '{action_name}'")
            raise PolicyViolationError(str(e))

    elif PolicyEngine.requires_approval(action_name, cost=cost, tags=tags, risk_level=risk_level):
        compliance_metadata.requires_human_approval = True
        compliance_metadata.approval_status = ApprovalStatus.PENDING

        if not ApprovalClient.is_configured():
            raise PolicyViolationError(
                f"Platform policy requires approval for '{action_name}' but ApprovalClient is not configured. "
                "Call ApprovalClient.configure() with platform_url and api_token."
            )

        ctx = ExecutionContext.current()
        platform_resp = ApprovalClient.request_approval_sync(
            action_name=action_name,
            action_description=approval_description,
            agent_id=agent_id or (ctx.agent_id if ctx else None),
            run_id=getattr(ctx, "run_id", None) if ctx else None,
            task_id=task_id or (ctx.task_id if ctx else None),
            mission_id=mission_id or (ctx.mission_id if ctx else None),
            estimated_cost=cost,
            timeout_seconds=PolicyEngine.get_approval_timeout(),
            action_inputs={"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
            context={"tags": tags or [], "risk_level": risk_level},
        )

        status_str = platform_resp.status.value if hasattr(platform_resp.status, "value") else str(platform_resp.status)
        try:
            compliance_metadata.approval_status = ApprovalStatus(status_str)
        except ValueError:
            compliance_metadata.approval_status = ApprovalStatus.PENDING
        compliance_metadata.human_in_the_loop_email = platform_resp.decided_by_email
        compliance_metadata.approval_timestamp = platform_resp.decided_at
        compliance_metadata.approval_notes = platform_resp.decision_notes

        approval_response = _PlatformApprovalAdapter(platform_resp)
        _record_approval_intervention(action_name, cost, approval_response, args, kwargs)

        if status_str != "approved":
            logger.warning(f"Action '{action_name}' not approved by platform: {platform_resp.status}")
            raise PolicyViolationError(
                f"Platform approval required but status is: {platform_resp.status}"
            )

    set_compliance_metadata(compliance_metadata)

    start_ns = time.perf_counter_ns()
    outcome = "success"
    error_message = None
    result = None

    try:
        # Execute the actual sync function
        result = func(*args, **kwargs)
        # Phase 7 A3: Store successful result in idempotency cache for future replays.
        if _idem_key is not None:
            from .guardrails.idempotency import DEFAULT_IDEMPOTENCY_CACHE
            _ctx_store = ExecutionContext.current()
            DEFAULT_IDEMPOTENCY_CACHE.store(
                _ctx_store.run_id if _ctx_store else None,
                _idem_key,
                result,
                idempotency_ttl_seconds,
            )
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

        # Cost was already recorded by check_and_reserve above — reserving is
        # what makes the budget a real cap, and it records as it reserves.
        # Adding here too would double-count every spend.

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
            impact=impact,
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
    run_id: Optional[str] = None,
    impact: str = "medium",
):
    """
    Isolate the logging logic to ensure fail-open behavior.

    This is the critical safety mechanism: if the ledger write fails,
    we log the error but NEVER crash the user's agent.
    """
    # Resolve run_id from ExecutionContext if not passed explicitly
    if run_id is None:
        ctx = ExecutionContext.current()
        if ctx:
            run_id = getattr(ctx, "run_id", None)

    try:
        if ledger_input_capture_enabled():
            redacted_args, redacted_kwargs, inputs_redacted = redact_inputs(args, kwargs)
            inputs = {"args": redacted_args, "kwargs": redacted_kwargs}
        else:
            inputs = {}
            inputs_redacted = False

        Ledger.record(
            action=action,
            inputs=inputs,
            outputs=result if outcome == "success" else error,
            cost_usd=cost,
            duration_ms=duration,
            outcome=outcome,
            tags=tags or [],
            compliance_metadata=compliance_metadata,
            agent_id=agent_id,
            task_id=task_id,
            mission_id=mission_id,
            run_id=run_id,
            inputs_redacted=inputs_redacted,
            impact=impact,
        )
    except Exception as e:
        # FAIL-OPEN: Log to stderr but never crash the agent
        logger.error(f"Agent Sentinel Ledger Failed: {e}")
        # In production, you might want to send this to a fallback monitoring service
