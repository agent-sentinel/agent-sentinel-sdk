from __future__ import annotations

from typing import Any
import logging

logger = logging.getLogger("agent_sentinel")


class AgentSentinelError(Exception):
    """Base exception for all Agent Sentinel errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ):
        """
        Initialize Agent Sentinel error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
            recoverable: Whether this error can be retried
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


class BudgetExceededError(AgentSentinelError):
    """Raised when an action or run exceeds the defined cost limit."""
    
    def __init__(
        self,
        message: str,
        spent: float,
        limit: float,
        budget_type: str = "action",
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize budget exceeded error.
        
        Args:
            message: Human-readable error message
            spent: Amount spent
            limit: Budget limit
            budget_type: Type of budget (action, run, session)
            details: Additional error context
        """
        self.spent = spent
        self.limit = limit
        self.budget_type = budget_type
        details = details or {}
        details.update({"spent": spent, "limit": limit, "budget_type": budget_type})

        # Format defensively: a None or non-numeric limit must not raise from
        # inside an exception constructor and mask the original budget error.
        try:
            amounts = (
                f"The {budget_type} budget of ${float(limit):.4f} USD is "
                f"exhausted (${float(spent):.4f} would be spent). "
            )
        except (TypeError, ValueError):
            amounts = f"The {budget_type} budget is exhausted. "

        self.remediation = RemediationPayload(
            reason_code="BUDGET_EXCEEDED",
            retry_guidance=amounts + _default_retry_guidance("BUDGET_EXCEEDED"),
        )
        details["remediation"] = self.remediation.to_dict()

        super().__init__(
            message=message,
            error_code="BUDGET_EXCEEDED",
            details=details,
            # Waiting or retrying does not refill a budget. The agent has to
            # change approach or stop.
            recoverable=False,
        )


class PolicyViolationError(AgentSentinelError):
    """
    Raised when an action violates a specific policy rule.

    Carries a RemediationPayload so a blocked agent receives guidance rather
    than a bare error string. Without it the agent sees only "denied" and
    typically retries the identical call, which cannot succeed. The payload
    names what to do instead: permitted alternatives for an allowlist miss, a
    wait time for a rate limit, a plain statement that a denied action will
    never succeed.
    """

    def __init__(
        self,
        message: str,
        policy_rule: str | None = None,
        action_name: str | None = None,
        details: dict[str, Any] | None = None,
        reason_code: str = "POLICY_VIOLATION",
        retry_guidance: str | None = None,
        safe_alternatives: list[str] | None = None,
        retry_after_seconds: float | None = None,
        recoverable: bool = False,
        error_code: str = "POLICY_VIOLATION",
    ):
        """
        Initialize policy violation error.

        Args:
            message: Human-readable error message
            policy_rule: Name of the violated policy rule
            action_name: Name of the action that violated policy
            details: Additional error context
            reason_code: Machine-readable cause, e.g. DENIED_ACTION,
                NOT_IN_ALLOWLIST, RATE_LIMITED
            retry_guidance: What the agent should do next. Auto-generated from
                reason_code when omitted.
            safe_alternatives: Actions the agent may call instead
            retry_after_seconds: For time-bounded blocks, when to try again
            recoverable: Whether retrying can ever succeed. True for rate
                limits (waiting works), False for deny lists (nothing works).
        """
        details = details or {}
        if policy_rule:
            details["policy_rule"] = policy_rule
        if action_name:
            details["action_name"] = action_name

        # The built-in guardrails (PII, moderation, loop) construct this error
        # with details={"reason_code": ...} rather than the kwarg. Honour that
        # so remediation.reason_code and details["reason_code"] cannot disagree.
        if reason_code == "POLICY_VIOLATION" and details.get("reason_code"):
            reason_code = details["reason_code"]
        details.setdefault("reason_code", reason_code)

        if not retry_guidance:
            retry_guidance = _default_retry_guidance(
                reason_code=reason_code,
                action_name=action_name,
                safe_alternatives=safe_alternatives,
                retry_after_seconds=retry_after_seconds,
            )

        self.remediation = RemediationPayload(
            reason_code=reason_code,
            retry_guidance=retry_guidance,
            safe_alternatives=safe_alternatives or [],
            retry_after_seconds=retry_after_seconds,
        )
        details["remediation"] = self.remediation.to_dict()

        AgentSentinelError.__init__(
            self,
            message=message,
            error_code=error_code,
            details=details,
            recoverable=recoverable,
        )


def _default_retry_guidance(
    reason_code: str,
    action_name: str | None = None,
    safe_alternatives: list[str] | None = None,
    retry_after_seconds: float | None = None,
) -> str:
    """
    Build agent-readable guidance for a block that has none supplied.

    Kept deliberately concrete: the text is fed back to a model as a tool
    result, so it must say what to do, not merely restate what happened.
    """
    name = f"'{action_name}'" if action_name else "This action"

    if reason_code == "DENIED_ACTION":
        text = (
            f"{name} is on the deny list and will never be permitted. "
            "Do not retry it. Accomplish the task another way, or report to "
            "the user that it is blocked by policy."
        )
    elif reason_code == "NOT_IN_ALLOWLIST":
        text = (
            f"{name} is not on the allowlist. Retrying it will not help. "
            "Use one of the permitted actions instead."
        )
    elif reason_code == "RATE_LIMITED":
        text = (
            f"{name} exceeded its rate limit. The same call will succeed "
            "later; wait before retrying, or do other useful work first."
        )
        if retry_after_seconds is not None:
            text += f" Retry after about {retry_after_seconds:.0f} seconds."
    elif reason_code == "BUDGET_EXCEEDED":
        text = (
            "The budget for this scope is exhausted. Retrying the same call "
            "will fail again. Use a cheaper approach, or stop and report the "
            "budget limit to the user."
        )
    elif reason_code == "POLICY_NOT_CONFIGURED":
        text = (
            "No policy is loaded and enforcement is fail-closed, so every "
            "action is blocked. This is an operator configuration problem, "
            "not something to retry."
        )
    elif reason_code == "NO_POLICY_FOR_WRITE":
        text = (
            f"{name} declares impact 'medium' or 'high' and no policy could "
            "be evaluated for it (unconfigured, unreachable with no cache, "
            "or a policy evaluation error), so it was blocked under the "
            "default 'fail_closed_writes' enforcement mode. This is an "
            "operator configuration problem, not something to retry as-is: "
            "if this action does not write anything, ask the operator to "
            "declare it with impact=\"read\" or impact=\"low\" instead."
        )
    elif reason_code == "EVALUATION_ERROR":
        text = (
            "Policy evaluation itself raised an unexpected error and "
            "enforcement is fail-closed, so the action was blocked. This is "
            "an operator/policy configuration bug, not something to retry."
        )
    else:
        text = (
            f"{name} was blocked by policy. Retrying the identical call is "
            "unlikely to succeed; try a different approach."
        )

    if safe_alternatives:
        text += f" Permitted alternatives: {', '.join(safe_alternatives)}."

    return text


class PolicyNotConfiguredError(PolicyViolationError):
    """
    Raised when an action is checked but no policy is loaded, under
    fail-closed enforcement.

    Subclasses PolicyViolationError deliberately: the action was refused by
    the policy layer, so every existing ``except (BudgetExceededError,
    PolicyViolationError)`` handler in guard.py, langchain.py and crewai.py
    catches it. Making it a sibling of PolicyViolationError instead meant a
    fail-closed block recorded no intervention, wrote no ledger row, and never
    cleared compliance metadata — the worst possible gap in the one mode a
    regulated deployment turns on.

    Raised under the strict ``"fail_closed"`` mode, which blocks every action
    while unconfigured regardless of declared impact. As of 0.4.0 the SDK's
    actual default is the softer ``"fail_closed_writes"`` (see
    ``NoPolicyForWriteError``): a telemetry or config problem still blocks
    writes, but does not break a caller's read-only tools. ``"fail_closed"``
    remains available for deployments that want every unconfigured action
    blocked, not just writes: opt in with
    ``PolicyEngine.set_enforcement_mode("fail_closed")`` or
    ``AGENT_SENTINEL_ENFORCEMENT=fail_closed``.
    """

    def __init__(
        self,
        message: str = (
            "Policy engine is not configured and enforcement mode is "
            "'fail_closed', so the action was blocked. Load a policy "
            "(PolicyEngine.configure / load_from_yaml / enable_remote_sync), "
            "or set enforcement mode to 'fail_open' to permit unpoliced actions."
        ),
        action_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        details["enforcement_mode"] = "fail_closed"
        PolicyViolationError.__init__(
            self,
            message=message,
            policy_rule="enforcement_mode",
            action_name=action_name,
            details=details,
            reason_code="POLICY_NOT_CONFIGURED",
            # Not recoverable by the agent: no prompt-level retry fixes a
            # missing policy. It is an operator problem.
            recoverable=False,
            error_code="POLICY_NOT_CONFIGURED",
        )


class NoPolicyForWriteError(PolicyViolationError):
    """
    Raised when a write-impact action (``guarded_action(impact=...)`` of
    ``"medium"`` or ``"high"``, the decorator's own default) is checked while
    the policy layer cannot vouch for it -- the engine is unconfigured, a
    remote sync has never produced a usable local cache, or policy
    evaluation itself raised -- under the default ``"fail_closed_writes"``
    enforcement mode.

    Subclasses PolicyViolationError for the same reason PolicyNotConfiguredError
    does: every existing ``except (BudgetExceededError, PolicyViolationError)``
    handler in guard.py must catch it, record an intervention, and clear
    compliance metadata. Missing that would be the worst possible gap in the
    SDK's own default mode.

    ``"fail_closed_writes"`` became the SDK default in 0.4.0: an unconfigured
    or unreachable engine now blocks declared-impact ``"medium"``/``"high"``
    actions while still permitting ``"read"``/``"low"`` ones with a warning,
    so read-only tools keep working without a policy. Set
    ``AGENT_SENTINEL_ENFORCEMENT=fail_open`` to restore the pre-0.4 behaviour
    for one more minor version, or ``"fail_closed"`` to block every action
    regardless of declared impact.
    """

    def __init__(
        self,
        message: str | None = None,
        action_name: str | None = None,
        impact: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        details["enforcement_mode"] = "fail_closed_writes"
        if impact:
            details["impact"] = impact

        if message is None:
            name = f"'{action_name}'" if action_name else "This action"
            message = (
                f"{name} declares impact={impact or 'medium'!r} and no policy "
                "could be evaluated for it, so it was blocked under the "
                "default 'fail_closed_writes' enforcement mode. Load a "
                "policy, declare impact=\"read\" or \"low\" if this action "
                "does not write anything, or set enforcement mode to "
                "'fail_open' to permit it."
            )

        PolicyViolationError.__init__(
            self,
            message=message,
            policy_rule="enforcement_mode",
            action_name=action_name,
            details=details,
            reason_code="NO_POLICY_FOR_WRITE",
            # Not recoverable by the agent: no prompt-level retry fixes a
            # missing policy. It is an operator problem, same as
            # PolicyNotConfiguredError.
            recoverable=False,
            error_code="NO_POLICY_FOR_WRITE",
        )


class RemediationPayload:
    """
    Machine-readable remediation information returned when an action is blocked.

    This enables LLM agents to self-repair by understanding exactly what
    went wrong and what actions to take to fix it.
    """

    __slots__ = (
        "reason_code", "missing_requirements", "required_prior_actions",
        "retry_guidance", "safe_alternatives", "stale_evidence",
        "argument_violations", "retry_after_seconds",
    )

    def __init__(
        self,
        reason_code: str,
        missing_requirements: list[str] | None = None,
        required_prior_actions: list[str] | None = None,
        retry_guidance: str | None = None,
        safe_alternatives: list[str] | None = None,
        stale_evidence: list[str] | None = None,
        argument_violations: list[str] | None = None,
        retry_after_seconds: float | None = None,
    ):
        self.reason_code = reason_code
        self.missing_requirements = missing_requirements or []
        self.required_prior_actions = required_prior_actions or []
        self.retry_guidance = retry_guidance or ""
        self.safe_alternatives = safe_alternatives or []
        self.stale_evidence = stale_evidence or []
        self.argument_violations = argument_violations or []
        # Only meaningful for time-bounded blocks such as rate limits, where
        # the same call succeeds later without any change by the agent.
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        payload: dict[str, Any] = {
            "reason_code": self.reason_code,
            "missing_requirements": self.missing_requirements,
            "required_prior_actions": self.required_prior_actions,
            "retry_guidance": self.retry_guidance,
            "safe_alternatives": self.safe_alternatives,
            "stale_evidence": self.stale_evidence,
            "argument_violations": self.argument_violations,
        }
        # Only emitted for time-bounded blocks. Omitting it when unset keeps
        # the payload byte-identical to prior releases for the evidence path,
        # so persisted rows and strict consumer schemas do not change shape.
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        return payload


class EvidenceViolationError(PolicyViolationError):
    """
    Raised when an action is blocked due to missing or stale evidence.

    Unlike hard policy blocks, evidence violations are recoverable — the agent
    can fix the issue by executing the required prerequisite actions first.

    The remediation payload provides machine-readable guidance for self-repair.
    """

    def __init__(
        self,
        message: str,
        action_name: str | None = None,
        missing_requirements: list[str] | None = None,
        required_prior_actions: list[str] | None = None,
        stale_evidence: list[str] | None = None,
        argument_violations: list[str] | None = None,
        safe_alternatives: list[str] | None = None,
        retry_guidance: str | None = None,
    ):
        # Determine reason code
        reason_code = "MISSING_EVIDENCE"
        if stale_evidence and not missing_requirements:
            reason_code = "STALE_EVIDENCE"
        elif argument_violations:
            reason_code = "ARGUMENT_CONSTRAINT_VIOLATED"

        # Auto-generate retry guidance if not provided
        if not retry_guidance:
            parts = []
            if missing_requirements:
                parts.append(f"Execute these actions first: {', '.join(missing_requirements)}")
            if stale_evidence:
                parts.append(f"Re-execute these actions (evidence expired): {', '.join(stale_evidence)}")
            if argument_violations:
                parts.append(f"Fix argument constraints: {', '.join(argument_violations)}")
            retry_guidance = ". ".join(parts) if parts else "Check action prerequisites."

        self.remediation = RemediationPayload(
            reason_code=reason_code,
            missing_requirements=missing_requirements or [],
            required_prior_actions=required_prior_actions or [],
            retry_guidance=retry_guidance,
            safe_alternatives=safe_alternatives or [],
            stale_evidence=stale_evidence or [],
            argument_violations=argument_violations or [],
        )

        details: dict[str, Any] = {"remediation": self.remediation.to_dict()}
        if action_name:
            details["action_name"] = action_name

        # Call AgentSentinelError directly to set our own error_code and recoverable
        AgentSentinelError.__init__(
            self,
            message=message,
            error_code="EVIDENCE_VIOLATION",
            details=details,
            recoverable=True,  # Agents CAN fix this by running required actions
        )


class ReplayDivergenceError(AgentSentinelError):
    """Raised during replay if the stored output does not match inputs."""
    
    def __init__(
        self,
        message: str,
        action_name: str | None = None,
        expected_inputs: Any | None = None,
        actual_inputs: Any | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize replay divergence error.
        
        Args:
            message: Human-readable error message
            action_name: Name of action that diverged
            expected_inputs: Expected inputs from recording
            actual_inputs: Actual inputs from current run
            details: Additional error context
        """
        details = details or {}
        if action_name:
            details["action_name"] = action_name
        if expected_inputs is not None:
            details["expected_inputs"] = str(expected_inputs)[:200]
        if actual_inputs is not None:
            details["actual_inputs"] = str(actual_inputs)[:200]
        super().__init__(
            message=message,
            error_code="REPLAY_DIVERGENCE",
            details=details,
            recoverable=False,
        )


class NetworkError(AgentSentinelError):
    """Raised when network communication fails."""
    
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        endpoint: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize network error.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code (if applicable)
            endpoint: API endpoint that failed
            details: Additional error context
        """
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        
        # Transient errors are retryable
        recoverable = status_code and 500 <= status_code < 600 or status_code in (408, 429)
        
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            details=details,
            recoverable=recoverable,
        )


class SyncError(AgentSentinelError):
    """Raised when background sync fails."""
    
    def __init__(
        self,
        message: str,
        batch_size: int | None = None,
        retry_count: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize sync error.
        
        Args:
            message: Human-readable error message
            batch_size: Size of batch that failed
            retry_count: Number of retries attempted
            details: Additional error context
        """
        details = details or {}
        if batch_size is not None:
            details["batch_size"] = batch_size
        if retry_count is not None:
            details["retry_count"] = retry_count
        super().__init__(
            message=message,
            error_code="SYNC_ERROR",
            details=details,
            recoverable=True,  # Sync can always be retried
        )


class TimeoutError(AgentSentinelError):
    """Raised when an operation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Human-readable error message
            timeout_seconds: Timeout duration in seconds
            operation: Operation that timed out
            details: Additional error context
        """
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(
            message=message,
            error_code="TIMEOUT",
            details=details,
            recoverable=True,  # Timeouts are usually retryable
        )


class AgentKilledError(AgentSentinelError):
    """Raised when an agent, run, or mission has been killed via kill switch."""

    def __init__(
        self,
        message: str,
        target_type: str | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if target_type:
            details["target_type"] = target_type
        if target_id:
            details["target_id"] = target_id
        super().__init__(
            message=message,
            error_code="AGENT_KILLED",
            details=details,
            recoverable=False,
        )


class KillSwitchUnavailableError(AgentKilledError):
    """
    Raised when the kill switch cannot be checked (platform unreachable, a
    non-200 response, or httpx not installed) under fail-closed enforcement.

    Subclasses AgentKilledError deliberately: every existing
    ``except AgentKilledError`` handler catches it. Under fail-closed, an
    agent that cannot be confirmed as NOT killed is treated the same as one
    confirmed killed — the caller cannot tell the difference between "not
    killed" and "don't know", and only one of those is safe to proceed on.
    """

    def __init__(
        self,
        message: str = (
            "Kill switch could not be checked and enforcement mode is "
            "'fail_closed', so the action was blocked."
        ),
        target_type: str | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        details["reason_code"] = "KILL_SWITCH_UNAVAILABLE"
        super().__init__(
            message=message,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )


class ConfigurationError(AgentSentinelError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Human-readable error message
            config_key: Configuration key that is invalid
            details: Additional error context
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            recoverable=False,  # Config errors don't recover
        )
