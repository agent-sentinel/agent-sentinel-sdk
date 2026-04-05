from __future__ import annotations

from typing import Optional, Any
import logging

logger = logging.getLogger("agent_sentinel")


class AgentSentinelError(Exception):
    """Base exception for all Agent Sentinel errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
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
        details: Optional[dict[str, Any]] = None,
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
        super().__init__(
            message=message,
            error_code="BUDGET_EXCEEDED",
            details=details,
            recoverable=False,  # Budget violations don't recover
        )


class PolicyViolationError(AgentSentinelError):
    """Raised when an action violates a specific policy rule."""
    
    def __init__(
        self,
        message: str,
        policy_rule: Optional[str] = None,
        action_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize policy violation error.
        
        Args:
            message: Human-readable error message
            policy_rule: Name of the violated policy rule
            action_name: Name of the action that violated policy
            details: Additional error context
        """
        details = details or {}
        if policy_rule:
            details["policy_rule"] = policy_rule
        if action_name:
            details["action_name"] = action_name
        super().__init__(
            message=message,
            error_code="POLICY_VIOLATION",
            details=details,
            recoverable=False,
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
        "argument_violations",
    )

    def __init__(
        self,
        reason_code: str,
        missing_requirements: Optional[list[str]] = None,
        required_prior_actions: Optional[list[str]] = None,
        retry_guidance: Optional[str] = None,
        safe_alternatives: Optional[list[str]] = None,
        stale_evidence: Optional[list[str]] = None,
        argument_violations: Optional[list[str]] = None,
    ):
        self.reason_code = reason_code
        self.missing_requirements = missing_requirements or []
        self.required_prior_actions = required_prior_actions or []
        self.retry_guidance = retry_guidance or ""
        self.safe_alternatives = safe_alternatives or []
        self.stale_evidence = stale_evidence or []
        self.argument_violations = argument_violations or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reason_code": self.reason_code,
            "missing_requirements": self.missing_requirements,
            "required_prior_actions": self.required_prior_actions,
            "retry_guidance": self.retry_guidance,
            "safe_alternatives": self.safe_alternatives,
            "stale_evidence": self.stale_evidence,
            "argument_violations": self.argument_violations,
        }


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
        action_name: Optional[str] = None,
        missing_requirements: Optional[list[str]] = None,
        required_prior_actions: Optional[list[str]] = None,
        stale_evidence: Optional[list[str]] = None,
        argument_violations: Optional[list[str]] = None,
        safe_alternatives: Optional[list[str]] = None,
        retry_guidance: Optional[str] = None,
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
        action_name: Optional[str] = None,
        expected_inputs: Optional[Any] = None,
        actual_inputs: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
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
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
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
        batch_size: Optional[int] = None,
        retry_count: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
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
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
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
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
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


class ConfigurationError(AgentSentinelError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
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
