"""
EU AI Act Compliance Module

Provides foundational support for EU AI Act Article 12 & 14 requirements:
- Human-in-the-Loop approval workflow
- Decision rationale tracking
- Data lineage documentation
- Compliance metadata collection

Enterprise Tier Feature (Foundation - full implementation pending)
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from collections.abc import Callable, Awaitable
from enum import Enum, StrEnum
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("agent_sentinel.compliance")


class ApprovalStatus(StrEnum):
    """Human oversight approval status (Article 14)"""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ComplianceLevel(StrEnum):
    """Risk levels according to EU AI Act classification"""
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH_RISK = "high_risk"
    UNACCEPTABLE = "unacceptable"


@dataclass
class ApprovalRequest:
    """Request for human approval of an action"""
    action_name: str
    action_description: str
    inputs: dict[str, Any]
    estimated_cost: float
    risk_level: ComplianceLevel = ComplianceLevel.MINIMAL
    timeout_seconds: int = 300  # 5 minutes default
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['risk_level'] = self.risk_level.value
        return data


@dataclass
class ApprovalResponse:
    """Response from human approver"""
    request_id: str
    status: ApprovalStatus
    approver_id: str | None = None
    approver_email: str | None = None
    notes: str | None = None
    approved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compliance metadata"""
        data = asdict(self)
        data['approved_at'] = self.approved_at.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class ComplianceMetadata:
    """
    Compliance metadata to be attached to actions.
    Maps to Action.compliance_metadata in platform.
    """
    # Human oversight (Article 14)
    requires_human_approval: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    human_in_the_loop_id: str | None = None
    human_in_the_loop_email: str | None = None
    approval_timestamp: datetime | None = None
    approval_notes: str | None = None
    
    # Decision transparency
    decision_rationale: str | None = None
    confidence_score: float | None = None
    alternative_actions_considered: list[str] = field(default_factory=list)
    
    # Data lineage
    input_data_sources: list[dict[str, str]] = field(default_factory=list)
    model_card: dict[str, Any] | None = None
    
    # Audit
    policy_violations_checked: list[str] = field(default_factory=list)
    override_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        data['approval_status'] = self.approval_status.value
        if self.approval_timestamp:
            data['approval_timestamp'] = self.approval_timestamp.isoformat()
        return data


class HumanApprovalHandler:
    """
    Handler for human-in-the-loop approval workflow.

    Can be used as an instance with a local callback, or as a class with
    class-level handlers.  When ``ApprovalClient`` is configured, instance
    methods ``request_approval_sync`` / ``request_approval_async`` delegate
    to the platform instead of invoking the local callback.

    Instance usage::

        handler = HumanApprovalHandler(callback=lambda req: True)
        approved = handler.request_approval_sync(
            action_name="drop_table",
            estimated_cost=0.05,
            context={"table": "users"},
        )

    Class usage (legacy)::

        HumanApprovalHandler.set_approval_handler(my_sync_handler)
        response = HumanApprovalHandler.request_approval_sync(...)
    """

    _approval_callback: Callable[[ApprovalRequest], Awaitable[ApprovalResponse]] | None = None
    _sync_approval_callback: Callable[[ApprovalRequest], ApprovalResponse] | None = None

    # ------------------------------------------------------------------
    # Instance API
    # ------------------------------------------------------------------

    def __init__(
        self,
        callback: Callable[..., Any] | None = None,
    ) -> None:
        """
        Create an instance-level handler.

        Args:
            callback: A callable that receives an ``ApprovalRequest`` (or any
                positional arg) and returns a truthy/falsy value indicating
                approval.  Only used when ``ApprovalClient`` is *not*
                configured.
        """
        self._instance_callback = callback

    def request_approval_sync(
        self,
        action_name: str,
        estimated_cost: float = 0.0,
        context: dict | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Request human approval for an action (blocking).

        When ``ApprovalClient`` is configured this delegates to the platform.
        Otherwise the local ``callback`` supplied at construction time is
        invoked with an ``ApprovalRequest`` instance.

        Returns:
            ``True`` if approved, ``False`` otherwise.
        """
        from agent_sentinel.approval import ApprovalClient, ApprovalStatus as _AS
        if ApprovalClient.is_configured():
            response = ApprovalClient.request_approval_sync(
                action_name=action_name,
                estimated_cost=estimated_cost,
                context=context,
                timeout_seconds=timeout,
            )
            return response.status == _AS.APPROVED

        # Fall back to local callback
        request = ApprovalRequest(
            action_name=action_name,
            action_description="",
            inputs=context or {},
            estimated_cost=estimated_cost,
        )
        if self._instance_callback is not None:
            result = self._instance_callback(request)
            return bool(result)
        logger.error("No approval callback configured and ApprovalClient is not set up.")
        return False

    async def request_approval_async(
        self,
        action_name: str,
        estimated_cost: float = 0.0,
        context: dict | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Request human approval for an action (async).

        When ``ApprovalClient`` is configured this delegates to the platform.
        Otherwise the local ``callback`` supplied at construction time is
        invoked with an ``ApprovalRequest`` instance.

        Returns:
            ``True`` if approved, ``False`` otherwise.
        """
        from agent_sentinel.approval import ApprovalClient, ApprovalStatus as _AS
        if ApprovalClient.is_configured():
            response = await ApprovalClient.request_approval_async(
                action_name=action_name,
                estimated_cost=estimated_cost,
                context=context,
                timeout_seconds=timeout,
            )
            return response.status == _AS.APPROVED

        # Fall back to local callback
        request = ApprovalRequest(
            action_name=action_name,
            action_description="",
            inputs=context or {},
            estimated_cost=estimated_cost,
        )
        if self._instance_callback is not None:
            result = self._instance_callback(request)
            return bool(result)
        logger.error("No approval callback configured and ApprovalClient is not set up.")
        return False

    @classmethod
    def set_approval_handler(
        cls,
        handler: Callable[[ApprovalRequest], ApprovalResponse]
    ):
        """
        Set synchronous approval handler.
        
        Example:
            def my_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
                # Show UI prompt or send to approval system
                print(f"Approval needed: {request.action_name}")
                user_input = input("Approve? (y/n): ")
                
                return ApprovalResponse(
                    request_id=request.request_id,
                    status=ApprovalStatus.APPROVED if user_input.lower() == 'y' else ApprovalStatus.REJECTED,
                    approver_email="user@example.com"
                )
            
            HumanApprovalHandler.set_approval_handler(my_approval_handler)
        """
        cls._sync_approval_callback = handler
        logger.info("Synchronous approval handler registered")
    
    @classmethod
    def set_async_approval_handler(
        cls,
        handler: Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]
    ):
        """
        Set asynchronous approval handler.
        
        Example:
            async def my_async_approval_handler(request: ApprovalRequest) -> ApprovalResponse:
                # Send to platform API and wait for webhook
                response = await send_approval_request_to_platform(request)
                return response
            
            HumanApprovalHandler.set_async_approval_handler(my_async_approval_handler)
        """
        cls._approval_callback = handler
        logger.info("Async approval handler registered")
    
    @classmethod
    async def _class_request_approval_async(
        cls,
        action_name: str,
        action_description: str,
        inputs: dict[str, Any],
        cost: float,
        risk_level: ComplianceLevel = ComplianceLevel.MINIMAL,
        timeout: int = 300
    ) -> ApprovalResponse:
        """
        Request human approval for an action via class-level handler (async version).

        Raises:
            RuntimeError: If no approval handler is configured
        """
        request = ApprovalRequest(
            action_name=action_name,
            action_description=action_description,
            inputs=inputs,
            estimated_cost=cost,
            risk_level=risk_level,
            timeout_seconds=timeout
        )

        if cls._approval_callback is None:
            logger.error("No async approval handler configured - action will be blocked")
            raise RuntimeError(
                "Human approval required but no handler configured. "
                "Call HumanApprovalHandler.set_async_approval_handler() to configure."
            )

        logger.info(f"Requesting approval for action: {action_name}")
        response = await cls._approval_callback(request)

        if response.status == ApprovalStatus.APPROVED:
            logger.info(f"Action '{action_name}' approved by {response.approver_email}")
        else:
            logger.warning(f"Action '{action_name}' rejected: {response.status}")

        return response

    @classmethod
    def _class_request_approval_sync(
        cls,
        action_name: str,
        action_description: str,
        inputs: dict[str, Any],
        cost: float,
        risk_level: ComplianceLevel = ComplianceLevel.MINIMAL,
        timeout: int = 300
    ) -> ApprovalResponse:
        """
        Request human approval for an action via class-level handler (sync version).

        Raises:
            RuntimeError: If no approval handler is configured
        """
        request = ApprovalRequest(
            action_name=action_name,
            action_description=action_description,
            inputs=inputs,
            estimated_cost=cost,
            risk_level=risk_level,
            timeout_seconds=timeout
        )

        if cls._sync_approval_callback is None:
            logger.error("No sync approval handler configured - action will be blocked")
            raise RuntimeError(
                "Human approval required but no handler configured. "
                "Call HumanApprovalHandler.set_approval_handler() to configure."
            )

        logger.info(f"Requesting approval for action: {action_name}")
        response = cls._sync_approval_callback(request)

        if response.status == ApprovalStatus.APPROVED:
            logger.info(f"Action '{action_name}' approved by {response.approver_email}")
        else:
            logger.warning(f"Action '{action_name}' rejected: {response.status}")

        return response


# Global compliance context for tracking metadata during execution
_current_compliance_metadata: ComplianceMetadata | None = None


def set_compliance_metadata(metadata: ComplianceMetadata):
    """Set compliance metadata for the current action"""
    global _current_compliance_metadata
    _current_compliance_metadata = metadata


def get_compliance_metadata() -> ComplianceMetadata | None:
    """Get current compliance metadata"""
    return _current_compliance_metadata


def clear_compliance_metadata():
    """Clear compliance metadata after action completes"""
    global _current_compliance_metadata
    _current_compliance_metadata = None


def add_data_lineage(source: str, version: str, **kwargs):
    """
    Add data lineage information to current action.
    
    Example:
        add_data_lineage("Postgres", version="v12", table="accounts", query="SELECT balance")
    """
    if _current_compliance_metadata:
        lineage_entry = {"source": source, "version": version, **kwargs}
        _current_compliance_metadata.input_data_sources.append(lineage_entry)
        logger.debug(f"Added data lineage: {lineage_entry}")


def set_decision_rationale(rationale: str, confidence: float | None = None):
    """
    Set decision rationale for current action (chain of thought).
    
    Example:
        set_decision_rationale(
            "User requested balance check. Verified user_id=123 has permission. "
            "Queried database and found balance=$4200.",
            confidence=0.95
        )
    """
    if _current_compliance_metadata:
        _current_compliance_metadata.decision_rationale = rationale
        if confidence is not None:
            _current_compliance_metadata.confidence_score = confidence
        logger.debug("Set decision rationale for compliance tracking")


def set_model_card(model_name: str, version: str, provider: str, **kwargs):
    """
    Set model card information for current action.
    
    Example:
        set_model_card(
            model_name="gpt-4-turbo",
            version="2024-04-09",
            provider="OpenAI",
            system_prompt_hash="sha256:abc123"
        )
    """
    if _current_compliance_metadata:
        _current_compliance_metadata.model_card = {
            "name": model_name,
            "version": version,
            "provider": provider,
            **kwargs
        }
        logger.debug(f"Set model card: {model_name} v{version}")

