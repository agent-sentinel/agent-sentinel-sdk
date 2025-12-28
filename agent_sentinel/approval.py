"""
Approval Client: SDK client for platform approval workflows.

This module provides the client-side implementation for the Approval Inbox feature:
- Submit approval requests to the platform
- Poll for approval status
- Integrate with the guard decorator for seamless approval workflows
- Support for both blocking (sync) and non-blocking (async) approval flows
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Optional, Any, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from .errors import PolicyViolationError, TimeoutError, NetworkError, ConfigurationError

logger = logging.getLogger("agent_sentinel.approval")

# Try to import httpx for remote sync
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.debug("httpx not installed. Remote approval disabled.")


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    INFO_REQUESTED = "info_requested"


class ApprovalPriority(str, Enum):
    """Priority level for approval requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Risk assessment level for the action."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    """Request for human approval of an action."""
    action_name: str
    action_description: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    estimated_cost: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    priority: ApprovalPriority = ApprovalPriority.MEDIUM
    timeout_seconds: int = 3600
    action_inputs: Dict[str, Any] | None = None
    context: Dict[str, Any] | None = None
    callback_url: str | None = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        # Avoid asdict() because it performs deepcopy which fails on locks
        return {
            "action_name": self.action_name,
            "action_description": self.action_description,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "estimated_cost": self.estimated_cost,
            "risk_level": self.risk_level.value,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "action_inputs": self.action_inputs,
            "context": self.context,
            "callback_url": self.callback_url,
        }


@dataclass
class ApprovalResponse:
    """Response from an approval request."""
    approval_id: str
    status: ApprovalStatus
    decided_at: datetime | None = None
    decided_by_email: str | None = None
    decision_notes: str | None = None
    info_request_message: str | None = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalResponse":
        """Create from API response dictionary."""
        status = ApprovalStatus(data.get("status", "pending"))
        decided_at = None
        if data.get("decided_at"):
            decided_at = datetime.fromisoformat(data["decided_at"].replace("Z", "+00:00"))
        
        return cls(
            approval_id=str(data.get("id", data.get("approval_id", ""))),
            status=status,
            decided_at=decided_at,
            decided_by_email=data.get("decided_by_email"),
            decision_notes=data.get("decision_notes"),
            info_request_message=data.get("info_request_message"),
        )


@dataclass
class ApprovalConfig:
    """Configuration for the approval client."""
    platform_url: str
    api_token: str
    poll_interval: float = 5.0  # Seconds between polls
    default_timeout: int = 3600  # Default timeout in seconds
    enabled: bool = True


class ApprovalClient:
    """
    Client for interacting with the platform's Approval Inbox.
    
    Usage:
        # Configure the client
        ApprovalClient.configure(
            platform_url="https://api.agentsentinel.dev",
            api_token="your-jwt-token",
        )
        
        # Submit an approval request and wait for result
        response = ApprovalClient.request_approval_sync(
            action_name="transfer_funds",
            action_description="Transfer $500 from account A to B",
            estimated_cost=0.10,
            timeout_seconds=300,
        )
        
        if response.status == ApprovalStatus.APPROVED:
            # Proceed with action
            pass
        else:
            # Handle rejection
            pass
    """
    
    _config: Optional[ApprovalConfig] = None
    _lock = threading.Lock()
    
    @classmethod
    def configure(
        cls,
        platform_url: str,
        api_token: str,
        poll_interval: float = 5.0,
        default_timeout: int = 3600,
        enabled: bool = True,
    ) -> None:
        """
        Configure the approval client.
        
        Args:
            platform_url: Platform API base URL
            api_token: JWT token for authentication
            poll_interval: Seconds between status polls
            default_timeout: Default timeout for approval requests
            enabled: Enable/disable approval workflows
        """
        with cls._lock:
            cls._config = ApprovalConfig(
                platform_url=platform_url.rstrip("/"),
                api_token=api_token,
                poll_interval=poll_interval,
                default_timeout=default_timeout,
                enabled=enabled,
            )
        logger.info(f"Approval client configured for {platform_url}")
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if the approval client is configured."""
        return cls._config is not None and cls._config.enabled
    
    @classmethod
    def check_approval_required(
        cls,
        action_name: str,
        cost_usd: float = 0.0,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if an action requires approval (pre-flight check).
        
        Args:
            action_name: Name of the action
            cost_usd: Estimated cost
            agent_id: Optional agent ID
            run_id: Optional run ID
            
        Returns:
            Dict with approval requirements if needed, None otherwise
        """
        if not cls.is_configured() or not HTTPX_AVAILABLE:
            return None
        
        url = f"{cls._config.platform_url}/api/v1/approvals/check"
        headers = {
            "Authorization": f"ApiKey {cls._config.api_token}",
        }
        params = {
            "action_name": action_name,
            "cost_usd": cost_usd,
        }
        if agent_id:
            params["agent_id"] = agent_id
        if run_id:
            params["run_id"] = run_id
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("requires_approval"):
                        return data
                    return None
                else:
                    logger.warning(f"Approval check failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error checking approval requirement: {e}")
            return None
    
    @classmethod
    def submit_request(cls, request: ApprovalRequest) -> str:
        """
        Submit an approval request to the platform.
        
        Args:
            request: ApprovalRequest with action details
            
        Returns:
            Approval request ID for polling
            
        Raises:
            ConfigurationError: If client not configured
            NetworkError: If request fails
        """
        if not cls.is_configured():
            raise ConfigurationError(
                "Approval client not configured. Call ApprovalClient.configure() first."
            )
        
        if not HTTPX_AVAILABLE:
            raise ConfigurationError(
                "httpx not installed. Install with: pip install agent-sentinel[remote]"
            )
        
        url = f"{cls._config.platform_url}/api/v1/approvals"
        headers = {
            "Authorization": f"ApiKey {cls._config.api_token}",
            "Content-Type": "application/json",
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                import json
                from .ledger import SafeEncoder
                
                # Use SafeEncoder to handle non-serializable objects in inputs
                content = json.dumps(request.to_dict(), cls=SafeEncoder)
                
                response = client.post(
                    url,
                    content=content,
                    headers=headers,
                )
                
                if response.status_code == 201:
                    data = response.json()
                    approval_id = str(data.get("id"))
                    logger.info(f"Approval request submitted: {approval_id}")
                    return approval_id
                else:
                    raise NetworkError(
                        f"Failed to submit approval request: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                    )
                    
        except httpx.RequestError as e:
            raise NetworkError(
                f"Network error submitting approval request: {e}",
                endpoint=url,
            )
    
    @classmethod
    def poll_status(cls, approval_id: str) -> ApprovalResponse:
        """
        Poll the status of an approval request.
        
        Args:
            approval_id: ID of the approval request
            
        Returns:
            ApprovalResponse with current status
            
        Raises:
            ConfigurationError: If client not configured
            NetworkError: If request fails
        """
        if not cls.is_configured():
            raise ConfigurationError(
                "Approval client not configured. Call ApprovalClient.configure() first."
            )
        
        if not HTTPX_AVAILABLE:
            raise ConfigurationError("httpx not installed")
        
        url = f"{cls._config.platform_url}/api/v1/approvals/{approval_id}/poll"
        headers = {
            "Authorization": f"ApiKey {cls._config.api_token}",
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers)
                
                if response.status_code == 200:
                    return ApprovalResponse.from_dict(response.json())
                elif response.status_code == 404:
                    raise NetworkError(
                        f"Approval request not found: {approval_id}",
                        status_code=404,
                        endpoint=url,
                    )
                else:
                    raise NetworkError(
                        f"Failed to poll approval status: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                    )
                    
        except httpx.RequestError as e:
            raise NetworkError(
                f"Network error polling approval status: {e}",
                endpoint=url,
            )
    
    @classmethod
    def cancel_request(cls, approval_id: str) -> ApprovalResponse:
        """
        Cancel a pending approval request.
        
        Args:
            approval_id: ID of the approval request
            
        Returns:
            ApprovalResponse with cancelled status
        """
        if not cls.is_configured():
            raise ConfigurationError(
                "Approval client not configured. Call ApprovalClient.configure() first."
            )
        
        if not HTTPX_AVAILABLE:
            raise ConfigurationError("httpx not installed")
        
        url = f"{cls._config.platform_url}/api/v1/approvals/{approval_id}/cancel"
        headers = {
            "Authorization": f"ApiKey {cls._config.api_token}",
            "Content-Type": "application/json",
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=headers)
                
                if response.status_code == 200:
                    return ApprovalResponse.from_dict(response.json())
                else:
                    raise NetworkError(
                        f"Failed to cancel approval request: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                    )
                    
        except httpx.RequestError as e:
            raise NetworkError(
                f"Network error cancelling approval request: {e}",
                endpoint=url,
            )
    
    @classmethod
    def respond_to_info_request(
        cls, 
        approval_id: str, 
        response_text: str
    ) -> ApprovalResponse:
        """
        Respond to an info request from an approver.
        
        Args:
            approval_id: ID of the approval request
            response_text: Response to the info request
            
        Returns:
            ApprovalResponse with updated status
        """
        if not cls.is_configured():
            raise ConfigurationError(
                "Approval client not configured. Call ApprovalClient.configure() first."
            )
        
        if not HTTPX_AVAILABLE:
            raise ConfigurationError("httpx not installed")
        
        url = f"{cls._config.platform_url}/api/v1/approvals/{approval_id}/respond"
        headers = {
            "Authorization": f"ApiKey {cls._config.api_token}",
            "Content-Type": "application/json",
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    url,
                    json={"response": response_text},
                    headers=headers,
                )
                
                if response.status_code == 200:
                    return ApprovalResponse.from_dict(response.json())
                else:
                    raise NetworkError(
                        f"Failed to respond to info request: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                    )
                    
        except httpx.RequestError as e:
            raise NetworkError(
                f"Network error responding to info request: {e}",
                endpoint=url,
            )
    
    @classmethod
    def request_approval_sync(
        cls,
        action_name: str,
        action_description: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        estimated_cost: float = 0.0,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        timeout_seconds: int | None = None,
        action_inputs: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> ApprovalResponse:
        """
        Request approval and block until decision is made.
        
        This is a convenience method that submits an approval request,
        then polls until a decision is made or timeout occurs.
        
        Args:
            action_name: Name of the action requiring approval
            action_description: Human-readable description
            agent_id: Optional agent identifier
            run_id: Optional run identifier
            estimated_cost: Estimated cost of the action
            risk_level: Risk level assessment
            priority: Priority for approvers
            timeout_seconds: How long to wait for approval
            action_inputs: Sanitized inputs for the action
            context: Additional context for approvers
            
        Returns:
            ApprovalResponse with the decision
            
        Raises:
            TimeoutError: If approval times out
            PolicyViolationError: If approval is rejected
        """
        timeout = timeout_seconds or cls._config.default_timeout if cls._config else 3600
        
        request = ApprovalRequest(
            action_name=action_name,
            action_description=action_description,
            agent_id=agent_id,
            run_id=run_id,
            estimated_cost=estimated_cost,
            risk_level=risk_level,
            priority=priority,
            timeout_seconds=timeout,
            action_inputs=action_inputs,
            context=context,
        )
        
        # Submit the request
        approval_id = cls.submit_request(request)
        
        # Poll until decision
        start_time = time.time()
        poll_interval = cls._config.poll_interval if cls._config else 5.0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Try to cancel the request
                try:
                    cls.cancel_request(approval_id)
                except Exception:
                    pass
                raise TimeoutError(
                    f"Approval request timed out after {timeout} seconds"
                )
            
            response = cls.poll_status(approval_id)
            
            if response.status == ApprovalStatus.PENDING:
                # Still waiting - continue polling
                time.sleep(poll_interval)
                continue
            
            elif response.status == ApprovalStatus.INFO_REQUESTED:
                # Info requested - agent operator needs to respond
                logger.warning(
                    f"Approval {approval_id} needs more info: "
                    f"{response.info_request_message}"
                )
                # Continue polling - operator might respond externally
                time.sleep(poll_interval)
                continue
            
            elif response.status == ApprovalStatus.APPROVED:
                logger.info(
                    f"Approval {approval_id} approved by {response.decided_by_email}"
                )
                return response
            
            elif response.status == ApprovalStatus.REJECTED:
                logger.warning(
                    f"Approval {approval_id} rejected: {response.decision_notes}"
                )
                raise PolicyViolationError(
                    f"Action '{action_name}' was rejected: {response.decision_notes}"
                )
            
            elif response.status == ApprovalStatus.EXPIRED:
                raise TimeoutError(
                    f"Approval request expired for action '{action_name}'"
                )
            
            elif response.status == ApprovalStatus.CANCELLED:
                raise PolicyViolationError(
                    f"Approval request was cancelled for action '{action_name}'"
                )
            
            else:
                # Unknown status - continue polling
                logger.warning(f"Unknown approval status: {response.status}")
                time.sleep(poll_interval)
    
    @classmethod
    async def request_approval_async(
        cls,
        action_name: str,
        action_description: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        estimated_cost: float = 0.0,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        timeout_seconds: int | None = None,
        action_inputs: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> ApprovalResponse:
        """
        Request approval asynchronously.
        
        Same as request_approval_sync but uses async/await.
        """
        import asyncio
        
        timeout = timeout_seconds or cls._config.default_timeout if cls._config else 3600
        
        request = ApprovalRequest(
            action_name=action_name,
            action_description=action_description,
            agent_id=agent_id,
            run_id=run_id,
            estimated_cost=estimated_cost,
            risk_level=risk_level,
            priority=priority,
            timeout_seconds=timeout,
            action_inputs=action_inputs,
            context=context,
        )
        
        # Submit the request (sync call in executor)
        loop = asyncio.get_event_loop()
        approval_id = await loop.run_in_executor(None, cls.submit_request, request)
        
        # Poll until decision
        start_time = time.time()
        poll_interval = cls._config.poll_interval if cls._config else 5.0
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                try:
                    await loop.run_in_executor(None, cls.cancel_request, approval_id)
                except Exception:
                    pass
                raise TimeoutError(
                    f"Approval request timed out after {timeout} seconds"
                )
            
            response = await loop.run_in_executor(None, cls.poll_status, approval_id)
            
            if response.status == ApprovalStatus.PENDING:
                await asyncio.sleep(poll_interval)
                continue
            
            elif response.status == ApprovalStatus.INFO_REQUESTED:
                logger.warning(
                    f"Approval {approval_id} needs more info: "
                    f"{response.info_request_message}"
                )
                await asyncio.sleep(poll_interval)
                continue
            
            elif response.status == ApprovalStatus.APPROVED:
                logger.info(
                    f"Approval {approval_id} approved by {response.decided_by_email}"
                )
                return response
            
            elif response.status == ApprovalStatus.REJECTED:
                logger.warning(
                    f"Approval {approval_id} rejected: {response.decision_notes}"
                )
                raise PolicyViolationError(
                    f"Action '{action_name}' was rejected: {response.decision_notes}"
                )
            
            elif response.status == ApprovalStatus.EXPIRED:
                raise TimeoutError(
                    f"Approval request expired for action '{action_name}'"
                )
            
            elif response.status == ApprovalStatus.CANCELLED:
                raise PolicyViolationError(
                    f"Approval request was cancelled for action '{action_name}'"
                )
            
            else:
                logger.warning(f"Unknown approval status: {response.status}")
                await asyncio.sleep(poll_interval)
    
    @classmethod
    def reset(cls) -> None:
        """Reset the approval client configuration."""
        with cls._lock:
            cls._config = None


