from __future__ import annotations

from .guard import guarded_action
from .errors import (
    AgentSentinelError,
    AgentKilledError,
    BudgetExceededError,
    PolicyViolationError,
    EvidenceViolationError,
    RemediationPayload,
    ReplayDivergenceError,
    NetworkError,
    SyncError,
    TimeoutError,
    ConfigurationError,
)
from .evidence import EvidenceTracker, EvidenceRecord
from .constraints import validate_constraints
from .retry import (
    RetryConfig,
    CircuitBreaker,
    with_retry,
    with_circuit_breaker,
)
from .ledger import Ledger
from .cost import CostTracker
from .policy import PolicyEngine, PolicyConfig
from .sync import BackgroundSync, ManualSync, SyncConfig, enable_remote_sync, manual_flush, flush_and_stop
from ._configure import configure, flush
from .replay import ReplayMode, ReplayEntry, replay_mode
from .compliance import (
    # Core classes
    HumanApprovalHandler,
    ComplianceMetadata,
    ApprovalRequest as ComplianceApprovalRequest,
    ApprovalResponse as ComplianceApprovalResponse,
    # Enums
    ApprovalStatus as ComplianceApprovalStatus,
    ComplianceLevel,
    # Utility functions
    set_compliance_metadata,
    get_compliance_metadata,
    clear_compliance_metadata,
    add_data_lineage,
    set_decision_rationale,
    set_model_card,
)

# Approval Inbox (Flagship Feature)
from .approval import (
    ApprovalClient,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    ApprovalPriority,
    RiskLevel,
    ApprovalConfig,
)

# Execution Context (runtime attribution)
from .context import ExecutionContext

# Kill Switch (emergency shutdown)
from .kill_switch import KillSwitchClient

# Interventions (CRITICAL - Core Value Proposition)
from .intervention import (
    InterventionTracker,
    InterventionRecord,
    InterventionType,
    InterventionOutcome,
)

# MCP support (optional - requires httpx)
try:
    from .mcp import (
        MCPClient,
        MCPTool,
        MCPResource,
        MCPPrompt,
        MCPToolCallResult,
        set_default_client,
        get_default_client,
    )
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    MCPClient = None  # type: ignore
    MCPTool = None  # type: ignore
    MCPResource = None  # type: ignore
    MCPPrompt = None  # type: ignore
    MCPToolCallResult = None  # type: ignore
    set_default_client = None  # type: ignore
    get_default_client = None  # type: ignore

# Framework integrations (optional - requires respective packages)
# Import integrations module to make it accessible
from . import integrations

__version__ = "0.1.0"
__all__ = [
    "guarded_action",
    "AgentSentinelError",
    "AgentKilledError",
    "BudgetExceededError",
    "PolicyViolationError",
    "EvidenceViolationError",
    "RemediationPayload",
    "EvidenceTracker",
    "EvidenceRecord",
    "validate_constraints",
    "ReplayDivergenceError",
    "NetworkError",
    "SyncError",
    "TimeoutError",
    "ConfigurationError",
    "RetryConfig",
    "CircuitBreaker",
    "with_retry",
    "with_circuit_breaker",
    "Ledger",
    "CostTracker",
    "PolicyEngine",
    "PolicyConfig",
    "BackgroundSync",
    "ManualSync",
    "SyncConfig",
    "enable_remote_sync",
    "manual_flush",
    "flush_and_stop",
    "configure",
    "flush",
    # Execution Context
    "ExecutionContext",
    # Kill Switch
    "KillSwitchClient",
    "ReplayMode",
    "ReplayEntry",
    "replay_mode",
    # EU Compliance (Enterprise Tier Foundation)
    "HumanApprovalHandler",
    "ComplianceMetadata",
    "ComplianceApprovalRequest",
    "ComplianceApprovalResponse",
    "ComplianceApprovalStatus",
    "ComplianceLevel",
    "set_compliance_metadata",
    "get_compliance_metadata",
    "clear_compliance_metadata",
    "add_data_lineage",
    "set_decision_rationale",
    "set_model_card",
    # Approval Inbox (Flagship Feature)
    "ApprovalClient",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    "ApprovalPriority",
    "RiskLevel",
    "ApprovalConfig",
    # Interventions (CRITICAL - Core Value Proposition)
    "InterventionTracker",
    "InterventionRecord",
    "InterventionType",
    "InterventionOutcome",
    # MCP Support (optional)
    "MCPClient",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPToolCallResult",
    "set_default_client",
    "get_default_client",
]
