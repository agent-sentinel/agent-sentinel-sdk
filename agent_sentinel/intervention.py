"""
Intervention tracking for the SDK.

This module records when Sentinel blocks, pauses, or modifies agent actions.
These interventions are the core value proposition - showing where Sentinel
prevented risky actions.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger("agent_sentinel")


class InterventionType(str, Enum):
    """Type of intervention performed by Sentinel."""
    HARD_BLOCK = "hard_block"              # Action completely denied
    APPROVAL_REQUIRED = "approval_required"  # Escalated for human approval
    RATE_LIMITED = "rate_limited"          # Blocked due to rate limiting
    BUDGET_EXCEEDED = "budget_exceeded"    # Blocked due to budget constraints
    DOWNGRADE = "downgrade"                # Action parameters modified/reduced
    WARNING = "warning"                    # Allowed but flagged as risky


class InterventionOutcome(str, Enum):
    """Final outcome of the intervention."""
    BLOCKED = "blocked"                    # Action was prevented
    ESCALATED = "escalated"                # Sent for approval (may be pending)
    APPROVED_AFTER_REVIEW = "approved_after_review"  # Was approved and executed
    REJECTED_AFTER_REVIEW = "rejected_after_review"  # Was rejected by approver
    MODIFIED = "modified"                  # Action was modified and allowed
    WARNED = "warned"                      # Allowed with warning


@dataclass
class InterventionRecord:
    """
    Record of an intervention performed by Sentinel.
    
    This captures the critical moment where Sentinel exercised authority
    over an autonomous agent's intended action.
    """
    # What happened
    intervention_type: InterventionType
    outcome: InterventionOutcome
    
    # Action details
    action_name: str
    action_description: Optional[str] = None
    
    # Context
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    policy_name: Optional[str] = None
    
    # Cost/Risk
    estimated_cost: float = 0.0
    actual_cost: float = 0.0  # If eventually executed
    risk_level: str = "medium"
    
    # The "blast radius" - what was avoided
    blast_radius: Dict[str, Any] = field(default_factory=dict)
    
    # Why it happened
    reason: Optional[str] = None
    agent_intent: Optional[str] = None
    
    # Original vs modified inputs
    original_inputs: Optional[Dict[str, Any]] = None
    modified_inputs: Optional[Dict[str, Any]] = None
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intervention_type": self.intervention_type.value,
            "outcome": self.outcome.value,
            "action_name": self.action_name,
            "action_description": self.action_description,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "policy_name": self.policy_name,
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "risk_level": self.risk_level,
            "blast_radius": self.blast_radius,
            "reason": self.reason,
            "agent_intent": self.agent_intent,
            "original_inputs": self.original_inputs,
            "modified_inputs": self.modified_inputs,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class InterventionTracker:
    """
    Singleton tracker for interventions.
    
    Records interventions to local file and syncs to platform.
    """
    
    _intervention_file: Optional[Path] = None
    _initialized: bool = False
    
    @classmethod
    def initialize(cls, intervention_file: Optional[str] = None) -> None:
        """
        Initialize the intervention tracker.
        
        Args:
            intervention_file: Path to interventions file. If None, uses default location.
        """
        if intervention_file:
            cls._intervention_file = Path(intervention_file)
        else:
            # Use .agent-sentinel directory
            sentinel_home = os.getenv("AGENT_SENTINEL_HOME", ".agent-sentinel")
            sentinel_dir = Path(sentinel_home)
            sentinel_dir.mkdir(parents=True, exist_ok=True)
            cls._intervention_file = sentinel_dir / "interventions.jsonl"
        
        cls._initialized = True
        logger.debug(f"Intervention tracker initialized: {cls._intervention_file}")
    
    @classmethod
    def record(
        cls,
        intervention_type: InterventionType,
        outcome: InterventionOutcome,
        action_name: str,
        action_description: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        estimated_cost: float = 0.0,
        actual_cost: float = 0.0,
        risk_level: str = "medium",
        blast_radius: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        agent_intent: Optional[str] = None,
        original_inputs: Optional[Dict[str, Any]] = None,
        modified_inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an intervention.
        
        This is called whenever Sentinel blocks, pauses, or modifies an action.
        """
        if not cls._initialized:
            cls.initialize()
        
        intervention = InterventionRecord(
            intervention_type=intervention_type,
            outcome=outcome,
            action_name=action_name,
            action_description=action_description,
            agent_id=agent_id,
            run_id=run_id,
            policy_name=policy_name,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            risk_level=risk_level,
            blast_radius=blast_radius or {},
            reason=reason,
            agent_intent=agent_intent,
            original_inputs=original_inputs,
            modified_inputs=modified_inputs,
            context=context or {},
        )
        
        # Write to local file
        try:
            from .ledger import SafeEncoder
            with open(cls._intervention_file, "a") as f:
                f.write(json.dumps(intervention.to_dict(), cls=SafeEncoder) + "\n")
            
            logger.info(
                f"Intervention recorded: {intervention_type.value} -> {outcome.value} "
                f"for '{action_name}'"
            )
        
        except Exception as e:
            logger.error(f"Failed to record intervention: {e}")
    
    @classmethod
    def get_interventions(cls, limit: Optional[int] = None) -> list[InterventionRecord]:
        """
        Read interventions from local file.
        
        Args:
            limit: Maximum number of interventions to return (most recent first)
        
        Returns:
            List of intervention records
        """
        if not cls._initialized:
            cls.initialize()
        
        if not cls._intervention_file.exists():
            return []
        
        interventions = []
        
        try:
            with open(cls._intervention_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Reconstruct the intervention record
                        interventions.append(InterventionRecord(
                            intervention_type=InterventionType(data["intervention_type"]),
                            outcome=InterventionOutcome(data["outcome"]),
                            action_name=data["action_name"],
                            action_description=data.get("action_description"),
                            agent_id=data.get("agent_id"),
                            run_id=data.get("run_id"),
                            policy_name=data.get("policy_name"),
                            estimated_cost=data.get("estimated_cost", 0.0),
                            actual_cost=data.get("actual_cost", 0.0),
                            risk_level=data.get("risk_level", "medium"),
                            blast_radius=data.get("blast_radius", {}),
                            reason=data.get("reason"),
                            agent_intent=data.get("agent_intent"),
                            original_inputs=data.get("original_inputs"),
                            modified_inputs=data.get("modified_inputs"),
                            context=data.get("context", {}),
                            timestamp=data.get("timestamp"),
                        ))
            
            # Return most recent first
            interventions.reverse()
            
            if limit:
                interventions = interventions[:limit]
            
            return interventions
        
        except Exception as e:
            logger.error(f"Failed to read interventions: {e}")
            return []
    
    @classmethod
    def clear(cls) -> None:
        """Clear all intervention records (for testing)."""
        if cls._intervention_file and cls._intervention_file.exists():
            cls._intervention_file.unlink()
        logger.debug("Intervention records cleared")

