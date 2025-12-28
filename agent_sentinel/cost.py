"""
Cost Tracking Module: Static counters for budget enforcement.

Phase 2 Implementation:
- Thread-safe cost accumulation
- Run-level and session-level tracking
- Support for nested contexts (runs within sessions)
- Simple reset mechanism for testing
"""
from __future__ import annotations

import threading
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CostCounter:
    """
    Accumulates costs at different scopes.
    
    Scopes:
    - session: Entire lifetime of the application
    - run: A single agent execution (can be reset between runs)
    - action: Individual action costs (tracked in ledger)
    """
    session_total: float = 0.0
    run_total: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_costs: Dict[str, float] = field(default_factory=dict)


class CostTracker:
    """
    Thread-safe singleton for tracking costs across the SDK.
    
    Usage:
        # Record a cost
        CostTracker.add_cost("search_web", 0.02)
        
        # Check current run total
        total = CostTracker.get_run_total()
        
        # Reset for new run
        CostTracker.reset_run()
    """
    
    _lock = threading.Lock()
    _counter = CostCounter()
    
    @classmethod
    def add_cost(cls, action: str, cost_usd: float) -> None:
        """
        Add a cost to both session and run totals.
        
        Thread-safe. Can be called from multiple threads simultaneously.
        
        Args:
            action: Name of the action being costed
            cost_usd: Cost in USD to add
        """
        with cls._lock:
            cls._counter.session_total += cost_usd
            cls._counter.run_total += cost_usd
            
            # Track per-action statistics
            cls._counter.action_counts[action] = \
                cls._counter.action_counts.get(action, 0) + 1
            cls._counter.action_costs[action] = \
                cls._counter.action_costs.get(action, 0.0) + cost_usd
    
    @classmethod
    def get_session_total(cls) -> float:
        """
        Get total cost for the entire session (application lifetime).
        
        Returns:
            Total cost in USD since application started
        """
        with cls._lock:
            return cls._counter.session_total
    
    @classmethod
    def get_run_total(cls) -> float:
        """
        Get total cost for the current run.
        
        A "run" is typically one agent execution. Call reset_run() between runs.
        
        Returns:
            Total cost in USD for current run
        """
        with cls._lock:
            return cls._counter.run_total
    
    @classmethod
    def get_action_stats(cls, action: Optional[str] = None) -> Dict:
        """
        Get statistics for actions.
        
        Args:
            action: If specified, returns stats for that action only.
                   If None, returns all action stats.
        
        Returns:
            Dict with 'count' and 'total_cost' for the action(s)
        """
        with cls._lock:
            if action:
                return {
                    "count": cls._counter.action_counts.get(action, 0),
                    "total_cost": cls._counter.action_costs.get(action, 0.0)
                }
            else:
                return {
                    "counts": dict(cls._counter.action_counts),
                    "costs": dict(cls._counter.action_costs)
                }
    
    @classmethod
    def reset_run(cls) -> None:
        """
        Reset the run-level counter.
        
        Call this at the start of each new agent run.
        Does NOT reset session total or action statistics.
        """
        with cls._lock:
            cls._counter.run_total = 0.0
    
    @classmethod
    def reset_all(cls) -> None:
        """
        Reset all counters.
        
        Useful for testing. In production, you typically only reset_run()
        between executions.
        """
        with cls._lock:
            cls._counter = CostCounter()
    
    @classmethod
    def get_snapshot(cls) -> Dict:
        """
        Get a complete snapshot of all cost data.
        
        Returns:
            Dict with session_total, run_total, and action statistics
        """
        with cls._lock:
            return {
                "session_total": cls._counter.session_total,
                "run_total": cls._counter.run_total,
                "action_counts": dict(cls._counter.action_counts),
                "action_costs": dict(cls._counter.action_costs)
            }

