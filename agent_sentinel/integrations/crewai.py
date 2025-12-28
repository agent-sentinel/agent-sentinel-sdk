"""
CrewAI Integration for AgentSentinel.

This module provides wrappers and utilities for integrating AgentSentinel
with CrewAI, enabling automatic tracking of crew actions, tasks, and costs.

Usage:
    from crewai import Agent, Task, Crew
    from agent_sentinel.integrations.crewai import SentinelCrew, wrap_crew_action
    
    # Option 1: Use SentinelCrew wrapper
    agents = [...]
    tasks = [...]
    
    crew = SentinelCrew(
        agents=agents,
        tasks=tasks,
        run_name="my_crew_execution",
        track_costs=True
    )
    
    result = crew.kickoff()
    
    # Option 2: Wrap individual actions
    @wrap_crew_action(name="research_task", cost_usd=0.05)
    def research_action(query):
        # Your action implementation
        return results
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..guard import guarded_action
from ..cost import CostTracker
from ..ledger import Ledger

try:
    from crewai import Crew, Agent, Task
    _CREWAI_AVAILABLE = True
except ImportError:
    # Provide stubs if CrewAI not installed
    Crew = object  # type: ignore
    Agent = object  # type: ignore
    Task = object  # type: ignore
    _CREWAI_AVAILABLE = False

logger = logging.getLogger("agent_sentinel.integrations.crewai")


def wrap_crew_action(
    name: Optional[str] = None,
    cost_usd: float = 0.0,
    tags: Optional[List[str]] = None,
    requires_human_approval: bool = False,
):
    """
    Decorator to wrap CrewAI actions with AgentSentinel tracking.
    
    This is a thin wrapper around @guarded_action that adds CrewAI-specific
    tags and metadata.
    
    Args:
        name: Optional name for the action
        cost_usd: Estimated cost for this action
        tags: Optional tags for categorization
        requires_human_approval: Whether this action requires approval
    
    Returns:
        Decorated function with AgentSentinel tracking
    
    Example:
        @wrap_crew_action(name="web_search", cost_usd=0.02)
        def search_web(query: str) -> str:
            # Perform web search
            return results
    """
    action_tags = ["crewai"] + (tags or [])
    
    return guarded_action(
        name=name,
        cost_usd=cost_usd,
        tags=action_tags,
        requires_human_approval=requires_human_approval,
    )


class SentinelCrew:
    """
    Wrapper around CrewAI Crew that adds AgentSentinel tracking.
    
    This class wraps CrewAI's Crew to automatically track:
    - Crew execution start/end
    - Individual task execution
    - Agent actions and tool usage
    - Total costs and duration
    
    Args:
        agents: List of CrewAI agents
        tasks: List of CrewAI tasks
        run_name: Optional name for this crew run
        track_costs: Whether to track costs (default True)
        track_tasks: Whether to track individual tasks (default True)
        auto_log: Whether to auto-log to ledger (default True)
        tags: Optional tags to apply to all actions
        **crew_kwargs: Additional arguments passed to Crew()
    
    Example:
        from crewai import Agent, Task
        from agent_sentinel.integrations.crewai import SentinelCrew
        
        researcher = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
        )
        
        task = Task(
            description="Research AI trends",
            agent=researcher,
        )
        
        crew = SentinelCrew(
            agents=[researcher],
            tasks=[task],
            run_name="ai_research_crew",
        )
        
        result = crew.kickoff()
        summary = crew.get_run_summary()
    """
    
    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        run_name: Optional[str] = None,
        track_costs: bool = True,
        track_tasks: bool = True,
        auto_log: bool = True,
        tags: Optional[List[str]] = None,
        **crew_kwargs: Any,
    ):
        """Initialize the SentinelCrew wrapper."""
        if not _CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. "
                "Install it with: pip install crewai"
            )
        
        self.agents = agents
        self.tasks = tasks
        self.run_name = run_name or f"crew_run_{int(time.time())}"
        self.track_costs = track_costs
        self.track_tasks = track_tasks
        self.auto_log = auto_log
        self.tags = tags or ["crewai"]
        
        # Create the underlying Crew
        self._crew = Crew(
            agents=agents,
            tasks=tasks,
            **crew_kwargs,
        )
        
        # Track execution state
        self._run_start_time: Optional[float] = None
        self._run_end_time: Optional[float] = None
        self._task_times: Dict[str, float] = {}
        
        # Cost tracking at run level
        self._run_start_cost = 0.0
        
        logger.info(
            f"SentinelCrew initialized: {self.run_name} | "
            f"Agents: {len(agents)} | Tasks: {len(tasks)}"
        )
    
    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the crew with AgentSentinel tracking.
        
        Args:
            inputs: Optional inputs to pass to the crew
        
        Returns:
            Result from crew execution
        """
        # Record starting cost
        self._run_start_cost = CostTracker.get_run_total()
        self._run_start_time = time.time()
        
        if self.auto_log:
            Ledger.log(
                action=f"crew:start:{self.run_name}",
                status="started",
                cost_usd=0.0,
                duration_ns=0,
                metadata={
                    "run_name": self.run_name,
                    "num_agents": len(self.agents),
                    "num_tasks": len(self.tasks),
                    "inputs": inputs,
                },
                tags=self.tags + ["crew_start"],
            )
        
        logger.info(f"Starting crew execution: {self.run_name}")
        
        try:
            # Execute the crew
            result = self._crew.kickoff(inputs=inputs)
            
            self._run_end_time = time.time()
            duration = self._run_end_time - self._run_start_time
            
            # Calculate cost for this run
            run_cost = CostTracker.get_run_total() - self._run_start_cost
            
            if self.auto_log:
                Ledger.log(
                    action=f"crew:complete:{self.run_name}",
                    status="completed",
                    cost_usd=run_cost,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "run_name": self.run_name,
                        "result_preview": str(result)[:200] if result else None,
                    },
                    tags=self.tags + ["crew_complete"],
                )
            
            logger.info(
                f"Crew execution completed: {self.run_name} | "
                f"Duration: {duration:.2f}s | Cost: ${run_cost:.6f}"
            )
            
            return result
            
        except Exception as e:
            self._run_end_time = time.time()
            duration = self._run_end_time - self._run_start_time
            
            if self.auto_log:
                Ledger.log(
                    action=f"crew:error:{self.run_name}",
                    status="failed",
                    cost_usd=0.0,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "run_name": self.run_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    tags=self.tags + ["crew_error", "error"],
                )
            
            logger.error(f"Crew execution failed: {self.run_name} | Error: {e}")
            raise
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the crew execution.
        
        Returns:
            Dict with execution statistics including costs and duration
        """
        duration = 0.0
        if self._run_start_time and self._run_end_time:
            duration = self._run_end_time - self._run_start_time
        
        run_cost = CostTracker.get_run_total() - self._run_start_cost
        cost_snapshot = CostTracker.get_snapshot()
        
        return {
            "run_name": self.run_name,
            "num_agents": len(self.agents),
            "num_tasks": len(self.tasks),
            "duration_seconds": duration,
            "run_cost_usd": run_cost,
            "total_cost_usd": cost_snapshot["run_total"],
            "action_counts": cost_snapshot["action_counts"],
            "action_costs": cost_snapshot["action_costs"],
            "started_at": self._run_start_time,
            "completed_at": self._run_end_time,
        }
    
    @property
    def crew(self) -> Crew:
        """Access the underlying CrewAI Crew object."""
        return self._crew


class SentinelAgent:
    """
    Wrapper around CrewAI Agent with built-in action tracking.
    
    This provides a more granular level of tracking for individual agents.
    
    Args:
        agent: CrewAI Agent instance
        track_actions: Whether to track all agent actions
        agent_id: Optional identifier for this agent
        tags: Optional tags for all agent actions
    
    Example:
        from crewai import Agent
        from agent_sentinel.integrations.crewai import SentinelAgent
        
        base_agent = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
        )
        
        sentinel_agent = SentinelAgent(
            agent=base_agent,
            agent_id="researcher_001",
            track_actions=True,
        )
    """
    
    def __init__(
        self,
        agent: Agent,
        track_actions: bool = True,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the SentinelAgent wrapper."""
        if not _CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. "
                "Install it with: pip install crewai"
            )
        
        self._agent = agent
        self.track_actions = track_actions
        self.agent_id = agent_id or getattr(agent, "role", "unknown_agent")
        self.tags = tags or ["crewai", "agent"]
        
        self._action_count = 0
        
        logger.debug(f"SentinelAgent initialized: {self.agent_id}")
    
    @property
    def agent(self) -> Agent:
        """Access the underlying CrewAI Agent object."""
        return self._agent
    
    def track_action(
        self,
        action_name: str,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Manually track an agent action.
        
        Args:
            action_name: Name of the action
            cost_usd: Cost of the action
            metadata: Optional metadata to log
        """
        if not self.track_actions:
            return
        
        self._action_count += 1
        
        combined_metadata = {
            "agent_id": self.agent_id,
            "action_number": self._action_count,
        }
        if metadata:
            combined_metadata.update(metadata)
        
        Ledger.log(
            action=f"agent_action:{action_name}",
            status="completed",
            cost_usd=cost_usd,
            duration_ns=0,
            metadata=combined_metadata,
            tags=self.tags + [f"agent:{self.agent_id}"],
        )
        
        if cost_usd > 0:
            CostTracker.add_cost(action_name, cost_usd)


# Utility function to wrap existing crew
def wrap_existing_crew(
    crew: Crew,
    run_name: Optional[str] = None,
    **sentinel_kwargs: Any,
) -> SentinelCrew:
    """
    Wrap an existing CrewAI Crew with Sentinel tracking.
    
    Args:
        crew: Existing Crew instance
        run_name: Optional run name
        **sentinel_kwargs: Additional arguments for SentinelCrew
    
    Returns:
        SentinelCrew wrapper around the existing crew
    
    Example:
        crew = Crew(agents=[...], tasks=[...])
        sentinel_crew = wrap_existing_crew(crew, run_name="my_run")
        result = sentinel_crew.kickoff()
    """
    sentinel = SentinelCrew(
        agents=crew.agents,
        tasks=crew.tasks,
        run_name=run_name,
        **sentinel_kwargs,
    )
    sentinel._crew = crew  # Use the existing crew instance
    return sentinel


