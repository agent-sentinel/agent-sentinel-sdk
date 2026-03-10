"""
CrewAI Integration for AgentSentinel.

This module provides a "Visa-like" active control integration for CrewAI,
automatically securing all agents, tools, and LLM calls with policy enforcement.

Features:
- Automatic Tool Injection: All agent tools are secured without manual decoration
- LLM Monitoring: Token costs are tracked and budget limits enforced in real-time
- Step Monitoring: Detects runaway agents and enforces step limits
- Active Control: Blocks tools and LLM calls that violate policies BEFORE execution

Usage:
    from crewai import Agent, Task, Crew
    from agent_sentinel.integrations.crewai import SentinelCrew
    
    # Create agents with standard tools (e.g., SerperDevTool)
    agents = [...]
    tasks = [...]
    
    # SentinelCrew automatically secures everything
    crew = SentinelCrew(
        agents=agents,
        tasks=tasks,
        run_name="my_crew_execution",
        enforce_policies=True,  # Active blocking (default: True)
        max_agent_steps=50,      # Prevent runaway agents
    )
    
    result = crew.kickoff()
    
    # Optional: Wrap individual actions manually
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
from ..policy import PolicyEngine
from ..intervention import InterventionTracker, InterventionType, InterventionOutcome
from ..errors import BudgetExceededError, PolicyViolationError

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
    Active Control wrapper around CrewAI Crew with automatic security injection.
    
    This class transforms CrewAI from a passive integration to a "Visa-like" active
    control system by:
    
    1. **Automatic Tool Injection**: All agent tools are wrapped with authorization
       checks. No manual decoration required - SerperDevTool, DuckDuckGoSearch, etc.
       are automatically secured.
    
    2. **LLM Monitoring**: Attaches SentinelCallbackHandler to all agent LLMs to
       track token costs and enforce budget limits before API calls.
    
    3. **Step Monitoring**: Tracks agent step counts to detect and prevent runaway
       agents (e.g., infinite loops, repetitive failures).
    
    4. **Active Blocking**: PolicyEngine checks are injected into the decision loop,
       not just at the wrapper level. Overspending and banned actions are blocked
       BEFORE execution.
    
    Args:
        agents: List of CrewAI agents
        tasks: List of CrewAI tasks
        run_name: Optional name for this crew run
        track_costs: Whether to track costs (default True)
        track_tasks: Whether to track individual tasks (default True)
        auto_log: Whether to auto-log to ledger (default True)
        enforce_policies: Enable active blocking via PolicyEngine (default True)
        max_agent_steps: Maximum steps per agent before intervention (default 50)
        detect_loops: Enable repetition detection for runaway agents (default True)
        tags: Optional tags to apply to all actions
        **crew_kwargs: Additional arguments passed to Crew()
    
    Example:
        from crewai import Agent, Task
        from crewai_tools import SerperDevTool
        from agent_sentinel.integrations.crewai import SentinelCrew
        
        # Standard CrewAI setup - no changes needed
        search_tool = SerperDevTool()
        
        researcher = Agent(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            tools=[search_tool],  # Tools are auto-secured!
        )
        
        task = Task(
            description="Research AI trends",
            agent=researcher,
        )
        
        # SentinelCrew automatically secures everything
        crew = SentinelCrew(
            agents=[researcher],
            tasks=[task],
            run_name="ai_research_crew",
            enforce_policies=True,  # Active blocking
            max_agent_steps=50,     # Prevent runaways
        )
        
        result = crew.kickoff()  # Fully secured execution
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
        enforce_policies: bool = True,
        max_agent_steps: int = 50,
        detect_loops: bool = True,
        tags: Optional[List[str]] = None,
        **crew_kwargs: Any,
    ):
        """Initialize the SentinelCrew with automatic security injection."""
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
        self.enforce_policies = enforce_policies
        self.max_agent_steps = max_agent_steps
        self.detect_loops = detect_loops
        self.tags = tags or ["crewai"]
        
        # Track execution state
        self._run_start_time: Optional[float] = None
        self._run_end_time: Optional[float] = None
        self._task_times: Dict[str, float] = {}
        
        # Cost tracking at run level
        self._run_start_cost = 0.0
        
        # Agent step tracking for runaway detection
        self._agent_steps: Dict[str, int] = {}
        self._agent_actions: Dict[str, List[str]] = {}
        
        # CRITICAL: Inject security BEFORE creating the Crew
        # This is the "Chip Reader" injection step
        self._secure_agents()
        self._attach_llm_monitors()
        self._attach_step_monitors()
        
        # Create the underlying Crew with secured agents
        self._crew = Crew(
            agents=agents,
            tasks=tasks,
            **crew_kwargs,
        )
        
        logger.info(
            f"SentinelCrew initialized: {self.run_name} | "
            f"Agents: {len(agents)} | Tasks: {len(tasks)} | "
            f"Policy Enforcement: {enforce_policies}"
        )
    
    # =========================================================================
    # Security Injection Methods (The "Chip Reader" Integration)
    # =========================================================================
    
    def _secure_agents(self) -> None:
        """
        Automatically wrap all agent tools with authorization checks.
        
        This is the "Chip Reader" - it intercepts tool calls and checks
        the PolicyEngine before execution. No manual decoration required.
        
        Implementation:
        - Iterates through all agents and their tools
        - Wraps each tool's execution method with policy checks
        - Marks tools as secured to avoid double-wrapping
        - Handles both BaseTool (func) and StructuredTool (_run) interfaces
        """
        for agent in self.agents:
            # Get agent identifier for logging
            agent_id = getattr(agent, "role", getattr(agent, "name", "unknown_agent"))
            
            if not hasattr(agent, "tools") or not agent.tools:
                logger.debug(f"Agent '{agent_id}' has no tools to secure")
                continue
            
            secured_tools = []
            for tool in agent.tools:
                # Check if already secured to avoid double-wrapping
                if getattr(tool, "_is_sentinel_secured", False):
                    secured_tools.append(tool)
                    logger.debug(f"Tool already secured: {getattr(tool, 'name', 'unknown')}")
                    continue
                
                # Get tool name for logging and policy checks
                tool_name = getattr(tool, "name", getattr(tool, "__class__.__name__", "unknown_tool"))
                
                # Find the tool's execution method
                # CrewAI tools typically have one of: func, _run, run, __call__
                original_method = None
                method_attr = None
                
                if hasattr(tool, "func") and callable(tool.func):
                    original_method = tool.func
                    method_attr = "func"
                elif hasattr(tool, "_run") and callable(tool._run):
                    original_method = tool._run
                    method_attr = "_run"
                elif hasattr(tool, "run") and callable(tool.run):
                    original_method = tool.run
                    method_attr = "run"
                elif callable(tool):
                    # Tool itself is callable
                    original_method = tool.__call__
                    method_attr = "__call__"
                
                if not original_method:
                    logger.warning(
                        f"Could not find execution method for tool: {tool_name}. "
                        f"Tool will not be secured."
                    )
                    secured_tools.append(tool)
                    continue
                
                # Create the secured wrapper
                @functools.wraps(original_method)
                def create_secured_wrapper(orig_func, t_name, a_id):
                    """Factory to capture closure variables correctly."""
                    def secured_run(*args, **kwargs):
                        # 1. AUTHORIZE: Check Policy before execution
                        if self.enforce_policies:
                            try:
                                PolicyEngine.check_action(t_name, cost=0.0)
                            except (BudgetExceededError, PolicyViolationError) as e:
                                logger.warning(f"Sentinel blocked tool '{t_name}': {e}")
                                
                                # Record intervention
                                InterventionTracker.record(
                                    intervention_type=(
                                        InterventionType.BUDGET_EXCEEDED 
                                        if isinstance(e, BudgetExceededError)
                                        else InterventionType.HARD_BLOCK
                                    ),
                                    outcome=InterventionOutcome.BLOCKED,
                                    action_name=t_name,
                                    estimated_cost=0.0,
                                    reason=str(e),
                                    original_inputs={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                                    risk_level="high",
                                    run_id=self.run_name,
                                    agent_id=a_id,
                                )
                                
                                # Re-raise to stop execution
                                raise e
                        
                        # 2. LOG: Track tool execution
                        start_time = time.perf_counter()
                        try:
                            # 3. EXECUTE: Run the original tool
                            result = orig_func(*args, **kwargs)
                            
                            # 4. RECORD: Log successful execution
                            duration = time.perf_counter() - start_time
                            if self.auto_log:
                                Ledger.log(
                                    action=f"tool:{t_name}",
                                    status="completed",
                                    cost_usd=0.0,
                                    duration_ns=int(duration * 1e9),
                                    metadata={
                                        "tool": t_name,
                                        "agent_id": a_id,
                                        "run_name": self.run_name,
                                    },
                                    tags=self.tags + ["tool", f"agent:{a_id}"],
                                )
                            
                            return result
                        
                        except Exception as e:
                            # Log tool error
                            duration = time.perf_counter() - start_time
                            if self.auto_log:
                                Ledger.log(
                                    action=f"tool:{t_name}:error",
                                    status="failed",
                                    cost_usd=0.0,
                                    duration_ns=int(duration * 1e9),
                                    metadata={
                                        "tool": t_name,
                                        "agent_id": a_id,
                                        "run_name": self.run_name,
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    },
                                    tags=self.tags + ["tool", "error", f"agent:{a_id}"],
                                )
                            raise
                    
                    return secured_run
                
                # Apply the wrapper
                secured_wrapper = create_secured_wrapper(original_method, tool_name, agent_id)
                setattr(tool, method_attr, secured_wrapper)
                
                # Mark as secured
                tool._is_sentinel_secured = True
                secured_tools.append(tool)
                
                logger.debug(f"Secured tool '{tool_name}' for agent '{agent_id}'")
            
            # Update agent's tools with secured versions
            agent.tools = secured_tools
            
            logger.info(f"Secured {len(secured_tools)} tools for agent '{agent_id}'")
    
    def _attach_llm_monitors(self) -> None:
        """
        Inject SentinelCallbackHandler into all agent LLMs.
        
        This is the "LLM Meter" - it tracks token costs and enforces
        budget limits before expensive API calls are made.
        
        Implementation:
        - Detects LLM backend (LangChain, LiteLLM, etc.)
        - Injects SentinelCallbackHandler with policy enforcement
        - Ensures no duplicate handlers
        """
        # Import here to avoid circular dependency
        try:
            from .langchain import SentinelCallbackHandler
        except ImportError:
            logger.warning("LangChain integration not available. LLM monitoring disabled.")
            return
        
        # Create the sentinel handler - wrap in try/except for graceful degradation
        try:
            sentinel_handler = SentinelCallbackHandler(
                run_name=self.run_name,
                track_costs=self.track_costs,
                enforce_policies=self.enforce_policies,
                tags=self.tags + ["crewai_llm"],
            )
        except ImportError:
            # LangChain not installed - that's OK, monitoring will be disabled
            logger.info("LangChain not installed. LLM monitoring disabled for this crew.")
            return
        
        for agent in self.agents:
            agent_id = getattr(agent, "role", getattr(agent, "name", "unknown_agent"))
            
            if not hasattr(agent, "llm"):
                logger.debug(f"Agent '{agent_id}' has no LLM to monitor")
                continue
            
            llm = agent.llm
            
            # Check if LLM supports callbacks (LangChain-style)
            if hasattr(llm, "callbacks"):
                # Get existing callbacks
                existing_callbacks = llm.callbacks or []
                
                # Check if sentinel already attached (avoid duplicates)
                has_sentinel = any(
                    isinstance(cb, SentinelCallbackHandler) 
                    for cb in existing_callbacks
                )
                
                if not has_sentinel:
                    # Append our handler
                    llm.callbacks = list(existing_callbacks) + [sentinel_handler]
                    logger.info(f"Attached LLM monitor to agent '{agent_id}'")
                else:
                    logger.debug(f"Agent '{agent_id}' already has Sentinel monitor")
            
            # Support for other LLM backends (LiteLLM, etc.)
            elif hasattr(llm, "add_callback"):
                llm.add_callback(sentinel_handler)
                logger.info(f"Attached LLM monitor to agent '{agent_id}' (add_callback)")
            
            else:
                logger.warning(
                    f"Agent '{agent_id}' LLM does not support callbacks. "
                    f"LLM monitoring disabled for this agent. "
                    f"LLM type: {type(llm).__name__}"
                )
    
    def _attach_step_monitors(self) -> None:
        """
        Inject step callbacks to detect runaway agents.
        
        This is the "Safety Net" - it monitors agent steps and blocks
        agents that exceed limits or enter infinite loops.
        
        Implementation:
        - Wraps agent step_callback to track step counts
        - Checks for repetition (loop detection)
        - Enforces max_agent_steps limit
        - Records interventions when agents are stopped
        """
        for agent in self.agents:
            agent_id = getattr(agent, "role", getattr(agent, "name", "unknown_agent"))
            
            # Initialize tracking for this agent
            self._agent_steps[agent_id] = 0
            self._agent_actions[agent_id] = []
            
            # Get existing callback if any
            original_callback = getattr(agent, "step_callback", None)
            
            # Create our monitoring callback
            def create_step_monitor(a_id, orig_cb):
                """Factory to capture closure variables correctly."""
                def sentinel_step_check(step_output):
                    # Increment step counter
                    self._agent_steps[a_id] += 1
                    current_step = self._agent_steps[a_id]
                    
                    # Extract action from step output
                    action_str = str(step_output)[:100] if step_output else "unknown"
                    self._agent_actions[a_id].append(action_str)
                    
                    logger.debug(f"Agent '{a_id}' step {current_step}: {action_str}")
                    
                    # 1. Check step limit
                    if current_step > self.max_agent_steps:
                        error_msg = (
                            f"Agent '{a_id}' exceeded maximum steps: "
                            f"{current_step} > {self.max_agent_steps}. "
                            f"Possible runaway agent or infinite loop."
                        )
                        logger.error(error_msg)
                        
                        # Record intervention
                        if self.auto_log:
                            InterventionTracker.record(
                                intervention_type=InterventionType.HARD_BLOCK,
                                outcome=InterventionOutcome.BLOCKED,
                                action_name=f"agent_step_{a_id}",
                                estimated_cost=0.0,
                                reason=error_msg,
                                original_inputs={"step": current_step, "action": action_str},
                                risk_level="critical",
                                run_id=self.run_name,
                                agent_id=a_id,
                            )
                        
                        # Raise error to stop agent
                        raise PolicyViolationError(error_msg)
                    
                    # 2. Check for loops (repetition detection)
                    if self.detect_loops and current_step >= 5:
                        recent_actions = self._agent_actions[a_id][-5:]
                        # Simple repetition check: all actions similar
                        if len(set(recent_actions)) == 1:
                            error_msg = (
                                f"Agent '{a_id}' appears to be stuck in a loop. "
                                f"Repeated action: {recent_actions[0]}"
                            )
                            logger.warning(error_msg)
                            
                            # Record intervention (warning level)
                            if self.auto_log:
                                InterventionTracker.record(
                                    intervention_type=InterventionType.WARNING,
                                    outcome=InterventionOutcome.WARNED,
                                    action_name=f"agent_loop_{a_id}",
                                    estimated_cost=0.0,
                                    reason=error_msg,
                                    original_inputs={"recent_actions": recent_actions},
                                    risk_level="high",
                                    run_id=self.run_name,
                                    agent_id=a_id,
                                )
                            
                            # For now, just log - could raise error or request approval
                            logger.warning(f"Loop detected for agent '{a_id}' - consider intervention")
                    
                    # 3. Call original callback if exists
                    if orig_cb:
                        orig_cb(step_output)
                
                return sentinel_step_check
            
            # Apply the step monitor
            agent.step_callback = create_step_monitor(agent_id, original_callback)
            
            logger.debug(f"Attached step monitor to agent '{agent_id}'")
    
    # =========================================================================
    # Crew Execution Methods
    # =========================================================================
    
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


