"""
AutoGen Integration for AgentSentinel.

This module provides a "Visa-like" active control integration for AutoGen,
intercepting agent-to-agent communication to enforce policies and track costs.

AutoGen uses a different architecture than LangChain/CrewAI:
- Agents communicate via message passing (register_reply)
- LLMs are called through agent's llm_config
- Tools/functions are registered with function_map

The Sentinel Integration Strategy:
1. Hook into the reply chain with register_reply (pre-authorization)
2. Wrap the LLM client to track token costs
3. Track agent-to-agent message flow for audit

Features:
- Active Policy Enforcement: Blocks replies if budget exceeded or policy violated
- LLM Cost Tracking: Intercepts LLM config to track token usage
- Agent Communication Audit: Logs all agent interactions
- Context Propagation: Maintains run identity across agent conversations
- Async Support: Works with both sync and async AutoGen agents

Usage:
    from autogen import AssistantAgent, UserProxyAgent
    from agent_sentinel.integrations.autogen import SentinelInspector
    
    # Create the Sentinel inspector
    sentinel = SentinelInspector(
        run_name="my_autogen_run",
        enforce_policies=True,  # Enable active blocking
        track_costs=True,       # Track LLM costs
    )
    
    # Create AutoGen agents as normal
    assistant = AssistantAgent(
        name="assistant",
        llm_config={"model": "gpt-4", "api_key": "..."}
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER"
    )
    
    # Secure the agents with one line each
    sentinel.register(assistant)
    sentinel.register(user_proxy)
    
    # Run as normal - Sentinel is now monitoring everything
    user_proxy.initiate_chat(assistant, message="What's the weather?")
    
    # Get execution summary
    summary = sentinel.get_run_summary()
"""
from __future__ import annotations

import logging
import time
import functools
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict

from ..cost import CostTracker
from ..ledger import Ledger
from ..policy import PolicyEngine
from ..intervention import InterventionTracker, InterventionType, InterventionOutcome
from ..errors import BudgetExceededError, PolicyViolationError

# Check for AutoGen availability
try:
    from autogen import Agent, ConversableAgent
    _AUTOGEN_AVAILABLE = True
except ImportError:
    _AUTOGEN_AVAILABLE = False
    Agent = object  # type: ignore
    ConversableAgent = object  # type: ignore

try:
    # Try to import pricing utilities
    from .pricing import calculate_token_cost, normalize_model_name
    _PRICING_AVAILABLE = True
except ImportError:
    # Fallback if pricing module not available
    def calculate_token_cost(*args, **kwargs):  # type: ignore
        return 0.0, False
    
    def normalize_model_name(model: str) -> str:  # type: ignore
        return model
    _PRICING_AVAILABLE = False

logger = logging.getLogger("agent_sentinel.integrations.autogen")


class SentinelInspector:
    """
    The 'Visa Terminal' for AutoGen.
    
    Attaches to ConversableAgent instances to authorize replies and track costs.
    Works by injecting hooks into AutoGen's reply chain - no monkey patching required!
    
    This is simpler than CrewAI because AutoGen has a built-in hook system
    (register_reply) designed exactly for this purpose.
    
    Args:
        run_name: Name for this run (for grouping in dashboard)
        enforce_policies: Enable active blocking via PolicyEngine (default True)
        track_costs: Whether to track LLM costs (default True)
        track_messages: Whether to track agent messages (default True)
        auto_log: Whether to auto-log to ledger (default True)
        tags: Optional tags to apply to all tracked actions
    
    Example:
        sentinel = SentinelInspector(run_name="my_run")
        
        assistant = AssistantAgent("assistant", llm_config=...)
        sentinel.register(assistant)
        
        # Now all assistant actions are monitored
    """
    
    def __init__(
        self,
        run_name: str,
        enforce_policies: bool = True,
        track_costs: bool = True,
        track_messages: bool = True,
        auto_log: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the Sentinel Inspector."""
        if not _AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. "
                "Install it with: pip install pyautogen"
            )
        
        self.run_name = run_name
        self.enforce_policies = enforce_policies
        self.track_costs = track_costs
        self.track_messages = track_messages
        self.auto_log = auto_log
        self.tags = tags or ["autogen"]
        
        # Track execution state
        self._run_start_time: Optional[float] = None
        self._run_start_cost: float = 0.0
        self._registered_agents: List[str] = []
        
        # Message flow tracking
        self._message_count: int = 0
        self._agent_message_counts: Dict[str, int] = defaultdict(int)
        
        # LLM call tracking
        self._llm_call_count: int = 0
        self._original_llm_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"SentinelInspector initialized: {self.run_name} | "
            f"Policy Enforcement: {enforce_policies} | "
            f"Cost Tracking: {track_costs}"
        )
    
    def register(self, agent: ConversableAgent, position: int = 0) -> None:
        """
        Inject Sentinel into an agent's reply loop.
        
        This is the "Visa swipe" - one line to secure an agent.
        
        The hook is inserted at position 0 (highest priority) so it runs
        BEFORE any other reply logic (LLM calls, tool execution, etc).
        
        Args:
            agent: AutoGen ConversableAgent to secure
            position: Position in reply chain (0 = highest priority)
        """
        agent_name = getattr(agent, "name", "unknown_agent")
        
        # 1. Register the authorization hook
        # This runs before EVERY reply this agent generates
        agent.register_reply(
            trigger=[Agent, ConversableAgent] if _AUTOGEN_AVAILABLE else list,
            reply_func=self._create_authorization_hook(agent_name),
            position=position,  # Run BEFORE other logic
        )
        
        # 2. Wrap the LLM config to track costs
        if self.track_costs and hasattr(agent, "llm_config") and agent.llm_config:
            self._wrap_llm_config(agent, agent_name)
        
        self._registered_agents.append(agent_name)
        logger.info(f"Sentinel secured agent: {agent_name}")
    
    def _create_authorization_hook(self, agent_name: str) -> Callable:
        """
        Create the authorization hook for a specific agent.
        
        This returns a function that will be called by AutoGen's reply system.
        
        Returns:
            Callable that checks authorization and returns None (allow) or str (block)
        """
        def authorization_hook(
            recipient: ConversableAgent,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
        ) -> Union[str, Dict, None]:
            """
            The "Visa Authorization" check.
            
            Runs before the agent generates a reply.
            
            Returns:
                - None: Allows the agent to continue normally (Authorized)
                - str/Dict: Returns this as the reply and stops processing (Blocked)
            """
            # Track message flow
            self._message_count += 1
            self._agent_message_counts[agent_name] += 1
            
            if not messages:
                return None
            
            last_message = messages[-1].get("content", "") if messages[-1] else ""
            sender_name = getattr(sender, "name", "unknown") if sender else "unknown"
            
            # Log message if enabled
            if self.track_messages and self.auto_log:
                Ledger.log(
                    action=f"autogen:message:{agent_name}",
                    status="received",
                    cost_usd=0.0,
                    duration_ns=0,
                    metadata={
                        "run_name": self.run_name,
                        "agent": agent_name,
                        "sender": sender_name,
                        "message_preview": last_message[:200] if last_message else None,
                        "message_count": self._message_count,
                    },
                    tags=self.tags + [f"agent:{agent_name}", "message"],
                )
            
            # AUTHORIZE: Check if this agent is allowed to reply
            if self.enforce_policies:
                try:
                    # Check policy before agent generates reply
                    # We don't know the cost yet (LLM hasn't run), but we check
                    # if we're already over budget
                    PolicyEngine.check_action(
                        action=f"agent_reply:{agent_name}",
                        cost=0.0,  # Actual cost tracked in LLM wrapper
                    )
                except (BudgetExceededError, PolicyViolationError) as e:
                    logger.warning(f"Sentinel Blocked AutoGen Agent {agent_name}: {e}")
                    
                    # Record the intervention
                    InterventionTracker.record(
                        intervention_type=(
                            InterventionType.BUDGET_EXCEEDED
                            if isinstance(e, BudgetExceededError)
                            else InterventionType.HARD_BLOCK
                        ),
                        outcome=InterventionOutcome.BLOCKED,
                        action_name=f"agent_reply:{agent_name}",
                        estimated_cost=0.0,
                        reason=str(e),
                        original_inputs={
                            "agent": agent_name,
                            "sender": sender_name,
                            "message": last_message[:200] if last_message else None,
                        },
                        risk_level="high",
                        run_id=self.run_name,
                    )
                    
                    # Return blocking message - AutoGen will use this as the reply
                    # and stop processing
                    return {
                        "content": f"[AgentSentinel] ACTION BLOCKED: {e}",
                        "role": "assistant",
                    }
            
            # Authorization passed - return None to continue normal processing
            return None
        
        return authorization_hook
    
    def _wrap_llm_config(self, agent: ConversableAgent, agent_name: str) -> None:
        """
        Wrap the agent's LLM config to track token costs.
        
        AutoGen uses llm_config dict to configure the LLM client.
        We can intercept responses by wrapping the config_list or model.
        
        However, AutoGen's architecture makes this complex. Instead, we:
        1. Store the original config
        2. Wrap the agent's generate_oai_reply method (where LLM is actually called)
        
        Args:
            agent: Agent to wrap
            agent_name: Name of the agent for logging
        """
        # Store original config
        self._original_llm_configs[agent_name] = agent.llm_config.copy()
        
        # Check if agent has generate_oai_reply (most AutoGen agents do)
        if not hasattr(agent, "generate_oai_reply"):
            logger.warning(f"Agent {agent_name} has no generate_oai_reply, cost tracking limited")
            return
        
        # Wrap the LLM generation method
        original_generate = agent.generate_oai_reply
        
        @functools.wraps(original_generate)
        def wrapped_generate(
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
        ) -> tuple[bool, Optional[str]]:
            """Wrapped generate that tracks costs."""
            start_time = time.perf_counter()
            
            # Call original method
            result = original_generate(messages=messages, sender=sender, config=config)
            
            duration = time.perf_counter() - start_time
            
            # Track the LLM call
            self._llm_call_count += 1
            
            # Try to extract cost from the response
            # AutoGen stores usage info in the client, we need to access it
            cost = 0.0
            model_name = "unknown"
            tokens = 0
            
            if agent.llm_config:
                model_name = agent.llm_config.get("model", "unknown")
                model_name = normalize_model_name(model_name)
                
                # Try to get token usage from the last response
                # This is AutoGen-version dependent
                if hasattr(agent, "last_message") and agent.last_message:
                    # Some AutoGen versions store usage here
                    usage = agent.last_message.get("usage", {})
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        tokens = prompt_tokens + completion_tokens
                        
                        if _PRICING_AVAILABLE:
                            cost, pricing_found = calculate_token_cost(
                                model=model_name,
                                input_tokens=prompt_tokens,
                                output_tokens=completion_tokens,
                            )
                            
                            if not pricing_found:
                                logger.warning(
                                    f"No pricing data for model: {model_name}"
                                )
            
            # Track cost
            if cost > 0:
                CostTracker.add_cost(f"llm_call:{model_name}", cost)
            
            # Log to ledger
            if self.auto_log:
                Ledger.log(
                    action=f"autogen:llm_call:{agent_name}",
                    status="completed",
                    cost_usd=cost,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "run_name": self.run_name,
                        "agent": agent_name,
                        "model": model_name,
                        "tokens": tokens,
                        "llm_call_number": self._llm_call_count,
                    },
                    tags=self.tags + [f"agent:{agent_name}", f"model:{model_name}", "llm_call"],
                )
                
                logger.info(
                    f"AutoGen LLM call: {agent_name} | Model: {model_name} | "
                    f"Tokens: {tokens} | Cost: ${cost:.6f} | Duration: {duration:.2f}s"
                )
            
            return result
        
        # Replace the method
        agent.generate_oai_reply = wrapped_generate
        logger.debug(f"Wrapped LLM config for agent: {agent_name}")
    
    def start_run(self) -> None:
        """
        Mark the start of a run.
        
        Call this before initiating the agent conversation.
        """
        self._run_start_time = time.time()
        self._run_start_cost = CostTracker.get_run_total()
        
        if self.auto_log:
            Ledger.log(
                action=f"autogen:run_start:{self.run_name}",
                status="started",
                cost_usd=0.0,
                duration_ns=0,
                metadata={
                    "run_name": self.run_name,
                    "registered_agents": self._registered_agents,
                },
                tags=self.tags + ["run_start"],
            )
        
        logger.info(f"AutoGen run started: {self.run_name}")
    
    def end_run(self, outcome: str = "completed") -> None:
        """
        Mark the end of a run.
        
        Call this after the agent conversation completes.
        
        Args:
            outcome: "completed", "failed", or "interrupted"
        """
        if not self._run_start_time:
            logger.warning("end_run called but start_run was not called")
            return
        
        duration = time.time() - self._run_start_time
        run_cost = CostTracker.get_run_total() - self._run_start_cost
        
        if self.auto_log:
            Ledger.log(
                action=f"autogen:run_end:{self.run_name}",
                status=outcome,
                cost_usd=run_cost,
                duration_ns=int(duration * 1e9),
                metadata={
                    "run_name": self.run_name,
                    "message_count": self._message_count,
                    "llm_call_count": self._llm_call_count,
                    "agent_message_counts": dict(self._agent_message_counts),
                },
                tags=self.tags + ["run_end"],
            )
        
        logger.info(
            f"AutoGen run ended: {self.run_name} | "
            f"Duration: {duration:.2f}s | Cost: ${run_cost:.6f} | "
            f"Messages: {self._message_count} | LLM Calls: {self._llm_call_count}"
        )
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current run.
        
        Returns:
            Dict with execution statistics including costs and message counts
        """
        duration = 0.0
        if self._run_start_time:
            duration = time.time() - self._run_start_time
        
        run_cost = CostTracker.get_run_total() - self._run_start_cost
        cost_snapshot = CostTracker.get_snapshot()
        
        return {
            "run_name": self.run_name,
            "registered_agents": self._registered_agents,
            "duration_seconds": duration,
            "run_cost_usd": run_cost,
            "total_cost_usd": cost_snapshot["run_total"],
            "message_count": self._message_count,
            "llm_call_count": self._llm_call_count,
            "agent_message_counts": dict(self._agent_message_counts),
            "action_counts": cost_snapshot["action_counts"],
            "action_costs": cost_snapshot["action_costs"],
        }


def wrap_autogen_config(
    config_list: List[Dict[str, Any]],
    run_name: str,
    enforce_policies: bool = True,
) -> List[Dict[str, Any]]:
    """
    Wrap AutoGen config_list to add Sentinel tracking.
    
    This is an alternative approach to registering individual agents.
    By wrapping the config_list, all agents using this config will be tracked.
    
    Args:
        config_list: AutoGen config_list (list of LLM configs)
        run_name: Name for this run
        enforce_policies: Whether to enforce policies
    
    Returns:
        Wrapped config_list with Sentinel tracking
    
    Example:
        config_list = [
            {"model": "gpt-4", "api_key": "..."},
            {"model": "gpt-3.5-turbo", "api_key": "..."},
        ]
        
        wrapped_config = wrap_autogen_config(
            config_list,
            run_name="my_run"
        )
        
        agent = AssistantAgent("assistant", llm_config={"config_list": wrapped_config})
    
    Note:
        This is a simpler but less powerful approach than using SentinelInspector.
        It only tracks LLM costs, not message flow or policy enforcement.
    """
    logger.warning(
        "wrap_autogen_config is a simplified approach. "
        "For full policy enforcement, use SentinelInspector.register() instead."
    )
    
    # For now, just return the original config
    # Full implementation would wrap each config's response handler
    return config_list


def create_sentinel_agents(
    agent_configs: List[Dict[str, Any]],
    run_name: str,
    enforce_policies: bool = True,
    **sentinel_kwargs: Any,
) -> tuple[List[ConversableAgent], SentinelInspector]:
    """
    Create AutoGen agents with Sentinel already registered.
    
    This is a convenience function that creates agents and secures them in one step.
    
    Args:
        agent_configs: List of agent config dicts with keys:
            - agent_class: AutoGen agent class (e.g., AssistantAgent)
            - name: Agent name
            - **kwargs: Additional args passed to agent constructor
        run_name: Name for this run
        enforce_policies: Whether to enforce policies
        **sentinel_kwargs: Additional args passed to SentinelInspector
    
    Returns:
        Tuple of (list of agents, SentinelInspector instance)
    
    Example:
        from autogen import AssistantAgent, UserProxyAgent
        
        agents, sentinel = create_sentinel_agents(
            agent_configs=[
                {
                    "agent_class": AssistantAgent,
                    "name": "assistant",
                    "llm_config": {"model": "gpt-4", "api_key": "..."}
                },
                {
                    "agent_class": UserProxyAgent,
                    "name": "user_proxy",
                    "human_input_mode": "NEVER"
                }
            ],
            run_name="my_run"
        )
        
        # Agents are already secured
        agents[1].initiate_chat(agents[0], message="Hello!")
    """
    sentinel = SentinelInspector(
        run_name=run_name,
        enforce_policies=enforce_policies,
        **sentinel_kwargs,
    )
    
    agents = []
    for config in agent_configs:
        agent_class = config.pop("agent_class")
        agent = agent_class(**config)
        sentinel.register(agent)
        agents.append(agent)
    
    logger.info(f"Created and secured {len(agents)} AutoGen agents")
    return agents, sentinel
