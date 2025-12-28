"""
LangChain Integration for AgentSentinel.

This module provides a callback handler that automatically tracks LangChain
chains, agents, and tool usage with cost tracking for LLM calls.

Usage:
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import initialize_agent, Tool
    from agent_sentinel.integrations.langchain import SentinelCallbackHandler
    
    # Create callback handler
    sentinel_handler = SentinelCallbackHandler(
        run_name="my_agent_run",
        track_costs=True
    )
    
    # Use with LangChain
    llm = ChatOpenAI(temperature=0, callbacks=[sentinel_handler])
    
    # Or with agents
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        callbacks=[sentinel_handler]
    )
    
    result = agent.run("What's the weather in SF?")
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from ..cost import CostTracker
from ..ledger import Ledger
from ..errors import AgentSentinelError

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    # Provide stub if LangChain not installed
    BaseCallbackHandler = object  # type: ignore
    _LANGCHAIN_AVAILABLE = False

try:
    from ..integrations.pricing import calculate_token_cost, normalize_model_name
except ImportError:
    # Fallback if pricing module not available
    def calculate_token_cost(*args, **kwargs):  # type: ignore
        return 0.0, False
    
    def normalize_model_name(model: str) -> str:  # type: ignore
        return model

logger = logging.getLogger("agent_sentinel.integrations.langchain")


class SentinelCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that integrates with AgentSentinel.
    
    Automatically tracks:
    - LLM calls with token usage and costs
    - Tool/action executions
    - Chain starts/ends
    - Agent actions
    - Errors and retries
    
    Args:
        run_name: Optional name for this run (for grouping in dashboard)
        track_costs: Whether to automatically track LLM costs (default True)
        track_tools: Whether to track tool/action calls (default True)
        auto_log: Whether to automatically log to ledger (default True)
        tags: Optional tags to apply to all tracked actions
    """
    
    def __init__(
        self,
        run_name: Optional[str] = None,
        track_costs: bool = True,
        track_tools: bool = True,
        auto_log: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the callback handler."""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install langchain"
            )
        
        super().__init__()
        self.run_name = run_name or f"langchain_run_{int(time.time())}"
        self.track_costs = track_costs
        self.track_tools = track_tools
        self.auto_log = auto_log
        self.tags = tags or ["langchain"]
        
        # Track call times and counts
        self._call_times: Dict[str, float] = {}
        self._call_counts: Dict[str, int] = {}
        
        # Session tracking
        self._session_start = time.time()
        
        logger.info(
            f"SentinelCallbackHandler initialized for run: {self.run_name}"
        )
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        self._call_times[str(run_id)] = time.perf_counter()
        
        model_name = serialized.get("name", "unknown_model")
        logger.debug(f"LLM started: {model_name} (run_id: {run_id})")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        if not self.track_costs:
            return
        
        # Extract token usage and calculate costs
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        
        if not token_usage:
            logger.debug(f"No token usage info for run_id: {run_id}")
            return
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        # Get model name
        model_name = llm_output.get("model_name", "unknown")
        model_name = normalize_model_name(model_name)
        
        # Calculate cost based on token usage
        cost, pricing_found = calculate_token_cost(
            model=model_name,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
        
        if not pricing_found:
            logger.warning(
                f"No pricing data found for model: {model_name}. "
                f"Cost tracking will be $0.00"
            )
        
        # Track in CostTracker
        action_name = f"llm_call:{model_name}"
        CostTracker.add_cost(action_name, cost)
        
        # Log to ledger if enabled
        if self.auto_log:
            combined_tags = list(self.tags) + (tags or [])
            combined_tags.append(f"model:{model_name}")
            
            Ledger.log(
                action=action_name,
                status="completed",
                cost_usd=cost,
                duration_ns=int(duration * 1e9),
                metadata={
                    "run_name": self.run_name,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "model": model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "pricing_found": pricing_found,
                },
                tags=combined_tags,
            )
        
        logger.info(
            f"LLM call completed: {model_name} | "
            f"Tokens: {total_tokens} | Cost: ${cost:.6f} | "
            f"Duration: {duration:.2f}s"
        )
    
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        if self.auto_log:
            combined_tags = list(self.tags) + (tags or [])
            combined_tags.append("error")
            
            Ledger.log(
                action=f"llm_call:error",
                status="failed",
                cost_usd=0.0,
                duration_ns=int(duration * 1e9),
                metadata={
                    "run_name": self.run_name,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                tags=combined_tags,
            )
        
        logger.error(f"LLM error (run_id: {run_id}): {error}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        if not self.track_tools:
            return
        
        self._call_times[str(run_id)] = time.perf_counter()
        
        tool_name = serialized.get("name", "unknown_tool")
        logger.debug(f"Tool started: {tool_name} (run_id: {run_id})")
    
    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        if not self.track_tools:
            return
        
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        if self.auto_log:
            combined_tags = list(self.tags) + (tags or [])
            combined_tags.append("tool")
            
            Ledger.log(
                action="tool_call",
                status="completed",
                cost_usd=0.0,  # Tools don't have inherent cost
                duration_ns=int(duration * 1e9),
                metadata={
                    "run_name": self.run_name,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "output_preview": output[:200] if output else None,
                },
                tags=combined_tags,
            )
        
        logger.debug(f"Tool completed (run_id: {run_id}) in {duration:.2f}s")
    
    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        if not self.track_tools:
            return
        
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        if self.auto_log:
            combined_tags = list(self.tags) + (tags or [])
            combined_tags.extend(["tool", "error"])
            
            Ledger.log(
                action="tool_call:error",
                status="failed",
                cost_usd=0.0,
                duration_ns=int(duration * 1e9),
                metadata={
                    "run_name": self.run_name,
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                tags=combined_tags,
            )
        
        logger.error(f"Tool error (run_id: {run_id}): {error}")
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when agent takes an action."""
        if not self.track_tools:
            return
        
        logger.debug(
            f"Agent action: {action.tool} with input: {action.tool_input}"
        )
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when agent finishes."""
        logger.debug(f"Agent finished (run_id: {run_id})")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts."""
        self._call_times[str(run_id)] = time.perf_counter()
        
        chain_type = serialized.get("name", "unknown_chain")
        logger.debug(f"Chain started: {chain_type} (run_id: {run_id})")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends."""
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        logger.debug(f"Chain completed (run_id: {run_id}) in {duration:.2f}s")
    
    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        duration = 0.0
        if str(run_id) in self._call_times:
            start_time = self._call_times.pop(str(run_id))
            duration = time.perf_counter() - start_time
        
        logger.error(f"Chain error (run_id: {run_id}): {error}")
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current run.
        
        Returns:
            Dict with run statistics including costs and counts
        """
        session_duration = time.time() - self._session_start
        cost_snapshot = CostTracker.get_snapshot()
        
        return {
            "run_name": self.run_name,
            "session_duration_seconds": session_duration,
            "total_cost_usd": cost_snapshot["session_total"],
            "run_cost_usd": cost_snapshot["run_total"],
            "action_counts": cost_snapshot["action_counts"],
            "action_costs": cost_snapshot["action_costs"],
        }


# Convenience function for quick setup
def create_sentinel_handler(
    run_name: Optional[str] = None,
    **kwargs: Any,
) -> SentinelCallbackHandler:
    """
    Create a SentinelCallbackHandler with sensible defaults.
    
    Args:
        run_name: Optional name for this run
        **kwargs: Additional arguments passed to SentinelCallbackHandler
    
    Returns:
        Configured SentinelCallbackHandler instance
    """
    return SentinelCallbackHandler(run_name=run_name, **kwargs)


