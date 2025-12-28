"""
AgentSentinel Framework Integrations.

This module provides out-of-the-box integrations for popular AI frameworks
and LLM providers, enabling automatic cost tracking and action monitoring.

Integrations:
- LangChain: Callback handler for tracing chains, agents, and tools
- CrewAI: Wrapper for crew actions and task execution
- LLM Providers: Instrumentation for OpenAI, Anthropic, Grok, and Gemini

Usage:
    # LangChain
    from agent_sentinel.integrations.langchain import SentinelCallbackHandler
    
    # CrewAI
    from agent_sentinel.integrations.crewai import SentinelCrew
    
    # LLM Instrumentation
    from agent_sentinel.integrations.llm import instrument_openai, instrument_anthropic
"""
from __future__ import annotations

__all__ = []

# Optional imports - only available if dependencies are installed
try:
    from .langchain import SentinelCallbackHandler
    __all__.append("SentinelCallbackHandler")
except ImportError:
    SentinelCallbackHandler = None  # type: ignore

try:
    from .crewai import SentinelCrew, wrap_crew_action
    __all__.extend(["SentinelCrew", "wrap_crew_action"])
except ImportError:
    SentinelCrew = None  # type: ignore
    wrap_crew_action = None  # type: ignore

try:
    from .llm import (
        instrument_openai,
        instrument_anthropic,
        instrument_grok,
        instrument_gemini,
        get_token_costs,
    )
    __all__.extend([
        "instrument_openai",
        "instrument_anthropic", 
        "instrument_grok",
        "instrument_gemini",
        "get_token_costs",
    ])
except ImportError:
    instrument_openai = None  # type: ignore
    instrument_anthropic = None  # type: ignore
    instrument_grok = None  # type: ignore
    instrument_gemini = None  # type: ignore
    get_token_costs = None  # type: ignore


