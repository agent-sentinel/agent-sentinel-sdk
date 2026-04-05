"""
AgentSentinel Framework Integrations.

This module provides out-of-the-box integrations for popular AI frameworks
and LLM providers, enabling automatic cost tracking and action monitoring.

Integrations:
- LangChain: Callback handler for tracing chains, agents, and tools
- CrewAI: Wrapper for crew actions and task execution
- AutoGen: Inspector for agent-to-agent communication monitoring
- LLM Providers: Instrumentation for OpenAI, Anthropic, Grok, and Gemini

Usage:
    # LangChain
    from agent_sentinel.integrations.langchain import SentinelCallbackHandler
    
    # CrewAI
    from agent_sentinel.integrations.crewai import SentinelCrew
    
    # AutoGen
    from agent_sentinel.integrations.autogen import SentinelInspector
    
    # LLM Instrumentation
    from agent_sentinel.integrations.llm import instrument_openai, instrument_anthropic
"""
from __future__ import annotations

__all__ = []

# Optional imports - only available if dependencies are installed.
# We catch Exception (not just ImportError) because some frameworks
# crash at import time on unsupported Python versions (e.g. chromadb
# on Python 3.14 via pydantic v1 ConfigError).
try:
    from .tools import sentinel_tool
    __all__.append("sentinel_tool")
except Exception:
    sentinel_tool = None  # type: ignore

try:
    from .tool_executor import SentinelToolExecutor, ToolResult
    __all__.extend(["SentinelToolExecutor", "ToolResult"])
except Exception:
    SentinelToolExecutor = None  # type: ignore
    ToolResult = None  # type: ignore

try:
    from .openai import OpenAISentinelTools
    __all__.append("OpenAISentinelTools")
except Exception:
    OpenAISentinelTools = None  # type: ignore

try:
    from .anthropic_tools import AnthropicSentinelTools
    __all__.append("AnthropicSentinelTools")
except Exception:
    AnthropicSentinelTools = None  # type: ignore

try:
    from .langgraph import SentinelToolNode
    __all__.append("SentinelToolNode")
except Exception:
    SentinelToolNode = None  # type: ignore

try:
    from .registry import auto_register_tools, discover_tools
    __all__.extend(["auto_register_tools", "discover_tools"])
except Exception:
    auto_register_tools = None  # type: ignore
    discover_tools = None  # type: ignore

try:
    from .langchain import SentinelCallbackHandler
    __all__.append("SentinelCallbackHandler")
except Exception:
    SentinelCallbackHandler = None  # type: ignore

try:
    from .crewai import SentinelCrew, wrap_crew_action
    __all__.extend(["SentinelCrew", "wrap_crew_action"])
except Exception:
    SentinelCrew = None  # type: ignore
    wrap_crew_action = None  # type: ignore

try:
    from .autogen import SentinelInspector, create_sentinel_agents
    __all__.extend(["SentinelInspector", "create_sentinel_agents"])
except Exception:
    SentinelInspector = None  # type: ignore
    create_sentinel_agents = None  # type: ignore

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
except Exception:
    instrument_openai = None  # type: ignore
    instrument_anthropic = None  # type: ignore
    instrument_grok = None  # type: ignore
    instrument_gemini = None  # type: ignore
    get_token_costs = None  # type: ignore


