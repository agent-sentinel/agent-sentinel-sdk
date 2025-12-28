"""
LLM Instrumentation for AgentSentinel.

This module provides automatic instrumentation for major LLM providers,
enabling transparent cost tracking without modifying application code.

Supported Providers:
- OpenAI (GPT-3.5, GPT-4, GPT-4o, o1)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- xAI/Grok (Grok-1, Grok-beta)
- Google Gemini (Gemini 1.5 Pro, Flash)

Usage:
    # OpenAI
    from agent_sentinel.integrations.llm import instrument_openai
    import openai
    
    instrument_openai()  # Patches OpenAI client
    
    client = openai.OpenAI()
    response = client.chat.completions.create(...)
    # Cost is automatically tracked!
    
    # Anthropic
    from agent_sentinel.integrations.llm import instrument_anthropic
    import anthropic
    
    instrument_anthropic()
    
    client = anthropic.Anthropic()
    message = client.messages.create(...)
    # Cost tracked automatically!
    
    # Get current costs
    from agent_sentinel.integrations.llm import get_token_costs
    costs = get_token_costs()
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

from ..cost import CostTracker
from ..ledger import Ledger

try:
    from ..integrations.pricing import calculate_token_cost, normalize_model_name
except ImportError:
    # Fallback if pricing module not available
    def calculate_token_cost(*args, **kwargs):  # type: ignore
        return 0.0, False
    
    def normalize_model_name(model: str) -> str:  # type: ignore
        return model

logger = logging.getLogger("agent_sentinel.integrations.llm")

# Track instrumentation state
_INSTRUMENTED: Dict[str, bool] = {
    "openai": False,
    "anthropic": False,
    "grok": False,
    "gemini": False,
}


def instrument_openai(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Instrument OpenAI client to automatically track costs.
    
    This patches the OpenAI client to intercept API calls and track
    token usage and costs transparently.
    
    Args:
        auto_log: Whether to automatically log to ledger (default True)
        tags: Optional tags to apply to all OpenAI calls
    
    Example:
        import openai
        from agent_sentinel.integrations.llm import instrument_openai
        
        instrument_openai()
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Cost automatically tracked in AgentSentinel!
    """
    if _INSTRUMENTED["openai"]:
        logger.warning("OpenAI already instrumented. Skipping.")
        return
    
    try:
        import openai
    except ImportError:
        raise ImportError(
            "OpenAI is not installed. "
            "Install it with: pip install openai"
        )
    
    _wrap_openai_client(openai, auto_log=auto_log, tags=tags or [])
    _INSTRUMENTED["openai"] = True
    logger.info("OpenAI instrumentation enabled")


def instrument_anthropic(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Instrument Anthropic client to automatically track costs.
    
    This patches the Anthropic client to intercept API calls and track
    token usage and costs transparently.
    
    Args:
        auto_log: Whether to automatically log to ledger (default True)
        tags: Optional tags to apply to all Anthropic calls
    
    Example:
        import anthropic
        from agent_sentinel.integrations.llm import instrument_anthropic
        
        instrument_anthropic()
        
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        # Cost automatically tracked!
    """
    if _INSTRUMENTED["anthropic"]:
        logger.warning("Anthropic already instrumented. Skipping.")
        return
    
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic is not installed. "
            "Install it with: pip install anthropic"
        )
    
    _wrap_anthropic_client(anthropic, auto_log=auto_log, tags=tags or [])
    _INSTRUMENTED["anthropic"] = True
    logger.info("Anthropic instrumentation enabled")


def instrument_grok(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Instrument Grok/xAI client to automatically track costs.
    
    Note: Grok uses OpenAI-compatible API, so this wraps the OpenAI client
    with Grok-specific configuration.
    
    Args:
        auto_log: Whether to automatically log to ledger (default True)
        tags: Optional tags to apply to all Grok calls
    
    Example:
        import openai
        from agent_sentinel.integrations.llm import instrument_grok
        
        instrument_grok()
        
        client = openai.OpenAI(
            api_key="your-grok-key",
            base_url="https://api.x.ai/v1"
        )
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    if _INSTRUMENTED["grok"]:
        logger.warning("Grok already instrumented. Skipping.")
        return
    
    # Grok uses OpenAI-compatible API
    instrument_openai(auto_log=auto_log, tags=(tags or []) + ["grok"])
    _INSTRUMENTED["grok"] = True
    logger.info("Grok instrumentation enabled")


def instrument_gemini(
    auto_log: bool = True,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Instrument Google Gemini client to automatically track costs.
    
    Args:
        auto_log: Whether to automatically log to ledger (default True)
        tags: Optional tags to apply to all Gemini calls
    
    Example:
        import google.generativeai as genai
        from agent_sentinel.integrations.llm import instrument_gemini
        
        instrument_gemini()
        
        genai.configure(api_key="your-api-key")
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content("Hello!")
        # Cost automatically tracked!
    """
    if _INSTRUMENTED["gemini"]:
        logger.warning("Gemini already instrumented. Skipping.")
        return
    
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "Google Generative AI is not installed. "
            "Install it with: pip install google-generativeai"
        )
    
    _wrap_gemini_client(genai, auto_log=auto_log, tags=tags or [])
    _INSTRUMENTED["gemini"] = True
    logger.info("Gemini instrumentation enabled")


def _wrap_openai_client(
    openai_module: Any,
    auto_log: bool,
    tags: list[str],
) -> None:
    """Wrap OpenAI client methods to track costs."""
    # Store original methods
    original_create = None
    
    try:
        # Try to get the chat completions create method
        from openai.resources.chat import completions
        original_create = completions.Completions.create
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to access OpenAI client methods: {e}")
        return
    
    def wrapped_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrapped create method that tracks costs."""
        start_time = time.perf_counter()
        
        # Call original method
        response = original_create(self, *args, **kwargs)
        
        duration = time.perf_counter() - start_time
        
        # Extract token usage
        try:
            usage = response.usage
            model = response.model
            
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            
            # Normalize model name
            model = normalize_model_name(model)
            
            # Calculate cost
            cost, pricing_found = calculate_token_cost(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                provider="openai",
            )
            
            if not pricing_found:
                logger.warning(
                    f"No pricing data found for OpenAI model: {model}"
                )
            
            # Track cost
            action_name = f"openai:{model}"
            CostTracker.add_cost(action_name, cost)
            
            # Log to ledger
            if auto_log:
                combined_tags = ["openai", "llm"] + tags
                combined_tags.append(f"model:{model}")
                
                Ledger.log(
                    action=action_name,
                    status="completed",
                    cost_usd=cost,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "provider": "openai",
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "pricing_found": pricing_found,
                    },
                    tags=combined_tags,
                )
            
            logger.debug(
                f"OpenAI call: {model} | Tokens: {total_tokens} | "
                f"Cost: ${cost:.6f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to track OpenAI costs: {e}")
        
        return response
    
    # Patch the method
    completions.Completions.create = wrapped_create


def _wrap_anthropic_client(
    anthropic_module: Any,
    auto_log: bool,
    tags: list[str],
) -> None:
    """Wrap Anthropic client methods to track costs."""
    # Store original method
    original_create = None
    
    try:
        from anthropic.resources import messages
        original_create = messages.Messages.create
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to access Anthropic client methods: {e}")
        return
    
    def wrapped_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrapped create method that tracks costs."""
        start_time = time.perf_counter()
        
        # Call original method
        message = original_create(self, *args, **kwargs)
        
        duration = time.perf_counter() - start_time
        
        # Extract token usage
        try:
            usage = message.usage
            model = message.model
            
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            
            # Normalize model name
            model = normalize_model_name(model)
            
            # Calculate cost
            cost, pricing_found = calculate_token_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider="anthropic",
            )
            
            if not pricing_found:
                logger.warning(
                    f"No pricing data found for Anthropic model: {model}"
                )
            
            # Track cost
            action_name = f"anthropic:{model}"
            CostTracker.add_cost(action_name, cost)
            
            # Log to ledger
            if auto_log:
                combined_tags = ["anthropic", "llm"] + tags
                combined_tags.append(f"model:{model}")
                
                Ledger.log(
                    action=action_name,
                    status="completed",
                    cost_usd=cost,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "provider": "anthropic",
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "pricing_found": pricing_found,
                    },
                    tags=combined_tags,
                )
            
            logger.debug(
                f"Anthropic call: {model} | Tokens: {input_tokens + output_tokens} | "
                f"Cost: ${cost:.6f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to track Anthropic costs: {e}")
        
        return message
    
    # Patch the method
    messages.Messages.create = wrapped_create


def _wrap_gemini_client(
    genai_module: Any,
    auto_log: bool,
    tags: list[str],
) -> None:
    """Wrap Gemini client methods to track costs."""
    try:
        original_generate = genai_module.GenerativeModel.generate_content
    except AttributeError as e:
        logger.error(f"Failed to access Gemini client methods: {e}")
        return
    
    def wrapped_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrapped generate_content method that tracks costs."""
        start_time = time.perf_counter()
        
        # Call original method
        response = original_generate(self, *args, **kwargs)
        
        duration = time.perf_counter() - start_time
        
        # Extract token usage
        try:
            model = self.model_name
            
            # Gemini returns usage metadata
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(
                    usage_metadata, "candidates_token_count", 0
                )
                total_tokens = getattr(usage_metadata, "total_token_count", 0)
            else:
                # Fallback: estimate from text
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
            
            # Normalize model name
            model = normalize_model_name(model)
            
            # Calculate cost
            cost, pricing_found = calculate_token_cost(
                model=model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                provider="gemini",
            )
            
            if not pricing_found:
                logger.warning(
                    f"No pricing data found for Gemini model: {model}"
                )
            
            # Track cost
            action_name = f"gemini:{model}"
            CostTracker.add_cost(action_name, cost)
            
            # Log to ledger
            if auto_log:
                combined_tags = ["gemini", "google", "llm"] + tags
                combined_tags.append(f"model:{model}")
                
                Ledger.log(
                    action=action_name,
                    status="completed",
                    cost_usd=cost,
                    duration_ns=int(duration * 1e9),
                    metadata={
                        "provider": "gemini",
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "pricing_found": pricing_found,
                    },
                    tags=combined_tags,
                )
            
            logger.debug(
                f"Gemini call: {model} | Tokens: {total_tokens} | "
                f"Cost: ${cost:.6f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to track Gemini costs: {e}")
        
        return response
    
    # Patch the method
    genai_module.GenerativeModel.generate_content = wrapped_generate


def get_token_costs() -> Dict[str, Any]:
    """
    Get current token costs across all providers.
    
    Returns:
        Dict with cost breakdown by provider and model
    
    Example:
        from agent_sentinel.integrations.llm import get_token_costs
        
        costs = get_token_costs()
        print(f"Total: ${costs['total_usd']:.6f}")
        print(f"By model: {costs['by_model']}")
    """
    snapshot = CostTracker.get_snapshot()
    
    # Group by provider
    by_provider: Dict[str, float] = {}
    by_model: Dict[str, float] = {}
    
    for action, cost in snapshot["action_costs"].items():
        # Extract provider from action name (e.g., "openai:gpt-4")
        if ":" in action:
            provider, model = action.split(":", 1)
            by_provider[provider] = by_provider.get(provider, 0.0) + cost
            by_model[action] = cost
    
    return {
        "total_usd": snapshot["run_total"],
        "session_total_usd": snapshot["session_total"],
        "by_provider": by_provider,
        "by_model": by_model,
        "action_counts": snapshot["action_counts"],
    }


def reset_instrumentation() -> None:
    """
    Reset instrumentation state.
    
    This allows re-instrumenting clients if needed.
    Warning: This does not un-patch already patched methods.
    """
    global _INSTRUMENTED
    _INSTRUMENTED = {
        "openai": False,
        "anthropic": False,
        "grok": False,
        "gemini": False,
    }
    logger.info("Instrumentation state reset")


def is_instrumented(provider: str) -> bool:
    """
    Check if a provider is currently instrumented.
    
    Args:
        provider: Provider name (openai, anthropic, grok, gemini)
    
    Returns:
        True if instrumented, False otherwise
    """
    return _INSTRUMENTED.get(provider.lower(), False)


