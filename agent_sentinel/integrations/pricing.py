"""
Token Pricing Database for LLM Providers.

This module maintains up-to-date pricing information for various LLM providers
to enable accurate cost tracking. Prices are in USD per 1M tokens.

Pricing data last updated: December 2025
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""
    
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    
    def calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """
        Calculate total cost for a given number of tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        return input_cost + output_cost


# OpenAI Pricing (as of December 2025)
OPENAI_PRICING: Dict[str, ModelPricing] = {
    # GPT-4 Models
    "gpt-4": ModelPricing(30.0, 60.0),
    "gpt-4-32k": ModelPricing(60.0, 120.0),
    "gpt-4-turbo": ModelPricing(10.0, 30.0),
    "gpt-4-turbo-preview": ModelPricing(10.0, 30.0),
    "gpt-4-1106-preview": ModelPricing(10.0, 30.0),
    "gpt-4-0125-preview": ModelPricing(10.0, 30.0),
    
    # GPT-4o Models (Optimized)
    "gpt-4o": ModelPricing(5.0, 15.0),
    "gpt-4o-2024-05-13": ModelPricing(5.0, 15.0),
    "gpt-4o-mini": ModelPricing(0.15, 0.6),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.6),
    
    # GPT-3.5 Models
    "gpt-3.5-turbo": ModelPricing(0.5, 1.5),
    "gpt-3.5-turbo-16k": ModelPricing(1.0, 2.0),
    "gpt-3.5-turbo-1106": ModelPricing(1.0, 2.0),
    "gpt-3.5-turbo-0125": ModelPricing(0.5, 1.5),
    
    # o1 Models (Reasoning)
    "o1-preview": ModelPricing(15.0, 60.0),
    "o1-mini": ModelPricing(3.0, 12.0),
    
    # Embedding Models (output tokens = 0)
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}

# Anthropic Pricing (as of December 2025)
ANTHROPIC_PRICING: Dict[str, ModelPricing] = {
    # Claude 3 Opus
    "claude-3-opus-20240229": ModelPricing(15.0, 75.0),
    
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": ModelPricing(3.0, 15.0),
    "claude-3-5-sonnet-20240620": ModelPricing(3.0, 15.0),
    "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0),
    
    # Claude 3 Haiku
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25),
    
    # Legacy models
    "claude-2.1": ModelPricing(8.0, 24.0),
    "claude-2.0": ModelPricing(8.0, 24.0),
    "claude-instant-1.2": ModelPricing(0.8, 2.4),
}

# Grok Pricing (xAI - as of December 2025)
GROK_PRICING: Dict[str, ModelPricing] = {
    "grok-1": ModelPricing(5.0, 15.0),  # Estimated pricing
    "grok-beta": ModelPricing(5.0, 15.0),
}

# Google Gemini Pricing (as of December 2025)
GEMINI_PRICING: Dict[str, ModelPricing] = {
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPricing(3.5, 10.5),
    "gemini-1.5-pro-latest": ModelPricing(3.5, 10.5),
    
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelPricing(0.075, 0.3),
    "gemini-1.5-flash-latest": ModelPricing(0.075, 0.3),
    "gemini-1.5-flash-8b": ModelPricing(0.0375, 0.15),
    
    # Gemini 1.0 Pro
    "gemini-1.0-pro": ModelPricing(0.5, 1.5),
    "gemini-pro": ModelPricing(0.5, 1.5),
}

# Consolidated pricing lookup
ALL_PRICING: Dict[str, Dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "grok": GROK_PRICING,
    "xai": GROK_PRICING,  # xAI alias
    "gemini": GEMINI_PRICING,
    "google": GEMINI_PRICING,  # Google alias
}


def get_model_pricing(
    model: str, 
    provider: Optional[str] = None
) -> Optional[ModelPricing]:
    """
    Get pricing information for a specific model.
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229")
        provider: Optional provider name (openai, anthropic, grok, gemini).
                 If not provided, searches all providers.
    
    Returns:
        ModelPricing object if found, None otherwise
    """
    if provider:
        provider = provider.lower()
        if provider in ALL_PRICING:
            return ALL_PRICING[provider].get(model)
        return None
    
    # Search all providers
    for pricing_dict in ALL_PRICING.values():
        if model in pricing_dict:
            return pricing_dict[model]
    
    return None


def calculate_token_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: Optional[str] = None,
) -> Tuple[float, bool]:
    """
    Calculate the cost for a given number of tokens.
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        provider: Optional provider name
    
    Returns:
        Tuple of (cost in USD, pricing_found)
        If pricing not found, returns (0.0, False)
    """
    pricing = get_model_pricing(model, provider)
    if pricing is None:
        return 0.0, False
    
    cost = pricing.calculate_cost(input_tokens, output_tokens)
    return cost, True


def estimate_cost_from_string(
    text: str,
    model: str,
    provider: Optional[str] = None,
    chars_per_token: int = 4,
) -> float:
    """
    Estimate cost based on string length (rough approximation).
    
    This is a fallback for when exact token counts aren't available.
    Uses rough heuristic of ~4 characters per token.
    
    Args:
        text: Input text
        model: Model identifier
        chars_per_token: Average characters per token (default 4)
        provider: Optional provider name
    
    Returns:
        Estimated cost in USD
    """
    estimated_tokens = len(text) // chars_per_token
    cost, _ = calculate_token_cost(model, estimated_tokens, 0, provider)
    return cost


# Model aliases for easier matching
MODEL_ALIASES: Dict[str, str] = {
    # OpenAI aliases
    "gpt4": "gpt-4",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "chatgpt-4o-latest": "gpt-4o",
    
    # Anthropic aliases  
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-opus": "claude-3-opus-20240229",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-haiku": "claude-3-haiku-20240307",
}


def normalize_model_name(model: str) -> str:
    """
    Normalize model name to canonical form.
    
    Args:
        model: Model identifier (possibly an alias)
    
    Returns:
        Canonical model name
    """
    return MODEL_ALIASES.get(model, model)


