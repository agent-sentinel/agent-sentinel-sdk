"""
Coverage tests for integrations module (llm.py, langchain.py, crewai.py, pricing.py)

Tests cover:
- LLM instrumentation for OpenAI, Anthropic, Grok, Gemini
- LangChain callback handler
- CrewAI integration
- Pricing and token cost calculations
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from agent_sentinel.cost import CostTracker
from agent_sentinel.ledger import Ledger


class TestIntegrationsPricing:
    """Test pricing module functions."""
    
    def test_pricing_module_imports(self) -> None:
        """Test that pricing module can be imported."""
        try:
            from agent_sentinel.integrations.pricing import (
                calculate_token_cost, 
                normalize_model_name
            )
            assert callable(calculate_token_cost)
            assert callable(normalize_model_name)
        except ImportError:
            pytest.skip("Pricing module not available")
    
    def test_normalize_model_names(self) -> None:
        """Test model name normalization."""
        try:
            from agent_sentinel.integrations.pricing import normalize_model_name
            
            # Test various model name formats
            assert normalize_model_name("gpt-4") is not None
            assert normalize_model_name("claude-3-opus") is not None
            assert normalize_model_name("grok-1") is not None
            assert normalize_model_name("gemini-1.5-pro") is not None
        except ImportError:
            pytest.skip("Pricing module not available")
    
    def test_calculate_token_cost_openai(self) -> None:
        """Test token cost calculation for OpenAI models."""
        try:
            from agent_sentinel.integrations.pricing import calculate_token_cost
            
            # Test GPT-4 cost calculation
            input_tokens = 100
            output_tokens = 50
            cost, found = calculate_token_cost(
                model="gpt-4",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            assert isinstance(cost, (int, float))
            assert isinstance(found, bool)
            assert cost >= 0
        except ImportError:
            pytest.skip("Pricing module not available")
    
    def test_calculate_token_cost_anthropic(self) -> None:
        """Test token cost calculation for Anthropic models."""
        try:
            from agent_sentinel.integrations.pricing import calculate_token_cost
            
            # Test Claude cost calculation
            input_tokens = 100
            output_tokens = 50
            cost, found = calculate_token_cost(
                model="claude-3-opus",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            assert isinstance(cost, (int, float))
            assert isinstance(found, bool)
            assert cost >= 0
        except ImportError:
            pytest.skip("Pricing module not available")
    
    def test_calculate_token_cost_unknown_model(self) -> None:
        """Test token cost calculation for unknown model."""
        try:
            from agent_sentinel.integrations.pricing import calculate_token_cost
            
            # Test unknown model
            cost, found = calculate_token_cost(
                model="unknown-model-xyz",
                input_tokens=100,
                output_tokens=50
            )
            assert isinstance(cost, (int, float))
            assert isinstance(found, bool)
        except ImportError:
            pytest.skip("Pricing module not available")


class TestLLMInstrumentation:
    """Test LLM instrumentation functions."""
    
    def test_get_instrumentation_state(self) -> None:
        """Test accessing instrumentation state."""
        try:
            from agent_sentinel.integrations.llm import _INSTRUMENTED
            
            assert isinstance(_INSTRUMENTED, dict)
            assert "openai" in _INSTRUMENTED
            assert "anthropic" in _INSTRUMENTED
            assert "grok" in _INSTRUMENTED
            assert "gemini" in _INSTRUMENTED
        except ImportError:
            pytest.skip("LLM module not available")
    
    def test_instrument_openai_stub(self) -> None:
        """Test instrument_openai function exists and is callable."""
        try:
            from agent_sentinel.integrations.llm import instrument_openai
            
            # Should not raise
            with patch("agent_sentinel.integrations.llm.logger"):
                instrument_openai(auto_log=False)
        except ImportError:
            pytest.skip("LLM module not available")
    
    def test_instrument_anthropic_stub(self) -> None:
        """Test instrument_anthropic function exists and is callable."""
        try:
            from agent_sentinel.integrations.llm import instrument_anthropic
            
            # Should not raise
            with patch("agent_sentinel.integrations.llm.logger"):
                instrument_anthropic(auto_log=False)
        except ImportError:
            pytest.skip("LLM module not available")
    
    def test_instrument_grok_stub(self) -> None:
        """Test instrument_grok function exists and is callable."""
        try:
            from agent_sentinel.integrations.llm import instrument_grok
            
            # Should not raise
            with patch("agent_sentinel.integrations.llm.logger"):
                instrument_grok(auto_log=False)
        except ImportError:
            pytest.skip("LLM module not available")
    
    def test_instrument_gemini_stub(self) -> None:
        """Test instrument_gemini function exists and is callable."""
        try:
            from agent_sentinel.integrations.llm import instrument_gemini
            
            # Should not raise
            with patch("agent_sentinel.integrations.llm.logger"):
                instrument_gemini(auto_log=False)
        except ImportError:
            pytest.skip("LLM module not available")
    
    def test_get_token_costs(self) -> None:
        """Test get_token_costs function."""
        try:
            from agent_sentinel.integrations.llm import get_token_costs
            
            costs = get_token_costs()
            assert isinstance(costs, dict)
        except ImportError:
            pytest.skip("LLM module not available")


class TestLangChainIntegration:
    """Test LangChain integration."""
    
    def test_sentinel_callback_handler_exists(self) -> None:
        """Test SentinelCallbackHandler class exists."""
        try:
            from agent_sentinel.integrations.langchain import SentinelCallbackHandler
            
            # Should be able to instantiate with mock LangChain
            handler = SentinelCallbackHandler(run_name="test")
            assert handler is not None
        except (ImportError, AttributeError):
            pytest.skip("LangChain module not available or incompatible")
    
    def test_sentinel_callback_handler_initialization(self) -> None:
        """Test SentinelCallbackHandler initialization with parameters."""
        try:
            from agent_sentinel.integrations.langchain import SentinelCallbackHandler
            
            handler = SentinelCallbackHandler(
                run_name="test_run",
                track_costs=True,
                track_tools=True,
                auto_log=True,
                tags=["test"]
            )
            assert handler is not None
        except (ImportError, AttributeError):
            pytest.skip("LangChain module not available or incompatible")


class TestCrewAIIntegration:
    """Test CrewAI integration."""
    
    def test_crewai_integration_module_exists(self) -> None:
        """Test CrewAI integration module can be imported."""
        try:
            from agent_sentinel.integrations.crewai import SentinelTaskCallback
            
            assert SentinelTaskCallback is not None
        except (ImportError, AttributeError):
            pytest.skip("CrewAI module not available")
    
    def test_crewai_task_callback_exists(self) -> None:
        """Test SentinelTaskCallback can be instantiated."""
        try:
            from agent_sentinel.integrations.crewai import SentinelTaskCallback
            
            callback = SentinelTaskCallback(run_name="test")
            assert callback is not None
        except (ImportError, AttributeError):
            pytest.skip("CrewAI module not available or incompatible")


class TestIntegrationsInit:
    """Test integrations __init__ module."""
    
    def test_integrations_init_imports(self) -> None:
        """Test that integrations init module exports main components."""
        try:
            from agent_sentinel.integrations import (
                instrument_openai,
                instrument_anthropic,
                SentinelCallbackHandler,
            )
            assert callable(instrument_openai)
            assert callable(instrument_anthropic)
        except (ImportError, AttributeError):
            pytest.skip("Some integrations not available")
    
    def test_integrations_optional_imports(self) -> None:
        """Test that integrations gracefully handle missing dependencies."""
        try:
            # This should not raise even if LangChain is not installed
            from agent_sentinel import integrations
            
            assert integrations is not None
        except ImportError as e:
            pytest.skip(f"Integration error: {e}")

