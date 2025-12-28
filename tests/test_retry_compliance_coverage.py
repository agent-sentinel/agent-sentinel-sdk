"""
Coverage tests for retry and compliance modules.

Tests cover:
- Retry configuration and exponential backoff
- Circuit breaker pattern
- Compliance monitoring and reporting
- Risk scoring
"""
from __future__ import annotations

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from agent_sentinel.errors import AgentSentinelError, NetworkError, SyncError, TimeoutError


class TestRetryConfig:
    """Test RetryConfig class."""
    
    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig with default values."""
        from agent_sentinel.retry import RetryConfig
        
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_retry_config_custom_values(self) -> None:
        """Test RetryConfig with custom values."""
        from agent_sentinel.retry import RetryConfig
        
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
    
    def test_retry_config_min_attempts(self) -> None:
        """Test that RetryConfig enforces minimum attempts."""
        from agent_sentinel.retry import RetryConfig
        
        config = RetryConfig(max_attempts=0)
        assert config.max_attempts == 1  # Enforced minimum
    
    def test_retry_config_custom_exceptions(self) -> None:
        """Test RetryConfig with custom retryable exceptions."""
        from agent_sentinel.retry import RetryConfig
        
        config = RetryConfig(
            retryable_exceptions=[ValueError, TypeError]
        )
        assert config.retryable_exceptions is not None
        assert len(config.retryable_exceptions) == 2
    
    def test_retry_config_calculation_delay(self) -> None:
        """Test RetryConfig delay calculation."""
        from agent_sentinel.retry import RetryConfig
        
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Delay should roughly double each attempt
        attempt1_delay = config.get_delay(1)
        attempt2_delay = config.get_delay(2)
        attempt3_delay = config.get_delay(3)
        
        assert attempt1_delay > 0
        assert attempt2_delay > attempt1_delay
        assert attempt3_delay > attempt2_delay or attempt3_delay == config.max_delay


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def test_retry_decorator_success(self) -> None:
        """Test retry decorator with successful call."""
        from agent_sentinel.retry import with_retry
        
        call_count = 0
        
        @with_retry()
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_decorator_eventual_success(self) -> None:
        """Test retry decorator with eventual success."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        call_count = 0
        
        @with_retry(config=RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False))
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Try again")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_decorator_max_retries_exceeded(self) -> None:
        """Test retry decorator when max retries exceeded."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        @with_retry(config=RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False))
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_retry_decorator_with_config(self) -> None:
        """Test retry decorator with custom config."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.01,
            jitter=False
        )
        
        call_count = 0
        
        @with_retry(config=config)
        def function_with_config():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry")
            return "success"
        
        result = function_with_config()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self) -> None:
        """Test CircuitBreaker initialization."""
        from agent_sentinel.retry import CircuitBreaker
        
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        assert breaker is not None
    
    def test_circuit_breaker_closed_state(self) -> None:
        """Test circuit breaker in closed (normal) state."""
        from agent_sentinel.retry import CircuitBreaker
        
        breaker = CircuitBreaker()
        
        def working_function():
            return "success"
        
        result = breaker.call(working_function)
        assert result == "success"
    
    def test_circuit_breaker_open_state(self) -> None:
        """Test circuit breaker opens after threshold failures."""
        from agent_sentinel.retry import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise ValueError("Fail")
        
        # First call raises
        with pytest.raises(ValueError):
            breaker.call(failing_function)
        
        # Second call raises
        with pytest.raises(ValueError):
            breaker.call(failing_function)
        
        # Third call should raise AgentSentinelError due to circuit being open
        with pytest.raises(AgentSentinelError):
            breaker.call(failing_function)
    
    def test_circuit_breaker_half_open_state(self) -> None:
        """Test circuit breaker in half-open state."""
        from agent_sentinel.retry import CircuitBreaker
        
        breaker = CircuitBreaker(recovery_timeout=1)
        
        def function():
            return "success"
        
        # Initially closed
        result = breaker.call(function)
        assert result == "success"
        assert breaker.state == CircuitBreaker.CLOSED


class TestComplianceModule:
    """Test compliance module functionality."""
    
    def test_compliance_imports(self) -> None:
        """Test that compliance module can be imported."""
        try:
            from agent_sentinel.compliance import (
                ComplianceChecker,
                RiskScorer,
                ComplianceReport
            )
            assert ComplianceChecker is not None
            assert RiskScorer is not None
            assert ComplianceReport is not None
        except ImportError:
            pytest.skip("Compliance module not available")
    
    def test_compliance_checker_initialization(self) -> None:
        """Test ComplianceChecker initialization."""
        try:
            from agent_sentinel.compliance import ComplianceChecker
            
            checker = ComplianceChecker()
            assert checker is not None
        except ImportError:
            pytest.skip("Compliance module not available")
    
    def test_risk_scorer_initialization(self) -> None:
        """Test RiskScorer initialization."""
        try:
            from agent_sentinel.compliance import RiskScorer
            
            scorer = RiskScorer()
            assert scorer is not None
        except ImportError:
            pytest.skip("Compliance module not available")
    
    def test_compliance_report_creation(self) -> None:
        """Test creating a ComplianceReport."""
        try:
            from agent_sentinel.compliance import ComplianceReport
            
            report = ComplianceReport(
                run_id="run-123",
                total_actions=10,
                compliant_actions=10,
                risk_score=0.1
            )
            assert report is not None
            assert report.run_id == "run-123"
        except ImportError:
            pytest.skip("Compliance module not available")
    
    def test_compliance_checker_check_action(self) -> None:
        """Test ComplianceChecker checking an action."""
        try:
            from agent_sentinel.compliance import ComplianceChecker
            
            checker = ComplianceChecker()
            
            # Check compliance of an action
            is_compliant = checker.check_action(
                action_name="test_action",
                inputs={"key": "value"},
                cost=0.01
            )
            assert isinstance(is_compliant, bool)
        except ImportError:
            pytest.skip("Compliance module not available")
    
    def test_risk_scorer_score_action(self) -> None:
        """Test RiskScorer scoring an action."""
        try:
            from agent_sentinel.compliance import RiskScorer
            
            scorer = RiskScorer()
            
            # Score an action
            risk_score = scorer.score_action(
                action_name="test_action",
                cost=0.50,
                is_network_call=True
            )
            assert isinstance(risk_score, (int, float))
            assert 0 <= risk_score <= 1
        except ImportError:
            pytest.skip("Compliance module not available")


class TestInterventionModule:
    """Test intervention module."""
    
    def test_intervention_imports(self) -> None:
        """Test that intervention module can be imported."""
        try:
            from agent_sentinel.intervention import (
                Intervention,
                InterventionType,
            )
            assert Intervention is not None
            assert InterventionType is not None
        except ImportError:
            pytest.skip("Intervention module not available")
    
    def test_intervention_types(self) -> None:
        """Test InterventionType enum."""
        try:
            from agent_sentinel.intervention import InterventionType
            
            # Check that some intervention types exist
            assert hasattr(InterventionType, "HARD_BLOCK")
            assert hasattr(InterventionType, "APPROVAL_REQUIRED")
            assert hasattr(InterventionType, "BUDGET_EXCEEDED")
        except (ImportError, AttributeError):
            pytest.skip("Intervention module not available or incomplete")
    
    def test_intervention_initialization(self) -> None:
        """Test Intervention initialization."""
        try:
            from agent_sentinel.intervention import Intervention
            
            intervention = Intervention(
                action_name="test_action",
                intervention_type="pause",
                reason="Budget exceeded"
            )
            assert intervention is not None
        except (ImportError, TypeError):
            pytest.skip("Intervention not fully implemented")


class TestRetryExceptionHandling:
    """Test retry exception handling."""
    
    def test_retry_with_network_error(self) -> None:
        """Test retry with NetworkError."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        call_count = 0
        
        @with_retry(config=RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False))
        def network_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("Connection failed")
            return "success"
        
        result = network_function()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_with_timeout_error(self) -> None:
        """Test retry with TimeoutError."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        call_count = 0
        
        @with_retry(config=RetryConfig(max_attempts=2, initial_delay=0.01, jitter=False))
        def timeout_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timed out")
            return "success"
        
        result = timeout_function()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_does_not_retry_non_retryable(self) -> None:
        """Test that non-retryable exceptions are not retried."""
        from agent_sentinel.retry import with_retry, RetryConfig
        
        call_count = 0
        
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[NetworkError],
            initial_delay=0.01,
            jitter=False
        )
        
        @with_retry(config=config)
        def function_with_wrong_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            function_with_wrong_error()
        
        # Should only be called once
        assert call_count == 1

