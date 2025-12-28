"""
Coverage tests for __init__ and error modules.

Tests cover:
- Main module exports
- Exception types and inheritance
- Error handling and context
"""
from __future__ import annotations

import pytest
from typing import Type

from agent_sentinel.errors import (
    AgentSentinelError,
    BudgetExceededError,
    PolicyViolationError,
    NetworkError,
    SyncError,
    TimeoutError,
    ConfigurationError,
    ReplayDivergenceError,
)


class TestMainModuleExports:
    """Test that main module exports all necessary components."""
    
    def test_main_imports_guard(self) -> None:
        """Test that guarded_action is exported from main module."""
        from agent_sentinel import guarded_action
        
        assert callable(guarded_action)
    
    def test_main_imports_cost_tracker(self) -> None:
        """Test that CostTracker is exported from main module."""
        from agent_sentinel import CostTracker
        
        assert CostTracker is not None
        assert hasattr(CostTracker, "get_run_total")
        assert hasattr(CostTracker, "get_session_total")
    
    def test_main_imports_policy_engine(self) -> None:
        """Test that PolicyEngine is exported from main module."""
        from agent_sentinel import PolicyEngine
        
        assert PolicyEngine is not None
        assert hasattr(PolicyEngine, "configure")
        assert hasattr(PolicyEngine, "load_from_yaml")
    
    def test_main_imports_errors(self) -> None:
        """Test that error classes are exported from main module."""
        from agent_sentinel import (
            AgentSentinelError,
            BudgetExceededError,
            PolicyViolationError,
        )
        
        assert AgentSentinelError is not None
        assert BudgetExceededError is not None
        assert PolicyViolationError is not None
    
    def test_main_imports_ledger(self) -> None:
        """Test that Ledger is exported from main module."""
        try:
            from agent_sentinel import Ledger
            
            assert Ledger is not None
            assert hasattr(Ledger, "record")
        except (ImportError, AttributeError):
            pytest.skip("Ledger not fully exported")
    
    def test_main_imports_sync(self) -> None:
        """Test that sync functions are exported from main module."""
        from agent_sentinel import enable_remote_sync, flush_and_stop
        
        assert callable(enable_remote_sync)
        assert callable(flush_and_stop)
    
    def test_main_version_available(self) -> None:
        """Test that version is available."""
        try:
            from agent_sentinel import __version__
            assert isinstance(__version__, str)
        except ImportError:
            pytest.skip("Version not exposed in __init__")


class TestErrorHierarchy:
    """Test error class hierarchy and inheritance."""
    
    def test_budget_exceeded_error_inheritance(self) -> None:
        """Test that BudgetExceededError inherits from AgentSentinelError."""
        assert issubclass(BudgetExceededError, AgentSentinelError)
    
    def test_policy_violation_error_inheritance(self) -> None:
        """Test that PolicyViolationError inherits from AgentSentinelError."""
        assert issubclass(PolicyViolationError, AgentSentinelError)
    
    def test_network_error_inheritance(self) -> None:
        """Test that NetworkError inherits from AgentSentinelError."""
        assert issubclass(NetworkError, AgentSentinelError)
    
    def test_sync_error_inheritance(self) -> None:
        """Test that SyncError inherits from AgentSentinelError."""
        assert issubclass(SyncError, AgentSentinelError)
    
    def test_timeout_error_inheritance(self) -> None:
        """Test that TimeoutError inherits from AgentSentinelError."""
        assert issubclass(TimeoutError, AgentSentinelError)
    
    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from AgentSentinelError."""
        assert issubclass(ConfigurationError, AgentSentinelError)
    
    def test_replay_divergence_error_inheritance(self) -> None:
        """Test that ReplayDivergenceError inherits from AgentSentinelError."""
        assert issubclass(ReplayDivergenceError, AgentSentinelError)


class TestErrorInstantiation:
    """Test instantiating error classes."""
    
    def test_agent_sentinel_error_basic(self) -> None:
        """Test creating AgentSentinelError."""
        error = AgentSentinelError("Test error")
        assert str(error) == "Test error"
    
    def test_budget_exceeded_error_message(self) -> None:
        """Test BudgetExceededError with message."""
        error = BudgetExceededError(
            "Budget of $10.00 exceeded",
            spent=10.0,
            limit=5.0
        )
        assert "Budget" in str(error)
        assert error.details["spent"] == 10.0
        assert error.details["limit"] == 5.0
    
    def test_policy_violation_error_message(self) -> None:
        """Test PolicyViolationError with message."""
        error = PolicyViolationError("Action 'delete_database' is denied")
        assert "delete_database" in str(error)
    
    def test_network_error_message(self) -> None:
        """Test NetworkError with message."""
        error = NetworkError("Connection refused")
        assert "Connection" in str(error)
    
    def test_sync_error_message(self) -> None:
        """Test SyncError with message."""
        error = SyncError("Sync failed")
        assert "Sync" in str(error)
    
    def test_timeout_error_message(self) -> None:
        """Test TimeoutError with message."""
        error = TimeoutError("Request timed out after 30s")
        assert "timed out" in str(error)
    
    def test_configuration_error_message(self) -> None:
        """Test ConfigurationError with message."""
        error = ConfigurationError("Invalid configuration")
        assert "configuration" in str(error).lower()


class TestErrorThrowingAndCatching:
    """Test throwing and catching error classes."""
    
    def test_catch_budget_exceeded_error(self) -> None:
        """Test catching BudgetExceededError."""
        with pytest.raises(BudgetExceededError):
            raise BudgetExceededError("Budget exceeded", spent=1.0, limit=0.5)
    
    def test_catch_policy_violation_error(self) -> None:
        """Test catching PolicyViolationError."""
        with pytest.raises(PolicyViolationError):
            raise PolicyViolationError("Policy violated")
    
    def test_catch_base_error(self) -> None:
        """Test catching specific error as base type."""
        with pytest.raises(AgentSentinelError):
            raise BudgetExceededError("Budget exceeded", spent=1.0, limit=0.5)
    
    def test_catch_network_error(self) -> None:
        """Test catching NetworkError."""
        with pytest.raises(NetworkError):
            raise NetworkError("Network failed")
    
    def test_multiple_error_types(self) -> None:
        """Test handling multiple error types."""
        errors_caught = []
        
        try:
            raise BudgetExceededError("Budget exceeded", spent=1.0, limit=0.5)
        except (BudgetExceededError, PolicyViolationError) as e:
            errors_caught.append(type(e).__name__)
        
        try:
            raise PolicyViolationError("Policy violated")
        except (BudgetExceededError, PolicyViolationError) as e:
            errors_caught.append(type(e).__name__)
        
        assert "BudgetExceededError" in errors_caught
        assert "PolicyViolationError" in errors_caught


class TestErrorContextAndAttributes:
    """Test error context and additional attributes."""
    
    def test_error_with_cause(self) -> None:
        """Test error with cause chain."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise BudgetExceededError(
                    "Budget exceeded",
                    spent=1.0,
                    limit=0.5
                ) from e
        except BudgetExceededError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
    
    def test_error_message_format(self) -> None:
        """Test error message formatting."""
        error = BudgetExceededError(
            "Action 'expensive_llm' cost $0.50 but budget is $0.25",
            spent=0.50,
            limit=0.25
        )
        error_dict = error.to_dict()
        assert error_dict["details"]["spent"] == 0.50
        assert error_dict["details"]["limit"] == 0.25
    
    def test_error_repr(self) -> None:
        """Test error representation."""
        error = PolicyViolationError("Denied action")
        repr_str = repr(error)
        assert "PolicyViolationError" in repr_str


class TestAgentSentinelInit:
    """Test agent_sentinel __init__ module."""
    
    def test_public_api_structure(self) -> None:
        """Test that public API is properly structured."""
        import agent_sentinel
        
        # Core components should be accessible
        assert hasattr(agent_sentinel, "guarded_action")
        assert hasattr(agent_sentinel, "CostTracker")
        assert hasattr(agent_sentinel, "PolicyEngine")
        assert hasattr(agent_sentinel, "Ledger")
    
    def test_all_exports_are_callable_or_classes(self) -> None:
        """Test that all public exports are callable or classes."""
        import agent_sentinel
        import inspect
        
        # Check main exports
        exports = [
            "guarded_action",
            "CostTracker",
            "PolicyEngine",
            "Ledger",
            "enable_remote_sync",
            "flush_and_stop",
        ]
        
        for export_name in exports:
            if hasattr(agent_sentinel, export_name):
                export = getattr(agent_sentinel, export_name)
                # Should be callable (function/class) or have methods
                assert callable(export) or hasattr(export, "__dict__")
    
    def test_error_imports_available(self) -> None:
        """Test that all error types are importable."""
        import agent_sentinel
        
        error_types = [
            "AgentSentinelError",
            "BudgetExceededError",
            "PolicyViolationError",
            "NetworkError",
            "SyncError",
            "TimeoutError",
            "ConfigurationError",
            "ReplayDivergenceError",
        ]
        
        for error_name in error_types:
            assert hasattr(agent_sentinel, error_name), f"{error_name} not exported"

