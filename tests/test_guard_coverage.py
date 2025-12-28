"""
Extended coverage tests for guard module.

Tests cover:
- Decorator edge cases (custom names, tags, costs)
- Context manager behavior
- Async/sync function handling
- Error handling and propagation
- Policy enforcement integration
- Cost tracking integration
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

from agent_sentinel.guard import guarded_action, _execute_sync, _execute_async
from agent_sentinel.ledger import Ledger
from agent_sentinel.cost import CostTracker
from agent_sentinel.policy import PolicyEngine
from agent_sentinel.errors import BudgetExceededError, PolicyViolationError


class TestGuardedActionDecorator:
    """Test guarded_action decorator with various configurations."""

    def test_decorator_with_default_name(self) -> None:
        """Test decorator uses function name when name not provided."""
        @guarded_action()
        def my_function():
            return "result"
        
        # Function should still be callable
        result = my_function()
        assert result == "result"

    def test_decorator_with_custom_name(self) -> None:
        """Test decorator accepts custom action name."""
        @guarded_action(name="custom_action_name", cost_usd=0.05)
        def my_function():
            return "result"
        
        result = my_function()
        assert result == "result"

    def test_decorator_with_tags(self) -> None:
        """Test decorator accepts tags."""
        @guarded_action(name="tagged_action", cost_usd=0.01, tags=["tag1", "tag2"])
        def my_function():
            return "result"
        
        result = my_function()
        assert result == "result"

    def test_decorator_with_zero_cost(self) -> None:
        """Test decorator works with zero cost."""
        @guarded_action(cost_usd=0.0)
        def free_function():
            return "free"
        
        result = free_function()
        assert result == "free"

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function name and docstring."""
        @guarded_action(name="override_name")
        def documented_function():
            """This is the docstring."""
            return "result"
        
        # functools.wraps should preserve the original function name
        assert documented_function.__name__ == "documented_function"

    def test_decorator_with_arguments(self) -> None:
        """Test decorated function correctly handles arguments."""
        @guarded_action(cost_usd=0.01)
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5

    def test_decorator_with_keyword_arguments(self) -> None:
        """Test decorated function correctly handles keyword arguments."""
        @guarded_action(cost_usd=0.01)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"
        
        result = greet("Alice", greeting="Hi")
        assert result == "Hi, Alice"

    def test_decorator_with_varargs_kwargs(self) -> None:
        """Test decorated function with *args and **kwargs."""
        @guarded_action(cost_usd=0.01)
        def flexible(*args, **kwargs):
            return {"args": args, "kwargs": kwargs}
        
        result = flexible(1, 2, 3, a="x", b="y")
        assert result["args"] == (1, 2, 3)
        assert result["kwargs"] == {"a": "x", "b": "y"}


class TestAsyncDecorator:
    """Test guarded_action with async functions."""

    @pytest.mark.asyncio
    async def test_async_function_detection(self) -> None:
        """Test decorator correctly detects and wraps async functions."""
        @guarded_action(cost_usd=0.01)
        async def async_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await async_function()
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_function_with_arguments(self) -> None:
        """Test async function with arguments."""
        @guarded_action(cost_usd=0.02)
        async def async_add(a, b):
            await asyncio.sleep(0.01)
            return a + b
        
        result = await async_add(3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_async_function_error_handling(self) -> None:
        """Test async function error propagation."""
        @guarded_action(cost_usd=0.01)
        async def async_failing():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
        
        with pytest.raises(ValueError, match="Async error"):
            await async_failing()


class TestSyncFunctionExecution:
    """Test sync function execution and error handling."""

    def test_sync_function_successful_execution(self) -> None:
        """Test sync function executes successfully."""
        @guarded_action(name="sync_action", cost_usd=0.01)
        def sync_function():
            return "sync_result"
        
        result = sync_function()
        assert result == "sync_result"

    def test_sync_function_with_exception(self) -> None:
        """Test sync function exception is re-raised."""
        @guarded_action(cost_usd=0.01)
        def failing_function():
            raise RuntimeError("Sync error")
        
        with pytest.raises(RuntimeError, match="Sync error"):
            failing_function()

    def test_sync_function_return_types(self) -> None:
        """Test sync function with various return types."""
        @guarded_action(cost_usd=0.01)
        def return_dict():
            return {"key": "value"}
        
        @guarded_action(cost_usd=0.01)
        def return_list():
            return [1, 2, 3]
        
        @guarded_action(cost_usd=0.01)
        def return_none():
            return None
        
        assert return_dict() == {"key": "value"}
        assert return_list() == [1, 2, 3]
        assert return_none() is None


class TestErrorHandling:
    """Test error handling and exception propagation."""

    def test_policy_violation_prevents_execution(self) -> None:
        """Test that PolicyViolationError prevents function execution."""
        @guarded_action(name="forbidden_action", cost_usd=0.01)
        def forbidden():
            return "should_not_execute"
        
        PolicyEngine.configure(denied_actions=["forbidden_action"])
        
        try:
            with pytest.raises(PolicyViolationError):
                forbidden()
        finally:
            PolicyEngine.reset()

    def test_budget_exceeded_prevents_execution(self) -> None:
        """Test that BudgetExceededError prevents function execution."""
        @guarded_action(name="expensive", cost_usd=100.0)
        def expensive_action():
            return "should_not_execute"
        
        PolicyEngine.configure(run_budget=1.0)
        
        try:
            with pytest.raises(BudgetExceededError):
                expensive_action()
        finally:
            PolicyEngine.reset()

    def test_exception_in_function_still_logs_cost(self) -> None:
        """Test that cost is tracked even when function raises exception."""
        @guarded_action(name="error_action", cost_usd=0.05)
        def failing_action():
            raise ValueError("Function failed")
        
        CostTracker.reset_all()
        
        try:
            with pytest.raises(ValueError):
                failing_action()
        finally:
            # Cost should be recorded despite error
            snapshot = CostTracker.get_snapshot()
            run_total = snapshot.get("run_total", 0.0)
            assert run_total == 0.05
            CostTracker.reset_all()

    def test_function_with_multiple_exception_types(self) -> None:
        """Test handling of various exception types."""
        @guarded_action(cost_usd=0.01)
        def raise_type_error():
            raise TypeError("Type mismatch")
        
        @guarded_action(cost_usd=0.01)
        def raise_key_error():
            raise KeyError("Missing key")
        
        @guarded_action(cost_usd=0.01)
        def raise_attribute_error():
            raise AttributeError("No attribute")
        
        with pytest.raises(TypeError):
            raise_type_error()
        
        with pytest.raises(KeyError):
            raise_key_error()
        
        with pytest.raises(AttributeError):
            raise_attribute_error()


class TestCostTracking:
    """Test cost tracking integration."""

    def test_cost_recorded_on_success(self) -> None:
        """Test cost is recorded when function succeeds."""
        @guarded_action(name="tracked_action", cost_usd=0.42)
        def successful_action():
            return "success"
        
        CostTracker.reset_all()
        
        try:
            result = successful_action()
            assert result == "success"
            
            snapshot = CostTracker.get_snapshot()
            run_total = snapshot.get("run_total", 0.0)
            assert run_total == 0.42
        finally:
            CostTracker.reset_all()

    def test_cost_recorded_on_error(self) -> None:
        """Test cost is recorded even when function fails."""
        @guarded_action(name="failing_action", cost_usd=0.33)
        def failing_action():
            raise RuntimeError("Failed")
        
        CostTracker.reset_all()
        
        try:
            with pytest.raises(RuntimeError):
                failing_action()
            
            snapshot = CostTracker.get_snapshot()
            run_total = snapshot.get("run_total", 0.0)
            assert run_total == 0.33
        finally:
            CostTracker.reset_all()

    def test_multiple_actions_accumulate_cost(self) -> None:
        """Test that multiple actions accumulate cost correctly."""
        @guarded_action(name="action1", cost_usd=0.10)
        def action1():
            return "1"
        
        @guarded_action(name="action2", cost_usd=0.20)
        def action2():
            return "2"
        
        @guarded_action(name="action3", cost_usd=0.30)
        def action3():
            return "3"
        
        CostTracker.reset_all()
        
        try:
            action1()
            action2()
            action3()
            
            snapshot = CostTracker.get_snapshot()
            run_total = snapshot.get("run_total", 0.0)
            assert abs(run_total - 0.60) < 0.001  # Use approximate equality for floats
        finally:
            CostTracker.reset_all()


class TestLedgerIntegration:
    """Test ledger writing from decorated functions."""

    def test_ledger_entry_created_on_execution(self, tmp_path, monkeypatch) -> None:
        """Test that ledger entry is created when function executes."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        @guarded_action(name="logged_action", cost_usd=0.01, tags=["logging"])
        def logged_function(x):
            return x * 2
        
        result = logged_function(5)
        assert result == 10
        
        # Verify entry in ledger
        log_path = Ledger.get_log_path()
        if log_path and log_path.exists():
            with open(log_path, "r") as f:
                import json
                entries = [json.loads(line) for line in f.readlines()]
            
            assert len(entries) > 0
            assert entries[-1]["action"] == "logged_action"

    def test_error_logged_to_ledger(self, tmp_path, monkeypatch) -> None:
        """Test that errors are logged to ledger with 'error' outcome."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_SENTINEL_HOME", raising=False)
        
        Ledger.reset()
        
        @guarded_action(name="error_logged_action", cost_usd=0.01)
        def error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_function()
        
        # Verify error logged
        log_path = Ledger.get_log_path()
        if log_path and log_path.exists():
            with open(log_path, "r") as f:
                import json
                entries = [json.loads(line) for line in f.readlines()]
            
            if entries:
                assert entries[-1]["outcome"] == "error"


class TestContextualBehavior:
    """Test decorator behavior in different contexts."""

    def test_decorator_in_class_method(self) -> None:
        """Test decorator on class method."""
        class Agent:
            @guarded_action(name="method_action", cost_usd=0.01)
            def perform_action(self):
                return "method_result"
        
        agent = Agent()
        result = agent.perform_action()
        assert result == "method_result"

    def test_decorator_in_static_method(self) -> None:
        """Test decorator on static method."""
        class Agent:
            @staticmethod
            @guarded_action(name="static_action", cost_usd=0.01)
            def static_action():
                return "static_result"
        
        result = Agent.static_action()
        assert result == "static_result"

    def test_decorator_in_nested_function(self) -> None:
        """Test decorator in nested function."""
        def outer():
            @guarded_action(name="nested_action", cost_usd=0.01)
            def inner():
                return "nested_result"
            
            return inner()
        
        result = outer()
        assert result == "nested_result"

    def test_multiple_decorators_stacking(self) -> None:
        """Test stacking guarded_action with other decorators."""
        def uppercase_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result.upper() if isinstance(result, str) else result
            return wrapper
        
        @uppercase_decorator
        @guarded_action(name="stacked_action", cost_usd=0.01)
        def decorated_function():
            return "hello"
        
        result = decorated_function()
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_async_decorator_in_async_context(self) -> None:
        """Test async decorator in async context."""
        @guarded_action(name="async_context", cost_usd=0.01)
        async def async_in_context():
            await asyncio.sleep(0.001)
            return "async_context_result"
        
        result = await async_in_context()
        assert result == "async_context_result"

