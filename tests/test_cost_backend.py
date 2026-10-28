"""
Tests for pluggable cost backends.

The behaviour under test is the one that matters for any multi-process
deployment: two workers spending against the same budget must not both observe
the pre-spend total. The original in-memory singleton could not do this, which
silently disabled budget enforcement outside a single process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import Mock, patch

import pytest

from agent_sentinel.context import ExecutionContext
from agent_sentinel.cost import CostTracker
from agent_sentinel.cost_backend import (
    CostSnapshot,
    FileCostBackend,
    InMemoryCostBackend,
    RemoteCostBackend,
    backend_from_env,
)
from agent_sentinel.errors import BudgetExceededError, NetworkError
from agent_sentinel.policy import PolicyEngine


@pytest.fixture(autouse=True)
def restore_default_backend():
    """Every test starts and ends on a clean default backend."""
    CostTracker.set_backend(None)
    CostTracker.set_backend(InMemoryCostBackend())
    yield
    CostTracker.set_backend(None)


class TestCostSnapshot:
    def test_roundtrip(self):
        original = CostSnapshot(
            session_total=1.5,
            run_total=0.5,
            action_counts={"a": 2},
            action_costs={"a": 1.5},
        )
        restored = CostSnapshot.from_dict(original.to_dict())
        assert restored.session_total == 1.5
        assert restored.run_total == 0.5
        assert restored.action_counts == {"a": 2}
        assert restored.action_costs == {"a": 1.5}

    def test_from_dict_tolerates_missing_keys(self):
        snapshot = CostSnapshot.from_dict({})
        assert snapshot.session_total == 0.0
        assert snapshot.action_counts == {}


class TestInMemoryBackend:
    def test_accumulates_session_and_run(self):
        backend = InMemoryCostBackend()
        backend.add_cost("search", 0.25)
        backend.add_cost("search", 0.25)
        snapshot = backend.snapshot()
        assert snapshot.session_total == 0.5
        assert snapshot.run_total == 0.5
        assert snapshot.action_counts["search"] == 2

    def test_reset_run_preserves_session(self):
        backend = InMemoryCostBackend()
        backend.add_cost("a", 1.0)
        backend.reset_run()
        snapshot = backend.snapshot()
        assert snapshot.run_total == 0.0
        assert snapshot.session_total == 1.0

    def test_reset_all_clears_everything(self):
        backend = InMemoryCostBackend()
        backend.add_cost("a", 1.0)
        backend.reset_all()
        snapshot = backend.snapshot()
        assert snapshot.session_total == 0.0
        assert snapshot.action_counts == {}

    def test_snapshot_is_a_copy(self):
        """Mutating a returned snapshot must not corrupt backend state."""
        backend = InMemoryCostBackend()
        backend.add_cost("a", 1.0)
        snapshot = backend.snapshot()
        snapshot.session_total = 999.0
        snapshot.action_counts["a"] = 999
        assert backend.snapshot().session_total == 1.0
        assert backend.snapshot().action_counts["a"] == 1


class TestFileBackend:
    def test_persists_across_instances(self, tmp_path):
        """A new process gets a new backend object; state must survive."""
        path = tmp_path / "costs.json"
        FileCostBackend(path).add_cost("a", 0.75)

        fresh = FileCostBackend(path)
        assert fresh.snapshot().session_total == 0.75

    def test_reset_run_preserves_session(self, tmp_path):
        path = tmp_path / "costs.json"
        backend = FileCostBackend(path)
        backend.add_cost("a", 1.0)
        backend.reset_run()
        snapshot = backend.snapshot()
        assert snapshot.run_total == 0.0
        assert snapshot.session_total == 1.0

    def test_creates_parent_directories(self, tmp_path):
        backend = FileCostBackend(tmp_path / "nested" / "deep" / "costs.json")
        backend.add_cost("a", 1.0)
        assert backend.snapshot().session_total == 1.0

    def test_corrupt_file_starts_from_zero_without_raising(self, tmp_path):
        """
        A truncated or hand-edited state file must not crash the caller's
        agent. Enforcement resumes from zero rather than taking the process
        down mid-run.
        """
        path = tmp_path / "costs.json"
        path.write_text("{not valid json")
        backend = FileCostBackend(path)
        assert backend.snapshot().session_total == 0.0
        backend.add_cost("a", 1.0)
        assert backend.snapshot().session_total == 1.0

    def test_writes_are_atomic_json(self, tmp_path):
        path = tmp_path / "costs.json"
        FileCostBackend(path).add_cost("a", 0.5)
        assert json.loads(path.read_text())["session_total"] == 0.5


class TestConcurrentProcesses:
    """The regression this whole module exists to prevent."""

    def test_no_lost_updates_across_processes(self, tmp_path):
        path = tmp_path / "costs.json"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        script = textwrap.dedent(
            """
            import sys
            from agent_sentinel.cost_backend import FileCostBackend
            backend = FileCostBackend(sys.argv[1])
            for _ in range(50):
                backend.add_cost("tool_call", 0.01)
            """
        )
        script_path = tmp_path / "spender.py"
        script_path.write_text(script)

        env = dict(os.environ, PYTHONPATH=repo_root)
        procs = [
            subprocess.Popen(
                [sys.executable, str(script_path), str(path)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(6)
        ]
        for proc in procs:
            assert proc.wait(timeout=60) == 0

        snapshot = FileCostBackend(path).snapshot()
        # 6 processes x 50 writes x $0.01. Any lost update shows up here.
        assert snapshot.session_total == pytest.approx(3.0, abs=1e-9)
        assert snapshot.action_counts["tool_call"] == 300


class TestBackendFromEnv:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("AGENT_SENTINEL_COST_BACKEND", raising=False)
        assert backend_from_env() is None

    def test_memory_returns_none(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "memory")
        assert backend_from_env() is None

    def test_file_backend(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "file")
        monkeypatch.setenv("AGENT_SENTINEL_COST_PATH", str(tmp_path / "c.json"))
        assert isinstance(backend_from_env(), FileCostBackend)

    def test_file_without_path_raises(self, monkeypatch):
        """
        Silently degrading to per-process memory would disable budget
        enforcement across a worker pool with no signal. Fail loudly instead.
        """
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "file")
        monkeypatch.delenv("AGENT_SENTINEL_COST_PATH", raising=False)
        with pytest.raises(ValueError, match="AGENT_SENTINEL_COST_PATH"):
            backend_from_env()

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "redis-ish")
        with pytest.raises(ValueError, match="Unknown"):
            backend_from_env()


class TestCostTrackerDelegation:
    """The public classmethod API must be unchanged by the backend seam."""

    def test_api_parity_with_memory_backend(self):
        CostTracker.reset_all()
        CostTracker.add_cost("a", 0.5)
        CostTracker.add_cost("a", 0.25)
        CostTracker.add_cost("b", 1.0)

        assert CostTracker.get_session_total() == 1.75
        assert CostTracker.get_run_total() == 1.75
        assert CostTracker.get_action_stats("a") == {"count": 2, "total_cost": 0.75}
        assert CostTracker.get_action_stats()["counts"] == {"a": 2, "b": 1}

        CostTracker.reset_run()
        assert CostTracker.get_run_total() == 0.0
        assert CostTracker.get_session_total() == 1.75

    def test_get_action_stats_unknown_action(self):
        CostTracker.reset_all()
        assert CostTracker.get_action_stats("nope") == {"count": 0, "total_cost": 0.0}

    def test_set_backend_swaps_storage(self, tmp_path):
        CostTracker.set_backend(FileCostBackend(tmp_path / "c.json"))
        CostTracker.reset_all()
        CostTracker.add_cost("a", 2.0)
        assert CostTracker.get_session_total() == 2.0
        assert isinstance(CostTracker.get_backend(), FileCostBackend)

    def test_invalid_env_falls_back_to_memory_without_raising(self, monkeypatch, caplog):
        """
        A typo in a container env var must not crash every agent at first
        spend. It degrades to memory and logs at ERROR, because enforcement
        is genuinely weaker than the operator asked for.
        """
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "bogus")
        CostTracker.set_backend(None)
        CostTracker.add_cost("a", 1.0)
        assert CostTracker.get_session_total() == 1.0
        assert any(r.levelname == "ERROR" for r in caplog.records)


class TestCorruptAndWedgedState:
    """Regressions from the pre-landing review."""

    @pytest.mark.parametrize(
        "payload", ['[1, 2, 3]', 'null', '"a string"', '123', '{not valid json']
    )
    def test_wrong_shape_state_file_starts_from_zero(self, tmp_path, payload):
        """
        Valid JSON of the wrong shape used to raise AttributeError straight out
        of snapshot(). A hand-edited or truncated state file must degrade to
        zero, not take the caller's agent down.
        """
        path = tmp_path / "costs.json"
        path.write_text(payload)
        backend = FileCostBackend(path)
        assert backend.snapshot().session_total == 0.0
        backend.add_cost("a", 1.0)
        assert backend.snapshot().session_total == 1.0

    def test_lock_timeout_raises_instead_of_hanging(self, tmp_path):
        """A wedged lock holder must not block every other worker forever."""
        fcntl = pytest.importorskip("fcntl")
        path = tmp_path / "costs.json"
        FileCostBackend(path).add_cost("a", 0.0)

        with open(str(path) + ".lock", "a+") as holder:
            fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
            from agent_sentinel.errors import TimeoutError as SentinelTimeoutError
            # Must be the SDK's TimeoutError, not the builtin: the builtin is
            # not an AgentSentinelError and escapes every SDK handler.
            with pytest.raises(SentinelTimeoutError, match="waiting for the cost state lock"):
                FileCostBackend(path, timeout=0.2).add_cost("a", 1.0)

    def test_timeout_none_still_blocks(self, tmp_path):
        """timeout=None is the documented opt-in to waiting indefinitely."""
        backend = FileCostBackend(tmp_path / "costs.json", timeout=None)
        backend.add_cost("a", 1.0)
        assert backend.snapshot().session_total == 1.0


class TestSharedBackendDrivesEnforcement:
    """
    The counter arithmetic is not the point; blocking is. This asserts that a
    budget check in one process actually sees spend recorded by another.
    """

    def test_budget_blocks_on_spend_from_another_process(self, tmp_path):
        from agent_sentinel.errors import BudgetExceededError
        from agent_sentinel.policy import PolicyEngine

        path = tmp_path / "costs.json"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = tmp_path / "spend.py"
        script.write_text(
            "import sys\n"
            "from agent_sentinel.cost_backend import FileCostBackend\n"
            "FileCostBackend(sys.argv[1]).add_cost('t', 1.5)\n"
        )
        env = dict(os.environ, PYTHONPATH=repo_root)
        proc = subprocess.run(
            [sys.executable, str(script), str(path)],
            env=env, capture_output=True, timeout=60,
        )
        assert proc.returncode == 0, proc.stderr.decode()

        CostTracker.set_backend(FileCostBackend(path))
        PolicyEngine.reset()
        PolicyEngine.configure(run_budget=1.0)
        try:
            with pytest.raises(BudgetExceededError):
                PolicyEngine.check_action("t", cost=0.01)
        finally:
            PolicyEngine.reset()


class TestAtomicReservationHoldsTheCap:
    """
    The property that matters for a shared budget, and the one the original
    concurrency test did NOT prove: with real work between the check and the
    record, does the cap actually hold?
    """

    RACER = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from agent_sentinel.cost import CostTracker\n"
        "from agent_sentinel.cost_backend import FileCostBackend\n"
        "from agent_sentinel.policy import PolicyEngine\n"
        "from agent_sentinel.errors import BudgetExceededError\n"
        "path, mode, sync_dir, worker, workers = sys.argv[1:]\n"
        "sync_dir = Path(sync_dir)\n"
        "def wait_for(predicate):\n"
        "    deadline = time.monotonic() + 30\n"
        "    while not predicate():\n"
        "        if time.monotonic() >= deadline:\n"
        "            raise TimeoutError('process barrier timed out')\n"
        "        time.sleep(0.01)\n"
        "CostTracker.set_backend(FileCostBackend(path))\n"
        "PolicyEngine.configure(run_budget=1.00)\n"
        "(sync_dir / ('ready-' + worker)).touch()\n"
        "wait_for(lambda: (sync_dir / 'go').exists())\n"
        "try:\n"
        "    if mode == 'reserve':\n"
        "        PolicyEngine.check_and_reserve('t', 0.50)\n"
        "    else:\n"
        "        PolicyEngine.check_action('t', 0.50)\n"
        "        (sync_dir / ('checked-' + worker)).touch()\n"
        "        wait_for(lambda: len(list(sync_dir.glob('checked-*'))) == int(workers))\n"
        "        CostTracker.add_cost('t', 0.50)\n"
        "except BudgetExceededError:\n"
        "    pass\n"
    )

    def _run_race(self, tmp_path, mode, workers=6):
        import time as _time

        path = tmp_path / f"race-{mode}.json"
        script = tmp_path / "racer.py"
        script.write_text(self.RACER)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=repo_root)

        sync_dir = tmp_path / f"sync-{mode}"
        sync_dir.mkdir()
        procs = [
            subprocess.Popen(
                [sys.executable, str(script), str(path), mode, str(sync_dir), str(worker), str(workers)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            for worker in range(workers)
        ]
        try:
            deadline = _time.monotonic() + 30
            while len(list(sync_dir.glob("ready-*"))) != workers:
                assert all(proc.poll() is None for proc in procs), "worker exited before barrier"
                assert _time.monotonic() < deadline, "workers did not reach ready barrier"
                _time.sleep(0.01)
            (sync_dir / "go").touch()
            for proc in procs:
                _, err = proc.communicate(timeout=60)
                assert proc.returncode == 0, err.decode()
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()

        return FileCostBackend(path).snapshot().session_total

    def test_check_and_reserve_holds_the_cap(self, tmp_path):
        spent = self._run_race(tmp_path, "reserve")
        assert spent <= 1.0 + 1e-9, f"cap blown: spent ${spent:.2f} against $1.00"

    def test_two_step_check_overshoots(self, tmp_path):
        """
        Documents WHY check_and_reserve exists. If this ever stops overshooting
        the two-step path became atomic by other means and the reserve
        primitive may be redundant — worth knowing either way.
        """
        spent = self._run_race(tmp_path, "check_action")
        assert spent > 1.0, (
            "expected the non-atomic path to overshoot; if it no longer does, "
            "re-examine whether check_and_reserve is still needed"
        )


class TestRemoteCostBackend:
    """
    t21: RemoteCostBackend.try_reserve calls POST /api/v1/budgets/reserve
    instead of comparing against local counters. The platform is the
    enforcement authority here; these tests mock the HTTP layer rather than
    running against a live server.
    """

    @patch("httpx.Client")
    def test_successful_reservation_returns_none_and_records_locally(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(status_code=201)

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        result = backend.try_reserve("search", 0.5)

        assert result is None
        assert backend.snapshot().session_total == 0.5
        assert backend.snapshot().run_total == 0.5

    @patch("httpx.Client")
    def test_request_shape_and_auth_header(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(status_code=201)

        backend = RemoteCostBackend("https://platform.test/", "sk_test")
        backend.try_reserve("search_web", 0.02)

        args, kwargs = mock_post.call_args
        assert args[0] == "https://platform.test/api/v1/budgets/reserve"
        assert kwargs["json"]["action_name"] == "search_web"
        assert kwargs["json"]["estimated_cost_usd"] == 0.02
        assert "group_id" not in kwargs["json"]
        assert "run_id" not in kwargs["json"]
        assert kwargs["headers"]["Authorization"] == "Bearer sk_test"

    @patch("httpx.Client")
    def test_sends_group_id_and_run_id_when_set(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(status_code=201)

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        CostTracker.set_group("mission-42")
        try:
            with ExecutionContext(run_id="11111111-1111-1111-1111-111111111111"):
                backend.try_reserve("search", 0.5)
        finally:
            CostTracker.set_group(None)

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["group_id"] == "mission-42"
        assert kwargs["json"]["run_id"] == "11111111-1111-1111-1111-111111111111"

    @patch("httpx.Client")
    def test_org_budget_402_returns_rejection_without_recording(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(
            status_code=402,
            json=lambda: {
                "detail": {
                    "error": "monthly_budget_exceeded",
                    "budget_type": "organization_monthly",
                    "spent": 12.5,
                    "limit": 10.0,
                }
            },
        )

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        rejection = backend.try_reserve("search", 0.5)

        assert rejection is not None
        assert rejection.budget_type == "organization_monthly"
        assert rejection.spent == 12.5
        assert rejection.limit == 10.0
        assert backend.snapshot().session_total == 0.0

    @patch("httpx.Client")
    def test_group_budget_402_reports_group_budget_type(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(
            status_code=402,
            json=lambda: {
                "detail": {
                    "error": "group_budget_exceeded",
                    "budget_type": "group",
                    "group_id": "m-1",
                    "spent": 6.0,
                    "limit": 5.0,
                }
            },
        )

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        rejection = backend.try_reserve("search", 1.0)

        assert rejection.budget_type == "group"
        assert rejection.spent == 6.0
        assert rejection.limit == 5.0

    @patch("httpx.Client")
    def test_network_error_fails_open_by_default(self, mock_client_cls):
        mock_client_cls.side_effect = RuntimeError("connection refused")

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        result = backend.try_reserve("search", 0.5)

        assert result is None
        assert backend.snapshot().session_total == 0.5

    @patch("httpx.Client")
    def test_network_error_raises_under_fail_closed(self, mock_client_cls):
        PolicyEngine.set_enforcement_mode("fail_closed")
        mock_client_cls.side_effect = RuntimeError("connection refused")

        backend = RemoteCostBackend("https://platform.test", "sk_test")
        with pytest.raises(NetworkError):
            backend.try_reserve("search", 0.5)
        assert backend.snapshot().session_total == 0.0

    @patch("httpx.Client")
    def test_unexpected_status_code_raises_under_fail_closed(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(status_code=500)

        PolicyEngine.set_enforcement_mode("fail_closed")
        backend = RemoteCostBackend("https://platform.test", "sk_test")

        with pytest.raises(NetworkError):
            backend.try_reserve("search", 0.5)

    def test_httpx_missing_fails_open_by_default(self, monkeypatch):
        monkeypatch.setattr("agent_sentinel.cost_backend.HTTPX_AVAILABLE", False)
        backend = RemoteCostBackend("https://platform.test", "sk_test")
        assert backend.try_reserve("search", 0.5) is None
        assert backend.snapshot().session_total == 0.5

    def test_httpx_missing_raises_under_fail_closed(self, monkeypatch):
        monkeypatch.setattr("agent_sentinel.cost_backend.HTTPX_AVAILABLE", False)
        PolicyEngine.set_enforcement_mode("fail_closed")
        backend = RemoteCostBackend("https://platform.test", "sk_test")
        with pytest.raises(NetworkError):
            backend.try_reserve("search", 0.5)


class TestRemoteCostBackendFromEnv:
    def test_remote_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", "https://platform.test")
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "sk_test")

        backend = backend_from_env()
        assert isinstance(backend, RemoteCostBackend)

    def test_remote_without_platform_url_uses_hosted_default(self, monkeypatch):
        # 0.5.0: the platform URL defaults to the hosted platform; only the
        # API key is still required.
        from agent_sentinel._defaults import DEFAULT_PLATFORM_URL

        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        monkeypatch.delenv("AGENT_SENTINEL_PLATFORM_URL", raising=False)
        monkeypatch.delenv("AGENT_SENTINEL_URL", raising=False)
        monkeypatch.setenv("AGENT_SENTINEL_API_KEY", "sk_test")

        backend = backend_from_env()
        assert isinstance(backend, RemoteCostBackend)
        assert backend._platform_url == DEFAULT_PLATFORM_URL

    def test_remote_without_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("AGENT_SENTINEL_COST_BACKEND", "remote")
        monkeypatch.setenv("AGENT_SENTINEL_PLATFORM_URL", "https://platform.test")
        monkeypatch.delenv("AGENT_SENTINEL_API_KEY", raising=False)

        with pytest.raises(ValueError, match="AGENT_SENTINEL_API_KEY"):
            backend_from_env()


class TestRemoteCostBackendRoundTripWithPolicyEngine:
    """AC4: SDK backend round-trip through the public check_and_reserve path,
    against a mocked server -- not just the backend in isolation."""

    @patch("httpx.Client")
    def test_check_and_reserve_permits_on_201(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(status_code=201)

        CostTracker.set_backend(RemoteCostBackend("https://platform.test", "sk_test"))
        PolicyEngine.configure(run_budget=100.0)

        PolicyEngine.check_and_reserve("search", 0.5)  # must not raise
        assert CostTracker.get_run_total() == 0.5

    @patch("httpx.Client")
    def test_check_and_reserve_raises_budget_exceeded_on_402(self, mock_client_cls):
        mock_post = mock_client_cls.return_value.__enter__.return_value.post
        mock_post.return_value = Mock(
            status_code=402,
            json=lambda: {
                "detail": {
                    "budget_type": "organization_monthly",
                    "spent": 12.5,
                    "limit": 10.0,
                }
            },
        )

        CostTracker.set_backend(RemoteCostBackend("https://platform.test", "sk_test"))
        PolicyEngine.configure(run_budget=100.0)

        with pytest.raises(BudgetExceededError) as exc:
            PolicyEngine.check_and_reserve("search", 1.0)

        assert exc.value.spent == 12.5
        assert exc.value.limit == 10.0
        assert CostTracker.get_run_total() == 0.0
