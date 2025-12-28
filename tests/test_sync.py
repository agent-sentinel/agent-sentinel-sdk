"""
Tests for sync.py module (Phase 2/3: Remote Sync)

Tests cover:
- Background thread/task that periodically uploads ledger entries
- Batch upload to minimize network calls
- Retry logic with exponential backoff
- Fail-open if network unavailable
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading

from agent_sentinel.sync import (
    BackgroundSync,
    SyncConfig,
    enable_remote_sync,
    flush_and_stop,
    get_sync,
)
from agent_sentinel.ledger import Ledger


class TestSyncConfig:
    """Test SyncConfig initialization and defaults."""
    
    def test_minimal_config(self):
        """Test creating config with minimal parameters."""
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        
        assert config.platform_url == "https://api.example.com"
        assert config.api_token == "test-token"
        assert config.run_id is not None  # Auto-generated UUID
        assert config.flush_interval == 10.0
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.enabled is True
    
    def test_full_config(self):
        """Test creating config with all parameters."""
        config = SyncConfig(
            platform_url="https://api.example.com/",  # Trailing slash
            api_token="test-token",
            run_id="custom-run-id",
            flush_interval=5.0,
            batch_size=50,
            max_retries=5,
            enabled=False,
        )
        
        assert config.platform_url == "https://api.example.com"  # Trailing slash removed
        assert config.run_id == "custom-run-id"
        assert config.flush_interval == 5.0
        assert config.batch_size == 50
        assert config.max_retries == 5
        assert config.enabled is False


class TestBackgroundSync:
    """Test BackgroundSync class functionality."""
    
    @pytest.fixture
    def temp_ledger(self, tmp_path):
        """Create a temporary ledger file with test data."""
        ledger_dir = tmp_path / ".agent-sentinel"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "ledger.jsonl"
        
        # Write test entries
        entries = [
            {
                "id": f"entry-{i}",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": f"test_action_{i}",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": ["test"],
                "payload": {"inputs": {}, "outputs": {}},
            }
            for i in range(5)
        ]
        
        with open(ledger_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        # Patch Ledger.get_log_path to return our temp path
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            yield ledger_path, entries
    
    def test_sync_initialization(self):
        """Test BackgroundSync initialization."""
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        sync = BackgroundSync(config)
        
        assert sync.config == config
        assert sync._thread is None
        assert sync._last_sync_offset == 0
        assert isinstance(sync._stop_event, threading.Event)
        assert isinstance(sync._lock, type(threading.Lock()))
    
    def test_sync_disabled_without_httpx(self):
        """Test that sync is disabled if httpx is not available."""
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        
        with patch("agent_sentinel.sync.HTTPX_AVAILABLE", False):
            sync = BackgroundSync(config)
            assert sync.config.enabled is False
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    def test_start_sync(self):
        """Test starting the background sync thread."""
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        sync = BackgroundSync(config)
        
        # Mock _sync_loop to block until stop event is set
        def mock_sync_loop():
            sync._stop_event.wait()
        
        with patch.object(sync, "_sync_loop", side_effect=mock_sync_loop):
            sync.start()
            
            assert sync._thread is not None
            time.sleep(0.1)  # Give thread time to start
            assert sync._thread.is_alive()
            assert sync._thread.daemon is True
            
            # Cleanup
            sync._stop_event.set()
            sync._thread.join(timeout=1.0)
    
    def test_start_sync_disabled(self, caplog):
        """Test that start() does nothing when sync is disabled."""
        import logging
        caplog.set_level(logging.INFO, logger="agent_sentinel")
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            enabled=False,
        )
        sync = BackgroundSync(config)
        
        sync.start()
        
        assert sync._thread is None
        assert "disabled" in caplog.text.lower()
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    def test_stop_sync(self):
        """Test stopping the background sync thread."""
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        sync = BackgroundSync(config)
        
        with patch.object(sync, "_sync_loop"):
            sync.start()
            time.sleep(0.1)  # Let thread start
            
            with patch.object(sync, "_flush_once"):
                sync.stop(timeout=1.0)
            
            assert not sync._thread.is_alive()
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    def test_batch_upload(self, temp_ledger):
        """Test that entries are uploaded in batches."""
        ledger_path, entries = temp_ledger
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            batch_size=2,  # Small batch size
        )
        
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            sync = BackgroundSync(config)
            
            # Mock successful uploads
            with patch.object(sync, "_upload_batch", return_value=True) as mock_upload:
                sync._flush_once()
                
                # Should be called 3 times: 2 + 2 + 1
                assert mock_upload.call_count == 3
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_upload_success(self, mock_httpx, temp_ledger):
        """Test successful upload returns True."""
        ledger_path, entries = temp_ledger
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.text = "Accepted"
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        sync = BackgroundSync(config)
        
        result = sync._upload_batch([entries[0]])
        
        assert result is True
        mock_client.post.assert_called_once()
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_upload_auth_failure(self, mock_httpx):
        """Test that auth failures (401) don't retry."""
        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="bad-token",
            max_retries=3,
        )
        sync = BackgroundSync(config)
        
        result = sync._upload_batch([{"id": "test"}])
        
        assert result is False
        # Should only call once (no retries for auth failures)
        assert mock_client.post.call_count == 1
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    @patch("agent_sentinel.sync.time")
    def test_exponential_backoff(self, mock_time, mock_httpx):
        """Test exponential backoff on retries."""
        # Mock server error response (500)
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            max_retries=3,
        )
        sync = BackgroundSync(config)
        
        result = sync._upload_batch([{"id": "test"}])
        
        assert result is False
        # Should retry 3 times
        assert mock_client.post.call_count == 3
        
        # Check exponential backoff: 2^0=1, 2^1=2 (with jitter up to 10%)
        sleep_calls = mock_time.sleep.call_args_list
        assert len(sleep_calls) == 2  # Sleep called between retries
        assert 1.0 <= sleep_calls[0][0][0] <= 1.1  # First backoff: 2^0 = 1 + jitter
        assert 2.0 <= sleep_calls[1][0][0] <= 2.2  # Second backoff: 2^1 = 2 + jitter
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    @patch("agent_sentinel.sync.time")
    def test_exponential_backoff_max_limit(self, mock_time, mock_httpx):
        """Test that exponential backoff caps at 10 seconds."""
        # Create a proper exception class
        class MockConnectError(Exception):
            pass
        
        # Mock connection error
        mock_httpx.ConnectError = MockConnectError
        mock_client = Mock()
        mock_client.post.side_effect = MockConnectError("Connection error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            max_retries=15,  # Many retries to test cap
        )
        sync = BackgroundSync(config)
        
        result = sync._upload_batch([{"id": "test"}])
        
        assert result is False
        
        # Check that backoff never exceeds 10 seconds
        sleep_calls = mock_time.sleep.call_args_list
        for call in sleep_calls:
            assert call[0][0] <= 10
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_fail_open_on_network_error(self, mock_httpx, temp_ledger, caplog):
        """Test that sync fails open when network is unavailable."""
        ledger_path, entries = temp_ledger
        
        # Create a proper exception class
        class MockConnectError(Exception):
            pass
        
        # Mock connection error
        mock_httpx.ConnectError = MockConnectError
        mock_client = Mock()
        mock_client.post.side_effect = MockConnectError("Network unavailable")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            max_retries=2,
        )
        
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            sync = BackgroundSync(config)
            
            # This should not raise an exception (fail-open)
            sync._flush_once()
            
            # Check that error was logged (either "Cannot reach platform" or "Failed to upload")
            assert ("Cannot reach platform" in caplog.text or 
                    "Failed to upload" in caplog.text)
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    def test_tracks_uploaded_entries(self, temp_ledger):
        """Test that sync tracks which entries have been uploaded."""
        ledger_path, entries = temp_ledger
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            sync = BackgroundSync(config)
            
            # First flush
            with patch.object(sync, "_upload_batch", return_value=True) as mock_upload:
                sync._flush_once()
                first_call_count = mock_upload.call_count
                uploaded_entries = sum(len(call[0][0]) for call in mock_upload.call_args_list)
            
            # Second flush (should not upload same entries)
            with patch.object(sync, "_upload_batch", return_value=True) as mock_upload:
                sync._flush_once()
                # Should not call upload since no new entries
                assert mock_upload.call_count == 0
    
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    def test_flush_now(self, temp_ledger):
        """Test manual flush_now() method."""
        ledger_path, entries = temp_ledger
        
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
        )
        
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            sync = BackgroundSync(config)
            
            with patch.object(sync, "_flush_once") as mock_flush:
                sync.flush_now()
                mock_flush.assert_called_once()


class TestGlobalSyncHelpers:
    """Test global sync helper functions."""
    
    def test_enable_remote_sync(self):
        """Test enable_remote_sync() creates and starts global sync."""
        with patch("agent_sentinel.sync.BackgroundSync") as MockSync:
            mock_instance = MockSync.return_value
            
            result = enable_remote_sync(
                platform_url="https://api.example.com",
                api_token="test-token",
            )
            
            MockSync.assert_called_once()
            mock_instance.start.assert_called_once()
            assert result == mock_instance
    
    def test_enable_remote_sync_no_auto_start(self):
        """Test enable_remote_sync() with auto_start=False."""
        with patch("agent_sentinel.sync.BackgroundSync") as MockSync:
            mock_instance = MockSync.return_value
            
            result = enable_remote_sync(
                platform_url="https://api.example.com",
                api_token="test-token",
                auto_start=False,
            )
            
            MockSync.assert_called_once()
            mock_instance.start.assert_not_called()
            assert result == mock_instance
    
    def test_get_sync(self):
        """Test get_sync() returns global instance."""
        with patch("agent_sentinel.sync.BackgroundSync") as MockSync:
            mock_instance = MockSync.return_value
            
            enable_remote_sync(
                platform_url="https://api.example.com",
                api_token="test-token",
            )
            
            result = get_sync()
            assert result == mock_instance
    
    def test_flush_and_stop(self):
        """Test flush_and_stop() stops global sync."""
        with patch("agent_sentinel.sync.BackgroundSync") as MockSync:
            mock_instance = MockSync.return_value
            
            enable_remote_sync(
                platform_url="https://api.example.com",
                api_token="test-token",
            )
            
            flush_and_stop()
            
            mock_instance.stop.assert_called_once()
            assert get_sync() is None


class TestIntegration:
    """Integration tests for sync functionality."""
    
    @pytest.mark.integration
    @patch("agent_sentinel.sync.HTTPX_AVAILABLE", True)
    @patch("httpx.Client")
    def test_end_to_end_sync_flow(self, mock_httpx, tmp_path):
        """Test complete sync flow from ledger to platform."""
        # Setup temporary ledger
        ledger_dir = tmp_path / ".agent-sentinel"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "ledger.jsonl"
        
        # Create ledger entries
        entries = [
            {
                "id": f"entry-{i}",
                "timestamp": "2025-01-01T00:00:00Z",
                "action": f"test_action_{i}",
                "cost_usd": 0.01,
                "duration_ms": 100.0,
                "outcome": "success",
                "tags": ["test"],
                "payload": {"inputs": {}, "outputs": {}},
            }
            for i in range(3)
        ]
        
        with open(ledger_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.text = "Accepted"
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_httpx.return_value = mock_client
        
        # Run sync
        config = SyncConfig(
            platform_url="https://api.example.com",
            api_token="test-token",
            flush_interval=0.5,
        )
        
        with patch("agent_sentinel.ledger.Ledger.get_log_path", return_value=ledger_path):
            sync = BackgroundSync(config)
            sync.start()
            
            # Wait for at least one flush
            time.sleep(1.0)
            
            sync.stop()
            
            # Verify upload was called
            assert mock_client.post.called
            
            # Verify payload structure
            call_args = mock_client.post.call_args
            payload = call_args.kwargs["json"]
            assert "run_id" in payload
            assert "entries" in payload
            assert len(payload["entries"]) == 3

