"""
Sync Module: Background flusher for sending ledger data to platform.

Phase 3 Implementation:
- Background thread that periodically flushes logs to platform
- Batch uploads to minimize network calls
- Retry logic with exponential backoff
- Fail-open: never crashes agent if platform unavailable
- Tracks uploaded entries to avoid duplicates
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
import random
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .ledger import Ledger
from .errors import NetworkError, SyncError, ConfigurationError
from .retry import RetryConfig, with_retry
from .intervention import InterventionTracker

logger = logging.getLogger("agent_sentinel")


class SyncConfig:
    """Configuration for platform sync."""
    
    def __init__(
        self,
        platform_url: str,
        api_token: str,
        run_id: Optional[str] = None,
        flush_interval: float = 10.0,
        batch_size: int = 100,
        max_retries: int = 3,
        enabled: bool = True,
    ):
        """
        Configure sync to platform.
        
        Args:
            platform_url: Base URL of platform (e.g. "https://api.agentsentinel.dev")
            api_token: JWT token or API key for authentication
            run_id: UUID for this agent run (auto-generated if None)
            flush_interval: Seconds between flushes (default 10)
            batch_size: Max entries per batch (default 100)
            max_retries: Max retry attempts per batch (default 3)
            enabled: Enable/disable sync (default True)
        """
        self.platform_url = platform_url.rstrip("/")
        self.api_token = api_token
        self.run_id = run_id or str(uuid.uuid4())
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.enabled = enabled


class BackgroundSync:
    """
    Background thread that periodically uploads ledger entries to platform.
    
    Usage:
        # Configure and start
        sync = BackgroundSync(SyncConfig(
            platform_url="https://api.agentsentinel.dev",
            api_token="your-jwt-token",
        ))
        sync.start()
        
        # Use your agent normally - logs are automatically synced
        
        # Graceful shutdown
        sync.stop()
    """
    
    def __init__(self, config: SyncConfig):
        """
        Initialize background sync.
        
        Args:
            config: SyncConfig instance
        """
        if not HTTPX_AVAILABLE:
            logger.warning(
                "httpx not installed. Remote sync disabled. "
                "Install with: pip install agent-sentinel[remote]"
            )
            config.enabled = False
        
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_sync_offset = 0  # Track how many lines we've uploaded
        self._last_intervention_sync_offset = 0  # Track interventions file position
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start the background sync thread."""
        if not self.config.enabled:
            logger.info("Remote sync is disabled")
            return
        
        if self._thread and self._thread.is_alive():
            logger.warning("Sync thread already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Background sync started: {self.config.platform_url}, "
            f"run_id={self.config.run_id}, interval={self.config.flush_interval}s"
        )
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the background sync thread gracefully.
        
        Args:
            timeout: Max seconds to wait for thread to finish
        """
        if not self._thread or not self._thread.is_alive():
            return
        
        logger.info("Stopping background sync...")
        self._stop_event.set()
        
        # Do one final flush before stopping
        try:
            self._flush_once()
        except Exception as e:
            logger.error(f"Final flush failed: {e}")
        
        self._thread.join(timeout=timeout)
        
        if self._thread.is_alive():
            logger.warning("Sync thread did not stop gracefully")
        else:
            logger.info("Background sync stopped")
    
    def flush_now(self) -> None:
        """
        Trigger an immediate flush (blocks until complete).
        
        Useful for ensuring logs are uploaded before agent exits.
        """
        if not self.config.enabled:
            return
        
        try:
            self._flush_once()
        except Exception as e:
            logger.error(f"Manual flush failed: {e}")
    
    def _sync_loop(self) -> None:
        """Main loop: flush periodically until stopped."""
        while not self._stop_event.is_set():
            try:
                self._flush_once()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
            
            # Wait for next interval (or until stopped)
            self._stop_event.wait(timeout=self.config.flush_interval)
    
    def _flush_once(self) -> None:
        """
        Read new ledger entries and upload them to platform.
        
        This is called periodically by the sync loop.
        """
        with self._lock:
            # Sync ledger entries
            ledger_path = Ledger.get_log_path()
            if not ledger_path or not ledger_path.exists():
                pass  # No ledger yet
            else:
                self._sync_ledger_entries(ledger_path)
            
            # Sync intervention records (CRITICAL - Core Value Proposition)
            self._sync_interventions()
            
    def _sync_ledger_entries(self, ledger_path: Path) -> None:
        """Sync ledger entries to platform."""
        # Read new entries (skip already-uploaded ones)
        entries = []
        try:
            with open(ledger_path, "r") as f:
                lines = f.readlines()
                new_lines = lines[self._last_sync_offset:]
                
                for line in new_lines:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON: {line[:50]}...")
        except Exception as e:
            logger.error(f"Failed to read ledger: {e}")
            return
        
        if not entries:
            return  # Nothing to sync
        
        # Upload in batches
        uploaded = 0
        for i in range(0, len(entries), self.config.batch_size):
            batch = entries[i:i + self.config.batch_size]
            
            if self._upload_batch(batch):
                uploaded += len(batch)
                self._last_sync_offset += len(batch)
            else:
                # Stop on first failed batch (will retry next interval)
                break
        
        if uploaded > 0:
            logger.info(f"Synced {uploaded} ledger entries to platform")
    
    def _sync_interventions(self) -> None:
        """
        Sync intervention records to platform.
        
        This is CRITICAL - interventions are the core value proposition.
        """
        # Initialize intervention tracker if needed
        if not hasattr(InterventionTracker, '_intervention_file'):
            InterventionTracker.initialize()
        
        intervention_file = InterventionTracker._intervention_file
        if not intervention_file or not intervention_file.exists():
            return  # No interventions yet
        
        # Read new intervention records
        interventions = []
        try:
            with open(intervention_file, "r") as f:
                lines = f.readlines()
                new_lines = lines[self._last_intervention_sync_offset:]
                
                for line in new_lines:
                    if line.strip():
                        try:
                            interventions.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid intervention JSON: {line[:50]}...")
        except Exception as e:
            logger.error(f"Failed to read interventions: {e}")
            return
        
        if not interventions:
            return  # Nothing to sync
        
        # Upload in batches
        uploaded = 0
        for i in range(0, len(interventions), self.config.batch_size):
            batch = interventions[i:i + self.config.batch_size]
            
            if self._upload_intervention_batch(batch):
                uploaded += len(batch)
                self._last_intervention_sync_offset += len(batch)
            else:
                # Stop on first failed batch (will retry next interval)
                break
        
        if uploaded > 0:
            logger.info(f"Synced {uploaded} interventions to platform")
    
    def _upload_intervention_batch(self, interventions: list[dict]) -> bool:
        """
        Upload a batch of interventions to platform.
        
        Args:
            interventions: List of intervention records (dicts)
        
        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE or not interventions:
            return False if not interventions else True
        
        url = f"{self.config.platform_url}/api/v1/interventions/"
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json",
        }
        
        # Upload each intervention individually (they have unique IDs)
        # In the future, we could batch this into a single endpoint
        success_count = 0
        for intervention in interventions:
            try:
                if not HTTPX_AVAILABLE:
                    return False
                
                import httpx
                response = httpx.post(
                    url,
                    headers=headers,
                    json=intervention,
                    timeout=10.0
                )
                
                if response.status_code in [200, 201]:
                    success_count += 1
                else:
                    logger.warning(f"Failed to upload intervention: {response.status_code}")
                    break  # Stop on first failure
            
            except Exception as e:
                logger.error(f"Error uploading intervention: {e}")
                break
        
        return success_count == len(interventions)
    
    def _upload_batch(self, entries: list[dict]) -> bool:
        """
        Upload a batch of entries to platform.
        
        Args:
            entries: List of ledger entries (dicts)
        
        Returns:
            True if successful, False otherwise
        """
        if not HTTPX_AVAILABLE:
            return False
        
        if not entries:
            return True  # Empty batch is "successful"
        
        url = f"{self.config.platform_url}/api/v1/ingest"
        headers = {
            "Authorization": f"ApiKey {self.config.api_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "run_id": self.config.run_id,
            "entries": entries,
        }
        
        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._make_upload_request(url, headers, payload)
                
                if response.status_code == 202:
                    # Success - entries accepted
                    logger.debug(f"Successfully uploaded {len(entries)} entries")
                    return True
                
                elif response.status_code == 401:
                    # Auth failure - don't retry
                    error = NetworkError(
                        f"Authentication failed: {response.text}",
                        status_code=401,
                        endpoint=url,
                    )
                    logger.error(f"Authentication failed: {error.message}")
                    return False
                
                elif response.status_code == 400:
                    # Client error - don't retry
                    error = NetworkError(
                        f"Invalid request: {response.text}",
                        status_code=400,
                        endpoint=url,
                    )
                    logger.error(f"Invalid request: {error.message}")
                    return False
                
                elif 500 <= response.status_code < 600:
                    # Server error - retry
                    error = NetworkError(
                        f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                        details={"response_text": response.text[:200]},
                    )
                    last_error = error
                    if attempt < self.config.max_retries - 1:
                        wait = self._calculate_backoff(attempt)
                        logger.warning(
                            f"Server error {response.status_code}, "
                            f"retry {attempt + 1}/{self.config.max_retries} in {wait:.2f}s"
                        )
                        time.sleep(wait)
                else:
                    # Other status - don't retry
                    error = NetworkError(
                        f"Unexpected status: {response.status_code}",
                        status_code=response.status_code,
                        endpoint=url,
                    )
                    logger.error(f"Upload failed: {error.message}")
                    return False
            
            except httpx.ConnectError as e:
                # Network connectivity issue - retry
                error = NetworkError(
                    f"Cannot reach platform: {str(e)}",
                    endpoint=url,
                )
                last_error = error
                if attempt < self.config.max_retries - 1:
                    wait = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Connection failed, retry {attempt + 1}/{self.config.max_retries} in {wait:.2f}s"
                    )
                    time.sleep(wait)
            
            except httpx.TimeoutException as e:
                # Request timed out - retry
                error = NetworkError(
                    f"Request timeout: {str(e)}",
                    endpoint=url,
                )
                last_error = error
                if attempt < self.config.max_retries - 1:
                    wait = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Request timeout, retry {attempt + 1}/{self.config.max_retries} in {wait:.2f}s"
                    )
                    time.sleep(wait)
            
            except Exception as e:
                # Unexpected error - log and retry
                error = SyncError(
                    f"Unexpected upload error: {str(e)}",
                    batch_size=len(entries),
                    retry_count=attempt + 1,
                )
                last_error = error
                if attempt < self.config.max_retries - 1:
                    wait = self._calculate_backoff(attempt)
                    logger.error(
                        f"Unexpected error, retry {attempt + 1}/{self.config.max_retries} in {wait:.2f}s: {e}"
                    )
                    time.sleep(wait)
        
        # All retries exhausted
        error_msg = f"Failed to upload {len(entries)} entries after {self.config.max_retries} attempts"
        if last_error:
            error_msg += f": {last_error.message}"
        logger.error(error_msg)
        
        # Raise error for monitoring/alerting purposes (won't crash agent due to fail-open)
        if last_error:
            logger.debug(f"Last error details: {last_error.to_dict()}")
        
        return False
    
    def _make_upload_request(
        self, url: str, headers: dict, payload: dict
    ) -> httpx.Response:
        """
        Make HTTP request with timeout.
        
        Args:
            url: Upload endpoint URL
            headers: HTTP headers
            payload: Request payload
            
        Returns:
            HTTP response
            
        Raises:
            httpx.ConnectError: If connection fails
            httpx.TimeoutException: If request times out
        """
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            return client.post(url, json=payload, headers=headers)
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.
        
        Args:
            attempt: Zero-based attempt number
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: 2^attempt with jitter, capped at 10s
        base_wait = 2 ** attempt
        jitter = random.uniform(0, base_wait * 0.1)  # +10% jitter
        return min(base_wait + jitter, 10.0)


# Global sync instance (optional - users can manage their own)
_global_sync: Optional[BackgroundSync] = None


def enable_remote_sync(
    platform_url: str,
    api_token: str,
    run_id: Optional[str] = None,
    flush_interval: float = 10.0,
    auto_start: bool = True,
) -> BackgroundSync:
    """
    Enable remote sync to platform (convenience function).
    
    This configures and starts a global background sync instance.
    
    Args:
        platform_url: Base URL of platform
        api_token: JWT token or API key
        run_id: Optional run ID (auto-generated if None)
        flush_interval: Seconds between flushes
        auto_start: Start sync immediately
    
    Returns:
        BackgroundSync instance
    
    Example:
        from agent_sentinel import enable_remote_sync
        
        sync = enable_remote_sync(
            platform_url="https://api.agentsentinel.dev",
            api_token="your-jwt-token",
        )
        
        # Use your agent - logs are synced automatically
        
        # At exit:
        sync.stop()
    """
    global _global_sync
    
    config = SyncConfig(
        platform_url=platform_url,
        api_token=api_token,
        run_id=run_id,
        flush_interval=flush_interval,
    )
    
    _global_sync = BackgroundSync(config)
    
    if auto_start:
        _global_sync.start()
    
    return _global_sync


def get_sync() -> Optional[BackgroundSync]:
    """
    Get the global sync instance.
    
    Returns:
        Global BackgroundSync instance or None if not configured
    """
    return _global_sync


def flush_and_stop() -> None:
    """
    Flush remaining logs and stop global sync.
    
    Call this before your agent exits to ensure all logs are uploaded.
    """
    global _global_sync
    if _global_sync:
        _global_sync.stop()
        _global_sync = None

