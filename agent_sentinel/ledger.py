"""
Ledger Module: Local file-based telemetry recording.

Phase 1 Implementation:
- Writes to local temporary directory (.agent-sentinel/)
- Uses JSONL format (newline-delimited JSON)
- Fail-open behavior: never crashes the agent if logging fails
- SafeEncoder to handle complex objects that aren't JSON serializable
- No network dependencies - completely offline
"""
from __future__ import annotations

import os
import json
import uuid
import logging
import datetime
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Constants
# By default, we write to a hidden folder in the user's current directory.
# This makes it easy to find logs: just look in .agent-sentinel/
# Falls back to system temp directory if current directory is not writable
DEFAULT_LOG_DIR = Path(".agent-sentinel")
DEFAULT_LOG_FILE = "ledger.jsonl"

logger = logging.getLogger("agent_sentinel")

class SafeEncoder(json.JSONEncoder):
    """
    A fireproof JSON encoder. 
    If an object is not serializable, it converts it to a string representation 
    instead of crashing the thread.
    """
    def default(self, obj):
        try:
            # Handle Pydantic models specifically if present
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"): # Pydantic v1 support
                return obj.dict()
            
            # Handle Dates
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
                
            # Handle UUIDs
            if isinstance(obj, uuid.UUID):
                return str(obj)

            # Attempt standard JSON serialization
            return super().default(obj)
        except Exception:
            # FALLBACK: If all else fails, return the string repr.
            # Example: <socket object at 0x102...>
            # This ensures we never raise a TypeError during logging.
            return str(obj)

class Ledger:
    """
    Local file-based ledger for recording agent actions.
    
    Phase 1: Writes to local temp file system
    - Default location: .agent-sentinel/ledger.jsonl (in current directory)
    - Fallback location: system temp directory if current dir not writable
    - Override via AGENT_SENTINEL_HOME environment variable
    
    The ledger uses JSONL (JSON Lines) format where each line is a complete
    JSON object representing one action. This format is:
    - Appendable (no need to read/parse entire file to add entries)
    - Streamable (can process line by line)
    - Recoverable (corrupted lines don't affect other entries)
    """
    _initialized = False
    _log_path: Optional[Path] = None

    @classmethod
    def setup(cls):
        """
        Idempotent setup. Creates the log directory if missing.
        
        Attempts in order:
        1. AGENT_SENTINEL_HOME env var if set
        2. .agent-sentinel/ in current working directory
        3. agent-sentinel/ in system temp directory (fallback)
        
        If all fail, disables writing (fail-open behavior).
        """
        if cls._initialized:
            return

        # Check for env var override, else use default
        base_dir_str = os.getenv("AGENT_SENTINEL_HOME")
        if base_dir_str:
            base_dir = Path(base_dir_str)
        else:
            base_dir = DEFAULT_LOG_DIR
        
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            cls._log_path = base_dir / DEFAULT_LOG_FILE
            cls._initialized = True
            logger.info(f"Agent Sentinel initialized: {cls._log_path}")
        except OSError as e:
            # Try fallback to system temp directory
            try:
                temp_dir = Path(tempfile.gettempdir()) / "agent-sentinel"
                temp_dir.mkdir(parents=True, exist_ok=True)
                cls._log_path = temp_dir / DEFAULT_LOG_FILE
                cls._initialized = True
                logger.warning(
                    f"Agent Sentinel using fallback temp directory: {cls._log_path} "
                    f"(original location {base_dir} failed: {e})"
                )
            except OSError as e2:
                # If we can't write to disk anywhere, we log to stderr and disable writing
                # so we don't spam errors on every action.
                logger.error(
                    f"Agent Sentinel disabled: Cannot create log directory. "
                    f"Tried {base_dir} and temp dir. Errors: {e}, {e2}"
                )
                cls._log_path = None  # Disable writing

    @classmethod
    def record(
        cls, 
        action: str, 
        inputs: Dict[str, Any], 
        outputs: Any, 
        cost_usd: float, 
        duration_ms: float, 
        outcome: str,
        tags: list[str],
        compliance_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Appends a structured log entry to the local JSONL file.
        
        Phase 1: Simple append to local file
        - Each entry is a complete JSON object on its own line
        - Includes timestamp, action name, cost, duration, outcome
        - Safely encodes complex objects that aren't JSON serializable
        - Fail-open: if write fails, logs error but doesn't crash
        
        Phase 5: EU Compliance
        - Optionally includes compliance_metadata for Enterprise Tier features
        - Metadata includes human approval, decision rationale, data lineage
        
        Args:
            action: Name of the action being recorded
            inputs: Dict containing args and kwargs passed to the function
            outputs: Return value or error message
            cost_usd: Cost in USD
            duration_ms: Duration in milliseconds
            outcome: "success" or "error"
            tags: List of tags for categorization
            compliance_metadata: Optional EU compliance metadata dict (Enterprise Tier)
        """
        # Lazy initialization
        if not cls._initialized:
            cls.setup()

        # If setup failed (e.g. permission error), do nothing (Fail Open)
        if not cls._log_path:
            return

        # Build the ledger entry with all metadata
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "cost_usd": cost_usd,
            "duration_ms": round(duration_ms, 3),
            "outcome": outcome,
            "tags": tags,
            # We nest inputs/outputs to keep top-level schema clean
            "payload": {
                "inputs": inputs,
                "outputs": outputs
            }
        }
        
        # Phase 5: Add compliance metadata if present (Enterprise Tier)
        if compliance_metadata:
            entry["compliance_metadata"] = compliance_metadata

        try:
            # We use 'a' (append) mode. 
            # This is generally atomic for small writes on POSIX, 
            # though extreme high concurrency might need file locking later.
            with open(cls._log_path, "a", encoding="utf-8") as f:
                # cls=SafeEncoder is the magic that prevents crashes
                json_str = json.dumps(entry, cls=SafeEncoder)
                f.write(json_str + "\n")
                
        except Exception as e:
            # Last line of defense: if disk is full or file is locked.
            # We drop the log entry rather than crashing the user's agent.
            logger.error(f"Agent Sentinel Drop: {e}")
    
    @classmethod
    def get_log_path(cls) -> Optional[Path]:
        """
        Get the current log file path.
        
        Returns:
            Path to the ledger file, or None if ledger is disabled
        """
        if not cls._initialized:
            cls.setup()
        return cls._log_path
    
    @classmethod
    def reset(cls):
        """
        Reset the ledger state. Useful for testing.
        
        Note: This does NOT delete the log file, only resets the state
        so that setup() will run again on next record() call.
        """
        cls._initialized = False
        cls._log_path = None
